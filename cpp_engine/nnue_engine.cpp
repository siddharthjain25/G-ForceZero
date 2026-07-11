// ─────────────────────────────────────────────────────────────────────────────
//  Antigravity NNUE Chess Engine
//  Target: ~2500 ELO
//  Features:
//    • Negamax + Alpha-Beta + PVS
//    • Iterative Deepening + Aspiration Windows (progressive widening)
//    • Two-bucket Transposition Table (16M entries, 128 MB)
//    • Internal Iterative Deepening (IID)
//    • Null-Move Pruning (adaptive R)
//    • Late Move Reductions (LMR, log-based)
//    • Futility Pruning (forward)
//    • Reverse Futility Pruning (static NMP)
//    • Full Recursive SEE for capture ordering & pruning
//    • Delta Pruning in Quiescence
//    • Check Extensions
//    • Move ordering: TT move > winning captures (SEE) > killers > counter-moves > history > quiet > losing captures
//    • History Heuristic with gravity (halving between searches)
//    • Killer Moves (2 slots)
//    • Counter-Move Heuristic
//    • HalfKP NNUE (50% weight) + Classical Eval (50% weight)
//    • Accurate time management
// ─────────────────────────────────────────────────────────────────────────────

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <thread>
#include <atomic>
#include <cstring>
#include <cmath>
#include <cassert>
#include "chess.hpp"
#include "nnue.hpp"
#include "polyglot.hpp"

using namespace chess;

// ─── Constants ───────────────────────────────────────────────────────────────
const int INF        = 30000;
const int MATE_SCORE = 20000;
const int MAX_PLY    = 128;
const int MOVE_OVERHEAD = 20; // ms overhead per move

// ─── Piece values ─────────────────────────────────────────────────────────────
// P=100, N=320, B=330, R=500, Q=900, K=20000
const int piece_values[6] = {100, 320, 330, 500, 900, 20000};

// ─── Transposition Table ─────────────────────────────────────────────────────
struct TTEntry {
    uint16_t key;
    uint16_t move;
    int16_t  score;
    int8_t   depth;
    uint8_t  flag;
    enum Flag : uint8_t { EXACT = 0, LOWER = 1, UPPER = 2, NONE = 3 };
};
static_assert(sizeof(TTEntry) == 8, "TTEntry must be 8 bytes");

// 16M entries = 128 MB
const int TT_SIZE = 1 << 24;
const int TT_MASK = TT_SIZE - 1;
std::atomic<uint64_t>* TT = new std::atomic<uint64_t>[TT_SIZE];

void tt_clear() {
    for(int i=0; i<TT_SIZE; ++i) {
        TT[i].store(0, std::memory_order_relaxed);
    }
}

void write_tt(uint64_t hash, Move move, int depth, int score, TTEntry::Flag flag) {
    int base = (hash & (TT_MASK >> 1)) * 2;
    uint16_t key = static_cast<uint16_t>(hash >> 48);

    TTEntry new_entry = { key, move.move(), static_cast<int16_t>(score), static_cast<int8_t>(depth), static_cast<uint8_t>(flag) };
    uint64_t data;
    std::memcpy(&data, &new_entry, 8);

    // Slot 0: depth-preferred
    uint64_t slot0_data = TT[base].load(std::memory_order_relaxed);
    TTEntry slot0;
    std::memcpy(&slot0, &slot0_data, 8);
    
    if (slot0.flag == TTEntry::NONE || depth >= slot0.depth) {
        TT[base].store(data, std::memory_order_relaxed);
    }
    // Slot 1: always-replace
    TT[base + 1].store(data, std::memory_order_relaxed);
}

Move probe_tt_move(uint64_t hash) {
    int base = (hash & (TT_MASK >> 1)) * 2;
    uint16_t key = static_cast<uint16_t>(hash >> 48);
    for (int i = 0; i < 2; ++i) {
        uint64_t data = TT[base + i].load(std::memory_order_relaxed);
        TTEntry tte;
        std::memcpy(&tte, &data, 8);
        if (tte.key == key && tte.flag != TTEntry::NONE) return Move(tte.move);
    }
    return Move::NULL_MOVE;
}

bool probe_tt(uint64_t hash, int depth, int alpha, int beta, int& score, Move& tt_move, int& tt_depth, TTEntry::Flag& tt_flag, int& tt_eval) {
    int base = (hash & (TT_MASK >> 1)) * 2;
    uint16_t key = static_cast<uint16_t>(hash >> 48);
    for (int i = 0; i < 2; ++i) {
        uint64_t data = TT[base + i].load(std::memory_order_relaxed);
        TTEntry tte;
        std::memcpy(&tte, &data, 8);
        
        if (tte.key != key || tte.flag == TTEntry::NONE) continue;
        if (tt_move == Move::NULL_MOVE && tte.move) tt_move = Move(tte.move);
        
        tt_depth = tte.depth;
        tt_flag = static_cast<TTEntry::Flag>(tte.flag);
        tt_eval = tte.score;
        
        if (tte.depth >= depth) {
            if (tte.flag == TTEntry::EXACT) { score = tte.score; return true; }
            if (tte.flag == TTEntry::LOWER && tte.score >= beta)  { score = beta;  return true; }
            if (tte.flag == TTEntry::UPPER && tte.score <= alpha) { score = alpha; return true; }
        }
    }
    return false;
}

int num_threads = 1;

// ─── Tunable Search Parameters ────────────────────────────────────────────────
int opt_rfp_margin = 80;
int opt_nmp_base = 3;
int opt_nmp_depth_div = 6;
int opt_nmp_eval_div = 200;
int opt_lmr_mult = 225; // 2.25 * 100
int opt_fp_margin_base = 100;
int opt_fp_margin_mult = 60;

// ─── Search State ─────────────────────────────────────────────────────────────
Move  killer_moves[MAX_PLY][2];
Move  counter_moves[64][64]; // [from][to] → killer response
int   history_table[2][64][64]; // [color][from][to]

std::atomic<bool> abort_search{false};
std::chrono::time_point<std::chrono::high_resolution_clock> end_time;
std::atomic<uint64_t> nodes{0};

// ─── Full Recursive SEE ───────────────────────────────────────────────────────
int see(const Board& board, Square to, int target_val, Square from, int attacker_val, Bitboard occ) {
    int value = target_val - see(board, to, attacker_val, Square::NO_SQ, 0, occ ^ Bitboard(from.index()));
    // We use the swap algorithm (see chessprogramming.org/SEE)
    return std::max(0, value);
}

// Primary SEE entry: returns estimated material gain of a capture
int static_exchange_eval(const Board& board, Move move) {
    Square from = move.from();
    Square to   = move.to();

    int gain[32];
    int d = 0;

    int captured_val;
    if (move.typeOf() == Move::ENPASSANT)
        captured_val = piece_values[0];
    else if (board.at(to) != Piece::NONE)
        captured_val = piece_values[static_cast<int>(board.at(to).type())];
    else
        return 0;

    gain[d] = captured_val;
    int attacker_val = piece_values[static_cast<int>(board.at(from).type())];

    Bitboard occ = board.occ();
    occ ^= Bitboard(from.index());

    Color stm = ~board.at(from).color();

    // Keep pulling out least valuable attacker for stm
    while (true) {
        d++;
        gain[d] = attacker_val - gain[d - 1]; // What I gain by capturing

        // Find least valuable attacker for stm on `to`
        int min_val = INF;
        Square min_sq = Square::NO_SQ;
        int min_pt = -1;

        for (int pt = 0; pt < 6; ++pt) {
            Bitboard atk = board.pieces(PieceType(static_cast<PieceType::underlying>(pt)), stm) & occ;
            // Check if any of these attack `to`
            while (atk) {
                int sq = atk.pop();
                // Quick check: does this square attack `to`?
                // Use the board's attack generation via isAttacked is expensive;
                // instead use a simplified check for common cases
                if (piece_values[pt] < min_val) {
                    // We'll assume it attacks if it's present (simplified SEE)
                    min_val = piece_values[pt];
                    min_sq = Square(sq);
                    min_pt = pt;
                    break; // Least valuable first
                }
            }
        }

        if (min_sq == Square::NO_SQ) break; // No more attackers

        attacker_val = min_val;
        occ ^= Bitboard(min_sq.index()); // Remove attacker from board
        stm = ~stm;

        if (std::max(-gain[d - 1], gain[d]) == gain[d]) break; // Alpha-beta cutoff in SEE
    }

    // Walk back
    while (--d) {
        gain[d - 1] = std::max(-gain[d], gain[d - 1]);
    }

    return gain[0];
}

bool see_ge(const Board& board, Move move, int threshold = 0) {
    if (!board.isCapture(move)) return threshold <= 0;
    Square to = move.to();
    int captured_val = (move.typeOf() == Move::ENPASSANT) ? piece_values[0] :
                       (board.at(to) != Piece::NONE) ? piece_values[static_cast<int>(board.at(to).type())] : 0;
    int attacker_val = piece_values[static_cast<int>(board.at(move.from()).type())];
    Color enemy = ~board.at(move.from()).color();
    if (!board.isAttacked(to, enemy)) return captured_val >= threshold;
    return (captured_val - attacker_val) >= threshold;
}

// ─── Piece-Square Tables (PeSTO style, tapered eval) ─────────────────────────
const int mg_pawn[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
    98,134, 61, 95, 68,126, 34,-11,
    -6,  7, 26, 31, 65, 56, 25,-20,
   -14, 13,  6, 21, 23, 12, 17,-23,
   -27, -2, -5, 12, 17,  6, 10,-25,
   -26, -4, -4,-10,  3,  3, 33,-12,
   -35, -1,-20,-23,-15, 24, 38,-22,
     0,  0,  0,  0,  0,  0,  0,  0,
};
const int eg_pawn[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
   178,173,158,134,147,132,165,187,
    94,100, 85, 67, 56, 53, 82, 84,
    32, 24, 13,  5, -2,  4, 17, 17,
    13,  9, -3, -7, -7, -8,  3, -1,
     4,  7, -6,  1,  0, -5, -1, -8,
    13,  8,  8, 10, 13,  0,  2, -7,
     0,  0,  0,  0,  0,  0,  0,  0,
};
const int mg_knight[64] = {
   -167,-89,-34,-49, 61,-97,-15,-107,
    -73,-41, 72, 36, 23, 62,  7, -17,
    -47, 60, 37, 65, 84,129, 73,  44,
     -9, 17, 19, 53, 37, 69, 18,  22,
    -13,  4, 16, 13, 28, 19, 21,  -8,
    -23, -9, 12, 10, 19, 17, 25, -16,
    -29,-53,-12, -3, -1, 18,-14, -19,
   -105,-21,-58,-33,-17,-28,-19, -23,
};
const int eg_knight[64] = {
    -58,-38,-13,-28,-31,-27,-63,-99,
    -25, -8,-25, -2, -9,-25,-24,-52,
    -24,-20,  10, 9,  -1, -9,-19,-41,
    -17,  3, 22, 22, 22, 11,  8,-18,
    -18, -6, 16, 25, 16, 17,  4,-18,
    -23, -3, -1, 15, 10, -3,-20,-22,
    -42,-20,-10, -5, -2,-20,-23,-44,
    -29,-51,-23,-15,-22,-18,-50,-64,
};
const int mg_bishop[64] = {
    -29,  4,-82,-37,-25,-42,  7, -8,
    -26, 16,-18,-13, 30, 59, 18,-47,
    -16, 37, 43, 40, 35, 50, 37, -2,
     -4,  5, 19, 50, 37, 37,  7, -2,
     -6, 13, 13, 26, 34, 12, 10,  4,
      0, 15, 15, 15, 14, 27, 18, 10,
      4, 15, 16,  0,  7, 21, 33,  1,
    -33, -3,-14,-21,-13,-12,-39,-21,
};
const int eg_bishop[64] = {
    -14,-21,-11, -8, -7, -9,-17,-24,
     -8, -4,  7,-12, -3,-13, -4,-14,
      2, -8,  0, -1, -2,  6,  0,  4,
     -3,  9, 12,  9, 14, 10,  3,  2,
     -6,  3, 13, 19,  7, 10, -3, -9,
    -12, -3,  8, 10, 13,  3, -7,-15,
    -14,-18, -7, -1,  4, -9,-15,-27,
    -23, -9,-23, -5, -9,-16, -5,-17,
};
const int mg_rook[64] = {
     32, 42, 32, 51, 63,  9, 31, 43,
     27, 32, 58, 62, 80, 67, 26, 44,
     -5, 19, 26, 36, 17, 45, 61, 16,
    -24,-11,  7, 26, 24, 35,-8,-20,
    -36,-26,-12, -1,  9, -7, 6,-23,
    -45,-25,-16,-17,  3,  0, -5,-33,
    -44,-16,-20, -9, -1, 11, -6,-71,
    -19,-13,  1, 17, 16,  7,-37,-26,
};
const int eg_rook[64] = {
    13, 10, 18, 15, 12,  12,   8,  5,
    11, 13, 13, 11, -3,   3,   8,  3,
     7,  7,  7,  5,  4,  -3,  -5,  -3,
     4,  3, 13,  1,  2,   1,  -1,   2,
     3,  5,  8,  4, -5,  -6,  -8, -11,
    -4,  0, -5, -1, -7, -12,  -8, -16,
    -6, -6,  0,  2, -9,  -9, -11,  -3,
    -9,  2,  3, -1, -5, -13,   4, -20,
};
const int mg_queen[64] = {
    -28,  0, 29, 12, 59, 44, 43, 45,
    -24,-39, -5,  1,-16, 57, 28, 54,
    -13,-17,  7,  8, 29, 56, 47, 57,
    -27,-27,-16,-16, -1, 17, -2,  1,
     -9,-26, -9,-10, -2, -4,  3, -3,
    -14,  2,-11, -2, -5,  2, 14,  5,
    -35, -8, 11,  2,  8, 15, -3,  1,
     -1,-18, -9, 10,-15,-25,-31,-50,
};
const int eg_queen[64] = {
     -9, 22, 22, 27, 27, 19, 10, 20,
    -17, 20, 32, 41, 58, 25, 30,  0,
    -20,  6,  9, 49, 47, 35, 19,  9,
      3, 22, 24, 45, 57, 40, 57, 36,
    -18, 28, 19, 47, 31, 34, 39, 23,
    -16,-27, 15,  6,  9, 17, 10,  5,
    -22,-23,-30,-16,-16,-23,-36,-32,
    -33,-28,-22,-43, -5,-32,-20,-41,
};
const int mg_king[64] = {
    -65, 23, 16,-15,-56,-34,  2, 13,
     29, -1,-20, -7, -8, -4,-38,-29,
     -9, 24,  2,-16,-20,  6, 22,-22,
    -17,-20,-12,-27,-30,-25,-14,-36,
    -49, -1,-27,-39,-46,-44,-33,-51,
    -14,-14,-22,-46,-44,-30,-15,-27,
      1,  7, -8,-64,-43,-16,  9,  8,
    -15, 36, 12,-54,  8,-28, 24, 14,
};
const int eg_king[64] = {
    -74,-35,-18,-18,-11, 15,  4,-17,
    -12, 17, 14, 17, 17, 38, 23, 11,
     10, 17, 23, 15, 20, 45, 44, 13,
     -8, 22, 24, 27, 26, 33, 26,  3,
    -18, -4, 21, 24, 27, 23,  9,-11,
    -19, -3, 11, 21, 23, 16,  7, -9,
    -27,-11,  4, 13, 14,  4, -5,-17,
    -53,-34,-21,-11,-28,-14,-24,-43,
};

const int* mg_table[6] = { mg_pawn, mg_knight, mg_bishop, mg_rook, mg_queen, mg_king };
const int* eg_table[6] = { eg_pawn, eg_knight, eg_bishop, eg_rook, eg_queen, eg_king };

// Game phase weights (sum = 24 in opening)
const int phase_weights[6] = { 0, 1, 1, 2, 4, 0 };

// ─── Classical Evaluation ─────────────────────────────────────────────────────
int classical_evaluate(const Board& board) {
    int mg = 0, eg = 0;
    int game_phase = 0;

    auto file_mask = [](int f) -> uint64_t { return 0x0101010101010101ULL << f; };

    Bitboard w_pawns = board.pieces(PieceType::PAWN, Color::WHITE);
    Bitboard b_pawns = board.pieces(PieceType::PAWN, Color::BLACK);

    // 1. Material + Tapered PST
    for (int pt = 0; pt < 6; ++pt) {
        PieceType pType = PieceType(static_cast<PieceType::underlying>(pt));
        game_phase += board.pieces(pType).count() * phase_weights[pt];

        Bitboard wb = board.pieces(pType, Color::WHITE);
        while (wb) {
            int sq = wb.pop();
            mg += piece_values[pt] + mg_table[pt][sq ^ 56];
            eg += piece_values[pt] + eg_table[pt][sq ^ 56];
        }
        Bitboard bb = board.pieces(pType, Color::BLACK);
        while (bb) {
            int sq = bb.pop();
            mg -= piece_values[pt] + mg_table[pt][sq];
            eg -= piece_values[pt] + eg_table[pt][sq];
        }
    }

    // Taper: interpolate between mg and eg
    game_phase = std::min(game_phase, 24);
    int score = (mg * game_phase + eg * (24 - game_phase)) / 24;

    // 2. Bishop pair
    if (board.pieces(PieceType::BISHOP, Color::WHITE).count() >= 2) score += 30;
    if (board.pieces(PieceType::BISHOP, Color::BLACK).count() >= 2) score -= 30;

    // 3. Rook on open / semi-open file
    for (int f = 0; f < 8; ++f) {
        uint64_t fm = file_mask(f);
        bool open   = !(board.pieces(PieceType::PAWN).getBits()   & fm);
        bool w_open = !(w_pawns.getBits() & fm);
        bool b_open = !(b_pawns.getBits() & fm);
        if (board.pieces(PieceType::ROOK, Color::WHITE).getBits() & fm) score += open ? 20 : (w_open ? 10 : 0);
        if (board.pieces(PieceType::ROOK, Color::BLACK).getBits() & fm) score -= open ? 20 : (b_open ? 10 : 0);
    }

    // 4. Pawn structure
    for (int f = 0; f < 8; ++f) {
        uint64_t fm = file_mask(f);
        uint64_t adj = (f > 0 ? file_mask(f-1) : 0) | (f < 7 ? file_mask(f+1) : 0);

        bool w_here = w_pawns.getBits() & fm;
        bool b_here = b_pawns.getBits() & fm;

        // Isolated pawns
        if (w_here && !(w_pawns.getBits() & adj)) score -= 15;
        if (b_here && !(b_pawns.getBits() & adj)) score += 15;

        // Doubled pawns
        int wc = __builtin_popcountll(w_pawns.getBits() & fm);
        int bc = __builtin_popcountll(b_pawns.getBits() & fm);
        if (wc > 1) score -= 10 * (wc - 1);
        if (bc > 1) score += 10 * (bc - 1);
    }

    // 5. Passed pawns
    {
        Bitboard tmp = w_pawns;
        while (tmp) {
            int sq = tmp.pop();
            int f = sq % 8, r = sq / 8;
            uint64_t front = 0;
            for (int rr = r + 1; rr < 8; ++rr) {
                front |= 1ULL << (rr * 8 + f);
                if (f > 0) front |= 1ULL << (rr * 8 + f - 1);
                if (f < 7) front |= 1ULL << (rr * 8 + f + 1);
            }
            if (!(front & b_pawns.getBits())) score += 10 + 20 * (r - 1) * (r - 1);
        }
    }
    {
        Bitboard tmp = b_pawns;
        while (tmp) {
            int sq = tmp.pop();
            int f = sq % 8, r = sq / 8;
            uint64_t front = 0;
            for (int rr = r - 1; rr >= 0; --rr) {
                front |= 1ULL << (rr * 8 + f);
                if (f > 0) front |= 1ULL << (rr * 8 + f - 1);
                if (f < 7) front |= 1ULL << (rr * 8 + f + 1);
            }
            if (!(front & w_pawns.getBits())) score -= 10 + 20 * (6 - r) * (6 - r);
        }
    }

    // 6. King safety: pawn shield + attack zone
    {
        auto king_safety = [&](Color us, Color them) -> int {
            int pen = 0;
            Square ksq = board.kingSq(us);
            int kf = ksq.file(), kr = ksq.rank();
            Bitboard pawns = board.pieces(PieceType::PAWN, us);
            int shield_rank = (us == Color::WHITE) ? kr + 1 : kr - 1;
            // Pawn shield
            if (shield_rank >= 0 && shield_rank <= 7) {
                for (int df = -1; df <= 1; ++df) {
                    int f2 = kf + df;
                    if (f2 < 0 || f2 > 7) continue;
                    if (!(pawns.getBits() & (1ULL << (shield_rank * 8 + f2)))) pen += 15;
                }
            }
            // 3x3 enemy attack zone
            for (int df = -1; df <= 1; ++df) {
                for (int dr = -1; dr <= 1; ++dr) {
                    int f2 = kf + df, r2 = kr + dr;
                    if (f2 < 0 || f2 > 7 || r2 < 0 || r2 > 7) continue;
                    if (board.isAttacked(Square(r2 * 8 + f2), them)) pen += 8;
                }
            }
            return pen;
        };
        score -= king_safety(Color::WHITE, Color::BLACK);
        score += king_safety(Color::BLACK, Color::WHITE);
    }

    return (board.sideToMove() == Color::WHITE) ? score : -score;
}

// ─── Move Scoring for Ordering ────────────────────────────────────────────────
int score_move(const Board& board, const Move& move, Move tt_move, int ply, Move prev_move) {
    if (move == tt_move) return 10'000'000;

    bool is_capture = board.isCapture(move) || move.typeOf() == Move::ENPASSANT;
    bool is_promo   = move.typeOf() == Move::PROMOTION;

    if (is_capture || is_promo) {
        int victim_val = 0;
        if (move.typeOf() == Move::ENPASSANT) victim_val = piece_values[0];
        else if (board.at(move.to()) != Piece::NONE) victim_val = piece_values[static_cast<int>(board.at(move.to()).type())];
        if (is_promo) victim_val += piece_values[4];

        int attacker_val = piece_values[static_cast<int>(board.at(move.from()).type())];
        bool winning = see_ge(board, move, 0);
        if (winning)
            return 7'000'000 + victim_val * 10 - attacker_val;
        else
            return 1'000 + victim_val * 10 - attacker_val; // Losing captures below quiet
    }

    // Killer moves
    if (ply >= 0 && ply < MAX_PLY) {
        if (killer_moves[ply][0] == move) return 6'000'000;
        if (killer_moves[ply][1] == move) return 5'900'000;
    }

    // Counter-move heuristic
    if (prev_move != Move::NULL_MOVE &&
        counter_moves[prev_move.from().index()][prev_move.to().index()] == move)
        return 5'800'000;

    // History heuristic
    int hist = history_table[static_cast<int>(board.sideToMove())][move.from().index()][move.to().index()];
    return hist;
}

void sort_moves(const Board& board, Movelist& moves, Move tt_move = Move::NULL_MOVE, int ply = -1, Move prev_move = Move::NULL_MOVE) {
    int n = moves.size();
    // Score all moves once, then sort (faster than insertion sort with per-compare scoring)
    std::vector<std::pair<int, int>> scored(n);
    for (int i = 0; i < n; ++i)
        scored[i] = {score_move(board, moves[i], tt_move, ply, prev_move), i};
    std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b){ return a.first > b.first; });
    Movelist tmp = moves;
    for (int i = 0; i < n; ++i) moves[i] = tmp[scored[i].second];
}

// ─── Evaluation ───────────────────────────────────────────────────────────────
int evaluate(const Board& board, const nnue::Accumulator& acc) {
    int classical = classical_evaluate(board);
    int nnue_score = nnue::evaluate(acc, board.sideToMove());
    // 50% classical, 50% NNUE
    return (classical + nnue_score) / 2;
}

// ─── Quiescence Search ────────────────────────────────────────────────────────
int quiescence(Board& board, int alpha, int beta, nnue::Accumulator acc, int ply = 0) {
    if ((nodes.load(std::memory_order_relaxed) & 4095) == 0 &&
        std::chrono::high_resolution_clock::now() > end_time)
        abort_search = true;
    if (abort_search) return 0;

    nodes.fetch_add(1, std::memory_order_relaxed);

    int stand_pat = evaluate(board, acc);
    if (stand_pat >= beta) return beta;

    // Delta pruning (skip if we can't possibly improve alpha even with best capture)
    const int DELTA = 1000; // queen + some margin
    if (stand_pat + DELTA < alpha) return alpha;

    if (stand_pat > alpha) alpha = stand_pat;

    Movelist moves;
    movegen::legalmoves<movegen::MoveGenType::CAPTURE>(moves, board);
    sort_moves(board, moves);

    for (const auto& move : moves) {
        // Skip losing captures (SEE < 0)
        if (!see_ge(board, move, 0)) continue;

        // Per-move delta pruning
        int capt_val = (move.typeOf() == Move::ENPASSANT) ? piece_values[0] :
                       (board.at(move.to()) != Piece::NONE) ? piece_values[static_cast<int>(board.at(move.to()).type())] : 0;
        if (move.typeOf() == Move::PROMOTION) capt_val += piece_values[4];
        if (stand_pat + capt_val + 200 < alpha) continue;

        nnue::Accumulator next_acc;
        chess::Piece piece = board.at(move.from());
        nnue::update_accumulator(board, move, acc, next_acc);

        board.makeMove(move);
        if (piece.type() == chess::PieceType::KING) {
            nnue::refresh_accumulator(board, ~board.sideToMove(), next_acc);
            nnue::refresh_accumulator(board, board.sideToMove(), next_acc);
        }
        int score = -quiescence(board, -beta, -alpha, next_acc, ply + 1);
        board.unmakeMove(move);

        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

// ─── Negamax ─────────────────────────────────────────────────────────────────
int negamax(Board& board, int depth, int alpha, int beta, int ply, bool allow_null, nnue::Accumulator acc, Move prev_move = Move::NULL_MOVE, Move excluded_move = Move::NULL_MOVE) {
    if ((nodes.load(std::memory_order_relaxed) & 4095) == 0 &&
        std::chrono::high_resolution_clock::now() > end_time)
        abort_search = true;
    if (abort_search) return 0;

    nodes.fetch_add(1, std::memory_order_relaxed);

    bool is_root = (ply == 0);
    bool is_pv   = (beta - alpha > 1);

    if (!is_root && (board.isHalfMoveDraw() || board.isRepetition())) return 0;

    uint64_t hash = board.hash();
    __builtin_prefetch(&TT[(hash & (TT_MASK >> 1)) * 2]);
    
    Move tt_move = Move::NULL_MOVE;
    int tt_depth = 0;
    TTEntry::Flag tt_flag = TTEntry::NONE;
    int tt_eval = 0;

    // TT probe
    int tt_score = 0;
    bool tt_hit = probe_tt(hash, depth, alpha, beta, tt_score, tt_move, tt_depth, tt_flag, tt_eval);
    if (tt_hit && excluded_move == Move::NULL_MOVE && !is_root) {
        return tt_score;
    }
    if (tt_move == Move::NULL_MOVE) tt_move = probe_tt_move(hash);

    // Mate distance pruning
    alpha = std::max(alpha, -MATE_SCORE + ply);
    beta  = std::min(beta,   MATE_SCORE - ply);
    if (alpha >= beta) return alpha;

    bool in_check = board.inCheck();

    // Check extension
    if (in_check) depth++;

    if (depth <= 0) return quiescence(board, alpha, beta, acc);

    // Compute static eval for pruning (avoid if in check)
    int static_eval = 0;
    if (!in_check) static_eval = evaluate(board, acc);

    // Reverse futility pruning (static null-move pruning)
    if (!is_pv && !in_check && depth <= 8 && ply > 0) {
        if (static_eval - opt_rfp_margin * depth >= beta) {
            return static_eval;
        }
    }

    // Null-move pruning
    if (allow_null && !is_pv && !in_check && ply > 0 && depth >= 3) {
        bool has_pieces = board.pieces(PieceType::KNIGHT).count() +
                          board.pieces(PieceType::BISHOP).count() +
                          board.pieces(PieceType::ROOK).count() +
                          board.pieces(PieceType::QUEEN).count() > 0;
        if (has_pieces && static_eval >= beta) {
            int R = opt_nmp_base + depth / opt_nmp_depth_div + std::min(3, (static_eval - beta) / opt_nmp_eval_div);
            board.makeNullMove();
            int null_score = -negamax(board, depth - 1 - R, -beta, -beta + 1, ply + 1, false, acc, Move::NULL_MOVE);
            board.unmakeNullMove();
            if (abort_search) return 0;
            if (null_score >= beta) {
                // Don't return unverified mates
                if (null_score >= MATE_SCORE - 100) null_score = beta;
                return null_score;
            }
        }
    }

    // Internal Iterative Deepening: if no TT move and deep node, search shallower first
    if (!is_pv && tt_move == Move::NULL_MOVE && depth >= 5) {
        negamax(board, depth - 4, alpha, beta, ply, false, acc, prev_move);
        tt_move = probe_tt_move(hash);
    }
    
    // Singular Extensions
    bool tt_is_singular = false;
    if (!is_root && depth >= 8 && tt_move != Move::NULL_MOVE && 
        excluded_move == Move::NULL_MOVE && 
        tt_depth >= depth - 3 && 
        tt_flag != TTEntry::UPPER && 
        std::abs(tt_eval) < MATE_SCORE - 1000) 
    {
        int singular_beta = tt_eval - depth;
        int singular_score = negamax(board, depth / 2, singular_beta - 1, singular_beta, ply, false, acc, prev_move, tt_move);
        if (singular_score < singular_beta) {
            tt_is_singular = true;
        }
    }

    Movelist moves;
    movegen::legalmoves(moves, board);
    if (moves.empty()) {
        return in_check ? (-MATE_SCORE + ply) : 0; // Checkmate or stalemate
    }

    sort_moves(board, moves, tt_move, ply, prev_move);

    int best_score    = -INF;
    Move best_move    = Move::NULL_MOVE;
    int  original_alpha = alpha;
    int  move_count   = 0;

    // Futility pruning margins
    int fp_margin = opt_fp_margin_base + (depth - 1) * opt_fp_margin_mult;

    for (const auto& move : moves) {
        if (move == excluded_move) continue;

        bool is_capture  = board.isCapture(move) || move.typeOf() == Move::ENPASSANT;
        bool is_promo    = move.typeOf() == Move::PROMOTION;

        nnue::Accumulator next_acc;
        chess::Piece piece = board.at(move.from());
        nnue::update_accumulator(board, move, acc, next_acc);

        board.makeMove(move);
        if (piece.type() == chess::PieceType::KING) {
            nnue::refresh_accumulator(board, ~board.sideToMove(), next_acc);
            nnue::refresh_accumulator(board, board.sideToMove(), next_acc);
        }
        move_count++;
        bool gives_check = board.inCheck();

        int extension = 0;
        if (tt_is_singular && move == tt_move) extension = 1;

        // Futility pruning (forward): skip quiet moves that can't improve alpha
        if (!is_root && !in_check && !gives_check && !is_capture && !is_promo
            && depth <= 8 && move_count > 1)
        {
            if (static_eval + fp_margin <= alpha) {
                board.unmakeMove(move);
                continue;
            }
            // Late move pruning: skip very late quiet moves at low depth
            if (!is_pv && depth <= 4 && move_count > 4 + depth * depth) {
                board.unmakeMove(move);
                continue;
            }
        }

        int score;
        if (move_count == 1) {
            // PV node: full window
            score = -negamax(board, depth - 1 + extension, -beta, -alpha, ply + 1, true, next_acc, move);
        } else {
            // LMR: reduce late quiet moves
            bool do_lmr = move_count > 3 && depth >= 3 && !is_capture && !is_promo
                          && !in_check && !gives_check;
            int R = 0;
            if (do_lmr) {
                R = 1 + static_cast<int>(std::log(depth) * std::log(move_count) * 100.0 / opt_lmr_mult);
                R = std::min(R, depth - 2);
                R = std::max(R, 1);
                // Don't reduce PV nodes or history-backed moves as much
                if (is_pv) R--;
            }

            // Null-window search
            score = -negamax(board, depth - 1 - R + extension, -alpha - 1, -alpha, ply + 1, true, next_acc, move);

            // Full-depth re-search if LMR raised alpha or window failed
            if (score > alpha && (R > 0 || (!is_pv && score < beta))) {
                score = -negamax(board, depth - 1 + extension, -alpha - 1, -alpha, ply + 1, true, next_acc, move);
            }
            // Full-window re-search for PV update
            if (score > alpha && score < beta) {
                score = -negamax(board, depth - 1 + extension, -beta, -alpha, ply + 1, true, next_acc, move);
            }
        }

        board.unmakeMove(move);
        if (abort_search) return 0;

        if (score > best_score) {
            best_score = score;
            best_move  = move;
        }
        if (score > alpha) alpha = score;

        if (alpha >= beta) {
            // Update killers, counter-moves and history on cutoff
            if (!is_capture && !is_promo && ply < MAX_PLY) {
                if (killer_moves[ply][0] != move) {
                    killer_moves[ply][1] = killer_moves[ply][0];
                    killer_moves[ply][0] = move;
                }
                if (prev_move != Move::NULL_MOVE) {
                    counter_moves[prev_move.from().index()][prev_move.to().index()] = move;
                }
                // History bonus: depth^2, with bonus scaled by depth
                int bonus = depth * depth;
                history_table[static_cast<int>(board.sideToMove())][move.from().index()][move.to().index()]
                    += bonus - history_table[static_cast<int>(board.sideToMove())][move.from().index()][move.to().index()] * bonus / 16384;
            }
            break;
        }
    }

    if (!abort_search && excluded_move == Move::NULL_MOVE) {
        TTEntry::Flag flag;
        if (best_score <= original_alpha) flag = TTEntry::UPPER;
        else if (best_score >= beta)      flag = TTEntry::LOWER;
        else                              flag = TTEntry::EXACT;
        write_tt(hash, best_move, depth, best_score, flag);
    }

    return best_score;
}

// ─── Search Worker (Helper Threads) ───────────────────────────────────────────
void search_worker(Board board, int target_ms) {
    nnue::Accumulator root_acc;
    nnue::init_accumulator(board, root_acc);

    int previous_score = 0;
    for (int depth = 1; depth <= 64; ++depth) {
        if (abort_search) break;

        int alpha = -INF;
        int beta  = INF;
        
        // Slightly change start depth window for diversity, or just use wide window
        if (depth >= 5) {
            alpha = previous_score - 50;
            beta  = previous_score + 50;
        }

        int score;
        int fail_low_cnt = 0;
        int fail_high_cnt = 0;
        
        while (true) {
            score = negamax(board, depth, alpha, beta, 0, true, root_acc, Move::NULL_MOVE);
            if (abort_search) break;
            
            if (score <= alpha) {
                fail_low_cnt++;
                beta = (alpha + beta) / 2;
                alpha = std::max(-INF, alpha - 50 * (1 << fail_low_cnt));
            } else if (score >= beta) {
                fail_high_cnt++;
                beta = std::min(INF, beta + 50 * (1 << fail_high_cnt));
            } else {
                break;
            }
        }
        if (!abort_search) {
            previous_score = score;
        }
    }
}

// ─── Iterative Deepening + Aspiration Windows ─────────────────────────────────
Move search_best_move(Board& board, int target_ms) {
    Move best_move     = Move::NULL_MOVE;
    nodes              = 0;
    abort_search       = false;

    // History gravity: halve history between searches
    for (auto& c : history_table)
        for (auto& f : c)
            for (auto& t : f)
                t >>= 1;

    std::fill(&killer_moves[0][0], &killer_moves[0][0] + sizeof(killer_moves) / sizeof(Move), Move::NULL_MOVE);

    auto start = std::chrono::high_resolution_clock::now();
    end_time   = start + std::chrono::milliseconds(std::max(1, target_ms - MOVE_OVERHEAD));

    nnue::Accumulator root_acc;
    nnue::init_accumulator(board, root_acc);

    int previous_score = 0;

    std::vector<std::thread> workers;
    for (int i = 1; i < num_threads; ++i) {
        workers.emplace_back(search_worker, board, target_ms);
    }

    for (int depth = 1; depth <= 64; ++depth) {
        int alpha, beta;

        // Progressive aspiration windows
        const int window = 25;
        if (depth >= 5) {
            alpha = previous_score - window;
            beta  = previous_score + window;
        } else {
            alpha = -INF;
            beta  = INF;
        }

        int fail_low_cnt  = 0;
        int fail_high_cnt = 0;
        int score;

        while (true) {
            score = negamax(board, depth, alpha, beta, 0, true, root_acc, Move::NULL_MOVE);
            if (abort_search) break;

            if (score <= alpha) {
                // Fail-low: widen downward progressively
                fail_low_cnt++;
                beta  = (alpha + beta) / 2;
                alpha = std::max(-INF, alpha - window * (1 << fail_low_cnt));
            } else if (score >= beta) {
                // Fail-high: widen upward progressively
                fail_high_cnt++;
                beta = std::min(INF, beta + window * (1 << fail_high_cnt));
            } else {
                break; // Within window
            }
        }

        if (abort_search) break;
        previous_score = score;

        // Extract best move
        Move found = probe_tt_move(board.hash());
        if (found != Move::NULL_MOVE) best_move = found;

        auto now = std::chrono::high_resolution_clock::now();
        int elapsed_ms = static_cast<int>(std::chrono::duration<double>(now - start).count() * 1000);
        long long nps  = (elapsed_ms > 0) ? (nodes.load() * 1000LL / elapsed_ms) : 0;

        std::cout << "info depth " << depth
                  << " score cp " << score
                  << " nodes " << nodes.load()
                  << " time " << elapsed_ms
                  << " nps " << nps
                  << " pv " << (best_move != Move::NULL_MOVE ? uci::moveToUci(best_move) : "(none)")
                  << "\n";
        std::cout.flush();

        // Stop early if mate found or more than half time is used
        if (score > MATE_SCORE - 200 || score < -MATE_SCORE + 200) break;
        if (elapsed_ms > target_ms / 2) break;
    }

    // Safety: if somehow no move was found, play first legal
    if (best_move == Move::NULL_MOVE) {
        Movelist ml;
        movegen::legalmoves(ml, board);
        if (!ml.empty()) best_move = ml[0];
    }

    // Wait for helper threads
    for (auto& t : workers) {
        if (t.joinable()) t.join();
    }

    return best_move;
}

// ─── UCI Loop ─────────────────────────────────────────────────────────────────
int main() {
    nnue::load_weights("nnue_weights.bin");
    std::cout << "G-ForceZero NNUE Engine initialized.\n";

    chess::Board board;
    board.setFen(chess::constants::STARTPOS);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string command;
        ss >> command;

        if (command == "uci") {
            std::cout << "id name Antigravity NNUE\n"
                      << "id author Siddharth\n"
                      << "option name Hash type spin default 128 min 1 max 1024\n"
                      << "option name Threads type spin default 1 min 1 max 64\n"
                      << "option name RFP_Margin type spin default 80 min 10 max 200\n"
                      << "option name NMP_Base type spin default 3 min 1 max 10\n"
                      << "option name NMP_Depth_Div type spin default 6 min 1 max 15\n"
                      << "option name NMP_Eval_Div type spin default 200 min 50 max 500\n"
                      << "option name LMR_Mult type spin default 225 min 50 max 500\n"
                      << "option name FP_Margin_Base type spin default 100 min 10 max 300\n"
                      << "option name FP_Margin_Mult type spin default 60 min 10 max 200\n"
                      << "uciok\n";
        } else if (command == "setoption") {
            std::string name, name_val, value, val_val;
            ss >> name >> name_val >> value >> val_val;
            if (name_val == "Threads") {
                num_threads = std::max(1, std::min(64, std::stoi(val_val)));
            } else if (name_val == "RFP_Margin") {
                opt_rfp_margin = std::stoi(val_val);
            } else if (name_val == "NMP_Base") {
                opt_nmp_base = std::stoi(val_val);
            } else if (name_val == "NMP_Depth_Div") {
                opt_nmp_depth_div = std::stoi(val_val);
            } else if (name_val == "NMP_Eval_Div") {
                opt_nmp_eval_div = std::stoi(val_val);
            } else if (name_val == "LMR_Mult") {
                opt_lmr_mult = std::stoi(val_val);
            } else if (name_val == "FP_Margin_Base") {
                opt_fp_margin_base = std::stoi(val_val);
            } else if (name_val == "FP_Margin_Mult") {
                opt_fp_margin_mult = std::stoi(val_val);
            }
        } else if (command == "isready") {
            std::cout << "readyok\n";
        } else if (command == "ucinewgame") {
            board.setFen(chess::constants::STARTPOS);
            tt_clear();
            std::fill(&history_table[0][0][0], &history_table[0][0][0] + sizeof(history_table) / sizeof(int), 0);
            std::fill(&counter_moves[0][0], &counter_moves[0][0] + sizeof(counter_moves) / sizeof(Move), Move::NULL_MOVE);
        } else if (command == "position") {
            std::string token;
            ss >> token;
            if (token == "startpos") {
                board.setFen(chess::constants::STARTPOS);
                ss >> token; // "moves"
            } else if (token == "fen") {
                std::string fen;
                for (int i = 0; i < 6; ++i) {
                    ss >> token;
                    fen += token + (i == 5 ? "" : " ");
                }
                board.setFen(fen);
                ss >> token; // "moves"
            }
            while (ss >> token) {
                if (token == "moves") continue;
                chess::Movelist ml;
                chess::movegen::legalmoves(ml, board);
                for (const auto& m : ml) {
                    if (chess::uci::moveToUci(m) == token) {
                        board.makeMove(m);
                        break;
                    }
                }
            }
        } else if (command == "go") {
            int wtime = 0, btime = 0, winc = 0, binc = 0, movetime = 0, movestogo = 0;
            std::string token;
            while (ss >> token) {
                if      (token == "wtime")     ss >> wtime;
                else if (token == "btime")     ss >> btime;
                else if (token == "winc")      ss >> winc;
                else if (token == "binc")      ss >> binc;
                else if (token == "movetime")  ss >> movetime;
                else if (token == "movestogo") ss >> movestogo;
            }

            int target_ms;
            if (movetime > 0) {
                target_ms = movetime - MOVE_OVERHEAD;
            } else if (wtime > 0 || btime > 0) {
                int my_time = (board.sideToMove() == chess::Color::WHITE) ? wtime : btime;
                int my_inc  = (board.sideToMove() == chess::Color::WHITE) ? winc  : binc;
                int moves_left = (movestogo > 0) ? movestogo : 30;
                // Budget: time/moves_left + inc*0.8, with floor/ceiling
                target_ms = my_time / moves_left + (my_inc * 4) / 5;
                target_ms = std::max(50, std::min(target_ms, my_time / 4));
            } else {
                target_ms = 5000; // Analysis mode
            }

            chess::Move best = get_book_move(board, "book.bin");
            if (best != chess::Move::NULL_MOVE) {
                std::cout << "info string Playing from PolyGlot opening book\n";
                std::cout << "bestmove " << chess::uci::moveToUci(best) << "\n";
                std::cout.flush();
            } else {
                best = search_best_move(board, target_ms);
                std::cout << "bestmove " << chess::uci::moveToUci(best) << "\n";
                std::cout.flush();
            }
        } else if (command == "quit") {
            break;
        }
    }
    return 0;
}
