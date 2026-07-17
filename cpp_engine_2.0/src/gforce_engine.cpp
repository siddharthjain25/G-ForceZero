// ─────────────────────────────────────────────────────────────────────────────
//  G-ForceZero NNUE Chess Engine
//  Features:
//    • Negamax + Alpha-Beta + PVS
//    • Iterative Deepening + Aspiration Windows (progressive widening)
//    • Two-bucket Transposition Table (16M entries, 128 MB)
//    • Internal Iterative Deepening (IID)
//    • Null-Move Pruning (adaptive R)
//    • Late Move Reductions (LMR, log-based table)
//    • Futility Pruning (forward)
//    • Reverse Futility Pruning (static NMP)
//    • Late Move Pruning
//    • Full Recursive SEE for capture ordering & pruning
//    • Delta Pruning in Quiescence
//    • Check Extensions + Singular Extensions
//    • Move ordering: TT move > winning captures (SEE) > killers > counter-moves > continuation history > history > quiet > losing captures
//    • History Heuristic with gravity (halving between searches)
//    • Continuation History (1-ply and 2-ply)
//    • Killer Moves (2 slots)
//    • Counter-Move Heuristic
//    • HalfKP NNUE (enabled, blended with Classical Eval)
//    • Accurate time management with soft/hard limits
//    • PV line tracking
// ─────────────────────────────────────────────────────────────────────────────

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <cmath>
#include <cassert>
#include <climits>
#include "chess.hpp"
#include "polyglot.hpp"
extern "C" {
#include "tbprobe.h"
}

using namespace chess;

// ─── Constants ───────────────────────────────────────────────────────────────
const int INF        = 30000;
const int MATE_SCORE = 20000;
const int MAX_PLY    = 128;
const int MOVE_OVERHEAD = 20; // ms overhead per move

// ─── Piece values ─────────────────────────────────────────────────────────────
// P=100, N=320, B=330, R=500, Q=900, K=20000
const int piece_values[6] = {100, 320, 330, 500, 900, 20000};
uint64_t nodes_searched = 0;
int last_search_score = 0;

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

int TT_SIZE = 1 << 21; // 16 MB default
int TT_MASK = TT_SIZE - 1;
std::atomic<uint64_t>* TT = nullptr;

#if defined(__linux__)
#include <sys/mman.h>
#ifndef MAP_HUGETLB
#define MAP_HUGETLB 0x40000
#endif
#endif

void init_tt(int mb_size = -1) {
    if (TT) {
        if (mb_size <= 0) return; // already initialized, no change
#if defined(__linux__)
        munmap(TT, TT_SIZE * sizeof(std::atomic<uint64_t>));
#else
        delete[] TT;
#endif
        TT = nullptr;
    }
    
    if (mb_size > 0) {
        size_t bytes = static_cast<size_t>(mb_size) * 1024 * 1024;
        TT_SIZE = 1024;
        while ((TT_SIZE * 8 * 2) <= bytes) TT_SIZE *= 2;
        TT_MASK = TT_SIZE - 1;
    }

    size_t size = TT_SIZE * sizeof(std::atomic<uint64_t>);
#if defined(__linux__)
    void* mem = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (mem == MAP_FAILED) {
        mem = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    }
    TT = static_cast<std::atomic<uint64_t>*>(mem);
#else
    TT = new std::atomic<uint64_t>[TT_SIZE];
#endif
}

void tt_clear() {
    std::memset(TT, 0, TT_SIZE * sizeof(std::atomic<uint64_t>));
}

uint8_t current_age = 0;

void write_tt(uint64_t hash, Move move, int depth, int score, TTEntry::Flag flag, int ply) {
    if (score >= MATE_SCORE - MAX_PLY) score += ply;
    else if (score <= -MATE_SCORE + MAX_PLY) score -= ply;
    int base = (hash & (TT_MASK >> 1)) * 2;
    uint16_t key = static_cast<uint16_t>(hash >> 48);

    uint8_t flag_age = static_cast<uint8_t>(flag) | (current_age << 2);
    TTEntry new_entry = { key, move.move(), static_cast<int16_t>(score), static_cast<int8_t>(depth), flag_age };
    uint64_t data;
    std::memcpy(&data, &new_entry, 8);

    uint64_t slot0_data = TT[base].load(std::memory_order_relaxed);
    TTEntry slot0;
    std::memcpy(&slot0, &slot0_data, 8);
    
    uint8_t slot0_flag = slot0.flag & 3;
    uint8_t slot0_age  = slot0.flag >> 2;
    
    if (slot0_flag == TTEntry::NONE || slot0_age != current_age || depth >= slot0.depth) {
        TT[base].store(data, std::memory_order_relaxed);
    } else {
        TT[base + 1].store(data, std::memory_order_relaxed);
    }
}

Move probe_tt_move(uint64_t hash) {
    int base = (hash & (TT_MASK >> 1)) * 2;
    uint16_t key = static_cast<uint16_t>(hash >> 48);
    for (int i = 0; i < 2; ++i) {
        uint64_t data = TT[base + i].load(std::memory_order_relaxed);
        TTEntry tte;
        std::memcpy(&tte, &data, 8);
        if (tte.key == key && (tte.flag & 3) != TTEntry::NONE) return Move(tte.move);
    }
    return Move::NULL_MOVE;
}

bool probe_tt(uint64_t hash, int depth, int alpha, int beta, int ply, int& score, Move& tt_move, int& tt_depth, TTEntry::Flag& tt_flag, int& tt_eval) {
    int base = (hash & (TT_MASK >> 1)) * 2;
    uint16_t key = static_cast<uint16_t>(hash >> 48);
    for (int i = 0; i < 2; ++i) {
        uint64_t data = TT[base + i].load(std::memory_order_relaxed);
        TTEntry tte;
        std::memcpy(&tte, &data, 8);
        
        uint8_t flag = tte.flag & 3;
        if (tte.key != key || flag == TTEntry::NONE) continue;
        if (tt_move == Move::NULL_MOVE && tte.move) tt_move = Move(tte.move);
        
        tt_depth = tte.depth;
        tt_flag = static_cast<TTEntry::Flag>(flag);
        tt_eval = tte.score;
        if (tt_eval >= MATE_SCORE - MAX_PLY) tt_eval -= ply;
        else if (tt_eval <= -MATE_SCORE + MAX_PLY) tt_eval += ply;
        
        if (tte.depth >= depth) {
            if (flag == TTEntry::EXACT) { score = tt_eval; return true; }
            if (flag == TTEntry::LOWER && tt_eval >= beta)  { score = beta;  return true; }
            if (flag == TTEntry::UPPER && tt_eval <= alpha) { score = alpha; return true; }
        }
    }
    return false;
}

int num_threads = 1;

// ─── Syzygy Tablebase ─────────────────────────────────────────────────────────
std::string syzygy_path = "";
bool tb_enabled = false;
// TB_LARGEST is set by tbprobe after init; we mirror it here for convenience
int tb_pieces = 0;

// Convert board to tablebase probe bitboards and call tb_probe_wdl.
// Returns TB_WIN/TB_DRAW/TB_LOSS/TB_BLESSED_LOSS/TB_CURSED_WIN, or -1 on failure.
static int probe_wdl(const Board& board) {
    if (!tb_enabled) return -1;
    // Count pieces; skip if too many
    int cnt = board.occ().count();
    if (cnt > tb_pieces) return -1;
    // Castling rights disqualify the probe
    if (!board.castlingRights().isEmpty()) return -1;


    uint64_t white   = board.us(Color::WHITE).getBits();
    uint64_t black   = board.us(Color::BLACK).getBits();
    uint64_t kings   = board.pieces(PieceType::KING).getBits();
    uint64_t queens  = board.pieces(PieceType::QUEEN).getBits();
    uint64_t rooks   = board.pieces(PieceType::ROOK).getBits();
    uint64_t bishops = board.pieces(PieceType::BISHOP).getBits();
    uint64_t knights = board.pieces(PieceType::KNIGHT).getBits();
    uint64_t pawns   = board.pieces(PieceType::PAWN).getBits();
    unsigned ep      = board.enpassantSq() == Square::underlying::NO_SQ ? 0
                     : board.enpassantSq().index();
    bool turn        = (board.sideToMove() == Color::WHITE);

    unsigned res = tb_probe_wdl(white, black, kings, queens, rooks, bishops, knights, pawns,
                                0 /*castling*/, ep, turn);
    if (res == TB_RESULT_FAILED) return -1;
    return static_cast<int>(res);
}

// ─── Tunable Search Parameters ────────────────────────────────────────────────
int opt_rfp_margin = 80;
int opt_nmp_base = 4;
int opt_nmp_depth_div = 3;
int opt_nmp_eval_div = 200;
int opt_lmr_mult = 225; // 2.25 * 100
int opt_fp_margin_base = 232;
int opt_fp_margin_mult = 217;
int opt_nnue_weight = 100; // NNUE blend weight (0-100)
bool is_selfplay = false;

// ─── LMR Table (precomputed) ──────────────────────────────────────────────────
int8_t lmr_table[MAX_PLY][64]; // [depth][move_count] — int8_t to fit in L1 cache
void init_lmr_table() {
    for (int d = 1; d < MAX_PLY; ++d)
        for (int m = 1; m < 64; ++m) {
            int v = static_cast<int>(std::log(d) * std::log(m) * 100.0 / opt_lmr_mult);
            lmr_table[d][m] = static_cast<int8_t>(std::min(v, 63));
        }
}

// ─── Search State ─────────────────────────────────────────────────────────────
Move  killer_moves[MAX_PLY][2];
Move  counter_moves[64][64]; // [from][to] → killer response
int   history_table[2][64][64]; // [color][from][to]
// Continuation history: [prev_piece][prev_to][piece][to]
int   cont_hist[12][64][12][64];
int   eval_history[MAX_PLY];

std::atomic<bool> abort_search{false};
std::chrono::time_point<std::chrono::high_resolution_clock> end_time;
std::atomic<uint64_t> nodes{0};

// ─── PV Line ─────────────────────────────────────────────────────────────────
Move pv_table[MAX_PLY][MAX_PLY];
int  pv_length[MAX_PLY];

// ─── Proper Full SEE ──────────────────────────────────────────────────────────
// Returns the least-valuable attacker of `to` by `stm`, removing it from `occ`
static inline int lva_attacker(const Board& board, Square to, Color stm, Bitboard& occ) {
    // Pawns
    {
        Bitboard pawns = board.pieces(PieceType::PAWN, stm) & occ;
        Bitboard pawn_atk = attacks::pawn(~stm, to) & pawns;
        if (pawn_atk) {
            int sq = pawn_atk.lsb();
            occ ^= Bitboard(1ULL << sq);
            return piece_values[0];
        }
    }
    // Knights — use reverse attack lookup: which knights attack `to`?
    {
        Bitboard knights = board.pieces(PieceType::KNIGHT, stm) & occ;
        Bitboard atkers  = attacks::knight(to) & knights; // O(1) lookup
        if (atkers) {
            int sq = atkers.lsb();
            occ ^= Bitboard(1ULL << sq);
            return piece_values[1];
        }
    }
    // Bishops
    {
        Bitboard bishops = board.pieces(PieceType::BISHOP, stm) & occ;
        Bitboard atkers  = attacks::bishop(to, occ) & bishops;
        if (atkers) {
            int sq = atkers.lsb();
            occ ^= Bitboard(1ULL << sq);
            return piece_values[2];
        }
    }
    // Rooks
    {
        Bitboard rooks  = board.pieces(PieceType::ROOK, stm) & occ;
        Bitboard atkers = attacks::rook(to, occ) & rooks;
        if (atkers) {
            int sq = atkers.lsb();
            occ ^= Bitboard(1ULL << sq);
            return piece_values[3];
        }
    }
    // Queens
    {
        Bitboard queens = board.pieces(PieceType::QUEEN, stm) & occ;
        Bitboard atkers = attacks::queen(to, occ) & queens;
        if (atkers) {
            int sq = atkers.lsb();
            occ ^= Bitboard(1ULL << sq);
            return piece_values[4];
        }
    }
    // King
    {
        Bitboard king   = board.pieces(PieceType::KING, stm) & occ;
        Bitboard atkers = attacks::king(to) & king;
        if (atkers) {
            int sq = atkers.lsb();
            occ ^= Bitboard(1ULL << sq);
            return piece_values[5];
        }
    }
    return INF; // no attacker
}

int static_exchange_eval(const Board& board, Move move) {
    Square from = move.from();
    Square to   = move.to();

    if (move.typeOf() == Move::CASTLING) return 0;

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
    
    // Handle promotions
    int attacker_val;
    if (move.typeOf() == Move::PROMOTION) {
        gain[d] += piece_values[static_cast<int>(move.promotionType())] - piece_values[0];
        attacker_val = piece_values[static_cast<int>(move.promotionType())];
    } else {
        attacker_val = piece_values[static_cast<int>(board.at(from).type())];
    }

    Bitboard occ = board.occ();
    occ ^= Bitboard(1ULL << from.index());
    if (move.typeOf() == Move::ENPASSANT) {
        int ep_sq = to.index() + (board.sideToMove() == Color::WHITE ? -8 : 8);
        occ ^= Bitboard(1ULL << ep_sq);
    }

    Color stm = ~board.at(from).color();

    while (true) {
        d++;
        if (d >= 31) break;
        gain[d] = attacker_val - gain[d - 1];

        int new_val = lva_attacker(board, to, stm, occ);
        if (new_val == INF) break; // no attacker

        attacker_val = new_val;
        stm = ~stm;

        // Alpha-beta cutoff in SEE
        if (std::max(-gain[d - 1], gain[d]) == gain[d]) break;
    }

    while (--d) {
        gain[d - 1] = -std::max(gain[d], -gain[d - 1]);
    }

    return gain[0];
}

bool see_ge(const Board& board, Move move, int threshold = 0) {
    if (move.typeOf() == Move::CASTLING) return threshold <= 0;
    if (!board.isCapture(move) && move.typeOf() != Move::ENPASSANT && move.typeOf() != Move::PROMOTION)
        return threshold <= 0;
    return static_exchange_eval(board, move) >= threshold;
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
        int pc = board.pieces(pType).count();
        game_phase += pc * phase_weights[pt];

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

    // 5. Passed pawns — use bitboard spans instead of loop-built masks
    {
        Bitboard tmp = w_pawns;
        while (tmp) {
            int sq  = tmp.pop();
            int f   = sq % 8, r = sq / 8;
            // Build forward + adjacent-file mask from rank r+1 to rank 7
            // Using fill-forward technique: shift up and OR adjacent files
            uint64_t file_bb = 0x0101010101010101ULL << f;
            uint64_t adj_bb  = ((f > 0) ? (file_bb >> 1) : 0) | ((f < 7) ? (file_bb << 1) : 0);
            uint64_t front   = (file_bb | adj_bb) & ~((1ULL << ((r + 1) * 8)) - 1); // mask above rank r
            if (!(front & b_pawns.getBits())) score += 10 + 20 * (r - 1) * (r - 1);
        }
    }
    {
        Bitboard tmp = b_pawns;
        while (tmp) {
            int sq  = tmp.pop();
            int f   = sq % 8, r = sq / 8;
            uint64_t file_bb = 0x0101010101010101ULL << f;
            uint64_t adj_bb  = ((f > 0) ? (file_bb >> 1) : 0) | ((f < 7) ? (file_bb << 1) : 0);
            // Mask below rank r (bits 0..r*8-1)
            uint64_t front   = (file_bb | adj_bb) & ((1ULL << (r * 8)) - 1);
            if (!(front & w_pawns.getBits())) score -= 10 + 20 * (6 - r) * (6 - r);
        }
    }

    // 6. King safety: pawn shield + attack zone count
    // Use lightweight attack counting via piece attack tables instead of board.isAttacked()
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
            // 3x3 enemy attack zone — count attacker pieces hitting king zone
            // Use king attacks bitboard to get zone squares, then check with piece attack bitboards
            Bitboard zone = attacks::king(ksq);
            zone |= Bitboard(1ULL << ksq.index()); // include king square itself
            while (zone) {
                int z = zone.pop();
                Square zsq(z);
                // Check if any enemy piece attacks this square (using precomputed attack tables)
                Bitboard occ = board.occ();
                if (attacks::pawn(us, zsq) & board.pieces(PieceType::PAWN, them)) pen += 8;
                if (attacks::knight(zsq)    & board.pieces(PieceType::KNIGHT, them)) pen += 8;
                if (attacks::bishop(zsq, occ) & (board.pieces(PieceType::BISHOP, them) | board.pieces(PieceType::QUEEN, them))) pen += 8;
                if (attacks::rook(zsq, occ)   & (board.pieces(PieceType::ROOK, them)   | board.pieces(PieceType::QUEEN, them))) pen += 8;
            }
            return pen;
        };
        score -= king_safety(Color::WHITE, Color::BLACK);
        score += king_safety(Color::BLACK, Color::WHITE);
    }

    return (board.sideToMove() == Color::WHITE) ? score : -score;
}

// ─── Custom Evaluation ────────────────────────────────────
int evaluate(const Board& board) {
    return classical_evaluate(board);
}

// ─── Piece index for continuation history ────────────────────────────────────
static inline int piece_idx(const Board& board, Square sq) {
    Piece p = board.at(sq);
    if (p == Piece::NONE) return 0;
    int color = (p.color() == Color::WHITE) ? 0 : 6;
    return color + static_cast<int>(p.type());
}

// ─── Move Scoring for Ordering ────────────────────────────────────────────────
// Returns encoded score based on MVV-LVA for captures, killers, and history.
int score_move(const Board& board, const Move& move, Move tt_move, int ply, Move prev_move, Move prev_prev_move) {
    if (move == tt_move) return 10'000'000;

    bool is_capture = board.isCapture(move) || move.typeOf() == Move::ENPASSANT;
    bool is_promo   = move.typeOf() == Move::PROMOTION;

    if (is_capture || is_promo) {
        int victim_val = 0;
        if (move.typeOf() == Move::ENPASSANT) victim_val = piece_values[0];
        else if (board.at(move.to()) != Piece::NONE) victim_val = piece_values[static_cast<int>(board.at(move.to()).type())];
        
        int attacker_val = piece_values[static_cast<int>(board.at(move.from()).type())];
        if (is_promo) victim_val += piece_values[4];

        // MVV-LVA ordering: 7 million + victim * 10 - attacker
        return 7'000'000 + victim_val * 10 - attacker_val;
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
    
    // Continuation history (1-ply)
    int cont1 = 0;
    if (prev_move != Move::NULL_MOVE) {
        int prev_piece = piece_idx(board, prev_move.to());
        int cur_piece  = piece_idx(board, move.from());
        cont1 = cont_hist[prev_piece][prev_move.to().index()][cur_piece][move.to().index()];
    }
    
    // Continuation history (2-ply)
    int cont2 = 0;
    if (prev_prev_move != Move::NULL_MOVE) {
        int pp_piece  = piece_idx(board, prev_prev_move.to());
        int cur_piece = piece_idx(board, move.from());
        cont2 = cont_hist[pp_piece][prev_prev_move.to().index()][cur_piece][move.to().index()];
    }

    return hist + cont1 / 2 + cont2 / 4;
}

// ─── Partial insertion sort: pick best move first, then sort rest lazily ──────
struct ScoredMove {
    Move move;
    int  score;
};

void score_moves(const Board& board, const Movelist& moves, ScoredMove* scored, int n, Move tt_move, int ply, Move prev_move, Move prev_prev_move) {
    for (int i = 0; i < n; ++i) {
        scored[i].move  = moves[i];
        scored[i].score = score_move(board, moves[i], tt_move, ply, prev_move, prev_prev_move);
    }
}

// Swap best remaining to position `start`
void partial_sort_next(ScoredMove* scored, int start, int n) {
    int best = start;
    for (int j = start + 1; j < n; ++j) {
        if (scored[j].score > scored[best].score) best = j;
    }
    if (best != start) std::swap(scored[start], scored[best]);
}

// ─── Quiescence Search ────────────────────────────────────────────────────────
int quiescence(Board& board, int alpha, int beta, int ply = 0) {
    // Use the centrally-managed abort flag set by negamax's time check
    if (abort_search) return 0;

    nodes.fetch_add(1, std::memory_order_relaxed);

    int stand_pat = evaluate(board);
    if (stand_pat >= beta) return beta;

    // Delta pruning (skip if we can't possibly improve alpha even with best capture)
    const int DELTA = 1000; // queen + some margin
    if (stand_pat + DELTA < alpha) return alpha;

    if (stand_pat > alpha) alpha = stand_pat;

    Movelist moves;
    Movelist all_moves;
    movegen::legalmoves<movegen::MoveGenType::ALL>(all_moves, board);
    for (int i = 0; i < all_moves.size(); ++i) {
        Move m = all_moves[i];
        bool is_passed_push = (m.typeOf() == Move::NORMAL && board.at(m.from()).type() == chess::PieceType::PAWN && (m.to().rank() == chess::Rank::RANK_7 || m.to().rank() == chess::Rank::RANK_2));
        if (board.isCapture(m) || m.typeOf() == Move::ENPASSANT || m.typeOf() == Move::PROMOTION || is_passed_push) {
            moves.add(m);
        }
    }
    
    int n = moves.size();
    ScoredMove scored[256];
    score_moves(board, moves, scored, n, Move::NULL_MOVE, -1, Move::NULL_MOVE, Move::NULL_MOVE);

    for (int i = 0; i < n; ++i) {
        partial_sort_next(scored, i, n);
        Move move = scored[i].move;
        
        // Skip losing captures (SEE < 0)
        if (!see_ge(board, move, 0)) continue;

        // Per-move delta pruning
        int capt_val = (move.typeOf() == Move::ENPASSANT) ? piece_values[0] :
                       (board.at(move.to()) != Piece::NONE) ? piece_values[static_cast<int>(board.at(move.to()).type())] : 0;
        if (move.typeOf() == Move::PROMOTION) capt_val += piece_values[4];
        if (stand_pat + capt_val + 200 < alpha) continue;

        bool is_king_move = (board.at(move.from()).type() == chess::PieceType::KING);
        
        
        // Always call update_accumulator to handle incremental updates (like captures) for the opponent

        board.makeMove(move);
        if (is_king_move) {
            // Only refresh the accumulator of the side that just moved (the opponent now)
            // nnue::refresh_accumulator(board, ~board.sideToMove());
        }
        int score = -quiescence(board, -beta, -alpha, ply + 1);
        board.unmakeMove(move);

        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

// ─── Negamax ─────────────────────────────────────────────────────────────────
int negamax(Board& board, int depth, int alpha, int beta, int ply, bool allow_null, Move prev_move = Move::NULL_MOVE, Move prev_prev_move = Move::NULL_MOVE, Move excluded_move = Move::NULL_MOVE) {
    if ((nodes.load(std::memory_order_relaxed) & 4095) == 0) {
        auto chk_now = std::chrono::high_resolution_clock::now();
        if (chk_now > end_time) {
            abort_search = true;
        }
    }
    if (abort_search) return 0;

    nodes.fetch_add(1, std::memory_order_relaxed);

    bool is_root = (ply == 0);
    bool is_pv   = (beta - alpha > 1);

    // Initialize PV
    if (ply < MAX_PLY) pv_length[ply] = 0;

    // Guard against excessive ply depth
    if (ply >= MAX_PLY - 1) return evaluate(board);

    if (!is_root && (board.isHalfMoveDraw() || board.isRepetition(1))) return 0;

    uint64_t hash = board.hash();
    __builtin_prefetch(&TT[(hash & (TT_MASK >> 1)) * 2]);

    // ── Syzygy WDL probe ──────────────────────────────────────────────────────
    if (!is_root && excluded_move == Move::NULL_MOVE) {
        int wdl = probe_wdl(board);
        if (wdl != -1) {
            int tb_score;
            if      (wdl == TB_WIN)          tb_score =  MATE_SCORE - 1000 + ply;
            else if (wdl == TB_CURSED_WIN)   tb_score =  1;
            else if (wdl == TB_LOSS)         tb_score = -(MATE_SCORE - 1000 + ply);
            else if (wdl == TB_BLESSED_LOSS) tb_score = -1;
            else                             tb_score =  0;

            TTEntry::Flag flag = (wdl == TB_WIN || wdl == TB_CURSED_WIN)   ? TTEntry::LOWER
                               : (wdl == TB_LOSS || wdl == TB_BLESSED_LOSS) ? TTEntry::UPPER
                               : TTEntry::EXACT;
            if (flag == TTEntry::EXACT ||
               (flag == TTEntry::LOWER && tb_score >= beta) ||
               (flag == TTEntry::UPPER && tb_score <= alpha)) {
                write_tt(hash, Move::NULL_MOVE, depth, tb_score, flag, ply);
                return tb_score;
            }
            if (flag == TTEntry::LOWER && tb_score > alpha) alpha = tb_score;
            if (flag == TTEntry::UPPER && tb_score < beta)  beta  = tb_score;
        }
    }


    
    Move tt_move = Move::NULL_MOVE;
    int tt_depth = 0;
    TTEntry::Flag tt_flag = TTEntry::NONE;
    int tt_eval = 0;

    // TT probe
    int tt_score = 0;
    bool tt_hit = probe_tt(hash, depth, alpha, beta, ply, tt_score, tt_move, tt_depth, tt_flag, tt_eval);
    if (tt_hit && excluded_move == Move::NULL_MOVE && !is_root) {
        return tt_score;
    }
    // probe_tt already fills tt_move from both buckets; no second scan needed.

    // Internal Iterative Reduction (IIR) - from GForce
    if (depth >= 6 && tt_move == Move::NULL_MOVE && !is_pv) {
        depth--;
    }

    // Mate distance pruning
    alpha = std::max(alpha, -MATE_SCORE + ply);
    beta  = std::min(beta,   MATE_SCORE - ply);
    if (alpha >= beta) return alpha;

    bool in_check = board.inCheck();

    // Check extension: only extend when we're not too deep (prevent infinite extension loops)
    // Standard practice: limit check extensions to not exceed 2x the root depth
    if (in_check && depth > 0 && ply < MAX_PLY * 2 / 3) depth++;

    if (depth <= 0) return quiescence(board, alpha, beta);

    // Compute static eval for pruning (avoid if in check)
    int static_eval = 0;
    if (!in_check) {
        // Use TT eval if available
        if (tt_flag != TTEntry::NONE) {
            static_eval = tt_eval;
        } else {
            static_eval = evaluate(board);
        }
        eval_history[ply] = static_eval;
    } else {
        eval_history[ply] = 0;
    }

    // Reverse futility pruning (static null-move pruning)
    if (!is_pv && !in_check && depth <= 8 && ply > 0 && excluded_move == Move::NULL_MOVE) {
        // "improving" check: if our position is better than 2 plies ago, we are improving
        bool improving = false;
        if (ply >= 2 && static_eval >= eval_history[ply - 2]) {
            improving = true;
        }
        
        // Adjust RFP margin based on improving status (GForce-like)
        int rfp_margin = opt_rfp_margin * depth;
        if (!improving) rfp_margin += 50; // Less aggressive pruning if not improving
        
        if (static_eval - rfp_margin >= beta) {
            return static_eval;
        }
    }

    // Null-move pruning
    if (allow_null && !is_pv && !in_check && ply > 0 && depth >= 3 && excluded_move == Move::NULL_MOVE) {
        // Avoid NMP in pure pawn/king endgames (zugzwang risk)
        // Use bitboard OR to check in one operation
        Bitboard non_pawn_kings = board.pieces(PieceType::KNIGHT) |
                                  board.pieces(PieceType::BISHOP) |
                                  board.pieces(PieceType::ROOK)   |
                                  board.pieces(PieceType::QUEEN);
        if (non_pawn_kings && static_eval >= beta) {
            int R = opt_nmp_base + depth / opt_nmp_depth_div + std::min(3, (static_eval - beta) / opt_nmp_eval_div);
            board.makeNullMove();
            int null_score = -negamax(board, depth - 1 - R, -beta, -beta + 1, ply + 1, false, Move::NULL_MOVE, prev_move);
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
    if (is_pv && tt_move == Move::NULL_MOVE && depth >= 5) {
        negamax(board, depth - 4, alpha, beta, ply, false, prev_move, prev_prev_move);
        tt_move = probe_tt_move(hash);
    }
    
    // Singular Extensions
    bool tt_is_singular = false;
    if (!is_root && depth >= 6 && tt_move != Move::NULL_MOVE && 
        excluded_move == Move::NULL_MOVE && 
        tt_depth >= depth - 3 && 
        tt_flag != TTEntry::UPPER && 
        std::abs(tt_eval) < MATE_SCORE - 1000) 
    {
        int singular_beta = tt_eval - depth;
        int singular_score = negamax(board, depth / 2, singular_beta - 1, singular_beta, ply, false, prev_move, prev_prev_move, tt_move);
        if (singular_score < singular_beta) {
            tt_is_singular = true;
        }
    }

    Movelist moves;
    movegen::legalmoves(moves, board);
    if (moves.empty()) {
        return in_check ? (-MATE_SCORE + ply) : 0; // Checkmate or stalemate
    }

    int n = moves.size();
    ScoredMove scored[256];
    score_moves(board, moves, scored, n, tt_move, ply, prev_move, prev_prev_move);

    int best_score    = -INF;
    Move best_move    = Move::NULL_MOVE;
    int  original_alpha = alpha;
    int  move_count   = 0;

    // Futility pruning margins
    int fp_margin = opt_fp_margin_base + (depth - 1) * opt_fp_margin_mult;

    Move quiets_searched[64];
    int num_quiets = 0;

    for (int mi = 0; mi < n; ++mi) {
        partial_sort_next(scored, mi, n);
        Move move = scored[mi].move;
        
        if (move == excluded_move) continue;

        move_count++;

        bool is_capture  = board.isCapture(move) || move.typeOf() == Move::ENPASSANT;
        bool is_promo    = move.typeOf() == Move::PROMOTION;
        chess::Piece piece = board.at(move.from());
        bool is_king_move  = (piece.type() == chess::PieceType::KING);

        // ── Pre-makeMove pruning (saves expensive board update for pruned moves) ──
        if (!is_root && !in_check && move_count > 1 && excluded_move == Move::NULL_MOVE) {
            if (!is_capture && !is_promo) {
                // Futility pruning: skip quiet moves that can't improve alpha
                if (depth <= 8 && static_eval + fp_margin <= alpha) continue;

                // Late move pruning: skip very late quiet moves at low depth
                if (!is_pv && depth <= 4 && move_count > 4 + depth * depth) continue;

                // SEE pruning for quiet moves at low depth
                if (!is_pv && depth <= 6 && !see_ge(board, move, -depth * 60)) continue;

                // Continuation history based pruning
                if (!is_pv && depth <= 6 && prev_move != Move::NULL_MOVE) {
                    int cp = (piece.color() == chess::Color::WHITE ? 0 : 6) + static_cast<int>(piece.type());
                    int pp;
                    if (move.from() == prev_move.to()) {
                        pp = cp;
                    } else {
                        chess::Piece prev_p = board.at(prev_move.to());
                        pp = prev_p == chess::Piece::NONE ? 0 : ((prev_p.color() == chess::Color::WHITE ? 0 : 6) + static_cast<int>(prev_p.type()));
                    }
                    if (cont_hist[pp][prev_move.to().index()][cp][move.to().index()] < -2000 * depth) continue;
                }
            } else if (is_capture && !is_promo && move_count > 1 && depth <= 6) {
                // SEE pruning for losing captures
                if (!see_ge(board, move, -depth * 100)) continue;
            }
        }

        if (!is_capture && !is_promo && num_quiets < 64) {
            quiets_searched[num_quiets++] = move;
        }

        // Update NNUE accumulator (lazy for king moves)
        
        
        // Always call update_accumulator to incrementally update opponent's features

        board.makeMove(move);
        if (is_king_move) {
            // Only refresh the side that made the king move
            // nnue::refresh_accumulator(board, ~board.sideToMove());
        }
        bool gives_check = board.inCheck();

        // Extensions: only apply when not too deep
        int extension = 0;
        if (ply < MAX_PLY * 2 / 3) {
            if (tt_is_singular && move == tt_move) extension = 1;
        }

        // Post-makeMove futility pruning for checks discovered during search
        if (!is_root && !in_check && !gives_check && !is_capture && !is_promo
            && depth <= 8 && move_count > 1 && static_eval + fp_margin <= alpha) {
        board.unmakeMove(move);
            continue;
        }

        // Cap effective depth to prevent runaway extensions
        int max_child_depth = std::max(0, std::min(depth - 1 + extension, MAX_PLY - ply - 1));
        
        int score;
        if (move_count == 1) {
            // PV node: full window
            score = -negamax(board, max_child_depth, -beta, -alpha, ply + 1, true, move, prev_move);
        } else {
            // LMR: reduce late quiet moves
            bool do_lmr = move_count > 3 && depth >= 3 && !is_capture && !is_promo
                          && !in_check && !gives_check;
            int R = 0;
            if (do_lmr) {
                int d = std::min(depth, MAX_PLY - 1);
                int m = std::min(move_count, 63);
                R = lmr_table[d][m];
                // Adjustments
                if (is_pv) R--;
                if (move == killer_moves[ply][0] || move == killer_moves[ply][1]) R--;
                if (gives_check) R--;
                
                // History-based LMR adjustment (GForce idea)
                int color = static_cast<int>(board.sideToMove());
                int hist_val = history_table[color][move.from().index()][move.to().index()];
                R -= hist_val / 4096; // Good history reduces less, bad history reduces more

                R = std::max(0, std::min(R, depth - 2));
            }

            // Null-window search (cap depth)
            int lmr_depth = std::max(0, std::min(depth - 1 - R + extension, MAX_PLY - ply - 1));
            score = -negamax(board, lmr_depth, -alpha - 1, -alpha, ply + 1, true, move, prev_move);

            // Full-depth re-search if LMR raised alpha or window failed
            if (score > alpha && R > 0) {
                score = -negamax(board, max_child_depth, -alpha - 1, -alpha, ply + 1, true, move, prev_move);
            }
            // Full-window re-search for PV update
            if (score > alpha && score < beta) {
                score = -negamax(board, depth - 1 + extension, -beta, -alpha, ply + 1, true, move, prev_move);
            }
        }
        board.unmakeMove(move);
        if (abort_search) return 0;

        if (score > best_score) {
            best_score = score;
            best_move  = move;
            
            // Update PV
            if (is_pv && score > alpha) {
                pv_table[ply][ply] = move;
                for (int i = ply + 1; i < pv_length[ply + 1]; ++i)
                    pv_table[ply][i] = pv_table[ply + 1][i];
                pv_length[ply] = std::max(ply + 1, pv_length[ply + 1]);
            }
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
                // History bonus: depth^2, with gravity
                int bonus = std::min(depth * depth, 400);
                int color = static_cast<int>(board.sideToMove());
                int fm = move.from().index(), tm = move.to().index();
                history_table[color][fm][tm]
                    += bonus - history_table[color][fm][tm] * std::abs(bonus) / 16384;
                
                // Continuation history update
                if (prev_move != Move::NULL_MOVE) {
                    int pp = piece_idx(board, prev_move.to());
                    int cp = piece_idx(board, move.from());
                    cont_hist[pp][prev_move.to().index()][cp][move.to().index()]
                        += bonus;
                }
                
                // History penalties for moves that didn't cause cutoff
                for (int i = 0; i < num_quiets - 1; ++i) {
                    Move q = quiets_searched[i];
                    int qf = q.from().index(), qt = q.to().index();
                    history_table[color][qf][qt]
                        -= bonus + history_table[color][qf][qt] * std::abs(bonus) / 16384;
                }
            }
            break;
        }
    }

    if (!abort_search && excluded_move == Move::NULL_MOVE) {
        TTEntry::Flag flag;
        if (best_score <= original_alpha) flag = TTEntry::UPPER;
        else if (best_score >= beta)      flag = TTEntry::LOWER;
        else                              flag = TTEntry::EXACT;
        write_tt(hash, best_move, depth, best_score, flag, ply);
    }

    return best_score;
}

// ─── Search Worker (Helper Threads) ───────────────────────────────────────────
void search_worker(Board board, int target_ms) {

    int previous_score = 0;
    for (int depth = 1; depth <= 64; ++depth) {
        if (abort_search) break;

        int alpha = -INF;
        int beta  = INF;
        
        if (depth >= 5) {
            alpha = previous_score - 50;
            beta  = previous_score + 50;
        }

        int score;
        int fail_low_cnt = 0;
        int fail_high_cnt = 0;
        
        while (true) {
            score = negamax(board, depth, alpha, beta, 0, true, Move::NULL_MOVE);
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
            last_search_score = score;
        }
    }
}

// ─── Thread Pool ──────────────────────────────────────────────────────────────
std::vector<std::thread> helper_threads;
std::mutex pool_mutex;
std::condition_variable pool_cv_threads;
std::condition_variable pool_cv_main;
bool pool_terminate = false;
int active_helpers = 0;
Board pool_board;
int pool_target_ms = 0;
uint64_t pool_job_id = 0;

void helper_thread_loop(int thread_id) {
    uint64_t my_job_id = 0;
    while (true) {
        std::unique_lock<std::mutex> lock(pool_mutex);
        pool_cv_threads.wait(lock, [&]{ return my_job_id != pool_job_id || pool_terminate; });
        if (pool_terminate) break;
        my_job_id = pool_job_id;
        
        Board b = pool_board;
        lock.unlock();
        
        search_worker(b, 0);
        
        lock.lock();
        active_helpers--;
        if (active_helpers == 0) {
            pool_cv_main.notify_one();
        }
    }
}

void stop_thread_pool() {
    {
        std::unique_lock<std::mutex> lock(pool_mutex);
        pool_terminate = true;
    }
    pool_cv_threads.notify_all();
    for (auto& t : helper_threads) {
        if (t.joinable()) t.join();
    }
    helper_threads.clear();
}

void resize_thread_pool(int new_threads) {
    stop_thread_pool();
    pool_terminate = false;
    num_threads = new_threads;
    for (int i = 1; i < num_threads; ++i) {
        helper_threads.emplace_back(helper_thread_loop, i);
    }
}

void start_helper_threads(const Board& b) {
    if (num_threads <= 1) return;
    std::unique_lock<std::mutex> lock(pool_mutex);
    pool_board = b;
    active_helpers = num_threads - 1;
    pool_job_id++;
    pool_cv_threads.notify_all();
}

void wait_helper_threads() {
    if (num_threads <= 1) return;
    std::unique_lock<std::mutex> lock(pool_mutex);
    pool_cv_main.wait(lock, []{ return active_helpers == 0; });
}

// ─── Iterative Deepening + Aspiration Windows ─────────────────────────────────
Move search_best_move(Board& board, int soft_limit, int hard_limit) {
    Move best_move     = Move::NULL_MOVE;
    nodes              = 0;
    abort_search       = false;
    current_age        = (current_age + 1) & 63;

    // History gravity: halve history between searches
    for (auto& c : history_table)
        for (auto& f : c)
            for (auto& t : f)
                t >>= 1;

    // Decay cont_hist every search (halve)
    // Use a single memset-based approach via arithmetic right shift in a tight loop
    for (int a = 0; a < 12; ++a)
        for (int b2 = 0; b2 < 64; ++b2)
            for (int c2 = 0; c2 < 12; ++c2)
                for (int d = 0; d < 64; ++d)
                    cont_hist[a][b2][c2][d] >>= 1;

    std::fill(&killer_moves[0][0], &killer_moves[0][0] + sizeof(killer_moves) / sizeof(Move), Move::NULL_MOVE);
    std::fill(&pv_length[0], &pv_length[0] + MAX_PLY, 0);

    auto start = std::chrono::high_resolution_clock::now();
    end_time   = start + std::chrono::milliseconds(std::max(1, hard_limit));

    int previous_score = 0;
    int best_move_stable_count = 0;
    Move last_best_move = Move::NULL_MOVE;
    int last_depth = 0;

    start_helper_threads(board);

    // ── Syzygy DTZ root probe ─────────────────────────────────────────────────
    // If the root position is in the tablebases, pick the best move immediately
    // using DTZ (Distance To Zero) for optimal conversion.
    if (tb_enabled) {
        int piece_cnt = board.occ().count();
        if (piece_cnt <= tb_pieces && board.castlingRights().isEmpty()) {
            uint64_t white   = board.us(Color::WHITE).getBits();
            uint64_t black   = board.us(Color::BLACK).getBits();
            uint64_t kings   = board.pieces(PieceType::KING).getBits();
            uint64_t queens  = board.pieces(PieceType::QUEEN).getBits();
            uint64_t rooks   = board.pieces(PieceType::ROOK).getBits();
            uint64_t bishops = board.pieces(PieceType::BISHOP).getBits();
            uint64_t knights = board.pieces(PieceType::KNIGHT).getBits();
            uint64_t pawns   = board.pieces(PieceType::PAWN).getBits();
            unsigned ep      = board.enpassantSq() == Square::underlying::NO_SQ ? 0
                             : board.enpassantSq().index();
            bool turn        = (board.sideToMove() == Color::WHITE);
            unsigned rule50  = board.halfMoveClock();

            unsigned results[TB_MAX_MOVES];
            unsigned tb_res = tb_probe_root(white, black, kings, queens, rooks, bishops, knights,
                                            pawns, rule50, 0, ep, turn, results);

            if (tb_res != TB_RESULT_FAILED) {
                // Find the result with best WDL, then lowest DTZ (fastest conversion)
                int best_wdl = -1;
                unsigned best_dtz = 0xFFF;
                unsigned best_res = TB_RESULT_FAILED;
                for (unsigned* r = results; *r != TB_RESULT_FAILED; ++r) {
                    int wdl = (int)TB_GET_WDL(*r);
                    unsigned dtz = TB_GET_DTZ(*r);
                    if (wdl > best_wdl || (wdl == best_wdl && dtz < best_dtz)) {
                        best_wdl = wdl;
                        best_dtz = dtz;
                        best_res = *r;
                    }
                }

                if (best_res != TB_RESULT_FAILED) {
                    unsigned from_sq  = TB_GET_FROM(best_res);
                    unsigned to_sq    = TB_GET_TO(best_res);
                    unsigned promotes = TB_GET_PROMOTES(best_res);
                    unsigned ep_flag  = TB_GET_EP(best_res);

                    Square from_s(from_sq);
                    Square to_s(to_sq);
                    Move tb_best = Move::NULL_MOVE;
                    if (promotes != TB_PROMOTES_NONE) {
                        PieceType promo = (promotes == TB_PROMOTES_QUEEN)  ? PieceType::QUEEN  :
                                          (promotes == TB_PROMOTES_ROOK)   ? PieceType::ROOK   :
                                          (promotes == TB_PROMOTES_BISHOP) ? PieceType::BISHOP :
                                                                             PieceType::KNIGHT;
                        tb_best = Move::make<Move::PROMOTION>(from_s, to_s, promo);
                    } else if (ep_flag) {
                        tb_best = Move::make<Move::ENPASSANT>(from_s, to_s);
                    } else {
                        tb_best = Move::make<Move::NORMAL>(from_s, to_s);
                    }

                    int tb_cp = (best_wdl == TB_WIN)          ?  (MATE_SCORE - 1000) :
                                (best_wdl == TB_CURSED_WIN)   ?   1                  :
                                (best_wdl == TB_LOSS)         ? -(MATE_SCORE - 1000) :
                                (best_wdl == TB_BLESSED_LOSS) ?  -1                  : 0;

                    std::string wdl_str = (best_wdl == TB_WIN)          ? "win"  :
                                          (best_wdl == TB_CURSED_WIN)   ? "draw" :
                                          (best_wdl == TB_LOSS)         ? "loss" :
                                          (best_wdl == TB_BLESSED_LOSS) ? "draw" : "draw";

                    std::string score_str;
                    if (tb_cp > MATE_SCORE - 200)
                        score_str = "mate " + std::to_string((MATE_SCORE - tb_cp + 1) / 2);
                    else if (tb_cp < -(MATE_SCORE - 200))
                        score_str = "mate -" + std::to_string((MATE_SCORE + tb_cp + 1) / 2);
                    else
                        score_str = "cp " + std::to_string(tb_cp);

                    std::cout << "info depth 1 score " << score_str
                              << " tbwdl " << wdl_str
                              << " tbdtz " << best_dtz
                              << " nodes 1 pv " << uci::moveToUci(tb_best) << "\n";
                    std::cout.flush();
                    std::cout << "bestmove " << uci::moveToUci(tb_best) << "\n";
                    std::cout.flush();
                    // Signal abort so the outer go() loop exits cleanly
                    abort_search = true;
                    return tb_best;
                }
            }
        }
    }


    for (int depth = 1; depth <= 64; ++depth) {
        int alpha, beta;

        // Progressive aspiration windows
        const int window = 30;  // slightly wider to reduce re-searches
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
            score = negamax(board, depth, alpha, beta, 0, true, Move::NULL_MOVE);
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
        last_depth = depth;

        // Extract best move (prefer PV table, fall back to TT)
        if (pv_length[0] > 0) {
            best_move = pv_table[0][0];
        } else {
            Move found = probe_tt_move(board.hash());
            if (found != Move::NULL_MOVE) best_move = found;
        }

        if (best_move == last_best_move) {
            best_move_stable_count++;
        } else {
            best_move_stable_count = 0;
            last_best_move = best_move;
        }

        auto now = std::chrono::high_resolution_clock::now();
        int elapsed_ms = static_cast<int>(std::chrono::duration<double>(now - start).count() * 1000);
        long long nps  = (elapsed_ms > 0) ? (nodes.load() * 1000LL / elapsed_ms) : 0;

        // Build PV string
        std::string pv_str;
        for (int i = 0; i < pv_length[0] && i < depth; ++i) {
            if (i > 0) pv_str += " ";
            pv_str += uci::moveToUci(pv_table[0][i]);
        }
        if (pv_str.empty() && best_move != Move::NULL_MOVE)
            pv_str = uci::moveToUci(best_move);

        // Score output: handle mates
        std::string score_str;
        if (score > MATE_SCORE - 200)
            score_str = "mate " + std::to_string((MATE_SCORE - score + 1) / 2);
        else if (score < -MATE_SCORE + 200)
            score_str = "mate " + std::to_string(-(MATE_SCORE + score + 1) / 2);
        else
            score_str = "cp " + std::to_string(score);

        if (!is_selfplay) {
            std::cout << "info depth " << depth
                      << " score " << score_str
                      << " nodes " << nodes.load()
                      << " time " << elapsed_ms
                      << " nps " << nps
                      << " pv " << pv_str
                      << "\n";
        }
        std::cout.flush();

        // Stop early if mate found
        if (score > MATE_SCORE - 200 || score < -MATE_SCORE + 200) break;
        
        // Time management: smart soft-limit based on elapsed + stability
        auto now_check = std::chrono::high_resolution_clock::now();
        int elapsed_check = static_cast<int>(
            std::chrono::duration<double>(now_check - start).count() * 1000);

        // Stability bonus: if best move hasn't changed for 3+ depths, stop earlier
        int effective_soft = soft_limit;
        if (best_move_stable_count >= 3) effective_soft = soft_limit * 6 / 10;

        if (elapsed_check >= effective_soft) { abort_search = true; break; }

        // Emergency: if we used >55% of soft budget and next depth will likely
        // exceed it (depth took more than 30% of soft), abort now
        if (elapsed_check > soft_limit * 55 / 100 && depth >= 6) {
            abort_search = true;
            break;
        }
    }

    // Safety: if somehow no move was found, play first legal
    if (best_move == Move::NULL_MOVE) {
        Movelist ml;
        movegen::legalmoves(ml, board);
        if (!ml.empty()) best_move = ml[0];
    }

    // Wait for helper threads
    wait_helper_threads();

    return best_move;
}

// ─── UCI Loop ─────────────────────────────────────────────────────────────────
int main() {
    init_tt();
    tt_clear();
    // Initialize LMR table
    init_lmr_table();
    
    // Custom evaluation does not require weight loading
    std::cout << "G-ForceZero Engine ready.\n";
    std::cout.flush();

    chess::Board board;
    board.setFen(chess::constants::STARTPOS);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string command;
        ss >> command;

        if (command == "uci") {
            std::cout << "id name G-ForceZero\n"
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
                      << "option name SyzygyPath type string default <empty>\n"
                      << "option name Hash type spin default 16 min 1 max 1048576\n"
                      << "uciok\n";
        } else if (command == "setoption") {
            std::string name, name_val, value, val_val;
            ss >> name >> name_val >> value >> val_val;
            if (name_val == "Threads") {
                resize_thread_pool(std::max(1, std::min(64, std::stoi(val_val))));
            } else if (name_val == "Hash") {
                init_tt(std::max(1, std::stoi(val_val)));
            } else if (name_val == "NNUE_Weight") {
                opt_nnue_weight = std::max(0, std::min(100, std::stoi(val_val)));
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
                init_lmr_table(); // Rebuild LMR table
            } else if (name_val == "FP_Margin_Base") {
                opt_fp_margin_base = std::stoi(val_val);
            } else if (name_val == "FP_Margin_Mult") {
                opt_fp_margin_mult = std::stoi(val_val);
            } else if (name_val == "SyzygyPath") {
                // val_val may only be the first token; rebuild full value from rest of line
                std::string full_path = val_val;
                std::string extra;
                while (ss >> extra) full_path += " " + extra;
                syzygy_path = full_path;
                if (!syzygy_path.empty() && syzygy_path != "<empty>") {
                    if (syzygy_tb_init(syzygy_path.c_str())) {
                        tb_enabled = true;
                        tb_pieces  = static_cast<int>(TB_LARGEST);
                        std::cout << "info string Syzygy TBs loaded from '" << syzygy_path
                                  << "' (" << TB_LARGEST << "-piece)\n";
                    } else {
                        tb_enabled = false;
                        std::cout << "info string Failed to load Syzygy TBs from '" << syzygy_path << "'\n";
                    }
                }
            }
        } else if (command == "isready") {
            std::cout << "readyok\n";
            board.setFen(chess::constants::STARTPOS);
            tt_clear();
            std::fill(&history_table[0][0][0], &history_table[0][0][0] + sizeof(history_table) / sizeof(int), 0);
            std::fill(&counter_moves[0][0], &counter_moves[0][0] + sizeof(counter_moves) / sizeof(Move), Move::NULL_MOVE);
            std::memset(cont_hist, 0, sizeof(cont_hist));
        } else if (command == "eval") {
            int score = classical_evaluate(board);
            std::cout << "Evaluation: " << score << "\n";
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

            int soft_limit, hard_limit;
            if (movetime > 0) {
                soft_limit = movetime - MOVE_OVERHEAD;
                hard_limit = movetime - MOVE_OVERHEAD;
            } else if (wtime > 0 || btime > 0) {
                int my_time = (board.sideToMove() == chess::Color::WHITE) ? wtime : btime;
                int my_inc  = (board.sideToMove() == chess::Color::WHITE) ? winc  : binc;

                // Emergency: if nearly out of time, respond instantly
                if (my_time < 100) {
                    chess::Move best = get_book_move(board, "book.bin");
                    if (best == chess::Move::NULL_MOVE) {
                        chess::Movelist ml;
                        chess::movegen::legalmoves(ml, board);
                        if (!ml.empty()) best = ml[0];
                    }
                    std::cout << "bestmove " << chess::uci::moveToUci(best) << "\n";
                    std::cout.flush();
                    continue;
                }

                // Smart time allocation:
                // moves_left: use movestogo if given, else estimate based on remaining time
                // In low-time situations, be more conservative (fewer moves_left assumed)
                int moves_left;
                if (movestogo > 0) {
                    moves_left = movestogo;
                } else {
                    // Heuristic: assume 30 moves left in the game, but clamp based on time
                    // Low time → assume fewer moves so we spend less per move
                    moves_left = 30;
                    if (my_time < 5000)  moves_left = 20;
                    if (my_time < 2000)  moves_left = 15;
                    if (my_time < 1000)  moves_left = 10;
                }

                // Base soft limit: proportional share + increment bonus
                int base = my_time / moves_left + (my_inc * 2) / 3;

                // Soft limit: don't exceed 20% of remaining time on one move
                soft_limit = std::min(base, my_time / 5);

                // Hard limit: absolute ceiling — never use more than 25% of clock
                hard_limit = std::min({
                    soft_limit * 4,           // up to 4× soft
                    my_time / 4,              // 25% of remaining time
                    my_time - MOVE_OVERHEAD   // leave at least overhead
                });
            } else {
                soft_limit = 5000; // Analysis mode
                hard_limit = 5000;
            }
            soft_limit = std::max(20, soft_limit - MOVE_OVERHEAD);
            hard_limit = std::max(20, hard_limit - MOVE_OVERHEAD);
            // Sanity: hard must be >= soft
            hard_limit = std::max(hard_limit, soft_limit);

            chess::Move best = get_book_move(board, "book.bin");
            if (best != chess::Move::NULL_MOVE) {
                std::cout << "info string Playing from PolyGlot opening book\n";
                std::cout << "bestmove " << chess::uci::moveToUci(best) << "\n";
                std::cout.flush();
            } else {
                best = search_best_move(board, soft_limit, hard_limit);
                std::cout << "bestmove " << chess::uci::moveToUci(best) << "\n";
                std::cout.flush();
            }
        } else if (command == "stop") {
            abort_search = true;
        } else if (command == "quit") {
            break;
        }
    }
    stop_thread_pool();
    return 0;
}
