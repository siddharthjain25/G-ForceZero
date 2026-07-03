#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <thread>
#include <atomic>
#include <cstring>
#include "chess.hpp"
#include "nnue.hpp"

using namespace chess;

const int INF = 30000;
const int MATE_SCORE = 20000;

// [NEW] Packed 8-byte TTEntry to prevent torn reads in multithreading (Lazy SMP)
struct TTEntry {
    uint16_t key;     // Upper 16 bits of the hash
    uint16_t move;    // 16-bit packed move
    int16_t score;    // Score
    int8_t depth;     // Search depth
    uint8_t flag;     // EXACT, LOWER, UPPER, NONE

    enum Flag : uint8_t { EXACT = 0, LOWER = 1, UPPER = 2, NONE = 3 };
};

// Guarantee it fits in exactly 8 bytes
static_assert(sizeof(TTEntry) == 8, "TTEntry must be 8 bytes for SMP safety!");

const int TT_SIZE = 1024 * 1024; // 1M entries
std::vector<TTEntry> TT(TT_SIZE, {0, 0, 0, 0, TTEntry::NONE});

// [NEW] Lockless TT writer wrapper
void write_tt(uint64_t hash, Move move, int depth, int score, TTEntry::Flag flag) {
    int index = hash % TT_SIZE;
    // Always replace if the new search is deeper or provides an exact score
    if (TT[index].flag == TTEntry::NONE || TT[index].depth <= depth || TT[index].flag == TTEntry::EXACT) {
        TT[index].key = static_cast<uint16_t>(hash >> 48); // Store top 16 bits
        TT[index].move = move.move();                      // Extract raw uint16_t move
        TT[index].score = score;
        TT[index].depth = depth;
        TT[index].flag = flag;
    }
}

// [NEW] Heuristic Arrays for Move Ordering
const int MAX_PLY = 128;
Move killer_moves[MAX_PLY][2];
int history_table[2][64][64]; // [Color][FromSq][ToSq]

std::atomic<bool> abort_search{false};
std::chrono::time_point<std::chrono::high_resolution_clock> end_time;
uint64_t nodes = 0;

int piece_values[6] = {100, 320, 330, 500, 900, 0}; // P, N, B, R, Q, K

// ── Standard PST tables (Chessprogramming Wiki / Fruit style) ─────────────────
// Indexed rank8→rank1, a→h. For White pieces: use pst[sq^56]. For Black: pst[sq].
const int pst_pawn[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
};
const int pst_knight[64] = {
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
};
const int pst_bishop[64] = {
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
};
const int pst_rook[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
};
const int pst_queen[64] = {
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
};
// King safety: penalize exposed king in middlegame, reward castled position
const int pst_king[64] = {
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
};

const int pst_king_endgame[64] = {
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50
};

const int* pst[6] = { pst_pawn, pst_knight, pst_bishop, pst_rook, pst_queen, pst_king };

// Classical material+PST + king safety + bishop pair + rook on open file
int classical_evaluate(const Board& board) {
    int score = 0;
    int phase = 0;

    // Calculate game phase
    phase += board.pieces(PieceType::KNIGHT).count() * 1;
    phase += board.pieces(PieceType::BISHOP).count() * 1;
    phase += board.pieces(PieceType::ROOK).count() * 2;
    phase += board.pieces(PieceType::QUEEN).count() * 4;
    int phase_weight = (phase * 256 + 12) / 24;
    if (phase_weight > 256) phase_weight = 256;

    // 1. Material + PST
    for (int pt = 0; pt < 6; ++pt) {
        PieceType pType = PieceType(static_cast<PieceType::underlying>(pt));
        Bitboard wb = board.pieces(pType, Color::WHITE);
        while (wb) { 
            int sq = wb.pop(); 
            score += piece_values[pt];
            if (pt == 5) { // KING
                int mg = pst_king[sq ^ 56];
                int eg = pst_king_endgame[sq ^ 56];
                score += (mg * phase_weight + eg * (256 - phase_weight)) / 256;
            } else {
                score += pst[pt][sq ^ 56]; 
            }
        }
        Bitboard bb = board.pieces(pType, Color::BLACK);
        while (bb) { 
            int sq = bb.pop(); 
            score -= piece_values[pt];
            if (pt == 5) { // KING
                int mg = pst_king[sq];
                int eg = pst_king_endgame[sq];
                score -= (mg * phase_weight + eg * (256 - phase_weight)) / 256;
            } else {
                score -= pst[pt][sq]; 
            }
        }
    }

    // 2. Bishop pair bonus
    if (board.pieces(PieceType::BISHOP, Color::WHITE).count() >= 2) score += 30;
    if (board.pieces(PieceType::BISHOP, Color::BLACK).count() >= 2) score -= 30;

    // 3. Rook on open/semi-open file bonus
    Bitboard all_pawns = board.pieces(PieceType::PAWN);
    Bitboard w_pawns   = board.pieces(PieceType::PAWN, Color::WHITE);
    Bitboard b_pawns   = board.pieces(PieceType::PAWN, Color::BLACK);
    auto file_mask = [](int f) -> Bitboard {
        uint64_t m = 0x0101010101010101ULL << f;
        return Bitboard(m);
    };
    for (int f = 0; f < 8; ++f) {
        Bitboard fm = file_mask(f);
        bool no_pawn   = (all_pawns & fm).empty();
        bool no_w_pawn = (w_pawns   & fm).empty();
        bool no_b_pawn = (b_pawns   & fm).empty();
        Bitboard wr = board.pieces(PieceType::ROOK, Color::WHITE) & fm;
        Bitboard br = board.pieces(PieceType::ROOK, Color::BLACK) & fm;
        if (wr.count() > 0) { score += no_pawn ? 20 : (no_w_pawn ? 10 : 0); }
        if (br.count() > 0) { score -= no_pawn ? 20 : (no_b_pawn ? 10 : 0); }
    }

    // 4. King safety: penalise enemy pieces attacking squares near the king
    auto king_safety = [&](Color us, Color them) -> int {
        int penalty = 0;
        Square ksq = board.kingSq(us);
        int kf = ksq.file(), kr = ksq.rank();
        // Count attackers to the 3x3 zone around the king
        int attacker_count = 0;
        for (int df = -1; df <= 1; ++df) {
            for (int dr = -1; dr <= 1; ++dr) {
                int f2 = kf + df, r2 = kr + dr;
                if (f2 < 0 || f2 > 7 || r2 < 0 || r2 > 7) continue;
                Square tsq = Square(r2 * 8 + f2);
                if (board.isAttacked(tsq, them)) attacker_count++;
            }
        }
        penalty = attacker_count * attacker_count * 4; // quadratic penalty
        // Bonus: pawn shield — pawns directly in front of king
        for (int df = -1; df <= 1; ++df) {
            int f2 = kf + df;
            if (f2 < 0 || f2 > 7) continue;
            int shield_rank = (us == Color::WHITE) ? kr + 1 : kr - 1;
            if (shield_rank < 0 || shield_rank > 7) continue;
            Square shield_sq = Square(shield_rank * 8 + f2);
            Piece p = board.at(shield_sq);
            if (p.type() == PieceType::PAWN && p.color() == us) penalty -= 10;
        }
        return penalty;
    };
    score -= king_safety(Color::WHITE, Color::BLACK);
    score += king_safety(Color::BLACK, Color::WHITE);

    // 4. Passed pawn bonuses
    Bitboard temp_w = w_pawns;
    while (temp_w) {
        int sq = temp_w.pop();
        int f = sq % 8;
        int r = sq / 8;
        uint64_t mask = 0;
        for (int rank = r + 1; rank < 8; ++rank) {
            mask |= (1ULL << (rank * 8 + f));
            if (f > 0) mask |= (1ULL << (rank * 8 + f - 1));
            if (f < 7) mask |= (1ULL << (rank * 8 + f + 1));
        }
        if ((mask & b_pawns.getBits()) == 0) {
            score += 20 * r * r; // exponential bonus: r=1: 20, r=4: 320, r=6: 720
        }
    }

    Bitboard temp_b = b_pawns;
    while (temp_b) {
        int sq = temp_b.pop();
        int f = sq % 8;
        int r = sq / 8;
        uint64_t mask = 0;
        for (int rank = r - 1; rank >= 0; --rank) {
            mask |= (1ULL << (rank * 8 + f));
            if (f > 0) mask |= (1ULL << (rank * 8 + f - 1));
            if (f < 7) mask |= (1ULL << (rank * 8 + f + 1));
        }
        if ((mask & w_pawns.getBits()) == 0) {
            int dist = 7 - r;
            score -= 20 * dist * dist; // exponential bonus
        }
    }

    // 5. Castling rights bonus
    if (board.castlingRights().has(Color::WHITE)) score += 20;
    if (board.castlingRights().has(Color::BLACK)) score -= 20;

    return (board.sideToMove() == Color::WHITE) ? score : -score;
}


// [NEW] Added 'ply' parameter to score Killer and History moves
int score_move(const Board& board, const Move& move, Move tt_move, int ply) {
    if (move == tt_move) return 1000000;

    if (board.isCapture(move) || move.typeOf() == Move::PROMOTION) {
        int victim_type = static_cast<int>(board.at(move.to()).type());
        int attacker_type = static_cast<int>(board.at(move.from()).type());

        if (move.typeOf() == Move::ENPASSANT) victim_type = static_cast<int>(PieceType::PAWN);

        // MVV-LVA: Most Valuable Victim, Least Valuable Attacker
        return 10000 + piece_values[victim_type] * 10 - piece_values[attacker_type];
    } else {
        // [NEW] Killer Heuristic (Recent quiet moves that caused cutoffs)
        if (ply >= 0 && ply < MAX_PLY) {
            if (killer_moves[ply][0] == move) return 9000;
            if (killer_moves[ply][1] == move) return 8000;
        }

        // [NEW] History Heuristic (Historical quiet moves that caused cutoffs)
        return history_table[board.sideToMove()][move.from().index()][move.to().index()];
    }
}

// [NEW] Added 'ply' parameter
void sort_moves(const Board& board, Movelist& moves, Move tt_move = Move::NULL_MOVE, int ply = -1) {
    std::vector<std::pair<int, Move>> scored_moves;
    scored_moves.reserve(moves.size());
    for (const auto& move : moves) {
        scored_moves.push_back({score_move(board, move, tt_move, ply), move});
    }

    std::sort(scored_moves.begin(), scored_moves.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    moves.clear();
    for (const auto& sm : scored_moves) {
        moves.add(sm.second);
    }
}

int evaluate(const Board& board, const nnue::Accumulator& acc) {
    auto [reason, result] = board.isGameOver();
    if (result != GameResult::NONE) {
        if (result == GameResult::DRAW) return 0;
        return (board.sideToMove() == Color::WHITE) ? -MATE_SCORE : MATE_SCORE;
    }

    // Hybrid: classical eval (material + PST) is the backbone;
    // NNUE provides positional bonus scaled to 15% to avoid overriding material signals.
    int classical = classical_evaluate(board);
    int nnue_score = nnue::evaluate(acc, board.sideToMove());
    return (classical * 17 + nnue_score * 3) / 20;
}

int quiescence(Board& board, int alpha, int beta, nnue::Accumulator acc) {
    if ((nodes & 2047) == 0 && std::chrono::high_resolution_clock::now() > end_time) abort_search = true;
    if (abort_search) return 0;

    nodes++;

    int stand_pat = evaluate(board, acc);
    if (stand_pat >= beta) return beta;
    if (alpha < stand_pat) alpha = stand_pat;

    Movelist moves;
    movegen::legalmoves<movegen::MoveGenType::CAPTURE>(moves, board);
    sort_moves(board, moves);

    for (const auto& move : moves) {
        nnue::Accumulator next_acc;
        chess::Piece piece = board.at(move.from());
        nnue::update_accumulator(board, move, acc, next_acc);

        board.makeMove(move);
        if (piece.type() == chess::PieceType::KING) {
            // After king moves, both sides' accumulators need a full refresh
            // because HalfKP features for BOTH perspectives depend on both king squares.
            nnue::refresh_accumulator(board, ~board.sideToMove(), next_acc); // mover
            nnue::refresh_accumulator(board, board.sideToMove(), next_acc);  // opponent
        }
        int score = -quiescence(board, -beta, -alpha, next_acc);
        board.unmakeMove(move);

        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

int negamax(Board& board, int depth, int alpha, int beta, int ply, bool allow_null, nnue::Accumulator acc) {
    if ((nodes & 2047) == 0 && std::chrono::high_resolution_clock::now() > end_time) abort_search = true;
    if (abort_search) return 0;

    nodes++;

    if (ply > 0 && (board.isHalfMoveDraw() || board.isRepetition())) return 0;

    uint64_t hash = board.hash();
    TTEntry& tte = TT[hash % TT_SIZE];
    Move tt_move = Move::NULL_MOVE;

    // [NEW] Unpack the compressed TT key correctly
    if (tte.key == static_cast<uint16_t>(hash >> 48)) {
        tt_move = Move(tte.move);
        if (tte.depth >= depth && ply > 0) {
            if (tte.flag == TTEntry::EXACT) return tte.score;
            if (tte.flag == TTEntry::LOWER && tte.score >= beta) return beta;
            if (tte.flag == TTEntry::UPPER && tte.score <= alpha) return alpha;
        }
    }

    if (depth <= 0) return quiescence(board, alpha, beta, acc);

    bool in_check = board.inCheck();

    // Null Move Pruning
    if (allow_null && depth >= 3 && !in_check && ply > 0) {
        if (board.pieces(PieceType::KNIGHT).count() > 0 ||
            board.pieces(PieceType::BISHOP).count() > 0 ||
            board.pieces(PieceType::ROOK).count() > 0 ||
            board.pieces(PieceType::QUEEN).count() > 0) {

            board.makeNullMove();
        int R = 2; // Depth reduction
        int null_score = -negamax(board, depth - 1 - R, -beta, -beta + 1, ply + 1, false, acc);
        board.unmakeNullMove();

        if (abort_search) return 0;
        if (null_score >= beta) {
            return beta; // Prune!
        }
            }
    }

    Movelist moves;
    movegen::legalmoves(moves, board);
    if (moves.empty()) {
        if (in_check) return -MATE_SCORE + ply;
        return 0; // Stalemate
    }

    // [NEW] Pass ply into sort_moves to utilize Killer/History heuristics
    sort_moves(board, moves, tt_move, ply);

    int best_score = -INF;
    Move best_move = Move::NULL_MOVE;
    int original_alpha = alpha;
    int move_count = 0;

    for (const auto& move : moves) {
        bool is_capture = board.isCapture(move);
        bool is_promotion = move.typeOf() == Move::PROMOTION;

        nnue::Accumulator next_acc;
        chess::Piece piece = board.at(move.from());
        nnue::update_accumulator(board, move, acc, next_acc);

        board.makeMove(move);
        if (piece.type() == chess::PieceType::KING) {
            // After king moves, both sides' accumulators need a full refresh
            nnue::refresh_accumulator(board, ~board.sideToMove(), next_acc); // mover
            nnue::refresh_accumulator(board, board.sideToMove(), next_acc);  // opponent
        }
        move_count++;
        int score;

        if (move_count == 1) {
            // First move: Full window
            score = -negamax(board, depth - 1, -beta, -alpha, ply + 1, true, next_acc); if (ply == 0) std::cout << "Root move " << uci::moveToUci(move) << " score=" << score << " alpha=" << alpha << " beta=" << beta << " depth=" << depth << "\n";
        } else {
            // LMR and PVS
            bool is_checking = board.inCheck();
            if (move_count > 3 && depth >= 3 && !is_capture && !is_promotion && !in_check && !is_checking) {
                // Reduced depth search
                int R = 1; // Reduction amount
                score = -negamax(board, depth - 1 - R, -alpha - 1, -alpha, ply + 1, true, next_acc);

                if (score > alpha && score < beta) {
                    // Re-search at full depth with PVS full window
                    score = -negamax(board, depth - 1, -beta, -alpha, ply + 1, true, next_acc); if (ply == 0) std::cout << "Root move " << uci::moveToUci(move) << " score=" << score << " alpha=" << alpha << " beta=" << beta << " depth=" << depth << "\n";
                }
            } else {
                // PVS: Null-window search
                score = -negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1, true, next_acc);
                if (score > alpha && score < beta) {
                    // PVS: Full-window search
                    score = -negamax(board, depth - 1, -beta, -alpha, ply + 1, true, next_acc); if (ply == 0) std::cout << "Root move " << uci::moveToUci(move) << " score=" << score << " alpha=" << alpha << " beta=" << beta << " depth=" << depth << "\n";
                }
            }
        }

        board.unmakeMove(move);

        if (abort_search) return 0;

        if (score > best_score) {
            best_score = score;
            best_move = move;
        }
        if (score > alpha) alpha = score;

        if (alpha >= beta) {
            // [NEW] Update Killer and History moves on beta cutoff
            if (!is_capture && !is_promotion && ply < MAX_PLY) {
                if (killer_moves[ply][0] != move) {
                    killer_moves[ply][1] = killer_moves[ply][0];
                    killer_moves[ply][0] = move;
                }
                history_table[board.sideToMove()][move.from().index()][move.to().index()] += depth * depth;
            }
            break; // Alpha-beta pruning
        }
    }

    if (!abort_search) {
        // [NEW] Use the lockless TT writer function
        TTEntry::Flag flag;
        if (best_score <= original_alpha) flag = TTEntry::UPPER;
        else if (best_score >= beta) flag = TTEntry::LOWER;
        else flag = TTEntry::EXACT;

        write_tt(hash, best_move, depth, best_score, flag);
    }

    return best_score;
}

Move search_best_move(Board& board, int target_ms) {
    Move best_move = Move::NULL_MOVE;
    Move iter_best_move = Move::NULL_MOVE;
    nodes = 0;
    abort_search = false;

    // [NEW] Reset heuristics for fresh search
    std::fill(&history_table[0][0][0], &history_table[0][0][0] + sizeof(history_table) / sizeof(int), 0);
    std::fill(&killer_moves[0][0], &killer_moves[0][0] + sizeof(killer_moves) / sizeof(Move), Move::NULL_MOVE);

    auto start = std::chrono::high_resolution_clock::now();
    end_time = start + std::chrono::milliseconds(target_ms);

    int num_threads = 1; // Hardcode 4 threads for now
    std::vector<std::thread> threads;

    nnue::Accumulator root_acc;
    nnue::init_accumulator(board, root_acc);

    // Worker threads (Lazy SMP)
    for (int i = 1; i < num_threads; ++i) {
        threads.emplace_back([&, i, board, root_acc]() mutable {
            for (int depth = 1 + (i % 2); depth <= 64; ++depth) {
                negamax(board, depth, -INF, INF, 0, true, root_acc);
                if (abort_search) break;
            }
        });
    }

    // [NEW] Main thread with Aspiration Windows
    int previous_score = 0;
    for (int depth = 1; depth <= 64; ++depth) {

        int alpha = -INF;
        int beta = INF;

        // Form the window around the previous search iteration
        if (depth >= 4) {
            alpha = std::max(-INF, previous_score - 50);
            beta = std::min(INF, previous_score + 50);
        }

        int score;
        while (true) {
            score = negamax(board, depth, alpha, beta, 0, true, root_acc);
            if (abort_search) break;

            // Re-search handling if score falls outside the aspiration window
            if (score <= alpha) {
                alpha = -INF;
                continue;
            } else if (score >= beta) {
                beta = INF;
                continue;
            }
            break;
        }

        if (abort_search) break;
        previous_score = score;

        uint64_t hash = board.hash();
        if (TT[hash % TT_SIZE].key == static_cast<uint16_t>(hash >> 48)) {
            iter_best_move = Move(TT[hash % TT_SIZE].move);
            best_move = iter_best_move;
        }

        auto current = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = current - start;

        std::cout << "info depth " << depth << " score cp " << score
        << " nodes " << nodes << " time " << static_cast<int>(diff.count() * 1000)
        << " nps " << static_cast<int>(nodes / (diff.count() + 0.0001))
        << " pv " << uci::moveToUci(best_move) << std::endl;

        if (score > MATE_SCORE - 100 || score < -MATE_SCORE + 100) break;
        if (diff.count() * 1000 > target_ms / 2.0) break;
    }

    abort_search = true; // Signal all workers to stop

    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }

    if (best_move == Move::NULL_MOVE) {
        Movelist ml;
        movegen::legalmoves(ml, board);
        if (!ml.empty()) best_move = ml[0];
    }

    return best_move;
}


