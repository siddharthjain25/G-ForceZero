#include "nnue.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <immintrin.h>

namespace nnue {

alignas(32) int16_t fc1_w[NUM_FEATURES][HIDDEN_SIZE];
alignas(32) int16_t fc1_b[HIDDEN_SIZE];

alignas(32) int32_t fc2_b[32];
alignas(32) int8_t  fc2_w[32][HIDDEN_SIZE * 2];

alignas(32) int32_t fc3_b[32];
alignas(32) int8_t  fc3_w[32][32];

alignas(32) int32_t fc4_b[1];
alignas(32) int8_t  fc4_w[1][32];

void load_weights(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open NNUE weights file: " + filepath);
    }
    
    file.read(reinterpret_cast<char*>(fc1_w), sizeof(fc1_w));
    file.read(reinterpret_cast<char*>(fc1_b), sizeof(fc1_b));
    
    file.read(reinterpret_cast<char*>(fc2_w), sizeof(fc2_w));
    file.read(reinterpret_cast<char*>(fc2_b), sizeof(fc2_b));

    file.read(reinterpret_cast<char*>(fc3_w), sizeof(fc3_w));
    file.read(reinterpret_cast<char*>(fc3_b), sizeof(fc3_b));

    file.read(reinterpret_cast<char*>(fc4_w), sizeof(fc4_w));
    file.read(reinterpret_cast<char*>(fc4_b), sizeof(fc4_b));
    
    if (!file) {
        throw std::runtime_error("Error reading NNUE weights!");
    }
}

int get_piece_idx(chess::Color perspective, chess::Piece piece) {
    int pt = static_cast<int>(piece.type()); // Pawn=0, Knight=1, Bishop=2, Rook=3, Queen=4
    bool is_mine = (piece.color() == perspective);
    return is_mine ? pt : pt + 5;
}

int get_feature_index(chess::Color perspective, chess::Piece piece, chess::Square sq, chess::Square king_sq) {
    if (piece == chess::Piece::NONE || piece.type() == chess::PieceType::KING) return -1; // King has no feature

    int sq_idx = sq.index();
    int k_sq_idx = king_sq.index();
    
    if (perspective == chess::Color::BLACK) {
        sq_idx ^= 56;
        k_sq_idx ^= 56;
    }
    
    int pt_idx = get_piece_idx(perspective, piece);
    return k_sq_idx * 641 + pt_idx * 64 + sq_idx;
}

void refresh_accumulator(const chess::Board& board, chess::Color perspective, Accumulator& acc) {
    int16_t* target = (perspective == chess::Color::WHITE) ? acc.white : acc.black;
    const int16_t* bias = fc1_b;
    
    // Copy biases using aligned AVX2 loads (both arrays are 32-byte aligned)
    for (int i = 0; i < HIDDEN_SIZE; i += 16) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(&target[i]),
                           _mm256_load_si256(reinterpret_cast<const __m256i*>(&bias[i])));
    }
    
    chess::Square king_sq = board.kingSq(perspective);
    
    for (int sq = 0; sq < 64; ++sq) {
        chess::Piece piece = board.at(chess::Square(sq));
        if (piece != chess::Piece::NONE && piece.type() != chess::PieceType::KING) {
            int idx = get_feature_index(perspective, piece, chess::Square(sq), king_sq);
            if (idx < 0 || idx >= NUM_FEATURES) continue; // guard instead of cout
            const int16_t* weight = fc1_w[idx];
            
            // Aligned vectorized addition
            for (int i = 0; i < HIDDEN_SIZE; i += 16) {
                __m256i t = _mm256_load_si256(reinterpret_cast<const __m256i*>(&target[i]));
                __m256i w = _mm256_load_si256(reinterpret_cast<const __m256i*>(&weight[i]));
                _mm256_store_si256(reinterpret_cast<__m256i*>(&target[i]), _mm256_add_epi16(t, w));
            }
        }
    }
}

void init_accumulator(const chess::Board& board, Accumulator& acc) {
    refresh_accumulator(board, chess::Color::WHITE, acc);
    refresh_accumulator(board, chess::Color::BLACK, acc);
}

// Update both white and black accumulators for one feature add/subtract.
// We interleave W and B in the same loop body for cache and ILP benefit.
static void update_feature(const chess::Board& board, Accumulator& acc, chess::Piece piece, chess::Square sq, int sign) {
    if (piece == chess::Piece::NONE || piece.type() == chess::PieceType::KING) return;
    
    chess::Square w_king_sq = board.kingSq(chess::Color::WHITE);
    chess::Square b_king_sq = board.kingSq(chess::Color::BLACK);

    int w_idx = get_feature_index(chess::Color::WHITE, piece, sq, w_king_sq);
    int b_idx = get_feature_index(chess::Color::BLACK, piece, sq, b_king_sq);
    if (w_idx < 0 || b_idx < 0) return;
    
    int16_t* w_acc = acc.white;
    int16_t* b_acc = acc.black;
    const int16_t* w_weight = fc1_w[w_idx];
    const int16_t* b_weight = fc1_w[b_idx];
    
    // Interleave W and B operations in one loop for better ILP and cache reuse
    if (sign == 1) {
        for (int i = 0; i < HIDDEN_SIZE; i += 16) {
            __m256i wa = _mm256_load_si256(reinterpret_cast<const __m256i*>(&w_acc[i]));
            __m256i ww = _mm256_load_si256(reinterpret_cast<const __m256i*>(&w_weight[i]));
            _mm256_store_si256(reinterpret_cast<__m256i*>(&w_acc[i]), _mm256_add_epi16(wa, ww));

            __m256i ba = _mm256_load_si256(reinterpret_cast<const __m256i*>(&b_acc[i]));
            __m256i bw = _mm256_load_si256(reinterpret_cast<const __m256i*>(&b_weight[i]));
            _mm256_store_si256(reinterpret_cast<__m256i*>(&b_acc[i]), _mm256_add_epi16(ba, bw));
        }
    } else {
        for (int i = 0; i < HIDDEN_SIZE; i += 16) {
            __m256i wa = _mm256_load_si256(reinterpret_cast<const __m256i*>(&w_acc[i]));
            __m256i ww = _mm256_load_si256(reinterpret_cast<const __m256i*>(&w_weight[i]));
            _mm256_store_si256(reinterpret_cast<__m256i*>(&w_acc[i]), _mm256_sub_epi16(wa, ww));

            __m256i ba = _mm256_load_si256(reinterpret_cast<const __m256i*>(&b_acc[i]));
            __m256i bw = _mm256_load_si256(reinterpret_cast<const __m256i*>(&b_weight[i]));
            _mm256_store_si256(reinterpret_cast<__m256i*>(&b_acc[i]), _mm256_sub_epi16(ba, bw));
        }
    }
}

void update_accumulator(const chess::Board& board, const chess::Move& move, const Accumulator& prev_acc, Accumulator& next_acc) {
    next_acc = prev_acc;
    
    chess::Square from = move.from();
    chess::Square to = move.to();
    chess::Piece piece = board.at(from);
    chess::Piece captured = board.at(to);
    
    // If the king moves, we must do a full refresh for that color's accumulator AFTER the move is made.
    // However, wait! `update_accumulator` is called BEFORE `board.makeMove()`.
    // So `board` is the state BEFORE the move.
    // But `refresh_accumulator` needs the state AFTER the move!
    // Actually, we can just apply the piece changes, and then if it's a King move, we flag it.
    // Let's modify `update_accumulator` to assume it's called BEFORE the move, BUT 
    // it computes the delta for piece moves.
    // Wait! If the king moves, the piece updates don't work because the king square changed.
    // We should just refresh the king's side accumulator. But we can't do it here because the board hasn't updated yet!
    // I will remove the logic here for king moves and handle it in nnue_engine.cpp.
    
    // 1. Remove piece from original square
    update_feature(board, next_acc, piece, from, -1);
    
    // 2. Handle captures (remove captured piece)
    if (captured != chess::Piece::NONE) {
        update_feature(board, next_acc, captured, to, -1);
    } else if (move.typeOf() == chess::Move::ENPASSANT) {
        chess::Square cap_sq = chess::Square(to.index() + (board.sideToMove() == chess::Color::WHITE ? -8 : 8));
        chess::Piece ep_pawn = chess::Piece(chess::PieceType::PAWN, ~board.sideToMove());
        update_feature(board, next_acc, ep_pawn, cap_sq, -1);
    }
    
    // 3. Handle promotion (add promoted piece instead of moving pawn)
    if (move.typeOf() == chess::Move::PROMOTION) {
        chess::Piece promoted = chess::Piece(move.promotionType(), board.sideToMove());
        update_feature(board, next_acc, promoted, to, 1);
    } else {
        // Normal move (add piece to new square)
        update_feature(board, next_acc, piece, to, 1);
    }
    
    // 4. Handle castling (move the rook)
    if (move.typeOf() == chess::Move::CASTLING) {
        chess::Square rook_from, rook_to;
        if (to.index() > from.index()) { // Kingside
            rook_from = chess::Square(from.index() + 3);
            rook_to = chess::Square(from.index() + 1);
        } else { // Queenside
            rook_from = chess::Square(from.index() - 4);
            rook_to = chess::Square(from.index() - 1);
        }
        chess::Piece rook = chess::Piece(chess::PieceType::ROOK, board.sideToMove());
        update_feature(board, next_acc, rook, rook_from, -1);
        update_feature(board, next_acc, rook, rook_to, 1);
    }
}

int evaluate(const Accumulator& acc, chess::Color side_to_move) {
    const int16_t* us_acc   = (side_to_move == chess::Color::WHITE) ? acc.white : acc.black;
    const int16_t* them_acc = (side_to_move == chess::Color::WHITE) ? acc.black : acc.white;

    // Output buffers for layers
    int8_t out2[32];
    int8_t out3[32];

    // Layer 2: 512 -> 32
    for (int i = 0; i < 32; ++i) {
        int32_t sum = fc2_b[i];
        for (int j = 0; j < 256; ++j) {
            // Clipped ReLU for inputs
            int8_t us_val   = std::max(0, std::min(127, static_cast<int>(us_acc[j])));
            int8_t them_val = std::max(0, std::min(127, static_cast<int>(them_acc[j])));
            
            sum += us_val * fc2_w[i][j];
            sum += them_val * fc2_w[i][256 + j];
        }
        sum >>= 6; // divide by 64
        out2[i] = static_cast<int8_t>(std::max(0, std::min(127, sum))); // Clipped ReLU
    }

    // Layer 3: 32 -> 32
    for (int i = 0; i < 32; ++i) {
        int32_t sum = fc3_b[i];
        for (int j = 0; j < 32; ++j) {
            sum += out2[j] * fc3_w[i][j];
        }
        sum >>= 6;
        out3[i] = static_cast<int8_t>(std::max(0, std::min(127, sum)));
    }

    // Layer 4: 32 -> 1
    int32_t final_sum = fc4_b[0];
    for (int j = 0; j < 32; ++j) {
        final_sum += out3[j] * fc4_w[0][j];
    }

    // Scale down by 16 to approximate standard centipawns
    int raw_cp = final_sum / 16;
    return raw_cp;
}

} // namespace nnue
