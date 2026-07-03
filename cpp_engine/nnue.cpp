#include "nnue.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>

namespace nnue {

int16_t fc1_w[HIDDEN_SIZE][NUM_FEATURES];
int16_t fc1_b[HIDDEN_SIZE];
int16_t fc2_w[1][HIDDEN_SIZE * 2];
int32_t fc2_b[1];

void load_weights(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open NNUE weights file: " + filepath);
    }
    
    file.read(reinterpret_cast<char*>(fc1_w), sizeof(fc1_w));
    file.read(reinterpret_cast<char*>(fc1_b), sizeof(fc1_b));
    file.read(reinterpret_cast<char*>(fc2_w), sizeof(fc2_w));
    file.read(reinterpret_cast<char*>(fc2_b), sizeof(fc2_b));
    
    if (!file) {
        throw std::runtime_error("Error reading NNUE weights!");
    }
}

int get_piece_idx(chess::Color perspective, chess::Piece piece) {
    int pt = static_cast<int>(piece.type()) - 1; // Pawn=0, Knight=1, Bishop=2, Rook=3, Queen=4
    bool is_mine = (piece.color() == perspective);
    return is_mine ? pt : pt + 5;
}

int get_feature_index(chess::Color perspective, chess::Piece piece, chess::Square sq, chess::Square king_sq) {
    if (piece.type() == chess::PieceType::KING) return -1; // King has no feature

    int sq_idx = sq.index();
    int k_sq_idx = king_sq.index();
    
    if (perspective == chess::Color::BLACK) {
        sq_idx ^= 56;
        k_sq_idx ^= 56;
    }
    
    int pt_idx = get_piece_idx(perspective, piece);
    return pt_idx * 4096 + k_sq_idx * 64 + sq_idx;
}

void refresh_accumulator(const chess::Board& board, chess::Color perspective, Accumulator& acc) {
    int16_t* target = (perspective == chess::Color::WHITE) ? acc.white : acc.black;
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        target[i] = fc1_b[i];
    }
    
    chess::Square king_sq = board.kingSq(perspective);
    
    for (int sq = 0; sq < 64; ++sq) {
        chess::Piece piece = board.at(chess::Square(sq));
        if (piece != chess::Piece::NONE && piece.type() != chess::PieceType::KING) {
            int idx = get_feature_index(perspective, piece, chess::Square(sq), king_sq);
            for (int i = 0; i < HIDDEN_SIZE; ++i) {
                target[i] += fc1_w[i][idx];
            }
        }
    }
}

void init_accumulator(const chess::Board& board, Accumulator& acc) {
    refresh_accumulator(board, chess::Color::WHITE, acc);
    refresh_accumulator(board, chess::Color::BLACK, acc);
}

static void update_feature(const chess::Board& board, Accumulator& acc, chess::Piece piece, chess::Square sq, int sign) {
    if (piece.type() == chess::PieceType::KING) return;
    
    chess::Square w_king_sq = board.kingSq(chess::Color::WHITE);
    chess::Square b_king_sq = board.kingSq(chess::Color::BLACK);

    int w_idx = get_feature_index(chess::Color::WHITE, piece, sq, w_king_sq);
    int b_idx = get_feature_index(chess::Color::BLACK, piece, sq, b_king_sq);
    
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        acc.white[i] += sign * fc1_w[i][w_idx];
        acc.black[i] += sign * fc1_w[i][b_idx];
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
    int32_t sum = fc2_b[0];
    
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        int16_t w = std::min(256, std::max(0, static_cast<int>(acc.white[i])));
        int16_t b = std::min(256, std::max(0, static_cast<int>(acc.black[i])));
        
        sum += w * fc2_w[0][i];
        sum += b * fc2_w[0][HIDDEN_SIZE + i];
    }
    
    // The network was trained with sigmoid output predicting White's win probability.
    // sum is the pre-sigmoid logit scaled by (256 * 64).
    // The training used K=0.003: prob = sigmoid(cp * 0.003)
    // So the raw network output logit = cp * 0.003 * (256 * 64)
    // Therefore: cp = sum / (0.003 * 256 * 64) = sum / 49.15
    int white_score = sum / 49;
    return (side_to_move == chess::Color::WHITE) ? white_score : -white_score;
}

} // namespace nnue
