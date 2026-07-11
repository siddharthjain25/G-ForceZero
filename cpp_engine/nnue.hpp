#ifndef NNUE_HPP
#define NNUE_HPP

#include <cstdint>
#include <string>
#include "chess.hpp"

namespace nnue {

// NNUE Configuration
constexpr int NUM_FEATURES = 41024;
constexpr int HIDDEN_SIZE = 256;

// Weights and biases
alignas(32) extern int16_t fc1_w[NUM_FEATURES][HIDDEN_SIZE];
alignas(32) extern int16_t fc1_b[HIDDEN_SIZE];

alignas(32) extern int32_t fc2_b[32];
alignas(32) extern int8_t  fc2_w[32][HIDDEN_SIZE * 2];

alignas(32) extern int32_t fc3_b[32];
alignas(32) extern int8_t  fc3_w[32][32];

alignas(32) extern int32_t fc4_b[1];
alignas(32) extern int8_t  fc4_w[1][32];

struct alignas(32) Accumulator {
    int16_t white[HIDDEN_SIZE];
    int16_t black[HIDDEN_SIZE];
};

void load_weights(const std::string& filepath);

// Get feature index for a piece on a square, from the perspective of the given color
int get_feature_index(chess::Color perspective, chess::Piece piece, chess::Square sq);

// Initialize accumulator from a complete board
void init_accumulator(const chess::Board& board, Accumulator& acc);

// Refresh accumulator for a specific perspective
void refresh_accumulator(const chess::Board& board, chess::Color perspective, Accumulator& acc);

// Update accumulator for a move
void update_accumulator(const chess::Board& board, const chess::Move& move, const Accumulator& prev_acc, Accumulator& next_acc);

// Evaluate the NNUE using the given accumulator
int evaluate(const Accumulator& acc, chess::Color side_to_move);

} // namespace nnue

#endif // NNUE_HPP
