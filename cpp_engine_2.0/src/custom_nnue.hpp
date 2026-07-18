#pragma once

#include "chess.hpp"
#include <array>
#include <cstdint>
#include <string>

namespace custom_nnue {

// ─── Architecture Constants (Stockfish HalfKAv2_hm + FullThreats) ──────────
constexpr int L1              = 1024;   // FeatureTransformer output dimensions
constexpr int L2              = 32;
constexpr int L3              = 32;
constexpr int PSQ_BUCKETS     = 8;      // Piece-count buckets (0..7)
constexpr int PSQ_DIMS        = 22528;  // HalfKAv2_hm: 64*64*11/2 = 22528
constexpr int THREAT_DIMS     = 60720;  // FullThreats feature dimensions
constexpr int FT_INPUT_DIMS   = PSQ_DIMS + THREAT_DIMS;
constexpr int FC0_OUT         = 32;
constexpr int FC1_IN          = 64;     // 32(sqr) + 32(linear) from fc_0
constexpr int FC1_OUT         = 32;
constexpr int FC2_IN          = 128;    // 64(fc_0 pair) + 64(fc_1 pair)
constexpr int FC2_OUT         = 1;
constexpr int WEIGHT_SCALE    = 6;

// Stockfish file format identifiers
constexpr uint32_t SF_VERSION       = 0x6A448AFAu;
constexpr uint32_t FT_COMBINED_HASH = 0x6165ddc9u;
constexpr uint32_t NET_ARCH_HASH    = 0x63337116u;
constexpr const char* LEB128_MAGIC  = "COMPRESSED_LEB128";
constexpr int LEB128_MAGIC_LEN      = 17;

// Evaluation scaling constants
constexpr int OUTPUT_SCALE   = 16;
constexpr int FT_MAX_VAL      = 255;
constexpr int HIDDEN_ONE_VAL  = 128;

// ─── Piece-Square feature index tables ─────────────────────────────────────
// HalfKAv2_hm: PieceSquareIndex[COLOR][PIECE] — offset into PS_NB groups
// Convention: viewed from side-to-move, W-us, B-them
constexpr int PS_NONE     = 0;
constexpr int PS_W_PAWN   = 0;
constexpr int PS_B_PAWN   = 1 * 64;
constexpr int PS_W_KNIGHT = 2 * 64;
constexpr int PS_B_KNIGHT = 3 * 64;
constexpr int PS_W_BISHOP = 4 * 64;
constexpr int PS_B_BISHOP = 5 * 64;
constexpr int PS_W_ROOK   = 6 * 64;
constexpr int PS_B_ROOK   = 7 * 64;
constexpr int PS_W_QUEEN  = 8 * 64;
constexpr int PS_B_QUEEN  = 9 * 64;
constexpr int PS_KING     = 10 * 64;
constexpr int PS_NB       = 11 * 64;   // 704

// clang-format off
constexpr int PieceSquareIndex[2][16] = {
    // WHITE perspective: W=us, B=them
    {PS_NONE, PS_W_PAWN, PS_W_KNIGHT, PS_W_BISHOP, PS_W_ROOK, PS_W_QUEEN, PS_KING, PS_NONE,
     PS_NONE, PS_B_PAWN, PS_B_KNIGHT, PS_B_BISHOP, PS_B_ROOK, PS_B_QUEEN, PS_KING, PS_NONE},
    // BLACK perspective: B=us, W=them
    {PS_NONE, PS_B_PAWN, PS_B_KNIGHT, PS_B_BISHOP, PS_B_ROOK, PS_B_QUEEN, PS_KING, PS_NONE,
     PS_NONE, PS_W_PAWN, PS_W_KNIGHT, PS_W_BISHOP, PS_W_ROOK, PS_W_QUEEN, PS_KING, PS_NONE}
};

// KingBuckets: per-square index into the 22528 feature space (32 buckets × 704)
#define B(v) (v * PS_NB)
constexpr int KingBuckets[64] = {
    B(28), B(29), B(30), B(31), B(31), B(30), B(29), B(28),
    B(24), B(25), B(26), B(27), B(27), B(26), B(25), B(24),
    B(20), B(21), B(22), B(23), B(23), B(22), B(21), B(20),
    B(16), B(17), B(18), B(19), B(19), B(18), B(17), B(16),
    B(12), B(13), B(14), B(15), B(15), B(14), B(13), B(12),
    B( 8), B( 9), B(10), B(11), B(11), B(10), B( 9), B( 8),
    B( 4), B( 5), B( 6), B( 7), B( 7), B( 6), B( 5), B( 4),
    B( 0), B( 1), B( 2), B( 3), B( 3), B( 2), B( 1), B( 0),
};
#undef B

// OrientTBL: horizontally mirror squares on files a-d (XOR with 7 = SQ_H1)
// This keeps the king always on files e-h after mirroring.
constexpr int OrientTBL[64] = {
    7, 7, 7, 7, 0, 0, 0, 0,
    7, 7, 7, 7, 0, 0, 0, 0,
    7, 7, 7, 7, 0, 0, 0, 0,
    7, 7, 7, 7, 0, 0, 0, 0,
    7, 7, 7, 7, 0, 0, 0, 0,
    7, 7, 7, 7, 0, 0, 0, 0,
    7, 7, 7, 7, 0, 0, 0, 0,
    7, 7, 7, 7, 0, 0, 0, 0,
};
// clang-format on

// ─── Accumulator ───────────────────────────────────────────────────────────
struct alignas(64) Accumulator {
    std::array<std::array<int16_t, L1>, 2> accumulation{};     // [color][1024]
    std::array<std::array<int32_t, PSQ_BUCKETS>, 2> psqtAccumulation{}; // [color][8]
    bool computed[2] = {false, false};
};

// ─── Feature Transformer Weights ───────────────────────────────────────────
struct FeatureTransformerWeights {
    alignas(64) int16_t biases[L1]{};                           // 1024
    alignas(64) int16_t weights[L1 * PSQ_DIMS]{};               // 1024 × 22528
    alignas(64) int8_t  threatWeights[L1 * THREAT_DIMS]{};      // 1024 × 60720
    alignas(64) int32_t psqtWeights[PSQ_DIMS * PSQ_BUCKETS]{};  // 22528 × 8
    alignas(64) int32_t threatPsqtWeights[THREAT_DIMS * PSQ_BUCKETS]{}; // 60720 × 8
};

// ─── Per-Bucket Layer Stacks ──────────────────────────────────────────────
struct LayerStacks {
    alignas(64) int32_t fc0_bias[FC0_OUT]{};           // 32
    alignas(64) int8_t  fc0_weights[FC0_OUT * L1]{};   // 32 × 1024
    alignas(64) int32_t fc1_bias[FC1_OUT]{};           // 32
    alignas(64) int8_t  fc1_weights[FC1_OUT * FC1_IN]{};// 32 × 64
    alignas(64) int32_t fc2_bias[FC2_OUT]{};           // 1
    alignas(64) int8_t  fc2_weights[FC2_OUT * FC2_IN]{};// 1 × 128
};

// ─── Full Network ──────────────────────────────────────────────────────────
struct Network {
    FeatureTransformerWeights ft;
    std::array<LayerStacks, PSQ_BUCKETS> layers;
    bool loaded = false;
};

// ─── Public API ────────────────────────────────────────────────────────────

// Load a Stockfish-format .nnue file and populate the global network
bool load_network(const std::string& path);

// Full (re)build of the accumulator from a board position
void refresh_accumulator(const chess::Board& board, Accumulator& acc);

// Mark accumulator as needing refresh after a move (incremental TBD)
void update_accumulator(Accumulator& acc, const chess::Board& board, const chess::Move& move);

// Evaluate a position given a fresh accumulator, from stm's perspective
int evaluate(Accumulator& acc, const chess::Board& board, chess::Color stm);

// Access the global network (populated by load_network)
extern Network g_network;

}  // namespace custom_nnue