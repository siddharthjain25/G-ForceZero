#include "custom_nnue.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <algorithm>

namespace custom_nnue {

Network g_network;

// ─── Little-Endian I/O ────────────────────────────────────────────────────
template<typename T>
static T read_le(std::istream& s) {
    T val;
    s.read(reinterpret_cast<char*>(&val), sizeof(T));
    return val;
}

template<typename T>
static void read_le_array(std::istream& s, T* out, size_t n) {
    s.read(reinterpret_cast<char*>(out), n * sizeof(T));
}

// ─── LEB128 Decoder ───────────────────────────────────────────────────────
static bool read_leb_128(std::istream& stream, int16_t* out, size_t count) {
    char magic[LEB128_MAGIC_LEN];
    stream.read(magic, LEB128_MAGIC_LEN);
    if (!stream || std::memcmp(magic, LEB128_MAGIC, LEB128_MAGIC_LEN) != 0) {
        std::cerr << "LEB128 magic mismatch\n";
        return false;
    }
    uint32_t bytes_left = read_le<uint32_t>(stream);
    if (!stream) return false;

    uint8_t buf[8192];
    uint32_t buf_pos = 8192;
    int16_t  result  = 0;
    int      shift   = 0;
    size_t   out_idx = 0;

    while (out_idx < count) {
        if (buf_pos >= 8192) {
            uint32_t n = std::min<uint32_t>(8192, bytes_left);
            stream.read(reinterpret_cast<char*>(buf), n);
            buf_pos = 0;
        }
        uint8_t byte = buf[buf_pos++];
        --bytes_left;
        result |= (byte & 0x7f) << (shift % 32);
        shift += 7;
        if ((byte & 0x80) == 0) {
            if (shift < 32 && (byte & 0x40))
                result |= ~((1 << shift) - 1);
            out[out_idx++] = result;
            result = 0;
            shift  = 0;
        }
    }
    return true;
}

// ─── Load Network ─────────────────────────────────────────────────────────
bool load_network(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open: " << path << "\n";
        return false;
    }

    uint32_t version  = read_le<uint32_t>(file);
    /* hash */         read_le<uint32_t>(file);
    uint32_t desc_len = read_le<uint32_t>(file);
    std::string desc(desc_len, '\0');
    file.read(&desc[0], desc_len);
    std::cout << "NNUE: " << desc << "\nVersion: 0x" << std::hex << version << std::dec << "\n";

    if (version != SF_VERSION) {
        std::cerr << "Version mismatch\n";
        return false;
    }

    // FeatureTransformer hash
    uint32_t ft_hash = read_le<uint32_t>(file);
    std::cout << "FT hash: 0x" << std::hex << ft_hash << std::dec << "\n";
    if (ft_hash != FT_COMBINED_HASH)
        std::cerr << "Warning: unexpected FT hash\n";

    auto& ft = g_network.ft;

    // 1. biases [L1] int16 — LEB128
    if (!read_leb_128(file, ft.biases, L1)) {
        std::cerr << "Failed: biases\n"; return false;
    }

    // 2. threatWeights [L1 * THREAT_DIMS] int8 — raw
    read_le_array<int8_t>(file, ft.threatWeights, L1 * THREAT_DIMS);

    // 3. threatPsqtWeights [THREAT_DIMS * PSQ_BUCKETS] int32 — LEB128
    if (!read_leb_128(file, reinterpret_cast<int16_t*>(ft.threatPsqtWeights),
                      THREAT_DIMS * PSQ_BUCKETS)) {
        std::cerr << "Failed: threatPsqtWeights\n";
        return false;
    }

    // 4. PSQ weights [L1 * PSQ_DIMS] int16 — LEB128
    if (!read_leb_128(file, ft.weights, L1 * PSQ_DIMS)) {
        std::cerr << "Failed: FT weights\n";
        return false;
    }

    // 5. PSQT weights [PSQ_DIMS * PSQ_BUCKETS] int32 — LEB128
    if (!read_leb_128(file, reinterpret_cast<int16_t*>(ft.psqtWeights),
                      PSQ_DIMS * PSQ_BUCKETS)) {
        std::cerr << "Failed: PSQT weights\n";
        return false;
    }
    std::cout << "FeatureTransformer loaded.\n";

    // ── 8 Network Layer Stacks ────────────────────────────────────────────
    for (int b = 0; b < PSQ_BUCKETS; ++b) {
        uint32_t nh = read_le<uint32_t>(file);
        if (nh != NET_ARCH_HASH && b == 0)
            std::cerr << "Net hash warning: 0x" << std::hex << nh << std::dec << "\n";

        auto& L = g_network.layers[b];

        // fc_0: AffineTransformSparseInput<1024, 32>
        read_le_array<int32_t>(file, L.fc0_bias, FC0_OUT);
        read_le_array<int8_t>(file, L.fc0_weights, FC0_OUT * L1);

        // fc_1: AffineTransform<64, 32>
        read_le_array<int32_t>(file, L.fc1_bias, FC1_OUT);
        read_le_array<int8_t>(file, L.fc1_weights, FC1_OUT * FC1_IN);

        // fc_2: AffineTransform<128, 1>
        read_le_array<int32_t>(file, L.fc2_bias, FC2_OUT);
        read_le_array<int8_t>(file, L.fc2_weights, FC2_IN);
    }

    g_network.loaded = true;
    std::cout << "Network loaded (" << int(PSQ_BUCKETS) << " buckets).\n";
    return true;
}

// ─── Feature Index Helpers ────────────────────────────────────────────────
static inline int piece_idx(const chess::Piece& p) {
    return static_cast<int>(p.color()) * 8 + static_cast<int>(p.type()) + 1;
}

static inline int color_idx(const chess::Color& c) {
    return c == chess::Color::WHITE ? 0 : 1;
}

// HalfKAv2_hm: feature = KingBuckets[ksq] + PieceSquareIndex + oriented_sq
static int make_feature(int perspective, int sq, int pi, int ksq) {
    return KingBuckets[ksq] + PieceSquareIndex[perspective][pi] + (sq ^ OrientTBL[sq]);
}

// ─── Accumulator Refresh ──────────────────────────────────────────────────
void refresh_accumulator(const chess::Board& board, Accumulator& acc) {
    if (!g_network.loaded) return;
    const auto& ft = g_network.ft;

    std::memset(&acc.accumulation, 0, sizeof(acc.accumulation));
    std::memset(&acc.psqtAccumulation, 0, sizeof(acc.psqtAccumulation));

    for (int p = 0; p < 2; ++p) {
        chess::Color pc = (p == 0) ? chess::Color::WHITE : chess::Color::BLACK;

        // Initialize with biases
        for (int d = 0; d < L1; ++d)
            acc.accumulation[p][d] = ft.biases[d];

        int ksq = static_cast<int>(board.kingSq(pc).index());

        chess::Bitboard occ = board.occ();
        while (occ) {
            int s = occ.pop();
            chess::Piece piece = board.at(chess::Square(s));
            if (piece == chess::Piece::NONE) continue;

            int feat = make_feature(p, s, piece_idx(piece), ksq);
            if ((unsigned)feat >= (unsigned)PSQ_DIMS) continue;

            for (int d = 0; d < L1; ++d)
                acc.accumulation[p][d] += ft.weights[feat * L1 + d];
            for (int b = 0; b < PSQ_BUCKETS; ++b)
                acc.psqtAccumulation[p][b] += ft.psqtWeights[feat * PSQ_BUCKETS + b];
        }
        acc.computed[p] = true;
    }
}

void update_accumulator(Accumulator& acc, const chess::Board&, const chess::Move&) {
    acc.computed[0] = false;
    acc.computed[1] = false;
}

// ─── Forward Pass ─────────────────────────────────────────────────────────
static void fc0_propagate(const int8_t* w, const int32_t* b,
                          const uint8_t* in, int32_t* out) {
    for (int o = 0; o < FC0_OUT; ++o) {
        int32_t s = b[o];
        for (int i = 0; i < L1; ++i)
            s += int32_t(w[o * L1 + i]) * int32_t(in[i]);
        out[o] = s;
    }
}

// SqrClippedReLU: out = clamp((in*in) >> (2*WEIGHT_SCALE + 7), 0, 127)
static void sqr_relu_32(const int32_t* in, uint8_t* out) {
    for (int i = 0; i < 32; ++i) {
        int64_t sq = int64_t(in[i]) * int64_t(in[i]);
        sq >>= (2 * WEIGHT_SCALE + 7);
        out[i] = (sq > 127) ? uint8_t(127) : uint8_t(sq);
    }
}

// ClippedReLU: out = clamp(in >> WEIGHT_SCALE, 0, 127)
static void clip_relu_32(const int32_t* in, uint8_t* out) {
    for (int i = 0; i < 32; ++i) {
        int v = in[i] >> WEIGHT_SCALE;
        if (v < 0) v = 0;
        else if (v > 127) v = 127;
        out[i] = uint8_t(v);
    }
}

// Generic dense affine layer
static void fc_propagate(const int8_t* w, const int32_t* b,
                         const uint8_t* in, int in_dim,
                         int32_t* out, int out_dim) {
    for (int o = 0; o < out_dim; ++o) {
        int32_t s = b[o];
        for (int i = 0; i < in_dim; ++i)
            s += int32_t(w[o * in_dim + i]) * int32_t(in[i]);
        out[o] = s;
    }
}

// ─── Evaluate ─────────────────────────────────────────────────────────────
int evaluate(Accumulator& acc, const chess::Board& board, chess::Color stm) {
    if (!g_network.loaded) return 0;

    if (!acc.computed[0] || !acc.computed[1])
        refresh_accumulator(board, acc);

    int bucket = (board.occ().count() - 1) / 4;
    if (bucket < 0) bucket = 0;
    if (bucket >= PSQ_BUCKETS) bucket = PSQ_BUCKETS - 1;

    const auto& L = g_network.layers[bucket];
    int us   = color_idx(stm);
    int them = us ^ 1;

    // PSQT
    int32_t psqt = acc.psqtAccumulation[us][bucket] - acc.psqtAccumulation[them][bucket];
    psqt /= 2;

    // Feature transform: pair accum halves → uint8
    uint8_t features[L1];
    for (int p = 0; p < 2; ++p) {
        int idx = (p == 0) ? us : them;
        int off = p * (L1 / 2);
        for (int j = 0; j < L1 / 2; ++j) {
            int a = acc.accumulation[idx][j];
            int b = acc.accumulation[idx][j + L1 / 2];
            if (a < 0) a = 0; else if (a > 255) a = 255;
            if (b < 0) b = 0; else if (b > 255) b = 255;
            features[off + j] = uint8_t((a * b) / 512);
        }
    }

    // fc0: 1024 → 32
    int32_t fc0_out[FC0_OUT];
    fc0_propagate(L.fc0_weights, L.fc0_bias, features, fc0_out);

    // Dual activation
    uint8_t fc0_sqr[32], fc0_clp[32];
    sqr_relu_32(fc0_out, fc0_sqr);
    clip_relu_32(fc0_out, fc0_clp);

    // fc1 input = [fc0_sqr, fc0_clp]
    uint8_t fc1_in[FC1_IN];
    for (int i = 0; i < 32; ++i) fc1_in[i]      = fc0_sqr[i];
    for (int i = 0; i < 32; ++i) fc1_in[32 + i] = fc0_clp[i];

    int32_t fc1_out[FC1_OUT];
    fc_propagate(L.fc1_weights, L.fc1_bias, fc1_in, FC1_IN, fc1_out, FC1_OUT);

    uint8_t fc1_sqr[32], fc1_clp[32];
    sqr_relu_32(fc1_out, fc1_sqr);
    clip_relu_32(fc1_out, fc1_clp);

    // fc2 input: [fc0_sqr, fc1_sqr, fc0_clp, fc1_clp] = 128
    uint8_t fc2_in[FC2_IN];
    int off = 0;
    for (int i = 0; i < 32; ++i) fc2_in[off++] = fc0_sqr[i];
    for (int i = 0; i < 32; ++i) fc2_in[off++] = fc1_sqr[i];
    for (int i = 0; i < 32; ++i) fc2_in[off++] = fc0_clp[i];
    for (int i = 0; i < 32; ++i) fc2_in[off++] = fc1_clp[i];

    int32_t fwdOut = L.fc2_bias[0];
    for (int i = 0; i < FC2_IN; ++i)
        fwdOut += int32_t(L.fc2_weights[i]) * int32_t(fc2_in[i]);

    // Skip connection
    fwdOut += fc0_out[FC0_OUT - 2] - fc0_out[FC0_OUT - 1];

    constexpr int64_t mul = 600 * OUTPUT_SCALE;
    constexpr int64_t div = int64_t(HIDDEN_ONE_VAL) * (1 << WEIGHT_SCALE) * 2;
    int positional = static_cast<int>((int64_t(fwdOut) * mul) / div);
    int score = (psqt + positional) / OUTPUT_SCALE;

    return (us == 0) ? score : -score;
}

}  // namespace custom_nnue