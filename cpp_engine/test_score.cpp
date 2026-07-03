#include "chess.hpp"
#include <iostream>
using namespace chess;

extern int piece_values[6];
extern const int pst_pawn[64];
// ... I'll just write my own copy of classical_evaluate to debug it.
int classical_evaluate_debug(const Board& board) {
    int score = 0;

    // 1. Material + PST
    for (int pt = 0; pt < 6; ++pt) {
        PieceType pType = PieceType(static_cast<PieceType::underlying>(pt));
        Bitboard wb = board.pieces(pType, Color::WHITE);
        while (wb) { int sq = wb.pop(); score += piece_values[pt]; } // IGNORING PST for simplicity
        Bitboard bb = board.pieces(pType, Color::BLACK);
        while (bb) { int sq = bb.pop(); score -= piece_values[pt]; }
    }
    std::cout << "Material score: " << score << "\n";

    Bitboard w_pawns = board.pieces(PieceType::PAWN, Color::WHITE);
    Bitboard b_pawns = board.pieces(PieceType::PAWN, Color::BLACK);

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
            score += 20 * r * r; 
            std::cout << "White passed pawn at " << f << "," << r << " added " << 20 * r * r << "\n";
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
            score -= 20 * dist * dist;
            std::cout << "Black passed pawn at " << f << "," << r << " subtracted " << 20 * dist * dist << "\n";
        }
    }
    return score;
}

int main() {
    Board b;
    std::string moves_str = "e2e4 c7c5 b1c3 b8c6 g1f3 d7d6 f1b5 e7e5 b5c4 f8e7 c3d5 g8f6 d5e7 d8e7 d2d3 c8e6 c4e6 f7e6 e1g1 f6d7 c1g5 e7f7 g5e3 e8g8 f3g5 f7e7 d1e2 a7a5 f1e1 e7e8 e2g4 f8f6 a1d1 h7h6 g5f3 e8f7 a2a3 a5a4 g4h3 a8f8 c2c3 b7b6 d3d4 c5d4 c3d4 e5d4 f3d4 c6d4 e3d4 e6e5 d4e3 d7c5 h3g4 f6e6 e3c5 b6c5 d1d2 e6f6 g4e2 g8h7 e1c1 f7g6 c1c4 f6f2 e2f2 f8f2 d2f2 g6e6 c4a4 e6g4 f2f3 d6d5 h2h3 g4d7 b2b3 d5d4 a4c4 d7c6 g1h2 c6e4 c4c5 d4d3 c5c7 e4d4 f3f7 e5e4 c7d7 d4e5 g2g1 e4e3 f7g7 e5g7 d7g7 h7g7";
    b.setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    
    std::string token;
    size_t pos = 0;
    while ((pos = moves_str.find(" ")) != std::string::npos) {
        token = moves_str.substr(0, pos);
        b.makeMove(uci::uciToMove(b, token));
        moves_str.erase(0, pos + 1);
    }
    b.makeMove(uci::uciToMove(b, moves_str));

    std::cout << "Score: " << classical_evaluate_debug(b) << "\n";
    return 0;
}
