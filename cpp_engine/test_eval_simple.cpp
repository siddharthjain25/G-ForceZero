#include "chess.hpp"
#include <iostream>
using namespace chess;

extern int classical_evaluate(const Board& board);

int main() {
    Board b;
    b.setFen("5rk1/3n1qpp/1p2pr2/p1pp4/P2P2Q1/2P1P1P1/1P1B3P/3RR1K1 b - - 0 22"); 
    std::string moves_str = "e2e4 c7c5 b1c3 b8c6 g1f3 d7d6 f1b5 e7e5 b5c4 f8e7 c3d5 g8f6 d5e7 d8e7 d2d3 c8e6 c4e6 f7e6 e1g1 f6d7 c1g5 e7f7 g5e3 e8g8 f3g5 f7e7 d1e2 a7a5 f1e1 e7e8 e2g4 f8f6 a1d1 h7h6 g5f3 e8f7 a2a3 a5a4 g4h3 a8f8 c2c3 b7b6 d3d4 c5d4 c3d4 e5d4 f3d4 c6d4 e3d4 e6e5 d4e3 d7c5 h3g4 f6e6 e3c5 b6c5 d1d2 e6f6 g4e2 g8h7 e1c1 f7g6";
    std::string start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    b.setFen(start_fen);
    
    std::string token;
    size_t pos = 0;
    while ((pos = moves_str.find(" ")) != std::string::npos) {
        token = moves_str.substr(0, pos);
        b.makeMove(uci::uciToMove(b, token));
        moves_str.erase(0, pos + 1);
    }
    b.makeMove(uci::uciToMove(b, moves_str));
    std::cout << "Classical Eval: " << classical_evaluate(b) << "\n";
    return 0;
}
