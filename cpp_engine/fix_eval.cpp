#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

int main() {
    ifstream in("nnue_engine.cpp");
    string code((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
    in.close();

    string king_pst = R"(const int pst_king[64] = {
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
)";

    size_t pos = code.find("const int pst_king[64]");
    size_t end_pos = code.find("};", pos) + 2;
    code.replace(pos, end_pos - pos, king_pst);

    string eval_start = R"(// Classical material+PST + king safety + bishop pair + rook on open file
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
            if (pt == 5) {
                int mg = pst_king[sq];
                int eg = pst_king_endgame[sq];
                score -= (mg * phase_weight + eg * (256 - phase_weight)) / 256;
            } else {
                score -= pst[pt][sq]; 
            }
        }
    })";

    pos = code.find("// Classical material+PST + king safety + bishop pair + rook on open file");
    end_pos = code.find("    // 2. King Safety", pos);
    code.replace(pos, end_pos - pos, eval_start + "\n\n");

    ofstream out("nnue_engine.cpp");
    out << code;
    out.close();
    
    return 0;
}
