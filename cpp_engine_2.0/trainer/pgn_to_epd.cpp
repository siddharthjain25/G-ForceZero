#include "chess.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace chess;

class EpdExtractor : public pgn::Visitor {
public:
    std::string result_str = "0.5";
    Board board;
    std::vector<std::string> game_fens;
    std::ofstream& out;
    int move_count = 0;
    bool skip_game = false;

    EpdExtractor(std::ofstream& out_stream) : out(out_stream) {}

    void startPgn() override {
        result_str = "0.5";
        board.setFen(constants::STARTPOS);
        game_fens.clear();
        move_count = 0;
        skip_game = false;
    }

    void header(std::string_view key, std::string_view value) override {
        if (key == "Result") {
            if (value == "1-0") result_str = "1.0";
            else if (value == "0-1") result_str = "0.0";
            else if (value == "1/2-1/2") result_str = "0.5";
            else skip_game = true;
        }
    }

    void startMoves() override {
    }

    void move(std::string_view move_str, std::string_view comment) override {
        if (skip_game) return;
        try {
            Move m = uci::parseSan(board, move_str);
            board.makeMove(m);
            // Skip the first 16 ply (8 full moves) of theory to get diverse positions
            if (move_count >= 16) {
                game_fens.push_back(board.getFen());
            }
            move_count++;
        } catch (...) {
            skip_game = true; 
        }
    }

    void endPgn() override {
        if (skip_game || game_fens.empty()) return;
        
        // Output every 10th move from the game to avoid huge files and correlated positions
        for (size_t i = 0; i < game_fens.size(); i += 10) {
            out << game_fens[i] << " c9 \"" << result_str << "\"\n";
        }
    }
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.pgn> <output.epd>\n";
        return 1;
    }
    
    std::ifstream pgn_file(argv[1]);
    if (!pgn_file.is_open()) {
        std::cerr << "Failed to open input file.\n";
        return 1;
    }

    std::ofstream epd_file(argv[2]);
    if (!epd_file.is_open()) {
        std::cerr << "Failed to open output file.\n";
        return 1;
    }
    
    EpdExtractor extractor(epd_file);
    pgn::StreamParser parser(pgn_file);
    
    std::cout << "Converting PGN to EPD... This is heavily optimized and will be fast.\n";
    parser.readGames(extractor);
    std::cout << "Done.\n";
    
    return 0;
}
