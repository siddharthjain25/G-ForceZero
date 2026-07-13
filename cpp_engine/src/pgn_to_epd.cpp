#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include "chess.hpp"

using namespace chess;

class PgnVisitor : public Visitor {
public:
    PgnVisitor(std::ofstream& out, int max_games) 
        : out_(out), max_games_(max_games), games_processed_(0) {}

    void startPgn() override {
        board_.setFen(constants::STARTPOS);
        result_str_ = "";
        w_elo_ = 0;
        b_elo_ = 0;
        skipPgn(false);
    }

    void header(std::string_view key, std::string_view value) override {
        if (key == "Result") {
            if (value == "1-0") result_str_ = "\"1.0\"";
            else if (value == "0-1") result_str_ = "\"0.0\"";
            else if (value == "1/2-1/2") result_str_ = "\"0.5\"";
            else skipPgn(true);
        } else if (key == "WhiteElo") {
            try { w_elo_ = std::stoi(std::string(value)); } catch(...) { skipPgn(true); }
        } else if (key == "BlackElo") {
            try { b_elo_ = std::stoi(std::string(value)); } catch(...) { skipPgn(true); }
        }
    }

    void startMoves() override {
        if (w_elo_ < 2500 || b_elo_ < 2500 || result_str_.empty()) {
            skipPgn(true);
        }
    }

    void move(std::string_view move_str, std::string_view comment) override {
        if (skip()) return;
        try {
            Move m = board_.parseSan(move_str);
            board_.makeMove(m);
            // Skip first 10 plies to avoid heavy draw bias from openings
            if (board_.plies() > 10) {
                out_ << board_.getFen() << " c9 " << result_str_ << "\n";
            }
        } catch (...) {
            // Illegal move or corrupt SAN string, skip this game
            skipPgn(true);
        }
    }

    void endPgn() override {
        if (!skip()) {
            games_processed_++;
            if (games_processed_ % 1000 == 0) {
                std::cout << "." << std::flush;
            }
        }
        if (games_processed_ >= max_games_) {
            // Throw exception to break the parsing loop gracefully
            throw std::runtime_error("MAX_GAMES_REACHED");
        }
    }

private:
    Board board_;
    std::string result_str_;
    int w_elo_;
    int b_elo_;
    std::ofstream& out_;
    int max_games_;
    int games_processed_;
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.pgn> <output.epd>\n";
        return 1;
    }

    std::ifstream in(argv[1]);
    std::ofstream out(argv[2]);

    if (!in.is_open() || !out.is_open()) {
        std::cerr << "Error opening files.\n";
        return 1;
    }

    PgnVisitor visitor(out, 1000000); // 1 Million Game Limit
    StreamParser parser(in);

    std::cout << "Starting high-speed C++ PGN conversion...\n";
    try {
        parser.readGames(visitor);
    } catch (const std::runtime_error& e) {
        if (std::string(e.what()) == "MAX_GAMES_REACHED") {
            std::cout << "\n[INFO] Reached 1,000,000 games. Stopped safely.\n";
        } else {
            std::cerr << "\nError: " << e.what() << "\n";
        }
    }

    std::cout << "Conversion complete!\n";
    return 0;
}
