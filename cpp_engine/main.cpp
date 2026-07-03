#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include "chess.hpp"
#include "agent.hpp"
#include "mcts.hpp"

// Splits a string into tokens
std::vector<std::string> split(const std::string &s) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (tokenStream >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

int main(int argc, char* argv[]) {
    std::string model_path = "../model.onnx";
    if (argc > 1) {
        model_path = argv[1];
    }

    // Initialize Neural Network and MCTS
    Agent agent(model_path);
    MCTS mcts(&agent);
    
    chess::Board board;
    board.setFen(chess::constants::STARTPOS);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        
        std::vector<std::string> tokens = split(line);
        std::string command = tokens[0];

        if (command == "uci") {
            std::cout << "id name Antigravity C++" << std::endl;
            std::cout << "id author Siddharth" << std::endl;
            std::cout << "uciok" << std::endl;
        } 
        else if (command == "isready") {
            std::cout << "readyok" << std::endl;
        } 
        else if (command == "ucinewgame") {
            board.setFen(chess::constants::STARTPOS);
        } 
        else if (command == "position") {
            int move_idx = -1;
            if (tokens.size() > 1 && tokens[1] == "startpos") {
                board.setFen(chess::constants::STARTPOS);
                move_idx = 2;
            } else if (tokens.size() > 1 && tokens[1] == "fen") {
                std::string fen = tokens[2] + " " + tokens[3] + " " + tokens[4] + " " + tokens[5] + " " + tokens[6] + " " + tokens[7];
                board.setFen(fen);
                move_idx = 8;
            }

            // Apply moves if any
            if (move_idx != -1 && move_idx < tokens.size() && tokens[move_idx] == "moves") {
                for (size_t i = move_idx + 1; i < tokens.size(); ++i) {
                    chess::Movelist moves;
                    chess::movegen::legalmoves(moves, board);
                    for (const auto& m : moves) {
                        if (chess::uci::moveToUci(m) == tokens[i]) {
                            board.makeMove(m);
                            break;
                        }
                    }
                }
            }
        } 
        else if (command == "go") {
            int simulations = -1; // Default
            int wtime = 0, btime = 0;
            
            // Basic parsing of nodes or movetime
            for (size_t i = 1; i < tokens.size(); ++i) {
                if (tokens[i] == "nodes" && i + 1 < tokens.size()) {
                    simulations = std::stoi(tokens[i + 1]);
                }
                else if (tokens[i] == "movetime" && i + 1 < tokens.size()) {
                    int ms = std::stoi(tokens[i + 1]);
                    simulations = std::max(10, (ms * 50) / 1000); // Assume ~50 sims/sec on CPU
                }
                else if (tokens[i] == "wtime" && i + 1 < tokens.size()) {
                    wtime = std::stoi(tokens[i + 1]);
                }
                else if (tokens[i] == "btime" && i + 1 < tokens.size()) {
                    btime = std::stoi(tokens[i + 1]);
                }
            }

            if (simulations == -1) {
                // If wtime is parsed (e.g. from lichess-bot), calculate movetime
                if (wtime > 0 || btime > 0) {
                    int my_time = (board.sideToMove() == chess::Color::WHITE) ? wtime : btime;
                    int target_ms = my_time / 40; // Allocate 1/40th of time
                    // Assume 500 sims/sec on CUDA. 
                    // This allows ~800 nodes in Rapid (1.5 seconds) but scales down gracefully in Blitz.
                    simulations = std::max(100, (target_ms * 50) / 1000); 
                } else {
                    simulations = 800; // Fallback for testing
                }
            }

            chess::Move best_move = mcts.get_best_move(board, simulations);
            std::cout << "bestmove " << chess::uci::moveToUci(best_move) << std::endl;
        } 
        else if (command == "quit") {
            break;
        }
    }

    return 0;
}
