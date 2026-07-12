#pragma once
#include "chess.hpp"
#include "polyglot.hpp"
#include <fstream>
#include <vector>
#include <iostream>
#include <random>

struct PackedPosition {
    uint64_t white;
    uint64_t black;
    uint64_t kings;
    uint64_t queens;
    uint64_t rooks;
    uint64_t bishops;
    uint64_t knights;
    uint64_t pawns;
    uint8_t ep;
    uint8_t turn;
    uint8_t castling;
    uint8_t padding;
    int16_t score;
    int8_t result; // 0=Black win, 1=White win, 2=Draw
    int8_t padding2;
} __attribute__((packed));

extern chess::Move search_best_move(chess::Board& board, int soft_limit, int hard_limit);
extern int last_search_score;

inline void generate_selfplay(int num_games, int depth, const std::string& output_file) {
    std::ofstream out(output_file, std::ios::binary | std::ios::app);
    if (!out) {
        std::cerr << "Failed to open " << output_file << " for writing\n";
        return;
    }

    std::mt19937 rng(1337);
    std::cout << "Starting self-play generation of " << num_games << " games...\n";
    
    int games_completed = 0;
    while (games_completed < num_games) {
        chess::Board board;
        board.setFen(chess::constants::STARTPOS);

        // Randomize opening using the book
        int book_moves = (rng() % 4) + 6; // 6 to 9 halfmoves
        for (int i = 0; i < book_moves; ++i) {
            chess::Move m = get_book_move(board, "book.bin");
            if (m == chess::Move::NULL_MOVE) break;
            board.makeMove(m);
        }

        std::vector<PackedPosition> game_positions;
        int result = 2; // Default to draw
        
        while (true) {
            // Check terminal states
            if (board.isHalfMoveDraw() || board.isRepetition()) {
                result = 2; break;
            }
            chess::Movelist ml;
            chess::movegen::legalmoves(ml, board);
            if (ml.size() == 0) {
                if (board.inCheck()) {
                    result = (board.sideToMove() == chess::Color::WHITE) ? 0 : 1; // Mated
                } else {
                    result = 2; // Stalemate
                }
                break;
            }

            // Search move
            chess::Move best = search_best_move(board, depth, depth); // fixed depth/nodes
            if (best == chess::Move::NULL_MOVE) break;
            
            // Do not record checkmate scores (too extreme for MSE training)
            if (std::abs(last_search_score) < 30000) {
                PackedPosition pos;
                pos.white = board.us(chess::Color::WHITE).getBits();
                pos.black = board.us(chess::Color::BLACK).getBits();
                pos.kings = board.pieces(chess::PieceType::KING).getBits();
                pos.queens = board.pieces(chess::PieceType::QUEEN).getBits();
                pos.rooks = board.pieces(chess::PieceType::ROOK).getBits();
                pos.bishops = board.pieces(chess::PieceType::BISHOP).getBits();
                pos.knights = board.pieces(chess::PieceType::KNIGHT).getBits();
                pos.pawns = board.pieces(chess::PieceType::PAWN).getBits();
                pos.ep = board.enpassantSq() == chess::Square::underlying::NO_SQ ? 0 : board.enpassantSq().index();
                pos.turn = board.sideToMove() == chess::Color::WHITE ? 1 : 0;
                
                uint8_t castling = 0;
                if (board.castlingRights().has(chess::Color::WHITE, chess::Board::CastlingRights::Side::KING_SIDE)) castling |= 1;
                if (board.castlingRights().has(chess::Color::WHITE, chess::Board::CastlingRights::Side::QUEEN_SIDE)) castling |= 2;
                if (board.castlingRights().has(chess::Color::BLACK, chess::Board::CastlingRights::Side::KING_SIDE)) castling |= 4;
                if (board.castlingRights().has(chess::Color::BLACK, chess::Board::CastlingRights::Side::QUEEN_SIDE)) castling |= 8;
                pos.castling = castling;
                
                pos.score = last_search_score; // Engine eval from white's perspective! Wait! last_search_score is side-to-move relative.
                // Convert eval to absolute white perspective:
                if (pos.turn == 0) pos.score = -pos.score; 

                game_positions.push_back(pos);
            }
            board.makeMove(best);
        }

        // Apply result and save game
        for (auto& pos : game_positions) {
            pos.result = result;
            out.write(reinterpret_cast<const char*>(&pos), sizeof(PackedPosition));
        }

        games_completed++;
        if (games_completed % 100 == 0) {
            std::cout << "Generated " << games_completed << " / " << num_games << " games.\n";
            out.flush();
        }
    }
    std::cout << "Self-play generation complete. Data written to " << output_file << "\n";
}
