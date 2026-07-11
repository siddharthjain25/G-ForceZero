#pragma once
#include "chess.hpp"
#include "polyglot_keys.hpp"
#include <fstream>
#include <vector>
#include <cstdlib>
#include <string>

struct PolyglotEntry {
    uint64_t key;
    uint16_t move;
    uint16_t weight;
    uint32_t learn;
};

inline uint64_t compute_polyglot_hash(const chess::Board& board) {
    uint64_t hash = 0;
    
    // Pieces
    for (int sq = 0; sq < 64; ++sq) {
        chess::Piece piece = board.at(chess::Square(sq));
        if (piece.type() != chess::PieceType::NONE) {
            int pc = static_cast<int>(piece.type());
            if (piece.color() == chess::Color::BLACK) pc += 6;
            hash ^= PolyGlotRandom[pc * 64 + sq];
        }
    }
    
    // Castling
    if (board.castlingRights().has(chess::Color::WHITE, chess::Board::CastlingRights::Side::KING_SIDE))   hash ^= PolyGlotRandom[768];
    if (board.castlingRights().has(chess::Color::WHITE, chess::Board::CastlingRights::Side::QUEEN_SIDE))  hash ^= PolyGlotRandom[769];
    if (board.castlingRights().has(chess::Color::BLACK, chess::Board::CastlingRights::Side::KING_SIDE))   hash ^= PolyGlotRandom[770];
    if (board.castlingRights().has(chess::Color::BLACK, chess::Board::CastlingRights::Side::QUEEN_SIDE))  hash ^= PolyGlotRandom[771];
    
    // En-passant
    if (board.enpassantSq() != chess::Square::NO_SQ) {
        int ep_file = board.enpassantSq().file();
        int ep_rank = board.enpassantSq().rank();
        int cap_rank = (ep_rank == 2) ? 3 : 4;
        chess::Color us = board.sideToMove();
        
        bool can_capture = false;
        if (ep_file > 0) {
            chess::Piece p = board.at(chess::Square(cap_rank * 8 + ep_file - 1));
            if (p.type() == chess::PieceType::PAWN && p.color() == us) can_capture = true;
        }
        if (ep_file < 7) {
            chess::Piece p = board.at(chess::Square(cap_rank * 8 + ep_file + 1));
            if (p.type() == chess::PieceType::PAWN && p.color() == us) can_capture = true;
        }
        if (can_capture) {
            hash ^= PolyGlotRandom[772 + ep_file];
        }
    }
    
    // Turn
    if (board.sideToMove() == chess::Color::WHITE) {
        hash ^= PolyGlotRandom[780];
    }
    
    return hash;
}

inline chess::Move polyglot_to_move(uint16_t move, const chess::Board& board) {
    int to_file = move & 7;
    int to_rank = (move >> 3) & 7;
    int from_file = (move >> 6) & 7;
    int from_rank = (move >> 9) & 7;
    int promo = (move >> 12) & 7;
    
    char f_file = 'a' + from_file;
    char f_rank = '1' + from_rank;
    char t_file = 'a' + to_file;
    char t_rank = '1' + to_rank;
    
    chess::Square from(from_rank * 8 + from_file);
    if (board.at(from).type() == chess::PieceType::KING) {
        if (f_file == 'e' && f_rank == '1' && t_file == 'h' && t_rank == '1') { t_file = 'g'; }
        if (f_file == 'e' && f_rank == '1' && t_file == 'a' && t_rank == '1') { t_file = 'c'; }
        if (f_file == 'e' && f_rank == '8' && t_file == 'h' && t_rank == '8') { t_file = 'g'; }
        if (f_file == 'e' && f_rank == '8' && t_file == 'a' && t_rank == '8') { t_file = 'c'; }
    }
    
    std::string uci = "";
    uci += f_file; uci += f_rank; uci += t_file; uci += t_rank;
    if (promo == 1) uci += "n";
    else if (promo == 2) uci += "b";
    else if (promo == 3) uci += "r";
    else if (promo == 4) uci += "q";
    
    return chess::uci::uciToMove(board, uci);
}

inline uint64_t swap64(uint64_t val) { return __builtin_bswap64(val); }
inline uint16_t swap16(uint16_t val) { return __builtin_bswap16(val); }

inline chess::Move get_book_move(const chess::Board& board, const std::string& book_path) {
    std::ifstream file(book_path, std::ios::binary);
    if (!file) return chess::Move::NULL_MOVE;
    
    uint64_t hash = compute_polyglot_hash(board);
    
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    if (size % 16 != 0 || size == 0) return chess::Move::NULL_MOVE;
    
    size_t num_entries = size / 16;
    long long low = 0, high = num_entries - 1;
    long long first_match = -1;
    
    while (low <= high) {
        long long mid = low + (high - low) / 2;
        file.seekg(mid * 16, std::ios::beg);
        uint64_t key;
        file.read(reinterpret_cast<char*>(&key), 8);
        key = swap64(key);
        
        if (key < hash) {
            low = mid + 1;
        } else if (key > hash) {
            high = mid - 1;
        } else {
            first_match = mid;
            high = mid - 1;
        }
    }
    
    if (first_match == -1) return chess::Move::NULL_MOVE;
    
    std::vector<PolyglotEntry> matches;
    file.seekg(first_match * 16, std::ios::beg);
    while (true) {
        PolyglotEntry entry;
        if (!file.read(reinterpret_cast<char*>(&entry.key), 8)) break;
        file.read(reinterpret_cast<char*>(&entry.move), 2);
        file.read(reinterpret_cast<char*>(&entry.weight), 2);
        file.read(reinterpret_cast<char*>(&entry.learn), 4);
        
        entry.key = swap64(entry.key);
        if (entry.key != hash) break;
        
        entry.move = swap16(entry.move);
        entry.weight = swap16(entry.weight);
        matches.push_back(entry);
    }
    
    if (matches.empty()) return chess::Move::NULL_MOVE;
    
    int total_weight = 0;
    for (const auto& m : matches) total_weight += m.weight;
    
    if (total_weight == 0) {
        return polyglot_to_move(matches[0].move, board);
    }
    
    int r = rand() % total_weight;
    int acc = 0;
    for (const auto& m : matches) {
        acc += m.weight;
        if (acc > r) {
            return polyglot_to_move(m.move, board);
        }
    }
    
    return polyglot_to_move(matches[0].move, board);
}
