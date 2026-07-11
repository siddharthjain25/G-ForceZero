#pragma once
#include "chess.hpp"
#include "Fathom/src/tbprobe.h"
#include <string>
#include <iostream>

inline void init_syzygy(const std::string& path) {
    if (tb_init(path.c_str())) {
        std::cout << "info string Syzygy tablebases initialized. Max pieces: " << TB_LARGEST << "\n";
    } else {
        std::cout << "info string Failed to load Syzygy tablebases from " << path << "\n";
    }
}

inline chess::Move probe_syzygy_root(const chess::Board& board) {
    if (TB_LARGEST == 0) return chess::Move::NULL_MOVE;
    if (board.occ().count() > TB_LARGEST) return chess::Move::NULL_MOVE;
    if (!board.castlingRights().isEmpty()) return chess::Move::NULL_MOVE;
    if (board.halfMoveClock() > 0) return chess::Move::NULL_MOVE; // Fathom fails if rule50 > 0

    unsigned res = tb_probe_root(
        board.us(chess::Color::WHITE).getBits(),
        board.us(chess::Color::BLACK).getBits(),
        board.pieces(chess::PieceType::KING).getBits(),
        board.pieces(chess::PieceType::QUEEN).getBits(),
        board.pieces(chess::PieceType::ROOK).getBits(),
        board.pieces(chess::PieceType::BISHOP).getBits(),
        board.pieces(chess::PieceType::KNIGHT).getBits(),
        board.pieces(chess::PieceType::PAWN).getBits(),
        0, 0,
        board.enpassantSq() == chess::Square::NO_SQ ? 0 : board.enpassantSq().index(),
        board.sideToMove() == chess::Color::WHITE,
        nullptr
    );

    if (res == TB_RESULT_FAILED) return chess::Move::NULL_MOVE;
    
    int from = TB_GET_FROM(res);
    int to = TB_GET_TO(res);
    int promo = TB_GET_PROMOTES(res);
    
    char f_file = 'a' + (from & 7);
    char f_rank = '1' + (from >> 3);
    char t_file = 'a' + (to & 7);
    char t_rank = '1' + (to >> 3);
    
    std::string uci = "";
    uci += f_file; uci += f_rank; uci += t_file; uci += t_rank;
    if (promo == TB_PROMOTES_QUEEN) uci += "q";
    else if (promo == TB_PROMOTES_ROOK) uci += "r";
    else if (promo == TB_PROMOTES_BISHOP) uci += "b";
    else if (promo == TB_PROMOTES_KNIGHT) uci += "n";
    
    return chess::uci::uciToMove(board, uci);
}
