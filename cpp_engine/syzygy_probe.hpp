#pragma once
#include "chess.hpp"
#include <string>
#include <iostream>
#include <cstdio>

inline void init_syzygy(const std::string& path) {
    std::cout << "info string Lichess Cloud API initialized for 7-piece endgames.\n";
}

inline chess::Move probe_syzygy_root(const chess::Board& board) {
    if (board.occ().count() > 7) return chess::Move::NULL_MOVE;
    if (!board.castlingRights().isEmpty()) return chess::Move::NULL_MOVE;

    std::string fen = board.getFen();
    for (char& c : fen) {
        if (c == ' ') c = '_';
    }
    
    std::string cmd = "wget -qO- \"https://tablebase.lichess.ovh/standard?fen=" + fen + "\"";
    
    char buffer[1024];
    std::string result = "";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return chess::Move::NULL_MOVE;
    
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    pclose(pipe);
    
    // Extract the top recommended move from the JSON array
    size_t uci_pos = result.find("\"uci\":\"");
    if (uci_pos == std::string::npos) return chess::Move::NULL_MOVE;
    
    uci_pos += 7; // length of "uci":"
    size_t end_pos = result.find("\"", uci_pos);
    if (end_pos == std::string::npos) return chess::Move::NULL_MOVE;
    
    std::string uci = result.substr(uci_pos, end_pos - uci_pos);
    
    if (uci.length() < 4 || uci.length() > 5) return chess::Move::NULL_MOVE;
    
    return chess::uci::uciToMove(board, uci);
}
