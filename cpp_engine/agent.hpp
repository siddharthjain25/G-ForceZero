#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <onnxruntime_cxx_api.h>
#include "chess.hpp"

class Agent {
private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session* session = nullptr;
    Ort::AllocatorWithDefaultOptions allocator;

public:
    Agent(const std::string& model_path) : env(ORT_LOGGING_LEVEL_WARNING, "ChessNet") {
        try {
            OrtCUDAProviderOptions cuda_options;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not enable CUDA. Falling back to CPU. " << e.what() << std::endl;
        }
        
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = new Ort::Session(env, model_path.c_str(), session_options);
        std::cout << "[Agent] Successfully loaded ONNX model: " << model_path << std::endl;
    }

    ~Agent() {
        delete session;
    }

    std::vector<float> board_to_tensor(const chess::Board& board) {
        std::vector<float> tensor(19 * 8 * 8, 0.0f);
        bool is_black = board.sideToMove() == chess::Color::BLACK;
        
        for (int sq = 0; sq < 64; ++sq) {
            chess::Piece piece = board.at(chess::Square(sq));
            if (piece != chess::Piece::NONE) {
                int effective_sq = is_black ? (sq ^ 56) : sq;
                int p_type = static_cast<int>(piece.type());
                int p_color = (piece.color() == chess::Color::WHITE) ? 1 : 0;
                
                int channel_offset;
                if (is_black) {
                    channel_offset = (p_color == 0) ? 0 : 6;
                } else {
                    channel_offset = (p_color == 1) ? 0 : 6;
                }
                
                int r = 7 - (effective_sq / 8);
                int c = effective_sq % 8;
                tensor[(channel_offset + p_type) * 64 + (r * 8 + c)] = 1.0f;
            }
        }
        
        for (int i = 14*64; i < 15*64; ++i) tensor[i] = 1.0f;
        
        auto cr = board.castlingRights();
        bool wK = cr.has(chess::Color::WHITE, chess::Board::CastlingRights::Side::KING_SIDE);
        bool wQ = cr.has(chess::Color::WHITE, chess::Board::CastlingRights::Side::QUEEN_SIDE);
        bool bK = cr.has(chess::Color::BLACK, chess::Board::CastlingRights::Side::KING_SIDE);
        bool bQ = cr.has(chess::Color::BLACK, chess::Board::CastlingRights::Side::QUEEN_SIDE);
        
        bool aK = is_black ? bK : wK;
        bool aQ = is_black ? bQ : wQ;
        bool oK = is_black ? wK : bK;
        bool oQ = is_black ? wQ : bQ;
        
        if (aK) {
            for (int r=0; r<4; ++r) for (int c=0; c<4; ++c) tensor[16*64 + r*8 + c] = 1.0f;
        }
        if (aQ) {
            for (int r=0; r<4; ++r) for (int c=4; c<8; ++c) tensor[16*64 + r*8 + c] = 1.0f;
        }
        if (oK) {
            for (int r=4; r<8; ++r) for (int c=0; c<4; ++c) tensor[16*64 + r*8 + c] = 1.0f;
        }
        if (oQ) {
            for (int r=4; r<8; ++r) for (int c=4; c<8; ++c) tensor[16*64 + r*8 + c] = 1.0f;
        }
        
        if (board.enpassantSq() != chess::Square::NO_SQ) {
            int ep = board.enpassantSq().index();
            if (is_black) ep ^= 56;
            int r = 7 - (ep / 8);
            int c = ep % 8;
            tensor[17*64 + r*8 + c] = 1.0f;
        }
        
        float full_move = std::min(1.0f, board.fullMoveNumber() / 100.0f);
        for (int i=15*64; i<16*64; ++i) tensor[i] = full_move;
        
        float hm = std::min(1.0f, board.halfMoveClock() / 50.0f);
        for (int i=18*64; i<19*64; ++i) tensor[i] = hm;
        
        static bool black_dumped = false;
        if (!black_dumped && board.sideToMove() == chess::Color::BLACK && board.fullMoveNumber() >= 30) {
            std::ofstream out("cpp_tensor_black.txt");
            for (float f : tensor) {
                out << std::fixed << std::setprecision(6) << f << "\n";
            }
            out.close();
            black_dumped = true;
        }
        
        return tensor;
    }

    std::pair<std::vector<float>, float> evaluate(const chess::Board& board) {
        std::vector<float> input_tensor_values = board_to_tensor(board);
        
        std::vector<int64_t> input_node_dims = {1, 19, 8, 8};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(), 
            input_node_dims.data(), input_node_dims.size());

        const char* input_names[] = {"input"};
        const char* output_names[] = {"policy", "value"};

        auto output_tensors = session->Run(Ort::RunOptions{nullptr}, 
            input_names, &input_tensor, 1, 
            output_names, 2);

        auto shape0 = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        float* policy_data;
        float* value_data;
        
        if (shape0.size() > 1 || (shape0.size() == 1 && shape0[0] > 1)) {
            policy_data = output_tensors[0].GetTensorMutableData<float>();
            value_data = output_tensors[1].GetTensorMutableData<float>();
        } else {
            policy_data = output_tensors[1].GetTensorMutableData<float>();
            value_data = output_tensors[0].GetTensorMutableData<float>();
        }
        
        std::vector<float> policy(policy_data, policy_data + 4096);
        float value = value_data[0];

        return {policy, value};
    }
};
