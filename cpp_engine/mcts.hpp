#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include "chess.hpp"
#include "agent.hpp"

class Node {
public:
    chess::Board board;
    chess::Move move;
    float prior;
    int visits;
    float value_sum;
    std::vector<std::shared_ptr<Node>> children;
    bool is_expanded;

    Node(const chess::Board& b, chess::Move m, float p = 0.0f) 
        : board(b), move(m), prior(p), visits(0), value_sum(0.0f), is_expanded(false) {}

    float q_value() const {
        if (visits == 0) return 0.0f;
        return value_sum / visits;
    }
};

class MCTS {
private:
    Agent* agent;
    float c_puct;

    // Softmax function
    std::vector<float> softmax(const std::vector<float>& logits) {
        std::vector<float> probs(logits.size());
        float max_logit = *std::max_element(logits.begin(), logits.end());
        float sum = 0.0f;
        for (size_t i = 0; i < logits.size(); ++i) {
            probs[i] = std::exp(logits[i] - max_logit);
            sum += probs[i];
        }
        for (size_t i = 0; i < probs.size(); ++i) {
            probs[i] /= sum;
        }
        return probs;
    }

    // Convert move to integer index [0, 4095]
    int move_to_idx(const chess::Move& m, bool is_black) {
        
        int from = m.from().index();
        int to = m.to().index();
        if (is_black) {
            from ^= 56;
            to ^= 56;
        }
        return from * 64 + to;

    }

public:
    MCTS(Agent* a, float cpuct = 1.25f) : agent(a), c_puct(cpuct) {}

    std::shared_ptr<Node> search(const chess::Board& root_board, int num_simulations) {
        auto root = std::make_shared<Node>(root_board, chess::Move::NULL_MOVE);

        for (int i = 0; i < num_simulations; ++i) {
            std::shared_ptr<Node> node = root;
            std::vector<std::shared_ptr<Node>> search_path;
            search_path.push_back(node);

            // 1. SELECT
            while (node->is_expanded && !node->children.empty()) {
                float best_ucb = -1e9f;
                std::shared_ptr<Node> best_child = nullptr;

                for (const auto& child : node->children) {
                    float u = c_puct * child->prior * std::sqrt(node->visits) / (1.0f + child->visits);
                    float q = child->q_value();
                    
                    // We flip Q because child's Q is relative to the opponent
                    float ucb = -q + u; 

                    if (ucb > best_ucb) {
                        best_ucb = ucb;
                        best_child = child;
                    }
                }
                node = best_child;
                search_path.push_back(node);
            }

            // 2. EXPAND AND EVALUATE
            auto status = node->board.isGameOver();
            
            float value = 0.0f;

            if (status.first != chess::GameResultReason::NONE) {
                // Terminal node
                if (status.second == chess::GameResult::WIN) value = 1.0f;
                else if (status.second == chess::GameResult::LOSE) value = -1.0f;
                else value = 0.0f;
            } else {
// Forward declaration of tactical evaluator
int alphaBeta(chess::Board& board, int depth, int alpha, int beta, int current_ply);

                // Network Evaluation
                auto eval_result = agent->evaluate(node->board);
                std::vector<float> policy_logits = eval_result.first;
                float nn_value = std::tanh(eval_result.second); // Bound between -1 and 1

                // Blend with tactical search (3 plies deep)
                // We use a copy of the board because alphaBeta mutates it
                chess::Board temp_board = node->board;
                int tactical_cp = alphaBeta(temp_board, 3, -20000, 20000, 0);
                
                // Use a wider divisor (1500) to prevent squashing huge material advantages
                // and strictly map mates to 1.0/-1.0
                float tactical_val = 0.0f;
                if (tactical_cp > 9000) tactical_val = 1.0f;
                else if (tactical_cp < -9000) tactical_val = -1.0f;
                else tactical_val = std::tanh(tactical_cp / 1500.0f);
                
                value = (0.6f * tactical_val) + (0.4f * nn_value);

                std::vector<float> policy_probs = softmax(policy_logits);

                // Generate legal moves and expand
                chess::Movelist moves;
                chess::movegen::legalmoves(moves, node->board);

                float legal_prob_sum = 0.0f;
                std::vector<std::pair<chess::Move, float>> legal_moves_probs;

                for (const auto& m : moves) {
                    int idx = move_to_idx(m, node->board.sideToMove() == chess::Color::BLACK);
                    float p = policy_probs[idx];
                    legal_moves_probs.push_back({m, p});
                    legal_prob_sum += p;
                }

                for (auto& mp : legal_moves_probs) {
                    chess::Board child_board = node->board;
                    child_board.makeMove(mp.first);
                    
                    // Normalize probability
                    float norm_p = (legal_prob_sum > 0) ? (mp.second / legal_prob_sum) : (1.0f / legal_moves_probs.size());
                    
                    
                    auto child_node = std::make_shared<Node>(child_board, mp.first, norm_p);
                    node->children.push_back(child_node);
                }
                node->is_expanded = true;
            }

            // 3. BACKPROPAGATE
            for (int j = search_path.size() - 1; j >= 0; --j) {
                search_path[j]->value_sum += value;
                search_path[j]->visits += 1;
                // Flip value for the parent node and apply a ply-discount to strongly prefer faster mates
                value = -value * 0.999f;
            }
        }

        return root;
    }

    chess::Move get_best_move(const chess::Board& root_board, int num_simulations, float temperature = 0.0f) {
        auto root = search(root_board, num_simulations);
        
        if (root->children.empty()) return chess::Move::NULL_MOVE;

        std::shared_ptr<Node> best_child = nullptr;
        
        if (temperature == 0.0f) {
            // Greedy choice: max visits
            int max_visits = -1;
            for (const auto& child : root->children) {
                if (child->visits > max_visits) {
                    max_visits = child->visits;
                    best_child = child;
                }
            }
        } else {
            // Not implemented for brevity, would do stochastic sampling
            best_child = root->children[0];
        }

        return best_child->move;
    }
};