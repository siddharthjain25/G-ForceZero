import re

with open('src/nnue_engine.cpp', 'r') as f:
    code = f.read()

# Remove nnue include
code = re.sub(r'#include "nnue\.hpp"\n', '', code)

# Remove accumulator logic from evaluate
eval_pattern = r'// ─── Blended Evaluation \(NNUE \+ Classical\) ────────────────────────────────────\nint evaluate\(const Board& board.*?\}\n'
replacement = """// ─── Custom Evaluation ────────────────────────────────────
int evaluate(const Board& board) {
    return classical_evaluate(board);
}
"""
code = re.sub(eval_pattern, replacement, code, flags=re.DOTALL)

# Remove opt_nnue_weight
code = re.sub(r'int opt_nnue_weight = 100;\n', '', code)

# Replace evaluate calls
code = code.replace('evaluate(board, acc)', 'evaluate(board)')

# Update function signatures for quiescence and negamax
code = re.sub(r'int quiescence\(Board& board, int alpha, int beta, int ply = 0\)', r'int quiescence(Board& board, int alpha, int beta, int ply = 0)', code)
code = re.sub(r'int negamax\(Board& board, int depth, int alpha, int beta, int ply, bool allow_null, Move prev_move = Move::NULL_MOVE, Move prev_prev_move = Move::NULL_MOVE, Move excluded_move = Move::NULL_MOVE\)', r'int negamax(Board& board, int depth, int alpha, int beta, int ply, bool allow_null, Move prev_move = Move::NULL_MOVE, Move prev_prev_move = Move::NULL_MOVE, Move excluded_move = Move::NULL_MOVE)', code)

# Note: The previous sed removed ", nnue::Accumulator& acc" from some lines.
# But `root_acc` initialization and updates remain.
code = re.sub(r'\s*nnue::update_accumulator\(.*?\);\n', '\n', code)
code = re.sub(r'\s*nnue::undo_accumulator\(.*?\);\n', '\n', code)
code = re.sub(r'\s*nnue::make_null_move_accumulator\(.*?\);\n', '\n', code)
code = re.sub(r'\s*nnue::undo_null_move_accumulator\(.*?\);\n', '\n', code)
code = re.sub(r'\s*nnue::init_accumulator\(.*?\);\n', '\n', code)
code = re.sub(r'\s*nnue::Accumulator root_acc;\n', '\n', code)
code = re.sub(r'\s*nnue::Accumulator acc;\n', '\n', code)

# Fix any remaining acc arguments in negamax/quiescence calls (e.g. `negamax(..., true, root_acc, ...)` if they existed)
# It's safer to just regex replace `, root_acc` and `, acc` from negamax and quiescence calls.
# I will do that via search_best_move and the uci loop.
code = code.replace(', root_acc,', ',')
code = code.replace(', acc,', ',')
code = code.replace(', acc)', ')')

# Handle UCI eval
uci_eval = r'\} else if \(command == "eval"\) \{.*?\} else if \(command == "position"\)'
uci_eval_replacement = r"""} else if (command == "eval") {
            int score = classical_evaluate(board);
            std::cout << "Evaluation: " << score << "\n";
        } else if (command == "position")"""
code = re.sub(uci_eval, uci_eval_replacement, code, flags=re.DOTALL)

with open('src/nnue_engine.cpp', 'w') as f:
    f.write(code)

