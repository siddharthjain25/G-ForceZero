import sys

with open('src/gforce_engine.cpp', 'r') as f:
    content = f.read()

# 1. Add include
inc_target = 'extern "C" {\n#include "tbprobe.h"\n}\n'
if inc_target in content:
    content = content.replace(inc_target, inc_target + '#include "custom_nnue.hpp"\n\ncustom_nnue::Accumulator global_nnue_acc;\nbool nnue_loaded = false;\n')

# 2. Modify evaluate function
eval_target = '''// ─── Custom Evaluation ────────────────────────────────────
int evaluate(const Board& board) {
    return classical_evaluate(board);
}'''
eval_replace = '''// ─── Custom Evaluation ────────────────────────────────────
int evaluate(const Board& board) {
    if (nnue_loaded) {
        custom_nnue::init_accumulator(global_nnue_acc, board);
        return custom_nnue::evaluate(global_nnue_acc, board.sideToMove());
    }
    return classical_evaluate(board);
}'''
if eval_target in content:
    content = content.replace(eval_target, eval_replace)

# 3. Add load_weights to main()
main_target = '    // Custom evaluation does not require weight loading\n    std::cout << "G-ForceZero Engine ready.\\n";'
main_replace = '    if (!nnue_loaded) {\n        nnue_loaded = custom_nnue::load_weights("brain.nnue");\n    }\n    std::cout << "G-ForceZero Engine ready.\\n";'
if main_target in content:
    content = content.replace(main_target, main_replace)

with open('src/gforce_engine.cpp', 'w') as f:
    f.write(content)
