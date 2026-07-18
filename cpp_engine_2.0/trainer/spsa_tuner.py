import asyncio
import chess
import chess.engine
import random
import math
import sys
import time

ENGINE_PATH = "./nnue_engine"
CONCURRENCY = 12
GAMES_PER_ITER = 100
TIME_LIMIT = 0.1 # Very fast games for tuning

# SPSA hyperparameters
a_base = 0.5
c_base = 2.0
A = 10
alpha = 0.602
gamma = 0.101

params = {
    "RFP_Margin": {"val": 80.0, "min": 10, "max": 200, "step": 5},
    "NMP_Base": {"val": 3.0, "min": 1, "max": 10, "step": 1},
    "NMP_Depth_Div": {"val": 6.0, "min": 1, "max": 15, "step": 1},
    "NMP_Eval_Div": {"val": 200.0, "min": 50, "max": 500, "step": 10},
    "LMR_Mult": {"val": 225.0, "min": 50, "max": 500, "step": 10},
    "FP_Margin_Base": {"val": 100.0, "min": 10, "max": 300, "step": 10},
    "FP_Margin_Mult": {"val": 60.0, "min": 10, "max": 200, "step": 5},
}

async def play_game(engine1_options, engine2_options):
    transport, engine1 = await chess.engine.popen_uci(ENGINE_PATH)
    transport2, engine2 = await chess.engine.popen_uci(ENGINE_PATH)
    
    # Need to convert int values to string for chess.engine
    opts1 = {k: str(v) for k, v in engine1_options.items()}
    opts2 = {k: str(v) for k, v in engine2_options.items()}
    
    await engine1.configure(opts1)
    await engine2.configure(opts2)
    
    board = chess.Board()
    # Play random opening moves to create variety
    for _ in range(4):
        moves = list(board.legal_moves)
        if not moves: break
        board.push(random.choice(moves))
        
    res = 0.5
    try:
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                result = await engine1.play(board, chess.engine.Limit(time=TIME_LIMIT))
            else:
                result = await engine2.play(board, chess.engine.Limit(time=TIME_LIMIT))
            board.push(result.move)
            
        outcome = board.outcome(claim_draw=True)
        if outcome.winner == chess.WHITE:
            res = 1.0
        elif outcome.winner == chess.BLACK:
            res = 0.0
    except Exception as e:
        print(f"Game error: {e}")
        pass
        
    await engine1.quit()
    await engine2.quit()
    return res

async def run_iteration(iter_num):
    print(f"\n--- Iteration {iter_num} ---")
    ak = a_base / ((iter_num + A) ** alpha)
    ck = c_base / (iter_num ** gamma)
    
    delta = {}
    for p in params:
        # Bernoulli +/- 1
        delta[p] = 1 if random.random() > 0.5 else -1
        
    theta_plus = {}
    theta_minus = {}
    
    for p, p_data in params.items():
        step = p_data["step"]
        # Perturb by ck * step
        perturb = ck * step * delta[p]
        theta_plus[p] = max(p_data["min"], min(p_data["max"], int(p_data["val"] + perturb)))
        theta_minus[p] = max(p_data["min"], min(p_data["max"], int(p_data["val"] - perturb)))
        
    print(f"Testing Theta+ vs Theta-")
    
    # Run games concurrently
    tasks = []
    # Half games engine1 is white, half engine1 is black
    for i in range(GAMES_PER_ITER):
        if i % 2 == 0:
            tasks.append(play_game(theta_plus, theta_minus))
        else:
            # reverse roles, so invert score
            tasks.append(play_game(theta_minus, theta_plus))
            
    # Batch the tasks to avoid spawning too many engines at once
    results = []
    for i in range(0, len(tasks), CONCURRENCY):
        batch = tasks[i:i+CONCURRENCY]
        batch_results = await asyncio.gather(*batch)
        
        for j, res in enumerate(batch_results):
            # If engine1 was black (i%2 != 0), invert result
            idx = i + j
            if idx % 2 != 0:
                res = 1.0 - res
            results.append(res)
        print(f"Completed {len(results)}/{GAMES_PER_ITER} games...")
        
    score_plus = sum(results) / len(results)
    print(f"Theta+ Win Rate: {score_plus:.3f}")
    
    # Update parameters
    for p, p_data in params.items():
        step = p_data["step"]
        # Gradient estimate
        g_k = (score_plus - (1.0 - score_plus)) / (2.0 * ck * delta[p])
        # We want to MAXIMIZE score_plus, so we add the gradient
        change = ak * g_k * step
        p_data["val"] = max(p_data["min"], min(p_data["max"], p_data["val"] + change))
        print(f"{p}: {p_data['val']:.2f}")

async def main():
    for i in range(1, 101):
        await run_iteration(i)
        
if __name__ == "__main__":
    asyncio.run(main())
