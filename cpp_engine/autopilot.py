import os
import subprocess
import time

GAMES_PER_ITERATION = 2000
DEPTH = 5
PROCESSES = 4

print("==============================================")
print("   G-ForceZero Automated Training Pipeline    ")
print("==============================================\n")

MAX_ITERATIONS = 50 # Prevents Kaggle from hard-crashing at 12 hours
start_time = time.time()

# To start fresh like AlphaZero, delete best_model.pth if you want to start from scratch
if os.path.exists("best_model.pth"):
    print("[INFO] Resuming from existing best_model.pth...")
    iteration = 2 # Assuming we already bootstrapped
else:
    print("[INFO] Starting from scratch (Iteration 1). Bootstrapping required!")
    iteration = 1

while iteration <= MAX_ITERATIONS:
    # Stop early if we are approaching Kaggle's 12-hour limit (11.5 hours = 41400 seconds)
    if time.time() - start_time > 41400:
        print("\n[INFO] Approaching 12-hour limit! Exiting gracefully to save Kaggle outputs.")
        break
    print(f"\n--- [ ITERATION {iteration} ] ---")
    
    # Bootstrap: If it's the very first iteration, we don't have a smart NNUE yet.
    # We must use Classical Evaluation (Piece Values) to generate logical games!
    # Otherwise, we use our newly trained NNUE (Weight 100).
    nnue_weight = 0 if iteration == 1 else 100
    
    print(f"Phase 1: Generating {GAMES_PER_ITERATION} games (NNUE_Weight = {nnue_weight})...")
    
    games_per_proc = GAMES_PER_ITERATION // PROCESSES
    procs = []
    
    for i in range(PROCESSES):
        # We inject the NNUE_Weight setting before starting selfplay
        cmd = f"echo -e 'setoption name NNUE_Weight value {nnue_weight}\\nselfplay {games_per_proc} {DEPTH} out{i}.bin\\nquit' | ./build/nnue_engine"
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL)
        procs.append(p)
        
    for i, p in enumerate(procs):
        p.wait()
        
    print("Phase 1 Complete! Merging datasets...")
    
    # Combine the parallel files into one master dataset for this iteration
    os.system("cat out0.bin out1.bin out2.bin out3.bin > current_dataset.bin")
    os.system("rm out0.bin out1.bin out2.bin out3.bin")
    
    print(f"Phase 2: Training Neural Network...")
    
    # Call the PyTorch trainer. 
    # train_halfkp.py automatically handles atomic saving (os.replace) to brain.nnue
    # so even if you Ctrl-C during training, brain.nnue is never corrupted.
    import sys
    ret = os.system(f"{sys.executable} trainer/train_halfkp.py current_dataset.bin")
    
    if ret != 0:
        print("\n[!] Training was interrupted (Ctrl-C) or failed. Safely exiting autopilot.")
        break
        
    print(f"\nIteration {iteration} complete! Engine brain.nnue updated safely.")
    iteration += 1
    time.sleep(2) # Brief pause before next cycle
