# G-ForceZero ♟️

A custom, high-performance C++ Chess Engine built from scratch, powered by a custom **HalfKP NNUE** (Neural Network Updated Efficiently) and designed for autonomous cloud deployment.

## Acknowledgements
Some critical search algorithms, heuristic structures, and function references in this engine were inspired by and adapted from the [Caissa](https://github.com/Witek902/Caissa) chess engine. We are deeply grateful to its developers for their open-source contributions to computer chess.

---

## 📑 Table of Contents
1. [Features](#1-features)
2. [Compilation](#2-compilation)
3. [Training the Neural Network](#3-training-the-neural-network-nnue)
4. [SPSA Tuning](#4-spsa-tuning-optimizing-search)
5. [Cloud Deployment (Lichess Bot)](#5-cloud-deployment-lichess-bot)

---

## 1. Features
* **Search Architecture:** C++ based alpha-beta search with highly tuned heuristics including Null Move Pruning (NMP), Late Move Reductions (LMR), and Reverse Futility Pruning (RFP).
* **Multi-Threading:** Lazy SMP support allowing the engine to effortlessly scale across multiple CPU cores.
* **SIMD Acceleration:** AVX2 vectorized inference for blazing-fast neural network evaluations.
* **Custom NNUE:** HalfKP network architecture (40960 input features → 256 hidden neurons → 1 evaluation output) evaluated natively in C++.
* **Cloud Ready:** Fully Dockerized and configured with automated matchmaking to run 24/7 as an autonomous Lichess Bot on free-tier cloud hardware.

---

## 2. Compilation
To compile the C++ engine natively on Linux:
```bash
cd cpp_engine
g++ -O3 -march=native nnue_engine.cpp nnue.cpp -o nnue_engine
```
This generates the highly optimized `nnue_engine` binary. The `-O3` and `-march=native` flags ensure that the compiler utilizes AVX2 instructions specifically optimized for your processor.

---

## 3. Training the Neural Network (NNUE)

G-ForceZero uses a custom PyTorch trainer that can parse massive datasets (100GB+) directly from your hard drive into the GPU with extremely flat memory usage (~300MB RAM).

### A. Download the Training Data
The neural network learns fastest from positions that contain perfect Stockfish evaluations. 
1. Download a massive Lichess Evaluation JSONL dataset (e.g., `lichess_db_eval.jsonl`) from the official [Lichess Open Database](https://database.lichess.org/#evals).
2. Decompress the file (these files are massive and can easily exceed 100GB).

### B. PyTorch Training
1. Activate your Python virtual environment (ensure PyTorch and Numpy are installed).
2. Run the highly-optimized trainer:
```bash
cd cpp_engine
python3 train_halfkp.py /path/to/lichess_db_eval.jsonl
```
*Note: Because the trainer is designed for massive datasets, it will silently count the millions of lines in the file for the first 5-15 minutes before the GPU spins up to calculate the progress bar.*

### C. Safe Interrupts
You can press `Ctrl+C` at any time during training. The script will safely intercept the signal, instantly freeze the neural network, and export your progress to a quantized `nnue_weights.bin` before shutting down!

---

## 4. SPSA Tuning (Optimizing Search)
Once the neural network is highly trained, you can mathematically optimize the search heuristics (like determining the absolute perfect margin for Null Move Pruning).

The included `spsa_tuner.py` script makes the engine play thousands of automated blitz matches against itself. It dynamically tweaks internal UCI variables, calculates ELO differences, and converges on the perfect mathematical parameters.
```bash
cd cpp_engine
python3 spsa_tuner.py
```

---

## 5. Cloud Deployment (Lichess Bot)

G-ForceZero is fully configured to run autonomously on Lichess using cloud hosting platforms like Render, HuggingFace Spaces, or Oracle Cloud.

### Infrastructure
* **Dockerfile:** Automatically compiles the engine and sets up the official `lichess-bot` Python environment.
* **Keep-Alive Server:** The `render/` folder contains a Flask dummy web-server designed to bind to cloud ports and prevent platforms like Render from putting the bot to sleep.

### Configured for Cloud Survival
Cloud hardware is often heavily limited (e.g., 0.1 vCPU). The `config.yml` has been meticulously optimized to prevent the engine from losing on time:
* **Concurrency = 1:** Forces the bot to dedicate all CPU power to a single game at a time.
* **Strict Increments:** The bot automatically rejects Bullet games and any games without at least a +2 second increment. 
* **Auto-Matchmaking:** When idle, the bot automatically creates challenges strictly targeting safe time controls (e.g., 3+2, 5+5, 10+10).
* **Polite Declines:** Automatically messages users who send incompatible challenges to explain the hardware limitations.

### How to Deploy
1. Commit the code and your trained `nnue_weights.bin` file to GitHub.
2. Link the repository to your chosen Cloud Platform (selecting Docker as the environment).
3. Set the `LICHESS_BOT_TOKEN` environment variable in your cloud dashboard.
4. Deploy! The bot will instantly compile in the cloud and come alive on Lichess.
