#!/usr/bin/env python3
"""
pgn_to_epd_wdl.py - Convert a Lichess PGN database to EPD training data
using GAME OUTCOME as the training label (no Stockfish needed).

This is the approach used by Leela Chess Zero and modern NNUE trainers.
The label is the game result (1.0 = white win, 0.5 = draw, 0.0 = black win),
converted to a centipawn approximation using the logit function:
    cp = logit(result) / K   where K = 0.003 (matching our sigmoid formula)

Usage:
    python3 pgn_to_epd_wdl.py \\
        --pgn /path/to/lichess.pgn \\
        --out training_data.epd \\
        --games 2000000 \\
        --min-elo 1800 \\
        --workers 4
"""

import argparse
import chess
import chess.pgn
import io
import math
import os
import sys
import random
import time
from multiprocessing import Pool, cpu_count

# ─── Configuration ─────────────────────────────────────────────────────────────
MIN_PLY        = 16    # Skip first 8 moves (opening theory)
MAX_PLY        = 300   # Ignore extremely long games
SAMPLE_RATE    = 4     # Take every Nth position
MAX_PER_GAME   = 6     # Hard cap positions per game
K              = 0.003 # Sigmoid scaling constant (matches train_halfkp.py)
CP_CLAMP       = 800   # Clamp synthetic CP to ±800 (realistic range)

def result_to_cp(result_str):
    """Convert game result to synthetic centipawn score."""
    if result_str == "1-0":
        wdl = 1.0
    elif result_str == "0-1":
        wdl = 0.0
    elif result_str == "1/2-1/2":
        wdl = 0.5
    else:
        return None

    # Inverse sigmoid: cp = ln(p/(1-p)) / K
    # Clamp to avoid ±inf at p=0 or p=1
    p = max(0.01, min(0.99, wdl))
    cp = math.log(p / (1.0 - p)) / K
    return max(-CP_CLAMP, min(CP_CLAMP, int(cp)))


def process_game_text(game_text):
    """Process one PGN game string, return list of EPD lines."""
    try:
        pgn_io = io.StringIO(game_text)
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            return []
    except Exception:
        return []

    result = game.headers.get("Result", "*")
    cp = result_to_cp(result)
    if cp is None:
        return []

    board = game.board()
    moves = list(game.mainline_moves())
    n = len(moves)

    if n < MIN_PLY or n > MAX_PLY:
        return []

    # Build sample index set
    indices = list(range(MIN_PLY, max(MIN_PLY + 1, n - 4), SAMPLE_RATE))
    if len(indices) > MAX_PER_GAME:
        indices = random.sample(indices, MAX_PER_GAME)
    sample_set = set(indices)

    lines = []
    for i, move in enumerate(moves):
        is_capture = board.is_capture(move)
        board.push(move)

        if i not in sample_set:
            continue
        if board.is_check():
            continue
        if is_capture:
            continue
        if board.is_game_over():
            continue

        # Flip sign of cp if it's black's turn (score is from side-to-move perspective)
        score = cp if board.turn == chess.WHITE else -cp
        lines.append(f'{board.fen()} c9 "{score}";')

    return lines


def process_chunk(args):
    """Worker: process a chunk of PGN text blocks."""
    game_texts = args
    all_lines = []
    for gt in game_texts:
        all_lines.extend(process_game_text(gt))
    return all_lines


def stream_games(pgn_path, min_elo=1800):
    """Stream individual PGN game strings from a large file, with ELO filter."""
    buf = []
    white_elo = 0
    black_elo = 0

    with open(pgn_path, 'r', errors='replace', buffering=1 << 20) as f:
        for line in f:
            # Quick ELO extraction from header lines
            if line.startswith('[WhiteElo'):
                try: white_elo = int(line.split('"')[1])
                except: white_elo = 0
            elif line.startswith('[BlackElo'):
                try: black_elo = int(line.split('"')[1])
                except: black_elo = 0

            buf.append(line)

            # Detect end of game: empty line after moves
            if line.strip() == '' and len(buf) > 5:
                # Check if the buffer contains a game (has move text)
                text = ''.join(buf)
                if '1.' in text or '1...' in text:
                    if white_elo >= min_elo and black_elo >= min_elo:
                        yield text
                    buf = []
                    white_elo = 0
                    black_elo = 0
                else:
                    buf = []

    # Yield last game if any
    if buf:
        text = ''.join(buf)
        if '1.' in text and white_elo >= min_elo and black_elo >= min_elo:
            yield text


def main():
    parser = argparse.ArgumentParser(description='Convert PGN to EPD training data (WDL labels, no Stockfish)')
    parser.add_argument('--pgn',      required=True,              help='Input Lichess PGN file')
    parser.add_argument('--out',      required=True,              help='Output EPD file')
    parser.add_argument('--games',    type=int, default=2_000_000, help='Max games to process')
    parser.add_argument('--min-elo',  type=int, default=1800,     help='Min ELO filter for both players')
    parser.add_argument('--workers',  type=int, default=max(1, cpu_count() - 1), help='Worker processes')
    parser.add_argument('--chunk',    type=int, default=500,      help='Games per worker batch')
    args = parser.parse_args()

    if not os.path.exists(args.pgn):
        print(f"[ERROR] PGN file not found: {args.pgn}")
        sys.exit(1)

    print(f"Input:    {args.pgn} ({os.path.getsize(args.pgn) / 1e9:.1f} GB)")
    print(f"Output:   {args.out}")
    print(f"Games:    up to {args.games:,}")
    print(f"Min ELO:  {args.min_elo}")
    print(f"Workers:  {args.workers}")
    print()

    start_time = time.time()
    total_games = 0
    total_positions = 0
    chunk_buf = []

    with open(args.out, 'w', buffering=1 << 20) as out_f:
        game_gen = stream_games(args.pgn, args.min_elo)

        for game_text in game_gen:
            if total_games >= args.games:
                break

            chunk_buf.append(game_text)

            if len(chunk_buf) >= args.chunk * args.workers:
                # Split into worker chunks
                chunks = [chunk_buf[i:i+args.chunk] for i in range(0, len(chunk_buf), args.chunk)]
                total_games += len(chunk_buf)
                chunk_buf = []

                with Pool(args.workers) as pool:
                    results = pool.map(process_chunk, chunks)

                for lines in results:
                    for line in lines:
                        out_f.write(line + '\n')
                        total_positions += 1

                elapsed = time.time() - start_time
                rate = total_games / elapsed if elapsed > 0 else 1
                eta_min = (args.games - total_games) / rate / 60
                print(f"  Games: {total_games:>8,} | Positions: {total_positions:>10,} | "
                      f"{rate:.0f} games/s | ETA: {eta_min:.1f} min", flush=True)

                out_f.flush()

        # Process remaining
        if chunk_buf:
            total_games += len(chunk_buf)
            chunks = [chunk_buf[i:i+args.chunk] for i in range(0, len(chunk_buf), args.chunk)]
            with Pool(args.workers) as pool:
                results = pool.map(process_chunk, chunks)
            for lines in results:
                for line in lines:
                    out_f.write(line + '\n')
                    total_positions += 1

    elapsed = time.time() - start_time
    print(f"\n✅ Done!")
    print(f"   Games processed:  {total_games:,}")
    print(f"   Positions saved:  {total_positions:,}")
    print(f"   Time elapsed:     {elapsed/60:.1f} minutes")
    print(f"   Output file:      {args.out}")
    print(f"   File size:        {os.path.getsize(args.out) / 1e6:.1f} MB")


if __name__ == '__main__':
    main()
