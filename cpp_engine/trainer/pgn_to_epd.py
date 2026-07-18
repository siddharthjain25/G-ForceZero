#!/usr/bin/env python3
"""
pgn_to_epd.py - Convert a Lichess PGN database to an EPD training dataset.

Pipeline:
  1. Stream PGN games (memory-efficient, no full load)
  2. Reconstruct board position-by-position
  3. Skip: openings (< 8 ply), game-over positions, checks, captures on last move
  4. Call Stockfish in batch to evaluate sampled FENs
  5. Write "FEN c9 \"cp\";\\n" format compatible with train_halfkp.py

Usage:
  python3 pgn_to_epd.py \\
      --pgn /path/to/lichess.pgn \\
      --out training_data.epd \\
      --stockfish stockfish \\
      --games 500000 \\
      --depth 12 \\
      --workers 4
"""

import argparse
import chess
import chess.pgn
import chess.engine
import io
import os
import sys
import random
import time
from multiprocessing import Pool, cpu_count

# ─── Settings ────────────────────────────────────────────────────────────────
MIN_PLY         = 16   # Skip opening (first 8 moves per side)
MAX_PLY         = 400  # Skip very long games
SAMPLE_RATE     = 3    # Sample every Nth position from a game
MIN_ELO         = 1800 # Only use high-quality games
SKIP_CHECKS     = True # Skip positions where side to move is in check
SKIP_CAPTURES   = True # Skip positions where last move was a capture
BATCH_SIZE      = 256  # Stockfish batch size for evaluation

def collect_fens_from_game(game, max_positions=5):
    """Extract sampled FENs from a PGN game."""
    fens = []
    board = game.board()
    moves = list(game.mainline_moves())
    n = len(moves)

    if n < MIN_PLY or n > MAX_PLY:
        return fens

    result = game.headers.get("Result", "*")
    if result == "1-0":
        result_val = 1.0
    elif result == "0-1":
        result_val = 0.0
    elif result == "1/2-1/2":
        result_val = 0.5
    else:
        return fens  # Unknown result

    # Sample positions
    sample_indices = range(MIN_PLY, n - 4, SAMPLE_RATE)
    sample_indices = list(sample_indices)
    if len(sample_indices) > max_positions:
        sample_indices = random.sample(sample_indices, max_positions)
    sample_set = set(sample_indices)

    last_move_was_capture = False
    for i, move in enumerate(moves):
        is_capture = board.is_capture(move)
        board.push(move)

        if i in sample_set:
            if SKIP_CHECKS and board.is_check():
                continue
            if SKIP_CAPTURES and is_capture:
                continue
            if board.is_game_over():
                continue
            fens.append((board.fen(), result_val))

        last_move_was_capture = is_capture

    return fens


def evaluate_fens_with_stockfish(fens, stockfish_path, depth=12):
    """Evaluate a list of FENs using Stockfish. Returns list of (fen, cp) tuples."""
    results = []
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        engine.configure({"Threads": 1, "Hash": 16})
    except Exception as e:
        print(f"[ERROR] Could not start Stockfish: {e}", file=sys.stderr)
        return results

    for fen, game_result in fens:
        board = chess.Board(fen)
        try:
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            score = info["score"].white()
            if score.is_mate():
                cp = 10000 * (1 if score.mate() > 0 else -1)
            else:
                cp = score.score()
            if cp is not None:
                results.append((fen, cp))
        except Exception:
            continue

    engine.quit()
    return results


def process_chunk(args):
    """Worker function: process a chunk of PGN text and return EPD lines."""
    pgn_text, stockfish_path, depth, max_per_game = args
    fens_with_results = []
    pgn_io = io.StringIO(pgn_text)
    while True:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            break
        fens_with_results.extend(collect_fens_from_game(game, max_per_game))

    if not fens_with_results:
        return []

    # Evaluate with Stockfish
    fens_only = [(f, r) for f, r in fens_with_results]
    evaluated = evaluate_fens_with_stockfish(fens_only, stockfish_path, depth)

    # Format as EPD
    lines = []
    for fen, cp in evaluated:
        cp = max(-3000, min(3000, cp))  # Clamp extreme values
        lines.append(f'{fen} c9 "{cp}";')
    return lines


def stream_game_chunks(pgn_path, chunk_size=100):
    """Stream PGN in chunks of `chunk_size` games."""
    chunk = []
    buf = []
    with open(pgn_path, 'r', errors='replace') as f:
        for line in f:
            buf.append(line)
            if line.strip() == '' and buf and any(b.startswith('1.') for b in buf):
                # End of a game
                chunk.append(''.join(buf))
                buf = []
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
        if buf:
            chunk.append(''.join(buf))
        if chunk:
            yield chunk


def main():
    parser = argparse.ArgumentParser(description='Convert PGN to EPD training data with Stockfish evals')
    parser.add_argument('--pgn',        required=True,  help='Input PGN file')
    parser.add_argument('--out',        required=True,  help='Output EPD file')
    parser.add_argument('--stockfish',  default='stockfish', help='Stockfish binary path')
    parser.add_argument('--games',      type=int, default=500_000, help='Max games to process')
    parser.add_argument('--depth',      type=int, default=12,      help='Stockfish analysis depth')
    parser.add_argument('--workers',    type=int, default=max(1, cpu_count() - 1), help='Worker processes')
    parser.add_argument('--min-elo',    type=int, default=MIN_ELO, help='Min player ELO to include')
    parser.add_argument('--max-per-game', type=int, default=5,   help='Max positions per game')
    args = parser.parse_args()

    # Verify stockfish
    import shutil
    sf_path = shutil.which(args.stockfish) or args.stockfish
    if not os.path.exists(sf_path):
        print(f"[ERROR] Stockfish not found at: {sf_path}")
        sys.exit(1)
    print(f"Using Stockfish: {sf_path}")
    print(f"Workers: {args.workers}, Depth: {args.depth}, Min ELO: {args.min_elo}")
    print(f"Processing up to {args.games:,} games from: {args.pgn}")

    start_time = time.time()
    total_positions = 0
    total_games = 0
    chunk_size = args.workers * 25  # Games per batch

    with open(args.out, 'w') as out_f:
        for chunk in stream_game_chunks(args.pgn, chunk_size):
            if total_games >= args.games:
                break

            # Filter by ELO before processing
            filtered = []
            for game_text in chunk:
                # Quick ELO filter without full parse
                white_elo = 0
                black_elo = 0
                for line in game_text.split('\n')[:20]:
                    if '[WhiteElo' in line:
                        try: white_elo = int(line.split('"')[1])
                        except: pass
                    if '[BlackElo' in line:
                        try: black_elo = int(line.split('"')[1])
                        except: pass
                if white_elo >= args.min_elo and black_elo >= args.min_elo:
                    filtered.append(game_text)

            if not filtered:
                continue

            # Split into worker sub-chunks
            sub_size = max(1, len(filtered) // args.workers)
            work_items = []
            for i in range(0, len(filtered), sub_size):
                sub = filtered[i:i + sub_size]
                pgn_text = '\n\n'.join(sub)
                work_items.append((pgn_text, sf_path, args.depth, args.max_per_game))

            # Process in parallel
            with Pool(args.workers) as pool:
                results = pool.map(process_chunk, work_items)

            for epd_lines in results:
                for line in epd_lines:
                    out_f.write(line + '\n')
                    total_positions += 1

            total_games += len(filtered)
            elapsed = time.time() - start_time
            rate = total_games / elapsed if elapsed > 0 else 0
            eta = (args.games - total_games) / rate if rate > 0 else 0
            print(f"Games: {total_games:>8,} | Positions: {total_positions:>10,} | "
                  f"Rate: {rate:.0f} games/s | ETA: {eta/60:.1f} min", flush=True)

            if total_games % (chunk_size * 10) == 0:
                out_f.flush()  # Flush periodically

    elapsed = time.time() - start_time
    print(f"\n✅ Done! {total_positions:,} positions from {total_games:,} games in {elapsed/60:.1f} minutes")
    print(f"Output: {args.out}")


if __name__ == '__main__':
    main()
