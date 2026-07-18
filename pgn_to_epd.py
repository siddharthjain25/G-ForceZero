import chess.pgn
import sys
import multiprocessing

def process_game(pgn_text):
    import chess.pgn
    import io
    
    epd_lines = []
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return []
        
    result = game.headers.get("Result", "*")
    w_elo = game.headers.get("WhiteElo", "0")
    b_elo = game.headers.get("BlackElo", "0")
    
    try:
        if int(w_elo) < 2500 or int(b_elo) < 2500:
            return [] # Skip games where either player is below 2500 Elo
    except ValueError:
        return [] # Skip games with missing or corrupted Elo data

    if result == "1-0":
        res_str = '"1.0"'
    elif result == "0-1":
        res_str = '"0.0"'
    elif result == "1/2-1/2":
        res_str = '"0.5"'
    else:
        return [] # Skip games without clear results

    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
        # Skip openings (first 10 plies) to avoid heavy draw bias
        if board.ply() > 10:
            # Create standard EPD string with c9 result
            epd = f'{board.fen()} c9 {res_str}'
            epd_lines.append(epd)
            
    return epd_lines

def main(pgn_file, output_file):
    print(f"Converting {pgn_file} to {output_file}...")
    
    # We read games as raw text first so we can use multiprocessing
    # python-chess is slow if we do it single-threaded!
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    
    games = []
    current_game = []
    
    MAX_GAMES = 1000000 # 1 Million games is more than enough for a huge training run!
    games_processed = 0
    
    with open(output_file, 'w') as out_f:
        with open(pgn_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                current_game.append(line)
                if line.startswith("1."): # End of headers, start of moves
                    pass 
                elif not line.strip() and len(current_game) > 5 and current_game[-2].startswith("1."):
                    # Game complete
                    games.append("".join(current_game))
                    current_game = []
                    
                    if len(games) >= 1000:
                        # Process chunk in parallel
                        results = pool.map(process_game, games)
                        for res in results:
                            for epd in res:
                                out_f.write(epd + "\n")
                        games_processed += len(games)
                        games = []
                        print(".", end="", flush=True)
                        
                        if games_processed >= MAX_GAMES:
                            print(f"\n[INFO] Reached {MAX_GAMES} games! Stopping parsing so PyTorch has time to train.")
                            break

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python pgn_to_epd.py <input.pgn> <output.epd>")
    else:
        main(sys.argv[1], sys.argv[2])
