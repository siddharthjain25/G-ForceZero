import chess.pgn
import io

pgn = io.StringIO("""[Event "casual bullet game"]
[Site "https://lichess.org/bmJ1AMId"]
[Date "2026.07.02"]
[Round "-"]
[White "G-ForceZero"]
[Black "cutecassia"]
[Result "0-1"]
[GameId "bmJ1AMId"]
[UTCDate "2026.07.02"]
[UTCTime "18:36:27"]
[WhiteElo "2168"]
[BlackElo "2340"]
[WhiteTitle "BOT"]
[BlackTitle "BOT"]
[Variant "Standard"]
[TimeControl "30+1"]
[ECO "B23"]
[Opening "Sicilian Defense: Closed, Traditional"]
[Termination "Normal"]

1. e4 c5 2. Nc3 Nc6 3. Nf3 d6 4. Bb5 e5 5. Bc4 Be7 6. Nd5 Nf6 7. Nxe7 Qxe7 8. d3 Be6 9. Bxe6 fxe6 10. O-O Nd7 11. Bg5 Qf7 12. Be3 O-O 13. Ng5 Qe7 14. Qe2 a5 15. Rfe1 Qe8 16. Qg4 Rf6 17. Rad1 h6 18. Nf3 Qf7 19. a3 a4 20. Qh3 Rf8 21. c3 b6 22. d4 cxd4 23. cxd4 exd4 24. Nxd4 Nxd4 25. Bxd4 e5 26. Be3 Nc5 27. Qg4 Re6 28. Bxc5 bxc5 29. Rd2 Rf6 30. Qe2 Kh7 31. Rc1 Qg6""")

game = chess.pgn.read_game(pgn)
board = game.board()
moves = []
for move in game.mainline_moves():
    moves.append(move.uci())
print(" ".join(moves))
