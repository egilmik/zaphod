import os, sys, argparse, numpy as np, chess, chess.pgn, chess.engine, hashlib, random
from pathlib import Path
import time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True, help="PGN file (can be large)")
    ap.add_argument("--engine", required=True, help="Path to UCI engine (e.g., stockfish)")
    ap.add_argument("--output", default="output.txt", help="Output folder")
    ap.add_argument("--movetime_ms", type=int, default=50, help="per-position analysis time (ms)")
    ap.add_argument("--depth", type=int, default=None, help="fixed depth instead of time")
    ap.add_argument("--max_games", type=int, default=None, help="limit games")
    ap.add_argument("--max_pos", type=int, default=None, help="limit total positions")
    ap.add_argument("--sample_every", type=int, default=4, help="sample every k plies")
    ap.add_argument("--skip_opening", type=int, default=6, help="ignore first N plies")
    args = ap.parse_args()

    start_time = time.time()
    
    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    limit = chess.engine.Limit(time=args.movetime_ms/1000.0) if args.depth is None else chess.engine.Limit(depth=args.depth)
    pgnFile = open(args.pgn)
    
    outputFile = open(args.output, "w+")

    posCounter = 0

    while True:
        game = chess.pgn.read_game(pgnFile)
        if game is None:
            break
        board = game.board()
        for move in game.mainline_moves():
            notCapture = not board.is_capture(move)            
            board.push(move)
            if notCapture and not board.is_check() and board.ply() > 10 and not board.is_checkmate() and not board.is_stalemate():
                try:
                    info = engine.analyse(board, limit, info=chess.engine.INFO_ALL)
                    sc = info["score"].pov(board.turn)
                    if sc.score() != None:
                        outputFile.write((board.fen() + " ; " + str(sc.score())) + "\n")
                        posCounter += 1
                    else:
                        break
                except Exception as e:
                    print(e)
            if args.max_pos is not None and args.max_pos <= posCounter:
                break
        if args.max_pos is not None and args.max_pos <= posCounter:
                break

    outputFile.close()
    engine.close()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time:.4f} seconds")



if __name__ == "__main__":
    main()
