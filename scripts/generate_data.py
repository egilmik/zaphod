import os, sys, argparse, numpy as np, chess, chess.pgn, chess.engine, hashlib, random
from pathlib import Path

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
            board.push(move)
            if not board.is_checkmate():
                try:
                    info = engine.analyse(board, limit, info=chess.engine.INFO_ALL)
                    sc = info["score"].pov(board.turn)
                    outputFile.write((board.fen() + " ; " + str(sc.score())) + "\n")
                    posCounter += 1
        #                positions.append(board,sc)
                except Exception as e:
                    print(e)
            if args.max_pos is not None and args.max_pos <= posCounter:
                break
        if args.max_pos is not None and args.max_pos <= posCounter:
                break

    outputFile.close()
    engine.close()


if __name__ == "__main__":
    main()
