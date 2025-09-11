# fen2idx.py
# Convert:  "FEN ; score"  ->  "i1 i2 i3 ... ; score"
# Example output line: "12 128 365 345 ; -34.5"

import argparse, sys, gzip
from pathlib import Path
import chess
import numpy as np

def open_text(path: str, mode: str):
    """Handles .gz seamlessly. Use text modes 'r' or 'w'."""
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, mode + "t", encoding="utf-8", newline="\n")
    return open(p, mode, encoding="utf-8", newline="\n")

def fen_to_indices(fen: str):
    """Return sorted list of active feature indices in [0, 768)."""
    board = chess.Board(fen)  # validates FEN
    idxs = []
    # Ensure deterministic order (A1..H8): sort by square index
    for sq, pc in sorted(board.piece_map().items()):
        base = 0 if pc.color == chess.WHITE else 6
        plane = base + (pc.piece_type - 1)  # PAWN..KING => 0..5 (+6 for black)
        idxs.append(plane * 64 + sq)        # A1=0..H8=63 (python-chess)
    return idxs

def process(in_path: str, out_path: str, skip_bad: bool):
    n_in = n_out = n_bad = 0
    with open_text(in_path, "r") as fin, open_text(out_path, "w") as fout:
        for line in fin:
            s = line.strip()
            if not s or s.startswith("#") or ";" not in s:
                continue
            n_in += 1
            fen_part, score_part = s.split(";", 1)
            fen   = fen_part.strip()
            score = score_part.strip()
            try:
                idxs = fen_to_indices(fen)
            except Exception as e:
                n_bad += 1
                if skip_bad:
                    continue
                raise
            fout.write("{} ; {}\n".format(" ".join(map(str, idxs)), score))
            n_out += 1
    print(f"read={n_in} wrote={n_out} bad={n_bad}", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  "-i", required=True, help="Input file (text or .gz) with 'FEN ; score'")
    ap.add_argument("--output", "-o", required=True, help="Output file (text or .gz)")
    ap.add_argument("--skip-bad", action="store_true", help="Skip invalid FEN lines instead of failing")
    args = ap.parse_args()
    process(args.input, args.output, args.skip_bad)

if __name__ == "__main__":
    main()
