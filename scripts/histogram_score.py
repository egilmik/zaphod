#!/usr/bin/env python3
"""
Parse generatedata lines like:
    10 101 329 409 425 430 554 738 ; -287

Each number before ';' is an NNUE input feature index in [0..767].
We decode piece type from index//64 (mod 6), map to material values
(P=100, N=320, B=330, R=500, Q=900, K=0), sum across all features in
the line to get total material (both sides, kings contribute 0), then
compute a normalized game phase φ ∈ [0,1].

φ normalization:
- dataset:   φ = (M - min(M)) / (max(M) - min(M))
- theoretical: φ = M / THEO_MAX, with THEO_MAX default 8000 cp
               (both sides, startpos material excluding kings).

Optionally overlay mean score per phase-bin.
"""

import argparse
import sys
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# Regex: "<ints...> ; <score>"
LINE_RE = re.compile(r"""
    ^\s*
    (?P<feat_str>(?:-?\d+\s+)+)   # one or more integers + space
    ;\s*
    (?P<score>-?\d+(?:\.\d+)?)    # final score (cp) or float
    \s*$
""", re.VERBOSE)


# Piece values in centipawns indexed by (plane % 6): P,N,B,R,Q,K
PIECE_VALUES = np.array([0, 1, 1, 2, 4, 0], dtype=np.float64)

INPUT_DIM = 12 * 64  # 768


def parse_line(line: str) -> float | None:
    m = LINE_RE.match(line)
    if not m:
        return None
    feats = [int(tok) for tok in m.group("feat_str").split()]
    score = float(m.group("score"))
    return  score


def read_lines(paths: List[str]) -> List[str]:
    if not paths or paths == ["-"]:
        return sys.stdin.read().splitlines()
    lines: List[str] = []
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            lines.extend(f.read().splitlines())
    return lines


def features_to_material(features: List[int]) -> float:
    # Deduplicate to be robust if upstream ever emits duplicates.
    uniq = set(features)
    total = 0.0
    for f in uniq:
        if 0 <= f < INPUT_DIM:
            plane = f // 64  # 0..11
            kind = plane % 6  # collapse color: 0=P,1=N,2=B,3=R,4=Q,5=K
            total += float(PIECE_VALUES[kind])
        # else: silently ignore out-of-range indices
    return total


def compute_phase(material: np.ndarray, mode: str, theoretical_max: float) -> np.ndarray:
    if mode == "dataset":
        lo = float(material.min()) if material.size else 0.0
        hi = float(material.max()) if material.size else 1.0
        denom = max(1e-9, hi - lo)
        return np.clip((material - lo) / denom, 0.0, 1.0)
    else:
        denom = max(1e-9, theoretical_max)
        return np.clip(material / denom, 0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Histogram positions by game phase derived from NNUE features' material.")
    ap.add_argument("inputs", nargs="*", help="Input file(s). Use '-' or no args to read stdin.")
    ap.add_argument("--bins", type=int, default=20, help="Histogram bins (default: 20).")
    ap.add_argument("--normalize", choices=["dataset", "theoretical"], default="theoretical",
                    help="Phase normalization: 'dataset' uses observed min/max; 'theoretical' uses --theoretical-max.")
    ap.add_argument("--theoretical-max", type=float, default=8000.0,
                    help="Max material for theoretical normalization (default: 8000 cp).")
    ap.add_argument("--mean-score-line", action="store_true",
                    help="Overlay mean score per phase-bin on a secondary y-axis.")
    ap.add_argument("--out", type=str, default="",
                    help="Output image (.png/.pdf). If omitted, shows window.")
    args = ap.parse_args()

    raw = read_lines(args.inputs)

    mats: List[float] = []
    scores: List[float] = []
    bad = 0
    for ln in raw:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parsed = parse_line(ln)
        if not parsed:
            bad += 1
            continue
        score = parsed
        scores.append(float(score))

    if bad:
        print(f"[warn] Skipped {bad} malformed line(s).", file=sys.stderr)
    if not mats:
        print("No valid data parsed.", file=sys.stderr)
        sys.exit(1)

    S = np.array(scores, dtype=np.float64)

    phi = compute_phase(S, args.normalize, args.theoretical_max)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(S, bins=args.bins)
    ax.set_xlabel("Game phase φ (0 = endgame, 1 = opening)")
    ax.set_ylabel("Positions")
    ax.set_title("Positions by Game Phase (from NNUE features' material)")

    if args.mean_score_line:
        # per-bin average of S
        bins = np.linspace(0.0, 1.0, args.bins + 1)
        idx = np.digitize(phi, bins) - 1  # 0..bins-1
        means = np.full(args.bins, np.nan, dtype=np.float64)
        for b in range(args.bins):
            mask = idx == b
            if np.any(mask):
                means[b] = np.nanmean(S[mask])
        centers = 0.5 * (bins[:-1] + bins[1:])
        ax2 = ax.twinx()
        ax2.plot(centers, means, marker='o', linestyle='-')
        ax2.set_ylabel("Mean score (cp)")

    fig.tight_layout()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"Saved histogram to: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
