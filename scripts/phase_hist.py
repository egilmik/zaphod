#!/usr/bin/env python3
"""
Parse generatedata output lines like:
    10 101 329 409 425 430 554 738 ; -287
Compute a game-phase value from the (pre-';') material numbers and
plot a histogram binned by game phase.

Game phase φ is normalized to [0, 1]. By default we normalize using
dataset min/max material totals (robust when you’re unsure whether the
numbers are per-side or both sides). Optionally, use a theoretical cap
(e.g., 8000 centipawns for both sides, excluding kings) with
--normalize theoretical --theoretical-max 8000.
"""

import argparse
import math
import sys
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


LINE_RE = re.compile(r"""
    ^\s*
    (?P<mats>(?:-?\d+\s+)+)   # one or more integers + space
    ;\s*
    (?P<score>-?\d+(?:\.\d+)?) # final score (cp) or float
    \s*$
""", re.VERBOSE)


def parse_line(line: str) -> Tuple[List[int], float] | None:
    m = LINE_RE.match(line)
    if not m:
        return None
    mats_str = m.group("mats").strip()
    score = float(m.group("score"))
    mats = [int(tok) for tok in mats_str.split()]
    return mats, score


def read_lines(paths: List[str]) -> List[str]:
    lines: List[str] = []
    if not paths or paths == ["-"]:
        lines = sys.stdin.read().splitlines()
    else:
        for p in paths:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                lines.extend(f.read().splitlines())
    return lines


def compute_phase(material_totals: np.ndarray, mode: str, theoretical_max: float) -> np.ndarray:
    if mode == "dataset":
        m_min = float(material_totals.min()) if material_totals.size else 0.0
        m_max = float(material_totals.max()) if material_totals.size else 1.0
        denom = max(1e-9, m_max - m_min)
        phi = (material_totals - m_min) / denom
        return np.clip(phi, 0.0, 1.0)
    else:
        denom = max(1e-9, theoretical_max)
        phi = material_totals / denom
        return np.clip(phi, 0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Histogram positions by game phase derived from material totals.")
    ap.add_argument("inputs", nargs="*", help="Input file(s). Use '-' or no args to read stdin.")
    ap.add_argument("--bins", type=int, default=20, help="Number of histogram bins (default: 20).")
    ap.add_argument("--normalize", choices=["dataset", "theoretical"], default="dataset",
                   help="How to scale phase to [0,1]. 'dataset' uses min/max of observed material totals. "
                        "'theoretical' divides by --theoretical-max (default 8000).")
    ap.add_argument("--theoretical-max", type=float, default=8000.0,
                   help="Max material used when --normalize theoretical (default: 8000).")
    ap.add_argument("--meanscoreline", action="store_true",
                   help="Overlay mean score per phase-bin as a line on a secondary y-axis.")
    ap.add_argument("--out", type=str, default="",
                   help="Output image path (.png/.pdf). If omitted, shows an interactive window.")
    args = ap.parse_args()

    raw_lines = read_lines(args.inputs)

    materials: List[float] = []
    scores: List[float] = []
    bad = 0
    for ln in raw_lines:
        parsed = parse_line(ln)
        if not parsed:
            bad += 1
            continue
        mats, sc = parsed
        materials.append(float(sum(mats)))
        scores.append(float(sc))

    if bad:
        print(f"[warn] Skipped {bad} malformed line(s).", file=sys.stderr)

    if not materials:
        print("No valid data parsed.", file=sys.stderr)
        sys.exit(1)

    m_tot = np.array(materials, dtype=np.float64)
    sc_arr = np.array(scores, dtype=np.float64)

    phi = compute_phase(m_tot, args.normalize, args.theoretical_max)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(phi, bins=args.bins)
    ax.set_xlabel("Game phase φ (0 = endgame, 1 = opening)")
    ax.set_ylabel("Positions")
    ax.set_title("Positions by Game Phase (derived from material totals)")

    if args.meanscoreline:
        # Compute per-bin means
        bins = np.linspace(0.0, 1.0, args.bins + 1)
        idxs = np.digitize(phi, bins) - 1  # bin indices 0..bins-1
        means = np.full(args.bins, np.nan, dtype=np.float64)
        for b in range(args.bins):
            mask = idxs == b
            if np.any(mask):
                means[b] = np.nanmean(sc_arr[mask])
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
