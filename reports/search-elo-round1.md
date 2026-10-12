# Zaphod search changes, round 1: report

Branch: `claude/hopeful-wozniak-bakobn`. Base: `c0cfa42`, the head after the two NPS
rounds (all non-functional, bench 2,472,970 nodes).

Playing strength can't be measured here. Bench runs at a fixed depth, so its node count
shows how much each change prunes or extends. It doesn't show whether the change is
stronger. All five changes are functional and need SPRT before they are kept.

## Summary

| | bench nodes | vs base | bench time (nodes / mean NPS) | vs base |
|---|---|---|---|---|
| **Baseline** `c0cfa42` | **2,472,970** | | 1.43 s | |
| 1. Singular extensions | 4,588,043 | +85.5% | 2.58 s | +80.7% |
| 2. Futility pruning (quiets) | 2,291,356 | −7.3% | 1.41 s | −1.1% |
| 3. SEE pruning (quiets + captures) | 2,039,137 | −17.5% | 1.24 s | −12.8% |
| 4. NMP reduction by depth and eval | 2,265,087 | −8.4% | 1.35 s | −5.0% |
| 5. History pruning (quiets) | 1,992,186 | −19.4% | 1.26 s | −11.6% |
| **All five** (branch head `0c71fe8`) | **3,194,503** | **+29.2%** | 2.20 s | +54.2% |

The four pruning changes each make the depth-12 search smaller. Singular extensions
make it much larger, because the search goes deeper on lines with one clearly best move.
That is what the feature is for, and it is why a fixed-depth bench can't judge it.

## Method

- Machine: cloud VM, 4 vCPUs. Clang 20.1.2, preset `linux-clang` (RelWithDebInfo, LTO).
- Each change is one commit. For the individual tests, each change sits directly on
  `c0cfa42`. These are local branches `exp3/singular`, `exp3/futility`, `exp3/see`,
  `exp3/nmp` and `exp3/histprune` (commits `3734efe`, `8ea0449`, `2128785`, `fcb827e`
  and `cba0d31`). They were not pushed, because this session can only push the one
  branch. The combined test is the branch head, where the same five commits are stacked
  in the order of the table.
- Every binary was built in its own clean worktree. Bench ran **7 interleaved rounds**:
  each round runs every binary once, and the order is reversed every other round.
- The node count was identical in all 7 runs of every binary, so search is still
  deterministic.
- NPS is noisy on this VM. The sd of a single run was 57k–181k (3–11%), so the NPS
  column below is only a rough guide. Fewer NPS doesn't mean the code got slower:
  pruned moves still cost move ordering, SEE and history lookups, but they no longer
  count as nodes.

| | mean NPS (7 runs) | sd | vs base |
|---|---|---|---|
| Baseline | 1,735,051 | 137,024 | |
| Singular extensions | 1,781,115 | 57,660 | +2.7% |
| Futility pruning | 1,624,766 | 180,573 | −6.4% |
| SEE pruning | 1,640,601 | 177,587 | −5.4% |
| NMP | 1,672,144 | 149,113 | −3.6% |
| History pruning | 1,581,195 | 114,086 | −8.9% |
| All five | 1,453,457 | 74,207 | −16.2% |

- Build: no new warnings. The only warnings are the four existing ones in
  `board.cpp:89–91`.
- Perft is not needed. `board.*`, `movegenerator.*`, `move.h` and `zobrist.h` are
  unchanged. The one file outside `search.*`/`params.h` that changed is `see.h` (change
  3).
- Sanity checks on the combined build: it finds mate in 1 in two positions and the
  KQ-vs-K mate, and plays normally from the start position (depth 21 in 5 s).

## How the five were chosen

What the search had before this round: TT cutoffs, IIR, razoring, RFP, NMP with a
nearly fixed R, LMP, LMR with history, check extensions, aspiration windows and
correction history. What it lacked, against a typical modern alpha-beta engine:

- no extensions other than check extensions (no singular extensions);
- no per-move pruning by static eval (a futility block existed, commented out);
- no SEE pruning in the main search, only in qsearch;
- no history-based pruning (the `History pruning` heading in the move loop was empty);
- an NMP reduction that barely grew with depth and ignored how far eval was above beta.

These five are the standard features with the largest payoff in other engines, and each
is self-contained. Two others were left out:

- **Killer moves.** They would change `movegenerator.*`, which needs perft, and the
  ordering already uses butterfly, piece-to and continuation history with threat
  buckets, so killers usually add less there.
- **Removing heap allocation from the move loop.** I had this as a candidate. NPS round
  2 already did it (`b202964`).

## The changes

### 1. Singular extensions with multi-cut and negative extension (`b216348`)

When the TT entry is a lower bound or exact, at depth ≥ `seMinDepth` (7), with
`ttDepth ≥ depth − 3` and no mate score, the node is first searched without the TT move,
at depth `(depth − 1) / 2`, against `singularBeta = ttScore − depth · seBetaMargin/16`
(that is, 2 cp per ply):

- **fails low:** no other move comes close, so the TT move is extended by one ply;
- **`singularBeta ≥ beta`:** another move beats beta as well, so the node returns
  `singularBeta` (multi-cut);
- **otherwise, if `ttScore ≥ beta`:** the TT move is searched one ply shallower
  (negative extension).

The excluded-move search does not cut off on the TT, does not write the TT, does not
update correction history, and skips IIR, razoring, RFP and NMP. It runs before the
node's move generator is set up, because it uses the same per-ply move generator.
Extensions only happen at `ply < 2 × root depth`, so they can't grow without limit.

New tunables: `seMinDepth` 7, `seBetaMargin` 32.

How sensitive the node count is (singular build only, set through `setoption`):

| setting | bench nodes |
|---|---|
| default (`seMinDepth` 7, `seBetaMargin` 32) | 4,588,043 |
| `seMinDepth` 8 | 4,271,759 |
| `seMinDepth` 10 | 3,954,206 |
| `seBetaMargin` 48 | 3,938,566 |
| `seBetaMargin` 64 | 3,426,516 |

Even with the verification search limited to depth ≥ 10, nodes rise by 60%, so most of
the cost comes from the extensions themselves, not the verification searches. If the
SPRT fails, `seBetaMargin` is the first thing to tune.

### 2. Futility pruning of quiet moves (`4b02c43`)

At non-PV nodes, not in check, with depth ≤ `fpDepth` (6), once
`staticEval + fpBase + fpMultiplier · depth ≤ alpha` (100 + 100·depth), the current quiet
move is skipped, and so is every quiet after it (`skipQuiet`). This replaces the
commented-out block that was already in the move loop.

New tunables: `fpDepth` 6, `fpBase` 100, `fpMultiplier` 100.

### 3. SEE pruning of quiet and noisy moves (`6b46c12`)

At depth ≤ `seePruneDepth` (8), a move is skipped when its SEE is below
`−seeQuietMargin · depth` (−50·depth) for a quiet, or `−seeNoisyMargin · depth`
(−90·depth) for a capture. Promotions are never pruned. The SEE value comes from
`moveGen.seeOfLastMove()`, so a capture's SEE that move ordering already worked out is
reused.

The change also fixes a bug in `see.h`. For a quiet move or an en passant capture the
target square is empty, `getPieceOnSquare` returns `All` (14), and `see()` read index 14
of a 14-entry value table. An empty target is now valued at 0. On its own, applied to
`329ac51`, the fix leaves the bench node count unchanged (2,472,970), so before this
round the bad read only affected en passant ordering and had no effect on bench.

New tunables: `seePruneDepth` 8, `seeQuietMargin` 50, `seeNoisyMargin` 90.

### 4. NMP reduction grows with depth and with eval over beta (`3a9f888`)

The reduction was `3 + (depth ≥ 6) + improving`. It is now
`nmpBase + depth / nmpDepthDivisor + min((staticEval − beta) / nmpEvalDivisor, nmpEvalMax)`,
that is `3 + depth/3 + min((eval − beta)/200, 3)`. The `improving` term is gone.

New tunables: `nmpBase` 3, `nmpDepthDivisor` 3, `nmpEvalDivisor` 200, `nmpEvalMax` 3.

### 5. History pruning of quiet moves (`0c71fe8`)

At depth ≤ `hpDepth` (4), not in check, a quiet move is skipped when
`butterfly + pieceTo + cont[1 ply] + cont[2 ply]` is below `−hpMargin · depth`
(−4096·depth). Each table is capped at about ±16k. It has the same guard as the rest of
the move-loop pruning: not the root, side to move has non-pawn material, and a move has
already been searched.

New tunables: `hpDepth` 4, `hpMargin` 4096.

## Notes for the SPRT

- Each commit on the branch is one idea. To test one on its own, cherry-pick it onto
  `c0cfa42`. The only conflict is in `params.h`, where every change adds its tunables at
  the same place: keep both sides.
- No existing tunable was changed. All 14 new ones are UCI spin options, so
  `scripts/gen_tune_config.py` picks them up for an SPSA run. Their values are
  conventional starting points, not tuned.
- Suggested order for the SPRT: singular extensions, SEE pruning, history pruning,
  futility pruning, NMP. That is roughly how much each change does in other engines.
  It is not a measured ranking.
- The combined build is more than the sum of the parts in one way: with singular
  extensions on, the four pruning changes cut nodes 30% compared with singular alone
  (4,588,043 → 3,194,503).
