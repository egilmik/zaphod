# Zaphod NPS optimisation, round 2: report

Branch: `claude/hopeful-wozniak-bakobn`. Base for this round: `a538ffc`, the round 1
changes (SPRT: +60 Elo).

## Summary

| | bench nodes | bench NPS (mean of 20) | search instructions | vs base |
|---|---|---|---|---|
| Round 2 base (`a538ffc`) | 2,472,970 | **940,167** | 8.697 G | |
| All 5 changes | 2,472,970 | **1,035,797** | 7.468 G | **+10.2% NPS, −14.1% instructions** |

A separate run of 30 interleaved rounds, base against all five only, gave +8.5%
(959,821 → 1,041,246). The standard error of that difference is about 1.8%. Over three
independent runs the combined gain is **+8.5% to +10.2%, about +9% NPS**.

All five changes are non-functional. Every binary built for this round, on every run,
produced the same bench node count, 2,472,970.

Round 1 was dominated by one large item: clearing the TT. With that gone, the profile is
flat. The five changes below each save 1–7% of the instructions the search executes,
and together they save 14%. Measured in time, that comes out at +10% NPS.

## Method

- Machine and compiler as in round 1: cloud VM, Intel Xeon @ 2.80 GHz, 4 vCPUs. Clang
  20.1.2, preset `linux-clang` (RelWithDebInfo, `-O2`, LTO).
- Profile: callgrind over the full bench of the round 2 base.
- Each change is one commit, and for the individual tests each sits directly on
  `a538ffc` (local branches `exp2/r1`…`exp2/r5`). The combined test is the branch head,
  where the five commits are stacked in order.
- Two measurements per binary:
  - **Search instructions (callgrind):** instructions executed inside
    `Search::search` during one full bench. This number is deterministic, so even 1%
    differences are real. It counts work, not time: cache misses and the like don't
    show up in it.
  - **Bench NPS:** 20 rounds, interleaved; every round runs every binary once, and the
    order is reversed every other round. Every binary was built in its own clean
    worktree.
- **The VM was noisier than in round 1.** Single runs varied by about 7% (sd), so even
  with 20 runs the standard error of a difference between two variants is about 2.5%.
  The individual NPS gains (2–5%) are mostly within that noise. The instruction counts
  are the reliable per-change measure, and the combined NPS gain is clearly outside the
  noise. The process's user CPU time (which excludes kernel and VM steal time) tracked
  the wall-clock NPS within 0.6 points, so the noise is real contention on the host, not
  steal time.

## Results

| Change | nodes | search instructions | Δ instructions | mean NPS (20 runs) | sd | Δ NPS |
|---|---|---|---|---|---|---|
| Base `a538ffc` | 2,472,970 | 8.697 G | | 940,167 | 61,794 | |
| 1. Leaner NNUE accumulator update | 2,472,970 | 8.122 G | **−6.6%** | 985,315 | 55,063 | +4.8% |
| 2. Pick moves on demand instead of sorting | 2,472,970 | 8.486 G | −2.4% | 961,653 | 72,963 | +2.3% |
| 3. Pins, snipers and threats computed on first use | 2,472,970 | 8.544 G | −1.8% | 980,137 | 92,424 | +4.3% |
| 4. Lighter `revertLastMove` | 2,472,970 | 8.505 G | −2.2% | 985,236 | 79,824 | +4.8% |
| 5. qsearch reuses the move-ordering SEE | 2,472,970 | 8.590 G | −1.2% | 961,798 | 72,287 | +2.3% |
| **All five** | 2,472,970 | **7.468 G** | **−14.1%** | **1,035,797** | 72,718 | **+10.2%** |

The instruction savings add up almost exactly: the individual ones sum to 14.2%, and
together they save 14.1%. The individual NPS gains sum to more than the combined +10%.
That is expected when each one is about the size of the noise, so I wouldn't read the
NPS ranking of changes 1–5 as exact. Rank them by instructions.

Confirmation runs of the combined change:

| Run | rounds | base NPS | all five NPS | Δ |
|---|---|---|---|---|
| 1 (all 7 binaries) | 12 | 953,324 | 1,042,125 | +9.3% |
| 2 (all 7 binaries, the table above) | 20 | 940,167 | 1,035,797 | +10.2% |
| 3 (base and all five only) | 30 | 959,821 | 1,041,246 | +8.5% |

In run 1 the individual changes measured between −1.8% and +4.8%, which again shows
that they are below the noise.

## The changes

### 1. Leaner NNUE accumulator update (`src/nnueq.cpp`)

**Finding:** `NNUEQ::applyPending` (the fused update from round 1) was 20.6% of all
instructions, and only about half of that was the SIMD loop. The rest was
bookkeeping:
- a `switch` mapping each piece to its feature plane;
- `encodeFeature` for each change and perspective;
- copying the queued changes into a second list while looking for pairs that cancel;
- separate passes over the white and black accumulators.

**Change:**
- A table `featureOffsets[piece][square][perspective]` holds the weight row offsets.
  It is built once at start-up.
- The search for cancelling pairs (only a promotion has one) only runs when four or
  more changes are queued: castling and promotions. Quiet moves and captures skip it.
- Both perspectives are updated in the same loop over the accumulator, so each chunk
  loads, updates and stores white and black together.

Lazy accumulator updates (skip nodes that never evaluate) looked promising in round 1.
The profile rules them out: 2.19 M of the 2.47 M accumulators built during bench are
evaluated, so at most ~11% of the update work could be skipped.

### 2. Pick moves on demand instead of sorting (`src/movegenerator.cpp`)

**Finding:** the insertion sort from round 1 cost ~3.7% of the instructions. Both move
lists were sorted in full as soon as they were generated, but most nodes cut off after
the first few moves, and late move pruning skips most of the quiet moves.

**Change:** `next()` now selects the best remaining move each time (`pickBest`). The
order (score, then move value) is a strict total order, so the moves come out in exactly
the same sequence as before. `sortNoisyMoves` is renamed `scoreNoisyMoves`, since it no
longer sorts.

### 3. Pins, snipers and threats computed on first use (`src/board.*`, `src/movegenerator.*`)

**Finding:** every `makeMove` computed:
- the checkers;
- the snipers and pins (sliders lined up with the king, and the pieces they pin);
- every square the opponent attacks.

Together that was ~5% of the instructions. Only the checkers are needed in every node.
The rest is only used to generate and order moves, and many nodes return before that:
TT cutoffs, stand pat in qsearch, repetitions.

**Change:**
- `getPins()`, `getSnipers()` and `getThreats()` now compute their value on first use in
  a position. Whether each value is valid is saved and restored with the rest of the
  undo information.
- The move generator reads pins and snipers once in `init()`, and the quiet scoring
  reads the threats once.

During bench, threats are now computed 0.44 M times instead of 2.47 M, and pins/snipers
1.29 M times instead of 2.47 M.

My first version saved only 0.2% of the instructions. The move generator called the
getters once per piece and twice per quiet move, and the check-and-branch in each call
ate most of the saving. Reading the values once per node brought it to −1.8%.

### 4. Lighter `revertLastMove` (`src/board.*`)

**Finding:** `revertLastMove` moved the pieces back with `addPiece` / `removePiece`.
Those also toggle the non-pawn, minor and major piece hash keys and queue NNUE changes,
and all of that is thrown away right after: the keys are restored from `keyHistory`,
and the accumulator is popped.

**Change:** `revertLastMove` uses new board-only helpers, `placePiece` and
`clearPiece`.

### 5. qsearch reuses the move-ordering SEE (`src/movegenerator.*`, `src/search.cpp`)

**Finding:** every capture in the noisy list gets a static exchange evaluation (SEE) to
order it. qsearch's futility pruning then ran SEE again, for the same capture in the
same position: 1.32 M extra calls during bench.

**Change:** `ScoredMove` keeps the SEE value, and `MoveGenerator::seeOfLastMove()` hands
it back. The TT move, and any move that wasn't scored, is still evaluated on the spot.
The result is the same because SEE only depends on the position, and the board is fully
restored before `next()` runs again.

## Tried and dropped

| Idea | Result | Why it was dropped |
|---|---|---|
| PGO (clang, bench as the training run) | +0.6% NPS (5 runs) | Within noise, and adds a two-stage build. |
| `-O3` instead of `-O2` | +1.9% NPS (5 runs) | Within noise. |
| Lazy NNUE accumulator updates | — | 89% of accumulators are evaluated anyway (see change 1). |

## Verification

- **Node count:** 2,472,970 for the base, each change on its own and the combined branch,
  on every bench run, about 340 in total this round, plus the callgrind runs.
- **Build:** `linux-clang` builds with no new warnings. The only warnings are the existing
  `-Wimplicitly-unsigned-literal` ones in `board.cpp`.
- **`scripts/bench_history.py -n 6 --runs 5`:** see the appendix.
- **Not done:**
  - **Perft.** Changes 3 and 4 touch `board.*` and change 3 touches the pins and snipers
    the move generator uses for legality, so perft should be run before this is merged.
    It hasn't been run yet.
  - The ASan/UBSan build. It needs GCC 15, which this session doesn't have.
  - GCC and Windows builds.
  - Playing strength. It should be unaffected, since the search is identical node for
    node, but it was not measured.

## What's left

After this round, the largest items in the profile:
- the NNUE accumulator update, now ~14%;
- the NNUE output layer `forward()`, ~12%: one SCReLU pass over 2×512 values per
  evaluation. A different activation layout or an unrolled loop might take a few percent
  off, but the work itself is fixed by the network;
- move generation and scoring, ~12%;
- SEE in move ordering, ~6%;
- the history lookups.

No single item stands out any more. Further NPS gains will come in steps of 1–3% each.

## Appendix: `scripts/bench_history.py`

The repository's own tool, 5 runs per commit, run one commit after another rather than
interleaved. This time the spread *within* a single commit was 15–21%, so this table
can't resolve 2–10% effects. It confirms the node counts, but its NPS column is noise
(it puts the head level with the base). The interleaved runs above are the NPS result.

```
commit   date               nodes     avg nps     min nps     max nps  spread  vs older  subject
------------------------------------------------------------------------------------------------
6c18da0  2026-10-09       2472970      997521      969032     1043447    7.5%     -0.7%  qsearch: reuse the SEE computed for move ordering
8df6544  2026-10-09       2472970     1004479      906181     1076140   16.9%     -0.8%  Board: lighter revertLastMove
3e8999a  2026-10-09       2472970     1012622      914559     1086064   16.9%     +3.6%  Board: compute pins, snipers and threats on first use
803c5fd  2026-10-09       2472970      977903      864372     1046095   18.6%     -5.0%  Move picker: pick the best move on demand instead of sorting
863dbb3  2026-10-09       2472970     1029671      936376     1151289   20.9%     +3.3%  NNUE: leaner accumulator update
a538ffc  2026-10-09       2472970      996329      911862     1066394   15.5%         -  Add NPS optimisation report

Fastest: 863dbb3 at 1029671 nps  (NNUE: leaner accumulator update)
Slowest: 803c5fd at 977903 nps  (Move picker: pick the best move on demand instead of sorting)
```
