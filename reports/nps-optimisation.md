# Zaphod NPS optimisation: report

Branch: `claude/hopeful-wozniak-bakobn`, base commit `329ac51`.

## Summary

| | bench nodes | bench NPS (mean of 10) | vs base |
|---|---|---|---|
| Baseline (`329ac51`) | 2,472,970 | **437,694** | |
| All 5 changes | 2,472,970 | **951,424** | **+117%** |

All five changes are non-functional. Every one of the 15 binaries built for this
report produced the same bench node count, 2,472,970, on every run.

Most of that gain is not search speed. Bench clears the 256 MB transposition table
before each of its 35 positions, and those clears took about 2.2 s of the 5.6 s bench.
Change 1 removes them. Measured against a build that only has change 1, so the clear
is out of the picture, the other four changes make the search itself **~38% faster in
NPS** (691,554 → 951,424; the time per node drops by about 27%).

## Method

- Machine: cloud VM, Intel Xeon @ 2.80 GHz, 4 vCPUs, 33 MB L3. Clang 20.1.2,
  preset `linux-clang` (RelWithDebInfo, `-O2 -march=x86-64-v3`).
- Profile: callgrind over the full bench of the baseline, plus a separate timing of
  35 × 256 MB `memset`.
- Each change is a single commit. For the individual tests each change sits directly on
  the base commit (`exp/c1`…`exp/c5`). Changes 2–5 were also each applied on their own
  on top of change 1 (see below for why). The combined test is the branch head.
- Every binary was built in its own clean worktree. All binaries were benched
  **interleaved, 10 rounds**: each round runs every binary once, and the order is
  reversed every other round so that drift on the VM spreads evenly. NPS is the
  `<nodes> <nps>` line bench prints at the end.
- The VM is noisy: the standard deviation of a single run is 4–7% (up to 11% for the
  fastest builds). With 10 runs the standard error of a mean is about 1.5–3.5%.
  Differences smaller than about 3% are not reliable.

## Results

### Each change on its own, against the base

| Change | nodes | mean NPS | median | stdev | vs base |
|---|---|---|---|---|---|
| Baseline | 2,472,970 | 437,694 | 442,714 | 30,961 | |
| 1. O(1) TT clear | 2,472,970 | 691,554 | 700,977 | 48,408 | **+58.0%** |
| 2. NNUE accumulator stack | 2,472,970 | 456,954 | 461,806 | 18,697 | +4.4% |
| 3. TT huge pages + prefetch | 2,472,970 | 475,855 | 472,124 | 23,939 | +8.7% |
| 4. No heap allocation in search | 2,472,970 | 458,266 | 460,773 | 16,899 | +4.7% |
| 5. LTO | 2,472,970 | 448,356 | 447,883 | 19,070 | +2.4% |
| **All five** | 2,472,970 | **951,424** | 970,258 | 108,275 | **+117.4%** |

On the base, about 40% of the bench wall time goes to clearing the TT. That time is
bound by memory bandwidth and is noisy, and changes 2–5 do not touch it. So against the
base their gains are diluted, and they are also harder to tell apart from the noise.
In an earlier 8-round run, changes 2, 4 and 5 measured +0.1%, +0.3% and −0.9% against
the base; in a 6-round rerun they measured +4.5%, +1.1% and +2.4%. The table below is
the more reliable way to see what they do.

### Changes 2–5, each on its own on top of change 1 (search speed)

| Variant | mean NPS | stdev | vs change 1 alone |
|---|---|---|---|
| Change 1 alone | 691,554 | 48,408 | |
| 1 + 2. NNUE accumulator stack | 748,339 | 60,541 | **+8.2%** |
| 1 + 3. TT huge pages + prefetch | 777,882 | 60,271 | **+12.5%** |
| 1 + 4. No heap allocation in search | 745,501 | 54,141 | **+7.8%** |
| 1 + 5. LTO | 722,082 | 63,750 | +4.4% |
| All five | 951,424 | 108,275 | **+37.6%** |

### Cumulative (earlier 8-round run, same method)

| Commits | mean NPS | vs base | step |
|---|---|---|---|
| Baseline | 439,684 | | |
| 1 | 679,964 | +54.6% | +54.6% |
| 1–2 | 751,340 | +70.9% | +10.5% |
| 1–3 | 847,048 | +92.6% | +12.7% |
| 1–4 | 914,017 | +107.9% | +7.9% |
| 1–5 (all) | 985,355 | +124.1% | +7.8% |

The gains add up: the steps in the cumulative run match the "on top of change 1" numbers
closely, so the changes are largely independent.

## The changes

### 1. O(1) transposition table clear (`src/ttable.h`, `src/ttable.cpp`)

**Finding:** 46% of all instructions in the bench profile were `memset`, from `tt.clear()`
in `Search::setNewGame()`. Bench calls it before each of its 35 positions. Timed on its
own, 35 clears of the 256 MB table take 2.2 s.

**Change:** `clear()` now increments a 16-bit generation counter instead of wiping the
table. Each entry stores the generation it was written in, using 2 bytes of padding the
entry already had (it stays 16 bytes, which a `static_assert` now checks). `probe`
ignores entries from an older generation, and `put` treats them as empty slots. The
table is only really wiped with `memset` when the generation wraps, once every 65,536
clears.

**Why the node count cannot change:** in a cleared table the valid entries of a bucket
always form a prefix (`put` fills the first empty slot), and a stale entry behaves
exactly like a zeroed one in both `probe` and `put`.

**Caveat:** this is mostly a bench gain. In real play the TT is cleared once per game
(`ucinewgame`), which now takes no time instead of about 60 ms. It does not make the
search faster.

### 2. NNUE accumulator stack with fused updates (`src/nnueq.*`, `src/board.cpp`)

**Finding:** `NNUEQ::addPiece` and `removePiece` were 16% of the instructions. Every
added or removed piece was a separate read-modify-write pass over both 512-wide
accumulators. `revertLastMove` then ran the same passes again in reverse.

**Change:** there is one accumulator per ply, on a stack of 1,025. `Board::addPiece`
and `removePiece` only queue the feature change. At the end of `makeMove`,
`nnue.push()` builds the child accumulator from the parent in **one pass** per
perspective. There are kernels specialised at compile time for the common cases:
1 add + 1 sub (quiet move), 1 + 2 (capture) and 2 + 2 (castling). A promotion pawn that
is added and removed on the same square cancels out. `revertLastMove` is now just
`nnue.pop()`, and `parseFen` builds the accumulator from scratch with `nnue.refresh()`.

**Arithmetic:** the updates wrap (`_mm256_add_epi16`) instead of saturating
(`_mm256_adds_epi16`). That is what allows the changes to be applied in any order and
fused. It is bit-identical to the old code as long as no accumulator can leave the int16
range. For the current network: max |bias| = 199 and max |weight| = 487, so
199 + 32·487 = 15,783 < 32,767 (the tighter per-neuron bound is 10,457). The bound is
checked whenever a network is loaded, with a warning if a future network exceeds it.

A first version kept saturation and looped over the queued changes at runtime. It was
**4.5% slower** than the baseline, so the specialised wrapping kernels are what make
this change pay off.

### 3. TT on huge pages + prefetch (`src/ttable.h`, `src/search.cpp`)

**Finding:** the 256 MB table was allocated with `new[]`. Transparent huge pages on this
system are in `madvise` mode, so the table used 4 KB pages, and nearly every probe
missed the TLB as well as the cache. Callgrind cannot see this, since it counts
instructions and not memory stalls.

**Change:** the table is allocated 2 MB-aligned (`aligned_alloc` on POSIX,
`_aligned_malloc` on Windows) and marked `madvise(MADV_HUGEPAGE)` on Linux. While bench
runs, `AnonHugePages` for the process goes from 0 kB to about 208 MB. In addition, the
bucket of the new position is prefetched (`_mm_prefetch`) right after `makeMove` /
`makeNullMove`, so the cache miss overlaps with the rest of the node setup before the
probe.

This was the largest search-side gain (+12.5% on top of change 1). Part of its +8.7%
against the base comes from the `memset` of the clear also running faster on huge pages.

### 4. No heap allocation in the search (`src/movegenerator.cpp`, `src/search.cpp`)

**Finding:** `std::stable_sort` allocates a temporary buffer on every call. It sorted the
noisy and the quiet moves of most nodes: about 1.3 million buffer allocations during
bench. On top of that, the `failLowQuiet` / `failLowNoisy` lists in `negamax` were
`std::vector`s, which allocated on their first `push_back` in every node where a move
failed low.

**Change:** the sort is now an insertion sort using the same comparator. The comparator
is a strict total order (score, then move value), so every correct sort gives exactly the
same order. The fail-low lists are fixed-size `uint16_t[256]` arrays that are not
initialised.

### 5. Link-time optimisation (`CMakeLists.txt`)

**Change:** IPO/LTO is on for Release, RelWithDebInfo and MinSizeRel when
`CheckIPOSupported` accepts the toolchain. It stays off for Debug and sanitizer builds,
and `-DZAPHOD_LTO=OFF` turns it off. The board, NNUE, search and TT code call each other
on every node across source files, and without LTO none of those calls can be inlined.

This is the smallest and least certain gain (+2.4% against the base and +4.4% on top of
change 1, both within about 1.5 standard errors).

## Verification

- **Node count:** 2,472,970 for the baseline, for each change on its own, for each change
  on top of change 1, for every cumulative step and for the final branch, on every bench
  run (about 300 in total).
- **Build:** `linux-clang` (Clang 20) builds with no new warnings. The only warnings are
  the existing `-Wimplicitly-unsigned-literal` ones in `board.cpp`.
- **`scripts/bench_history.py -n 6 --runs 5 --preset linux-clang`** over the branch: see
  the appendix.
- **Not done:**
  - Perft. Changes 2 and 4 touch `board.cpp` (calls to `nnue.push/pop/refresh` only; no
    bitboard, hash or make/unmake logic changed) and `movegenerator.cpp` (only the sort).
    Perft has not been run yet.
  - The ASan/UBSan build. It needs GCC 15, which this session doesn't have.
  - GCC, Windows and macOS builds. The Windows `_aligned_malloc` path and LTO with GCC
    are written but untested.
  - Playing strength. It should be unaffected, since the search is identical node for
    node, but it was not measured, and no Elo gain is claimed.

## Further candidates (not done)

From the same profile, ranked by expected gain:

1. **Lazy accumulator updates:** queue the changes per ply and only apply them when a node
   is actually evaluated. Nodes with a static eval from the TT, or in check, would skip
   the update entirely. Builds on change 2.
2. **SEE in noisy move ordering** (~4% inclusive): it is computed for every capture,
   even when only the sign matters.
3. **`revertLastMove` work:** it still calls `toggleHashKeys` for every piece it moves
   back, then overwrites those keys from `keyHistory`.
4. **`hasPositionRepeated`** scans the whole history and checks every ply. It could step
   by 2 and stop at the last irreversible move. That needs care with null moves, which
   don't reset the halfmove clock, to keep the node count identical.
5. **`std::log` in LMR**, called twice per reduced move: a precomputed table gives the
   same integers.
6. **`calculateThreats` and `computeKingDanger`** compute nearly the same attack set,
   one in `makeMove` and one in `MoveGenerator::init`.

## Appendix: `scripts/bench_history.py`

The repository's own tool, 5 runs per commit, run sequentially rather than interleaved.
That makes it more exposed to drift on the VM than the tables above, but it shows the
same picture: the node count is identical at every commit, and NPS goes from 474k to
1,025k (+116%).

```
commit   date               nodes     avg nps     min nps     max nps  spread  vs older  subject
------------------------------------------------------------------------------------------------
af6ea3c  2026-10-09       2472970     1024590      961496     1055922    9.2%    +10.7%  Build: link time optimisation for optimised builds
b202964  2026-10-09       2472970      925824      871992      969032   10.5%     +3.4%  Search: no heap allocation in move ordering and negamax
ad54eb2  2026-10-09       2472970      895739      872607      924129    5.8%     +9.9%  TT: back the table with huge pages and prefetch buckets
64c00e8  2026-10-09       2472970      815242      755566      879121   15.2%     +8.8%  NNUE: accumulator stack with fused updates
891aef3  2026-10-09       2472970      749139      708790      787570   10.5%    +58.0%  TT: O(1) clear with a generation counter
329ac51  2026-10-09       2472970      474017      460087      487092    5.7%         -  CLAUDE.md: list files and values agents must not change

Fastest: af6ea3c at 1024590 nps  (Build: link time optimisation for optimised builds)
Slowest: 329ac51 at 474017 nps  (CLAUDE.md: list files and values agents must not change)
```
