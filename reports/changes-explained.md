# How the NPS changes work: step by step

This document walks through all ten speed changes on `claude/hopeful-wozniak-bakobn`:
five from round 1 (base `329ac51`) and five from round 2 (base `a538ffc`). For each one
it covers:

- the **problem** that the profile showed;
- the **idea** behind the fix;
- the **implementation**, step by step, with the functions involved;
- **why the search is unchanged**: none of these changes is allowed to move the bench
  node count (2,472,970), so each section argues why it can't.

The measured gains are in `nps-optimisation.md` (round 1) and
`nps-optimisation-round2.md` (round 2).

| # | Commit | Change | Files |
|---|---|---|---|
| R1.1 | `891aef3` | O(1) transposition table clear | `ttable.h`, `ttable.cpp` |
| R1.2 | `64c00e8` | NNUE accumulator stack with fused updates | `nnueq.*`, `board.cpp` |
| R1.3 | `ad54eb2` | TT on huge pages, plus prefetch | `ttable.h`, `search.cpp` |
| R1.4 | `b202964` | No heap allocation in the search | `movegenerator.cpp`, `search.cpp` |
| R1.5 | `af6ea3c` | Link-time optimisation | `CMakeLists.txt` |
| R2.1 | `863dbb3` | Leaner NNUE accumulator update | `nnueq.cpp` |
| R2.2 | `803c5fd` | Pick the best move on demand instead of sorting | `movegenerator.*` |
| R2.3 | `3e8999a` | Pins, snipers and threats computed on first use | `board.*`, `movegenerator.*` |
| R2.4 | `8df6544` | Lighter `revertLastMove` | `board.*` |
| R2.5 | `6c18da0` | qsearch reuses the move-ordering SEE | `movegenerator.*`, `search.cpp` |

---

## R1.1 O(1) transposition table clear

**Problem.** `Search::setNewGame()` calls `tt.clear()`, which wiped the whole 256 MB
table with `memset`. Bench calls `setNewGame()` before each of its 35 positions, so it
spent about 2.2 s of a 5.6 s run writing zeros. In real games this happens once per
`ucinewgame`.

**Idea.** Don't erase anything. Give the table a *generation* number and stamp each entry
with the generation it was written in. Clearing then just means "start a new
generation", and any entry from an older generation is treated as empty.

**Implementation.**

1. **Store a generation in every entry.** `TTable::InternalEntry` gets a field
   `uint16_t generation`. The entry already had 2 bytes of padding at the end, so it
   stays 16 bytes; a `static_assert(sizeof(InternalEntry) == 16)` guards that.
2. **Track the current generation.** `TTable` gets a member `uint16_t generation`.
   `setSize()` resets it to 0 when it allocates a fresh, zeroed table.
3. **Make `clear()` cheap.** It now increments `generation` and resets `tableAge`.
   Only when the 16-bit counter wraps around to 0 (once every 65,536 clears) does it
   still `memset` the table. That way an entry left over from exactly 65,536 clears ago
   can never look current.
4. **`probe()` skips old entries.** An entry only matches if
   `internal.generation == generation` **and** its key matches.
5. **`put()` treats old entries as empty.** While it scans the bucket, any entry from an
   older generation is first reset to a zeroed entry stamped with the current
   generation. From then on the existing logic sees it as an empty slot (`NO_TYPE`)
   and uses it.

**Why the search is unchanged.** After a real `memset`, every slot in a bucket is
zero. `put()` always fills the first empty slot, so the valid entries in a bucket
always form a prefix. With generations, a stale entry behaves exactly like a zeroed one:
`probe()` never returns it, and `put()` empties it before use. So every probe and every
replacement decision comes out the same as after a real wipe.

---

## R1.2 NNUE accumulator stack with fused updates

**Problem.** The accumulator (the 2 × 512 first-layer sums, one per perspective) was
updated in place. Every `Board::addPiece` / `removePiece` made a separate
read-modify-write pass over both perspectives. A quiet move cost 2 passes, a capture 3,
and `revertLastMove` did all of them again in reverse to restore the old values.
Together these were ~16% of all instructions.

**Idea.**

- Keep one accumulator per ply on a stack. Making a move computes the child from the
  parent in a single pass that applies all of the move's piece changes at once.
  Unmaking a move just steps back to the parent, which was never modified.
- Use plain wrapping 16-bit adds instead of saturating ones, so the changes can be
  applied in any order and combined.

**Implementation.**

1. **The stack** (`nnueq.h`). `struct Accumulator { int16_t pre[2][H]; }` holds both
   perspectives (`[0]` white, `[1]` black). `accStack` is an array of
   `ACC_STACK_SIZE = 1025` of them (the game-history limit of 1024 plus the root), and
   `top` is the index of the current one.
2. **Queue instead of compute.** `NNUEQ::addPiece` / `removePiece` no longer touch the
   accumulator. They append `{piece, square, add/remove}` to the `pending[64]` list via
   `queue()`. `Board::addPiece` / `removePiece` still call them exactly as before, so
   `makeMove` itself didn't change.
3. **`push()` at the end of `makeMove`.** `Board::makeMove` calls `nnue.push()` after the
   pieces have moved. `push()` calls `applyPending(accStack[top], accStack[top + 1])`
   and increments `top`.
4. **`pop()` in `revertLastMove`.** `Board::revertLastMove` calls `nnue.pop()`, which
   decrements `top` and drops anything queued while the move was taken back.
5. **`refresh()` for a new position.** `Board::parseFen` calls `nnue.refresh()` after
   placing the pieces. It resets the stack to the bias and applies the up to 32 queued
   piece additions. Null moves don't touch the stack, since the pieces don't move.
6. **The fused kernel.** `applyPending` turns the queued changes into lists of weight
   rows to add and to subtract. It then calls a kernel specialised at compile time for
   the common shapes:
   - `updateRows<1,1>`: quiet move;
   - `updateRows<1,2>`: capture;
   - `updateRows<2,2>`: castling.

   Each kernel walks the accumulator in 16-lane AVX2 chunks: it loads the parent chunk,
   applies the adds and subs, and stores the result into the child. Rarer shapes
   (promotions, FEN setup) copy the parent and then apply one row at a time.
7. **Promotions.** The pawn added on the promotion square and removed again in the same
   move cancels out, so that pair is dropped from the list before the kernel runs.
8. **Wrapping arithmetic.** The kernels use `_mm256_add_epi16` / `_mm256_sub_epi16`
   (wrapping) instead of the old `adds` / `subs` (saturating).

**Why the result is unchanged.** Wrapping and saturating arithmetic give the same result
as long as no value ever leaves the int16 range. For this network, max |bias| is 199
and max |weight| is 487. A legal position has at most 32 pieces, so no accumulator value
can exceed 199 + 32 × 487 = 15,783, well below 32,767. `checkAccumulatorRange()`
recomputes this bound whenever a network is loaded, and prints a warning if a future
net could overflow.

Within that range, addition is exact and order-independent, so the fused result matches
the old step-by-step result bit for bit. Unmaking now restores the parent's values
exactly instead of recomputing them.

A first version kept saturation and looped over the queued changes at run time; it was
4.5% slower than the original. The specialised wrapping kernels are what make this
change faster.

---

## R1.3 TT on huge pages, plus prefetch

**Problem.**
- **TLB misses:** the 256 MB table used ordinary 4 KB pages, because transparent huge
  pages on this system are `madvise`-only. Every probe lands somewhere random in the
  table, so nearly every probe also missed the TLB, the CPU's cache of page
  translations.
- **Cache misses:** the bucket itself was usually not in cache when the probe ran.

**Idea.**
- Back the table with 2 MB pages, so the TLB covers far more of it.
- Start loading the bucket as soon as the new position's hash key is known, so the cache
  miss overlaps with other work.

**Implementation.**

1. **Aligned allocation.** `TTable::allocateBuckets(count)` rounds the size up to a
   multiple of 2 MB and allocates it 2 MB-aligned: `std::aligned_alloc` on POSIX,
   `_aligned_malloc` on Windows.
2. **Request huge pages.** On Linux it calls `madvise(mem, bytes, MADV_HUGEPAGE)`.
   While bench runs, `AnonHugePages` for the process goes from 0 to ~208 MB.
3. **Construct and free properly.** The buckets are constructed with
   `std::uninitialized_value_construct_n`, which zeroes them as `new Bucket[n]` did.
   The table is held in a `unique_ptr<Bucket[], BucketDeleter>`, whose deleter calls
   `free` / `_aligned_free`.
4. **Prefetch.** `TTable::prefetch(key)` issues
   `_mm_prefetch(&table[index(key)], _MM_HINT_T0)`. `negamax` and `qsearch` call
   `tt.prefetch(board.getHashKey())` right after `board.makeMove(move)`, and `negamax`
   also does so after `makeNullMove()`. By the time the child node reaches
   `tt.probe()`, the bucket is usually already on its way into the cache.

**Why the search is unchanged.** Only where and how the memory is allocated changes.
The contents, indexing and replacement are identical, and a prefetch has no
architectural effect.

---

## R1.4 No heap allocation in the search

**Problem.**
- `std::stable_sort` allocates a temporary buffer on every call, and it sorted the noisy
  and quiet moves of most nodes: about 1.3 M allocations per bench.
- `negamax` also kept `std::vector<Move> failLowQuiet / failLowNoisy`, which allocated
  on their first `push_back` in every node where a move failed low.

**Implementation.**

1. **Sort without a buffer.** In `movegenerator.cpp` the `std::stable_sort` calls were
   replaced by an insertion sort, `sortScoredMoves`, using the same ordering: higher
   score first, ties broken by the lower move value. (Round 2's change R2.2 later
   replaced this sort with on-demand selection.)
2. **Fixed-size fail-low lists.** In `search.cpp` a small struct
   `FailedMoves { uint16_t moves[256]; int count; }` replaces the vectors.
   - `add()` stores the move's 16-bit value and `operator[]` rebuilds the `Move`.
   - The array is left uninitialised on purpose: an array of `Move` would zero 256
     entries in every node, because `Move` has a default member initialiser.
3. **Same loops.** The history-penalty loops now iterate `for (i < count)` instead of a
   range-for.

**Why the search is unchanged.** The comparator is a *strict total order*: the score,
then the unique move value. Every correct sort produces the same unique order, stable
or not. The fail-low lists hold the same moves in the same order.

---

## R1.5 Link-time optimisation

**Problem.** The hot path crosses source files on every node:
- `board.cpp` calls into `nnueq.cpp`;
- `search.cpp` calls into `board.cpp` and `ttable.cpp`.

Without LTO, none of those calls can be inlined.

**Implementation** (`CMakeLists.txt`).

1. **The option.** A new `option(ZAPHOD_LTO ... ON)`.
2. **Check the toolchain.** If it is on and `ZAPHOD_SANITIZE` is off,
   `check_ipo_supported()` checks that the compiler and linker can do LTO.
3. **Enable it for optimised builds.** If they can,
   `CMAKE_INTERPROCEDURAL_OPTIMIZATION_{RELEASE,RELWITHDEBINFO,MINSIZEREL}` is set to
   `ON`. Debug and sanitizer builds are unaffected, and `-DZAPHOD_LTO=OFF` turns it off.

**Why the search is unchanged.** It's the same code with more inlining. No floating-point
behaviour changes (no fast-math), so the results are identical.

---

## R2.1 Leaner NNUE accumulator update

**Problem.** After R1.2, `applyPending` was the hottest function (~20%), and only about
half of that was the SIMD loop. The rest was bookkeeping:
- a `switch` mapping each piece to its feature plane;
- `encodeFeature` for each change and each perspective;
- copying the queued changes into a second list while looking for cancelling pairs;
- a separate pass over the accumulator for each perspective.

**Implementation** (`nnueq.cpp`).

1. **Offset table.** A static `featureOffsets[15][64][2]` is built once at start-up. For
   each `BitBoardEnum` piece and square it holds the offset of the weight row from the
   white and from the black perspective. Looking up a change is now one table read.
2. **Cheaper cancellation.** Pairs that cancel (a promotion's pawn added and removed on
   the same square) can only occur when 4 or more changes are queued: promotions and
   castling. For shorter lists (quiet moves, captures) the search is skipped. When it
   does run, it works in place in `pending[]`: both members of the pair are removed by
   swapping in the last entries.
3. **Rows for both perspectives together.** The rows to add and subtract are collected
   as pairs: `add[i][0]` for white and `add[i][1]` for black.
4. **One pass for both perspectives.** `updateRows<Adds, Subs>` now loads the white and
   the black chunk at the same offset, applies all rows to both, and stores both. That
   is one loop instead of two, which halves the loop overhead and gives the CPU two
   independent chains to work on.

**Why the search is unchanged.** It's the same rows, added and subtracted in wrapping
arithmetic. Order and grouping don't matter within the int16 range (see R1.2).

---

## R2.2 Pick the best move on demand instead of sorting

**Problem.** Both move lists were insertion sorted in full as soon as they were
generated (~3.7% of instructions). But most nodes cut off after the first move or two,
and late move pruning skips most of the quiet moves, so most of that sorting was
wasted.

**Idea.** Don't sort. Each time `next()` needs a move, scan the remaining moves for the
best one (selection sort, one step at a time).

**Implementation** (`movegenerator.cpp`).

1. **`pickBest`.** `pickBest(moves, idx, count)` scans `moves[idx..count)` for the best
   entry under the same ordering (higher score, then lower move value). It swaps that
   entry into position `idx` and returns it.
2. **No more upfront sort.** The sort calls at the end of the two scoring functions are
   gone. `sortNoisyMoves` is renamed `scoreNoisyMoves`, since it now only scores.
3. **Call it in each stage of `next()`:**
   - **`GOOD_NOISY`:** `pickBest` brings the best remaining capture to the front. If its
     score is > 0 it is returned. Otherwise the stage moves on, and that move stays at
     the front for later.
   - **`QUIET`:** `return pickBest(quietMoves, quietIdx++, quietCount).move;`
   - **`BAD_NOISY`:** `pickBest` again, over the remaining noisy moves.

**Why the search is unchanged.** Repeatedly taking the maximum of what's left yields the
sorted order. The ordering is a strict total order (move values are unique), so the
sequence of moves is exactly the one the sort produced. Swapping entries around in the
unpicked part of the list doesn't matter, because only the maximum is ever taken from it.

---

## R2.3 Pins, snipers and threats computed on first use

**Problem.** Every `makeMove` / `makeNullMove` computed, for the new position:
- the **checkers**;
- the **snipers** (enemy sliders lined up with our king) and **pins**;
- the **threats** (every square the opponent attacks).

Together that was ~5% of instructions. Only the checkers are needed in every node, for
`inCheck`. The rest is only used to generate and order moves, and many nodes return
before that: TT cutoffs, stand pat in qsearch, repetitions. Qsearch nodes generate only
captures, which don't need threats at all.

**Idea.** Compute checkers eagerly. Compute pins/snipers and threats lazily, the first
time something asks for them in a position.

**Implementation.**

1. **Split the calculation** (`board.cpp`). `calculateCheckersSnipersPins()` now only
   computes the checkers, and it marks pins and threats as not yet known
   (`pinsValid = false; threatsValid = false;`). The pin and sniper code moved into a
   new `calculateSnipersPins()`, which sets `pinsValid = true`. `calculateThreats()`
   sets `threatsValid = true`.
2. **Lazy getters** (`board.h`). `getPins()`, `getSnipers()` and `getThreats()` check the
   flag and compute the value if it isn't known yet, then return the cached value. Every
   caller already went through these getters.
3. **No eager calls.** `makeMove`, `makeNullMove` and `parseFen` no longer call
   `calculateThreats()`.
4. **Undo keeps the flags in sync.** `MoveUndoInfo` gets `pinsValid` and `threatsValid`.
   `makeMove` / `makeNullMove` save them along with `pins` / `snipers` / `threats`, and
   `revertLastMove` / `revertNullMove` restore them. If the parent had already computed
   its threats before the move was made, they're still valid after the undo. If it
   hadn't, they're computed later if needed.
5. **Read once per node** (`movegenerator.*`).
   - `MoveGenerator::init()` reads `pins` and `snipers` into new members once.
   - All the piece generators, `isMoveLegal` and `isMoveLegalSliders` use those members
     instead of calling `board->getPins()` / `getSnipers()` for every piece.
   - `scoreQuietMoves()` reads `getThreats()` once into a local.

   Without this step the change saved only 0.2%, because the flag check in every getter
   call ate the gain. With it, the change saves 1.8%.

During bench, threats are now computed 0.44 M times instead of 2.47 M, and pins/snipers
1.29 M times instead of 2.47 M.

**Why the search is unchanged.** Each value is a pure function of the position. It is
computed in the same position it is used in, only later; the board is restored before
any later use, and the flags are saved and restored with the rest of the state. The
move generator's cached copies are valid for the whole node, just like the existing
`checkMask` and `kingDangerMask`. Because pins feed move legality, perft should still be
run on this change; see the round-2 report.

---

## R2.4 Lighter `revertLastMove`

**Problem.** `revertLastMove` put pieces back with `addPiece` / `removePiece`. Besides
the bitboards, those:
- toggle the non-pawn, minor and major piece hash keys (`toggleHashKeys`);
- queue NNUE changes.

All of that is thrown away right afterwards: all hash keys are restored from
`keyHistory`, and `nnue.pop()` discards the queue.

**Implementation** (`board.*`).

1. **Board-only helpers.** Two new inline private helpers:
   - `placePiece(sq, piece, color)`: sets the mailbox square and the `All`, piece and
     colour bitboards.
   - `clearPiece(sq, color)`: clears them again.
2. **Use them in `revertLastMove`.** Every `addPiece` / `removePiece` call there
   (castling rook, promotion, normal move, en-passant pawn, captured piece) now uses
   `placePiece` / `clearPiece`. `makeMove` and `parseFen` still use the full versions.

**Why the search is unchanged.** The bitboards and mailbox end up identical. The hash
keys are overwritten from `keyHistory` exactly as before. The accumulator is popped, and
`pop()` already cleared anything queued.

---

## R2.5 qsearch reuses the move-ordering SEE

**Problem.** `scoreNoisyMoves()` runs a static exchange evaluation (SEE) for every
capture to order it. qsearch's futility pruning then called `See::see` again for the same
capture in the same position: 1.32 M extra calls per bench.

**Implementation.**

1. **Keep the value** (`movegenerator.*`). `ScoredMove` gets an `int see` field.
   `addNoisy` / `addQuiet` initialise it to `SEE_UNKNOWN` (`INT32_MIN`), and
   `scoreNoisyMoves()` stores the SEE of each capture there before adding it to the
   score.
2. **Remember it for the move just returned.** `next()` resets `lastSee = SEE_UNKNOWN`
   on entry. When it returns a move from the noisy list (`GOOD_NOISY` or `BAD_NOISY`),
   it sets `lastSee` to that move's stored value.
3. **`seeOfLastMove(move)`.** Returns `lastSee` if it is known. Otherwise it computes
   `See::see` on the spot. That covers the TT move, which is returned before any scoring,
   and any move that wasn't scored.
4. **Use it in qsearch** (`search.cpp`). The futility condition
   `0 > See::see(board, ...)` becomes `0 > moveGen.seeOfLastMove(move)`.

**Why the search is unchanged.** SEE depends only on the position and the move. The
stored value was computed in the same position: every child move is unmade before
`next()` runs again, so the board is identical. Where no stored value exists, SEE is
computed exactly as before.

---

## How every change was checked

- **Node count.** `bench` must report 2,472,970 nodes for every change on its own and for
  the combined branch. It did, on every run.
- **Speed.** Bench NPS was measured over many runs, alternating between the builds, and
  in round 2 also with callgrind instruction counts. The numbers are in the two reports.
- **Still open: perft.** It has not been run yet. It should be run before merging,
  because R1.2, R2.3 and R2.4 touch `board.*`, and R2.3 changes how the move generator
  gets its pin information.
