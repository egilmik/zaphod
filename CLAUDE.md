# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Zaphod is a UCI chess engine in C++23 (bitboards + magic sliders, NNUE eval, negamax with
alpha-beta). It is a single-author hobby engine; see `README.MD` for the version/Elo history.

## Build

Requires CMake 3.25+ and a compiler that supports `#embed` — **GCC 15+ or Clang 19+**
(clang-cl included). The NNUE net is linked in via `#embed`, so an older toolchain fails at
configure time with an explicit `FATAL_ERROR`. Everything goes through `CMakePresets.json`:

```bash
cmake --preset linux-gcc && cmake --build --preset linux-gcc     # RelWithDebInfo, Ninja
cmake --preset linux-gcc-debug   && cmake --build --preset linux-gcc-debug
cmake --preset linux-gcc-asan    && cmake --build --preset linux-gcc-asan   # ASan + UBSan
cmake --preset linux-clang       && cmake --build --preset linux-clang
cmake --preset windows-clang-cl  && cmake --build --preset windows-clang-cl
```

Binaries land in `build/<preset>/bin/` (`Zaphod`, `GenerateData`).

Configure-time knobs (`-D...`):
- `ZAPHOD_NETWORK_NAME` / `ZAPHOD_NETWORK_SHA256` — the net downloaded into `nnue/` from
  `ZAPHOD_NETWORK_URL` (the `Zaphod-nnue` repo). The hash is enforced; a mismatch fails configure.
- `ZAPHOD_NETWORK_FILE` — embed a local net instead and skip the download. Use this when testing
  a freshly trained net.
- `ZAPHOD_ARCH` (default `x86-64-v3`) — AVX2 is **required**, `nnueq.cpp` `#error`s without it.

`#embed` is invisible to the dependency scanner, so CMake pins `nnueq.cpp`'s `OBJECT_DEPENDS` to
the net file. Re-running configure is still needed when the *name* changes.

## Test / verify

There is no unit-test framework. Verification is done through the UCI binary itself:

```bash
./build/linux-gcc/bin/Zaphod          # then type commands at the prompt
echo -e "perft\nquit" | ./build/linux-gcc/bin/Zaphod   # movegen correctness (minutes, depth 5-8)
echo -e "bench\nquit" | ./build/linux-gcc/bin/Zaphod   # 35 positions at fixed depth 12
```

- **`perft`** runs `PerftTest::runAllTest()` (`src/perft/perfttest.cpp`) — the 6 standard
  chessprogramming.org positions. It is the regression gate for any movegen/make-move change.
  Note the expected node counts are the **sum of perft(1..N)**, not perft(N), because
  `Perft::perft` accumulates nodes at every depth.
- **`bench`** is the search signature: total node count must be reproducible for a given commit.
  Any search change moves it; an unintended move means a bug. It also dumps per-heuristic hit
  counters (TT bounds, LMR, RFP, null move, razoring, qsearch pruning).
- `scripts/benchmark_epd_runner.py` runs an EPD suite against any UCI binary via python-chess.

To test one position: `position fen <FEN>` then `go depth N` / `go movetime N`, or `eval` for the
static NNUE score, `d` to print the board, `fen` to dump the current FEN.

## Architecture

`zaphod_core` (static lib, all of `src/` except the two mains) is linked into two executables:
`Zaphod` (`src/main.cpp` → UCI loop) and `GenerateData` (`src/generatedata.cpp` → self-play data
generation for NNUE training).

**Board (`board.h/.cpp`)** — dual representation: `bitBoardArray[15]` indexed by `BitBoardEnum`
plus a `mailBoxBoard[64]`. `BitBoardEnum` is `{White,R,N,B,Q,K,P,Black,r,n,b,q,k,p,All}`, so
`White == 0`, `Black == 7`, and code addresses pieces by colour arithmetic (`P + color`,
`K + sideToMove`). Square 0 is a1 (see the diagram at the bottom of `board.h`). Each `Board`
allocates its own ~4 MB of magic tables in the constructor, so constructing one is expensive —
`GenerateData` creates one per thread, search reuses a single `motherBoard`.

`makeMove`/`revertLastMove` push/pop a `MoveUndoInfo` (including cached `checkers`, `pins`,
`snipers`, `threats`) onto a fixed 1024-entry stack; `makeNullMove`/`revertNullMove` do the same
for null moves.

**NNUE (`nnueq.h/.cpp`)** — 768→H×2→1, perspective (both-side accumulators), SCReLU, AVX2
intrinsics, int16 quantized (`QA=255`, `QB=64`, `SCALE=400`). Architecture constants `NNUE::IN`
and `NNUE::H` in `nnueq.h` **must match the embedded net**; changing net size means editing both
`nnueq.h` and the `ZAPHOD_NETWORK_NAME`/`SHA256` cache vars. The `.bin` format is raw little-endian
int16 in order `l0w, l0b, l1w, l1b` with no header (trailing padding is tolerated).

There is **no accumulator stack**: `Board::addPiece`/`removePiece` update the two accumulators
incrementally, and unmake reverses them by replaying the inverse add/remove. Any new code path
that mutates the board must go through those two functions or eval silently drifts.

**MoveGenerator (`movegenerator.h/.cpp`)** has two modes:
- `static generateMoves(board, MoveList&)` — bulk legal generation, used by perft comparisons and
  by UCI move parsing.
- instance `init(...)` + `next()` — the staged, lazy picker used by search
  (`TT_MOVE → GEN_NOISY → GOOD_NOISY → KILLER → GEN_QUIET → QUIET → BAD_NOISY`). Killers are
  currently commented out in the `KILLER` stage even though `SearchStack` still carries them.
  Noisy moves are ordered by SEE/MVV-LVA (score > 0 = good noisy, the rest deferred to
  `BAD_NOISY`); quiets are scored from history. Search owns a `moveGenStack[MAXPLY+1]` so each ply
  reuses its generator instead of allocating.

**Search (`search.h/.cpp`)** — single-threaded iterative deepening with aspiration windows, and
`negamax` + `quinesence`. In roughly move order: TT cutoffs, mate-distance pruning, razoring,
reverse futility pruning, null-move pruning, late move pruning, LMR (log-based, adjusted by
pv/improving/gives-check/history), check extensions, PVS re-searches. Mate scores are folded in
and out of the TT with `scoreToTT`/`scoreFromTT`. Time control is set in `UCI::startSearch`
(`time/20 + inc/2`), and `isSearchStopped()` only checks the clock every 1000 nodes.

**History (`history.h`)** — two tables: a butterfly table indexed
`[stm][from][to][fromAttacked][toAttacked]` (threat-bucketed using `board.getThreats()`), and one
collapsed continuation table `int16_t[14][64][14][64]` shared by all plies, probed at offsets
{1,2,4,6} back with per-offset weights. Both are updated on a fail-high (bonus for the cutoff
move, penalty for the quiet moves that failed low). `history.age()` runs once per `search()` call.

**TTable (`ttable.h/.cpp`)** — 3-entry buckets padded to a power of two, multiply-shift indexing,
32-bit packed key, 5-bit age. Replacement prefers same-key/empty slots, else lowest
`depth - 2*relativeAge`. Sized via the UCI `Hash` option (default 256 MB).

## Tunable parameters and SPSA

`src/params.h` declares every tuned constant with `ZAP_TUNABLE_INT(Name, value, min, max, step)`.
The macro registers the parameter in a global registry *and* generates an accessor function, so
call sites read `lmrDividerQuiet()`, not a constant. Everything registered is automatically
exposed as a UCI spin option (see `UCI::sendID` / `UCI::setOption`) — that is how the SPSA tuner
drives it, and no per-parameter UCI plumbing is needed.

Workflow: `python scripts/gen_tune_config.py src/params.h -o tune.json` to emit the tuner config,
run SPSA, then write the results back as the new defaults in `params.h`
(`gen_tune_config.py --values best.txt` validates them against min/max first).

## NNUE data + training pipeline

1. `GenerateData -network <net.bin> -threads N -target_positions M -nodes 10000 -depth 4 -book <fens>`
   self-plays and writes `part_<i>.txt` per thread; each line is a list of active feature indices
   `; score`. Concatenate the parts.
2. `scripts/convert_txt_to_bin.py` packs those into the binary record format the trainer reads.
3. Training is done with [Bullet](https://github.com/jw1912/bullet) for the current nets;
   `scripts/binary_trainer.py` + `export_binary_model_quantized.py` are the older homegrown path
   and are **out of sync with the engine** (`HIDDEN = 256`, and the exporter writes a `NNUEQ1`
   headered int8 format the engine no longer parses). Prefer Bullet unless you are deliberately
   reviving that path.
4. `scripts/phase_hist*.py`, `histogram_score.py`, `eval.py` are dataset/eval inspection tools.

Note `src/nnue/768-256x2-1_*.bin` is a checked-in leftover from the 256-hidden era; the build
does not use it (it reads `nnue/` at the repo root, which is gitignored and populated by the
configure-time download).

## Conventions

- `Move` is a packed 16-bit value (`to` 0-5, `from` 6-11, promotion 12-13, type 14-15). Promotion
  pieces are stored as the *white* piece minus one and re-coloured on read; compare moves with
  `==`/`!=`, never by comparing `from`/`to` alone.
- Dead code is kept commented out in place (old pruning attempts, killers, futility pruning).
  That is deliberate — it is a record of what was tested. Don't delete it as cleanup.
- Commit messages are short and describe the Elo-relevant change ("Added lmp with history term",
  "Updated tunables from SPSA run").
- Any search or eval change needs an Elo measurement (SPRT against the previous build) before it
  is worth keeping; `bench` node counts alone only prove the change did something.
