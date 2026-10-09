# Zaphod

UCI chess engine in C++23 (hobby project). Alpha-beta search with an NNUE
evaluation (768→N×2→1); the network is baked into the binary at compile time.

## Build

Requires CMake 3.25+, Ninja and a compiler with `#embed` support: GCC 15+ or
Clang 19+. Older compilers fail at configure time with a clear FATAL_ERROR.
Targets x86-64-v3 (AVX2/BMI2 required).

    cmake --preset linux-clang && cmake --build --preset linux-clang   # RelWithDebInfo
    cmake --preset linux-gcc   && cmake --build --preset linux-gcc
    # also: linux-gcc-debug, linux-gcc-asan (ASan+UBSan), linux-clang-debug, windows-*

- Binaries land in `build/<preset>/bin/`: `Zaphod` (engine), `GenerateData` (datagen).
- Configure downloads the network into `nnue/` (gitignored) and verifies its
  SHA256. Needs network access, or pass `-DZAPHOD_NETWORK_FILE=/path/net.bin`.
- The network is selected by `ZAPHOD_NETWORK_NAME` + `ZAPHOD_NETWORK_SHA256` in
  CMakeLists.txt (see "Do not change" below).

**Claude Code cloud sessions:** the SessionStart hook (`.claude/hooks/session-start.sh`)
installs Clang 20 and makes it the default `clang++`, because the Ubuntu image
ships GCC 13 / Clang 18. Use the `linux-clang*` presets there; the `linux-gcc*`
presets will not configure.

## Run

The engine speaks UCI on stdin/stdout. Non-standard commands for development:
- `bench` — depth-12 search over 35 positions; last line is `<nodes> nodes <nps> nps`.
- `perft` — perft suite, 6 positions at depth 5–8. Slow: ~15 min single-threaded (billions of nodes); run it in the background.
- `eval`, `d` (print board), `fen`.

    echo -e "bench\nquit" | ./build/linux-clang/bin/Zaphod | tail -1

## Verify

There is no unit-test framework. Verify changes with:

1. **Build** with at least one preset, without new warnings.
2. **Bench node count**, checked with `scripts/bench_history.py`. Search is
   deterministic, so the node count is the signature of search behaviour.
   The script builds and benches *commits* (in a temporary worktree), not the
   working tree, so commit first, then compare against the parent:

       python3 scripts/bench_history.py -n 2 --preset linux-clang

   - Non-functional changes (refactors, speedups) must keep the node count identical.
   - Functional changes (search/eval) change it — report the old and new count.
   - The same table shows NPS; use `--runs 5` or more when judging speed changes.
3. **Perft** (`perft` command, every line must be `[Passed]`) after any change to
   `board.*`, `movegenerator.*`, `move.h` or `zobrist.h`.
4. **Sanitizers**: build `linux-gcc-asan` and run `bench` for memory/indexing changes
   (needs GCC 15; not available in cloud sessions).

Playing strength (SPRT / Elo) is tested outside this scope; agents cannot
measure it. Do not claim Elo gains.

## Code map (src/)

- `board.*` — bitboard position, make/unmake, Zobrist keys, incremental NNUE accumulator
- `movegenerator.*` — legal move generation and move ordering (history-based)
- `search.*` — iterative deepening, negamax, qsearch, LMR, RFP, NMP, razoring, aspiration windows
- `history.h` — butterfly / continuation / capture history and correction history
- `ttable.*` — transposition table; `see.h` — static exchange evaluation
- `nnueq.*` — quantized NNUE inference (AVX2 intrinsics) and `#embed` of the net
- `params.h` — SPSA-tunable parameters
- `uci.*` — UCI loop; `main.cpp` — engine entry; `generatedata.cpp` — self-play datagen
- `perft/`, `tools/` — perft suite, FEN helpers, opening book

## Do not change

Unless the maintainer explicitly asks:
- `README.MD` (including the version / Elo table).
- Values of existing tunables in `params.h` (value, min, max, step). These come
  from SPSA runs. Adding a new tunable for a new feature is fine.
- The network name / SHA256 in CMakeLists.txt.

## Guidelines

- **Tunable constants** go in `params.h` as `ZAP_TUNABLE_INT(name, value, min, max, step)`
  and are read via `params::name()`. They are exposed automatically as UCI spin
  options for SPSA tuning; `scripts/gen_tune_config.py` builds the tuner config
  from them. Don't hard-code new search magic numbers.
- **Performance matters**: no allocations, exceptions or virtual calls in
  search/movegen/eval hot paths. Check NPS before and after.
- Keep make/unmake, hash keys and the NNUE accumulator consistent; a mistake in
  one silently corrupts the others. Perft and bench will reveal it.
- Match the surrounding style: 4-space indentation, `camelCase` methods and
  variables, `PascalCase` types, `#ifndef` header guards.
- New source files must be added to `zaphod_core` in CMakeLists.txt.
- One idea per commit/branch, so each change can be strength-tested in isolation.
