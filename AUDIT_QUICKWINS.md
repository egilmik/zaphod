# Zaphod 2.0 Audit — Provably Behaviour-Neutral Quick Wins

Subset of `AUDIT.md` that cannot change search results: no hash values, no
scores, no move ordering, no pruning conditions are touched. Safe to batch.
Gate for every batch: clean build (GCC/Clang `-Wall -Wextra -Wpedantic
-Wshadow`, MSVC `/W4`), perft suite passes, `bench` prints the identical
node count.

## 1. Dead code removal

- `Search::see` (`search.cpp:632-763`) — drifted 130-line duplicate of `See::see`, never called; includes the `if (score[ply] < 0) { int x = 0; }` fossil.
- `Search::getPinned` (`search.cpp:765-789`), `Search::equal` (`search.cpp:798-802`), `Search::clearTTOnSearch` (`search.h:94`).
- `TranspositionEntry` + `TEType` (`transpositiontable.h:8-15`).
- `Board::generatePawnHashKey` (`board.cpp:559-573`, `board.h:235`).
- `Perft::dperft`, `dperftLeafNodeCounter`, `invalid*Move` statics (declared, never defined; `dperft` takes non-copyable `Board` by value and cannot be called) — `perft.h:24-29, 61-111`.
- `perftWithStats` dead `if (moveList.counter == 0)` block (`perft.h:122-128`; contains a latent piece-enum-as-square index).
- `initSqToBitMapping` / `initInvertedSqToBitMapping` (`board.cpp:9-33`).
- Unused `Material` members: `pieceSquareScoreArrayMG`, `passedPawnArray`, `getPassPawnScore`, `getMaterialScore(Board&)`, `gamePhaseArray`, `materialScoreArray`, `flip` (keep `pieceMaterialScoreArray`).
- Unused tunables `seeMarginQsearch`, `rfpQuadratic` (`params.h:53,57`).
- `Board::setBit(BitBoard&, bool, int)` overload that ignores `highLow` (`board.cpp:783-786`) — fold callers into the 2-arg overload.
- `Board::getGamePhase`/`setGamePhase`/`gamePhase` (`board.h:251-257,296`), `MoveGenerator::getLegalMoves` (`movegenerator.h:32-34`).
- `file(GLOB src ...)` and dead `${cmake_cxx_flags_relwithdebinfo}` line (`CMakeLists.txt:11,20`), bogus `target_include_directories(... "src/*.h")` (`:25`).
- Commented-out experiment blocks: `search.cpp:281-285, 318-330, 341-348, 350-363, 580-587`; `move.h:7-18`; `uci.cpp:219-246`; commented `sortMoveList` at `search.cpp:161`.
- Debug fossils: `board.cpp:468-470` ("How did this happen" — replace with nothing or an assert), `uci.cpp:20` string (superseded by the EOF fix below).

## 2. Compiler warnings (81 clang / 86 gcc, full clusters in AUDIT.md appendix)

- Delete/rename unused locals & set-but-unused vars across `movegenerator.cpp` (promo enums, `enemyBoard`, `allPieces`, `sideToMove`, `enPassantIncrement`, `pawnDoubleIncrement`, `doublePushRank` in noisy path, `captures` in quiet knight path), `search.cpp` (`inIteration`, `stop`, `r` in `search.cpp:422`), `see.h` (`us`, `otherSide`, `sideToMoveAttackers`), `uci.cpp` (`fixedSearchTime`).
- Remove unused parameters: `isMoveLegalSliders(pieceBoard)` (`movegenerator.cpp:144`), `generateKingMoves(checkers, pinned, snipers, kingSquare)` (`:1188`).
- Fix shadows: rename `setGamePhase` parameter (or delete with the dead member), `generatePawnMoves`/`generateKingMoves` parameter names vs members (`movegenerator.cpp:655,1188`), inner `magic` (`see.h:108`), `search.cpp:734`.
- Sign-compare: `uci.cpp:160,204` (`size_t i`), `search.cpp:825,854` (cast `nodeLimit` once), `movegenerator.cpp:160` (compare `uint32_t` consistently).
- `ull` suffixes on the four >INT64_MAX magic constants (`board.cpp:115,117`).
- `TTable::clear`: replace `memset` with value-init loop / `std::fill` (`ttable.h:60-63`) — byte-identical result, silences `-Wclass-memaccess`.
- Initialise `razoringEntryHit`, `qsearchFutilityPruningHit`, `qsearchMoveCounterPruningHit` (`search.h:48-50`).

## 3. Header hygiene

- Remove `using namespace NNUE;` from `nnueq.h:35` (leaks `H`/`IN`/`OUT` globally); qualify uses.
- `bitboard.h:4`: `#define BitBoard uint64_t` → `using BitBoard = std::uint64_t;` + `#include <cstdint>`.
- Add missing direct includes currently obtained transitively: `<memory>`, `<cstdint>`, `<limits>` in `search.h`/`search.cpp`; `<cmath>` (`std::log`) in `search.cpp`; `<algorithm>`/`<chrono>` in `uci.cpp`; `<string>` in `tools.h`.
- `search.h`: drop `<iostream>` (unused in header; keep in .cpp) — heavy header out of a hot include.

## 4. Const/nodiscard/noexcept (leaf-only, no semantic change)

- `Board` simple getters (`getBitboard`, `getSideToMove`, `getPieceOnSquare`, `getHashKey`, `getCheckers`, `getPins`, `getSnipers`, `getEnPassantSq`, clock getters) → `const noexcept` (+`[[nodiscard]]`).
- `Move` accessors already `constexpr`; add `[[nodiscard]]`.
- `popLsb`, `countSetBits`, `sqBB` users unchanged.
- Note: making members `const` does not affect codegen here; it is API hardening only.

## 5. Safe robustness fixes (no effect on any searched node)

- **EOF exit** (`uci.cpp:18-21`): `if (!std::getline(...)) break;` — C5.
- **`stoi` guards** in UCI token parsing — m12.
- **`ss` reset loop bound** `i <= MAXPLY` (`search.cpp:21`) — m15 (slot was never read before write in any reachable path; strictly neutral).
- **Version string** unification (`uci.cpp:153` / `CMakeLists.txt:2`) — m21.
- **README** thread-safety claim correction — m20.

## 6. CMake (build-system only; produced flags for the release config unchanged)

- Default `CMAKE_BUILD_TYPE=Release` when unset (this is what makes plain `cmake --build` reproduce the release binary — M7).
- Move `-march=x86-64-v3 -mtune=generic` to `target_compile_options` on both targets; delete the three redundant global appends.
- Give `GenerateData` the same `EVALFILE` definition as `Zaphod`.
- Fail configure with a clear message on GCC < 15 / Clang < 19 (no `#embed` support) — M8.
- Optional: `CheckIPOSupported` + `CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE` (flag change → binary changes, but bench nodes must still be identical; keep as its own commit and verify).

## Explicitly NOT quick wins (behaviour-changing, need isolated commits + bench diff)

TT mate-score ply adjustment (M1), side-to-move Zobrist key (M4), history widening (M2), NNUE wrapping ops (M3), time-management changes (C1/C2 — bench-neutral at fixed depth but affects play), depth-1 no-abort (C4), EP-key-only-when-legal (m4), 50-move-vs-mate order (m6), eval clamp (m8), bench FEN `moves` handling (m10), `givesCheck` via `getCheckers` (P5), any move-ordering or pruning change.
