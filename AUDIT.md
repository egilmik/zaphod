# Zaphod 2.0 Hardening Audit

Audit date: 2026-08-04. Read-only analysis pass; no code has been changed.
Line numbers refer to the tree at commit `fe8b690`.

---

## Phase 1 — Inventory

### Layout

| Area | Files |
|---|---|
| Build | `CMakeLists.txt` (two targets: `Zaphod`, `GenerateData`), `CMakeSettings.json` (VS/clang-cl presets) |
| Board / make-unmake / Zobrist | `src/board.h`, `src/board.cpp`, `src/bitboard.h`, `src/move.h` |
| Move generation (staged, legal) | `src/movegenerator.h/.cpp` |
| Search (negamax + qsearch, iterative deepening, aspiration) | `src/search.h/.cpp` |
| Transposition table | `src/ttable.h/.cpp` (active TT); `src/transpositiontable.h` (Zobrist keys + a dead legacy entry struct) |
| Evaluation (NNUE 768→256x2→1, SCReLU, AVX2) | `src/nnueq.h/.cpp`, embedded net `src/nnue/*.bin` via `#embed` |
| SEE | `src/see.h` (used) and a duplicate `Search::see` in `src/search.cpp` (dead) |
| History / killers | `src/history.h`, `SearchStack` in `src/search.h` |
| UCI + time management | `src/uci.h/.cpp`, `src/main.cpp` |
| Testing | `src/perft/perft.h`, `src/perft/perfttest.h/.cpp` (6-position perft suite), `bench` command in `uci.cpp` (35 FENs, fixed depth 12) |
| Data generation tool | `src/generatedata.cpp`, `src/tools/openingbook.*`, `src/tools/fentools.h` |
| Dead-ish helpers | `src/material.h` (PSQT/material tables, mostly unused since NNUE), `src/tools.h` (consistency checkers, debug only) |

### Toolchain facts

- `CMAKE_CXX_STANDARD 23`, extensions off. **The code actually requires more than C++23**: `src/nnueq.cpp:18` uses `#embed`, which needs GCC ≥ 15, Clang ≥ 19, or a very recent MSVC. Verified locally: GCC 13.3 and Clang 18.1 both fail with *invalid preprocessing directive* — see H1.
- AVX2 is mandatory: `-march=x86-64-v3` is hardcoded, and `nnueq.cpp:33` `#error`s without `__AVX2__`. There is no fallback build.
- MSVC/clang-cl path: `CMakeSettings.json` uses clang-cl; `ttable.h:15-20` has an MSVC-specific `u128` path using the internal header `<__msvc_int128.hpp>`.
- Single-threaded engine. `GenerateData` is multi-threaded but shares no engine state between threads (one `Board`+`Search` per worker).
- Interesting quirk: rook magics are precomputed constants, but **bishop magics are re-derived with `std::random_device` at every `Board` construction** (`board.cpp:122-123, 164`). Attack tables are identical regardless of which magic is found, so this is a determinism/startup-time issue, not a correctness one.
- The file/rank mask names in `board.h:35-44` are **inverted relative to the square mapping** (a1=bit 0 … h1=bit 7, so `FileAMask` = bits 7,15,… is actually the h-file). Usage is self-consistent throughout, so nothing is wrong at runtime, but every future reader/writer of shift code is one sign error away from a bug. Flagged as hygiene (H13).

---

## Phase 2 — Findings

Severity: **Critical** = can lose games/crash/corrupt state; **Major** = wrong results or serious robustness/strength issue in realistic conditions; **Minor** = edge case, latent, or quality issue; **Nit** = cosmetic.

### Critical

**C1. Time management: engine searches forever when the clock is low — guaranteed time loss.**
`src/uci.cpp:128-141` + `src/search.cpp:36-41`.
`limits.timeLimit = wTime / 20` — with `wtime < 20` ms this is **0**, and `search()` treats `timeLimit <= 0` as *no limit* (`maxSearchTime = INT_MAX`), so at near-zero clock the engine starts an unbounded search and flags. Same if a GUI sends only the side-to-move's clock: the allocator requires `wTime > 0 && bTime > 0`, otherwise no limit is set at all.
*Minimal fix:* compute the budget, then clamp: `limits.timeLimit = std::max(1, ...)`; require only the side-to-move's clock; treat "has clock but computed 0" as 1 ms.

**C2. Time management: increment can exceed remaining time; no hard-limit safety margin; `movestogo` ignored.**
`src/uci.cpp:130-140`.
Budget is `time/20 + inc/2` with no cap against remaining time (e.g. `wtime 200 winc 10000` → 5.1 s budget with 0.2 s on the clock). There is also no move-overhead margin for GUI/connection latency, and `movestogo` is parsed nowhere, so at `X moves in Y` controls the engine allocates as if sudden death. The soft/hard limit are the same value (`maxSearchTime`), checked every 1000 nodes (`search.cpp:842`) — that part is fine, but the budget itself can exceed the clock.
*Minimal fix:* `budget = min(time/20 + inc/2, time - overhead)` with `overhead ≈ 10–50 ms`, floor at 1 ms; parse `movestogo` and use `time/movestogo + inc/2` (capped) when present. This changes time behaviour only, never node counts at fixed depth.

**C3. `stop`, `go infinite` and `ponder` are unsupported, and search blocks the UCI thread.**
`src/uci.cpp:15-42, 88-122`.
`go` runs the search synchronously; no input is read until it returns. `go infinite` (token not parsed → no limits → depth-255 search) effectively hangs the engine, and `stop` can never be processed. Any GUI analysis session or ponder-on match is broken.
*Minimal fix* consistent with a hardening pass: run the search on a `std::thread`, make `stopSearch` a `std::atomic<bool>` (relaxed loads in `isSearchStopped`), have the UCI loop keep reading and set the flag on `stop`/`quit`. Without threads, at minimum parse `infinite`/`stop` and refuse to search unbounded (document limitation). Note `SearchLimits.nodeLimit` exists but `go nodes` is also never parsed — trivial to wire while in there.

**C4. Search can emit an illegal/garbage best move when stopped before depth 1 completes, and when the root position has no legal moves.**
`src/search.cpp:155-165`.
If `bestMoveIteration.bestMove` is empty (stop during the first root iteration), the fallback plays `list.moves[0]` — but if the position is checkmate/stalemate `list.counter == 0` and `moves[0]` is a default-constructed `Move` (value 0 → printed as `a1a1`). The commit message on `fe8b690` acknowledges invalid moves still escape. Also the root loop's stop-check (`search.cpp:439-441`) sits *before* the best-move update, so a fully-searched root move that finished exactly at the deadline is discarded (conservative, but contributes to "no move yet").
*Minimal fix:* never abort the depth-1 iteration (skip time checks while `currentTargetDepth == 1`), so `bestMoveIteration` is always a legal move; guard the fallback with `list.counter > 0` and otherwise print `bestmove 0000`.

**C5. UCI loop spins at 100% CPU forever on EOF.**
`src/uci.cpp:18-21`.
When the GUI closes stdin, `getline` fails, the loop prints "Is this a problem in the UCI loop?" and iterates again — an infinite busy loop that also floods stdout. Engines must exit on EOF.
*Minimal fix:* `if (!std::getline(std::cin, cmd)) break;`

### Major

**M1. Mate scores are stored in / read from the TT without ply adjustment.**
`src/search.cpp:210-226, 486-489, 535-545`; `src/ttable.cpp`.
`tt.put(key, bestScore, …)` stores root-relative mate scores (`±(MATESCORE − ply)`), and probe-side cutoffs (`negamax` lines 212-226, `quinesence` line 542) return them verbatim at a *different* ply. Result: wrong mate distances in `info score mate`, and wrong cutoff decisions near mates (a "mate in 3" entry probed deeper in the tree still claims mate-in-3-from-root). Mate-distance pruning (`search.cpp:201-205`) is correct; only the TT interchange is wrong.
*Minimal fix:* on store, `if (score > MATESCORE−MAXPLY) score += ply; else if (score < −(MATESCORE−MAXPLY)) score −= ply;` and the inverse on probe. Behaviour-changing (bench nodes will move) — must be its own commit.

**M2. Quiet-history counters overflow `int16_t`.**
`src/history.h:7`, `src/search.cpp:468`.
`hist.quiet[side][from][to] += depth * depth` — at depth 60 one update adds 3600; ~9 updates of a popular refutation wrap 32767 → large negative → the *best* quiet move sorts last. This silently degrades move ordering in long searches; the periodic `/2` decay (`search.cpp:29-32, 93-96`) does not prevent it.
*Minimal fix:* widen to `int32_t` (table becomes 64 KiB, still fine) or clamp on update (`std::min(v + d*d, 16384)`). Behaviour-changing (ordering) — own commit.

**M3. NNUE accumulator uses *saturating* SIMD add/sub — make/unmake is not exactly reversible once any lane saturates.**
`src/nnueq.cpp:150-170` (`_mm256_adds_epi16` / `_mm256_subs_epi16`), scalar fallback uses wrapping `+=`/`-=` (lines 110, 135).
If an accumulator lane ever reaches ±32767 during a sequence of adds, the following subtract does not restore the previous value, and the accumulator diverges from a from-scratch refresh for the rest of the game (there is no refresh path in make/unmake; only `parseFen` rebuilds). With the current net the pre-activations presumably stay far from ±32767, so this is latent — but it is exactly the class of bug that surfaces only after retraining a net with bigger weights. Also: AVX2 vs scalar builds compute different things on overflow.
*Minimal fix:* use `_mm256_add_epi16`/`_mm256_sub_epi16` (wrapping, exactly reversible, matches scalar) — saturation gives no benefit here since `forward()` clamps to `[0, QA]` anyway. Verify bench nodes unchanged (they should be, unless saturation actually occurs today — which would itself be a finding).

**M4. Side-to-move Zobrist hashing XORs the *enum value* (0/7) instead of a random key; the real keys are dead code.**
`src/board.cpp:591, 739-746` (`hashKey ^= sideToMove`), unused keys at `src/transpositiontable.h:21, 35-36`.
White↔Black flips only bits 0-2 of the hash. The two sides-to-move of the same position therefore land in the same TT bucket with 32-bit `shortKey`s differing in 3 bits — key quality is measurably worse than it should be, raising both eviction pressure and the (already nonzero) 32-bit collision rate.
*Minimal fix:* XOR `ttable.sideToMoveKey[0]` (a single fixed key) on every side change, and include it in `generateHashKey()` for Black. Behaviour-changing (every hash changes) — own commit, bench nodes will move.

**M5. `MoveUndoInfo` truncates clocks to `int8_t`, and `revertNullMove` never restores `fullMoveClock`.**
`src/board.h:25-27`, `src/board.cpp:1110-1151` (null-move path saves/restores no `fullMoveClock`), `changeSideToMove` (`board.cpp:744`) increments it on every Black→White flip **including null moves**.
Consequences: (a) after any search, `fullMoveClock` is only correct again because every real make/unmake pair straddling the nulls restores it — but the *saved copy* is `int8_t`, so once the true value exceeds 127 (long games; `GenerateData` plays up to 200 moves) the restore writes a negative number; (b) `halfMoveClock` is capped at 100 inside search by the draw return, so its `int8_t` is safe today, but only by that accident.
Effects are on FEN output and on `GenerateData`'s `getFullMoveClock() > 200` termination (see M9) rather than play strength, but this is a make/unmake symmetry violation.
*Minimal fix:* make both fields `int16_t` in `MoveUndoInfo` (struct is `alignas(64)` with ≥16 spare bytes — layout cost zero), and save/restore `fullMoveClock` in `makeNullMove`/`revertNullMove`. Behaviour-neutral for search results.

**M6. Insufficient-material bishop-colour test uses the wrong parity.**
`src/board.cpp:412`.
`whiteBishopSq % 2 == blackBishopSq % 2` — square colour is `((sq >> 3) ^ sq) & 1`, not `sq & 1` (a1=0 and a2=8 get equal parity but are opposite colours). KB vs KB with opposite-coloured bishops is misclassified as a draw about half the time (and vice versa). Only `GenerateData` calls `hasInsufficientMaterial()` today, so this mislabels training games (`wdl = 0.5`), not engine play.
*Minimal fix:* `((wSq >> 3) ^ wSq) & 1) == (((bSq >> 3) ^ bSq) & 1)`.

**M7. A plain `cmake --build` does not reproduce the release binary — no optimisation flags at all by default.**
`CMakeLists.txt:10-13`.
No `CMAKE_BUILD_TYPE` default is set and no `-O` flag is ever added; line 11 references `${cmake_cxx_flags_relwithdebinfo}` (lowercase → empty, dead) and is immediately overwritten by line 12 anyway. Configure-and-build with no extra arguments produces an **-O0 engine** at (measured here) a fraction of release speed. Also: `-march=x86-64-v3` is appended to global `CMAKE_CXX_FLAGS` three redundant times, `target_include_directories(Zaphod PRIVATE ".../src/*.h")` is a bogus glob-as-path, `file(GLOB src ...)` result is never used, `GenerateData` never receives the `EVALFILE` define (works only because the fallback in `nnueq.cpp:14` happens to match), and there is no IPO/LTO configuration.
*Minimal fix:* default `CMAKE_BUILD_TYPE` to `Release` when unset; use `target_compile_options`/`target_compile_definitions` per target; delete the dead lines; optionally `CMAKE_INTERPROCEDURAL_OPTIMIZATION` behind `CheckIPOSupported`.

**M8. `#embed` requires bleeding-edge compilers and nothing checks for it.**
`src/nnueq.cpp:17-19`, `CMakeLists.txt`.
GCC 13 / Clang 18 (current Ubuntu LTS defaults) fail with *invalid preprocessing directive*. For a 2.0 release either document the compiler floor and add a CMake version check with a clear error, or provide a fallback (e.g. objcopy/cmake-generated byte array) — the latter is what OpenBench-style testing infra will need.
*Minimal fix:* `if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 15) message(FATAL_ERROR ...)` etc.; README note.

**M9. `GenerateData`: `-depth` is ignored, `-nodes` default is unlimited, and the monitor thread can hang the process.**
`src/generatedata.cpp:59-61` (only `nodeLimit` is set; `WorkerArgs.depth` is never applied), `:210` (`nodes = 0` default → `nodeLimit = 0` → `limits` all unlimited → the *first* search never terminates unless `-nodes` was passed), `:183` (`if (prod > target) return;` — workers stop at `>=` target, so when production lands exactly on target the monitor never exits and `join()` blocks forever), `:106+146` (game termination depends on the broken M5/M6 primitives).
*Minimal fix:* apply `a.depth` to `limits.depthLimit`, default nodes to 10000 if neither given, change monitor condition to `>=`, and join workers before/independently of the monitor (or make it detached/condition-signalled).

**M10. `OpeningBook` used after failed load / crashes on empty book.**
`src/generatedata.cpp:239-251` (book pointer is passed to workers even when `loadBook` returned false), `src/tools/openingbook.cpp:27-34` (`nextFen` indexes `fens[index++]` with no empty check → UB on empty vector; also `index` incremented before wrap check, benign but off-by-one style).
*Minimal fix:* `if (!success) { delete book; book = nullptr; }`; guard `nextFen` for empty.

**M11. Search-stop return values are bound-polluting.**
`src/search.cpp:186-188` (`return beta` on stop at node entry), `:519-521` (same in qsearch), `:439-441`/`:615-617` (`return 0` mid-loop).
Returning `beta` fabricates a fail-high; returning `0` fabricates a draw score. Today the root discards everything after `stopSearch` (the post-move check runs before the best-move update, and aborted iterations are not committed), so no wrong move is played *directly* — but any future code that trusts partial results (e.g. using the aborted iteration's `bestMoveIteration`, which line 155-156 **does** when the move exists) inherits scores contaminated by these fabrications. This is the standard pattern, but it deserves an explicit comment, and `bestMoveIteration` committed from an aborted iteration (line 142 not reached, but 155-165 uses it) should be understood as "move from an incomplete iteration".
*Minimal fix (report-only proposal):* keep, but document; optionally return `alpha` instead of `beta` on entry-stop.

### Minor

**m1. TT entry `depth` stored as `uint8_t` but probed back through `int8_t`.**
`src/ttable.h:32` (`TTEntry.depth` is `int8_t`) vs `:113` (`InternalEntry.depth` is `uint8_t`). Depths > 127 (reachable at long TCs with 255-iteration cap + check extensions) read back negative, disabling TT cutoffs (`tte.depth >= depth` false) — fail-safe direction, but inconsistent. Make both `uint8_t` (or clamp stored depth). Also `tt.put` narrows `int score`→`int16_t` unchecked; fine today (|score| ≤ 30000) — add an `assert`.

**m2. Bucket is 64 bytes but only `alignas(32)`.**
`src/ttable.h:134-139`. `InternalEntry` is 16 B (Move holds a needless `uint32_t`; 2 bytes wasted — see m3), 3×16=48, padded via `bit_ceil` to 64 — with `alignas(32)` a bucket may straddle two cache lines. `alignas(64)` is free. Also `clear()` memsets a non-trivial type (GCC `-Wclass-memaccess`); `std::fill` or value-init `Bucket{}` loop is equivalent and warning-clean. `setSize()` does not reset `tableAge` (only `clear()` does) — harmless but inconsistent.

**m3. `Move::value` is `uint32_t` for a 16-bit encoding.**
`src/move.h:64` (comments in `board.h:14` even say "2 byte"). Halving it shrinks `InternalEntry` to 14→(padded)16 B today but enables a 12-byte entry / 4-entry bucket later; also shrinks `MoveList`. Behaviour-neutral, wide but mechanical change; the `Move(uint16_t)` constructor already assumes 16 bits.

**m4. En-passant Zobrist key is hashed on every double push, even when no capture is possible.**
`src/board.cpp:882-896` (and `parseFen` accepts any EP field). Two positions identical except for an irrelevant EP right hash differently → missed repetition detections (draw scored as non-draw). Standard fix: only XOR the EP key when an enemy pawn can actually capture. Behaviour-changing; low priority for strength.

**m5. Repetition scan ignores the reversible-move bound and scans the whole game history.**
`src/board.cpp:366-382`. `int moves = std::min(halfMoveClock, historyPly)` is computed then unused; the loop walks all of `moveHistory` from 0. Correctness survives (equal hash across an irreversible move is a true collision, ~2⁻⁶⁴), but it is O(game length) per node instead of O(rule50), and the unused variable betrays the intent. Also requires two prior occurrences (true 3-fold) — most engines score the *first* repetition inside the search as a draw; current behaviour is legal but weaker (report-only proposal, do not change in hardening).

**m6. 50-move draw is returned before checkmate is tested.**
`src/search.cpp:192-194, 503-505`. At `halfMoveClock >= 100` with the side to move checkmated, the correct result is mate, not draw (the mating move completed before any draw claim). One-line fix: only return the draw when not in check (or when legal moves exist).

**m7. `info score mate` prints plies, not moves.**
`src/search.cpp:131-136`. UCI `mate N` is in *moves*; this prints `MATESCORE − score` (plies) and, combined with M1, often the wrong number anyway. `(plies + 1) / 2` with sign.

**m8. NNUE eval clamp collides with the mate-score space.**
`src/nnueq.cpp:86-87` clamps to ±30000 == `MATESCORE`; `search.cpp:131` treats anything above `MATESCORE − MAXPLY` (29745) as mate, and the RFP guard `search.cpp:276` compares against the same band. A pathological eval ≥ 29746 would print "mate" and dodge RFP. Clamp to ±29000 instead (behaviour-changing only in absurd positions).

**m9. `nps` division by zero → UB float→int conversion.**
`src/search.cpp:126-127` (µs duration can be 0 on a trivial depth-1 iteration → `inf` → undefined int conversion), `src/uci.cpp:344` same shape (ms). Guard with `std::max<int64_t>(1, count)` as `bench` (line 326) already does.

**m10. `bench` FENs with `moves ...` suffixes are silently not applied.**
`src/uci.cpp:277, 278, 291` pass the raw string (including `moves d4e6` etc.) to `parseFen`, which ignores the tail. Bench runs a *different position* than the Stockfish-derived list intends. Either strip the suffix or route through `setPosition` parsing. Behaviour-changing for bench numbers only (it *is* the bench definition).

**m11. `parseFen` counter parsing is fragile; `clearBoard` doesn't reset `fullMoveClock`/`gamePhase`.**
`src/board.cpp:505-546` (index gymnastics reading `fen[i+1]` at the string boundary — legal only via the `operator[](size())` null-char special case; breaks on 3-digit counters with trailing fields), `:322-341` (a FEN without counters inherits the previous position's `fullMoveClock`). Rewrite the tail parsing with `istringstream >> half >> full`.

**m12. `stoi` on unvalidated UCI tokens can throw and kill the engine.**
`src/uci.cpp:99-120, 198, 213`. A malformed `go wtime abc` terminates the process (no exception handler in `main`). Wrap in a small `parseInt(token, default)` helper.

**m13. `isSquareAttacked`/`popLsb` on an empty bitboard is UB.**
`src/board.cpp:807-812` (`__builtin_ctzll(0)` undefined), reachable from `isSquareAttacked` (`:1176`) if a side has no king (bad FEN), and `computeKingDanger` (`movegenerator.cpp:1126`). Not reachable in legal play; an `assert(board)` in `popLsb` would catch bad-FEN misuse in debug builds.

**m14. Redundant/misleading castling-rights code in `makeMove`.**
`src/board.cpp:974-999`: the `piece == R, fromSq == 0` branch clears `castleWQ` twice (once inside the guarded hash update, once unconditionally — the second is dead); the four capture-square blocks (`:1003-1029`) repeat the same pattern. Works, but is exactly the duplicated-logic-drift shape this audit is asked to flag. A single castle-mask table (`castleMask &= castleRightsMask[from] & castleRightsMask[to]`) with one hash update would collapse ~80 lines.

**m15. `SearchStack` slot `ss[255]` is never reset between searches.**
`src/search.cpp:21-27` loops `i < MAXPLY` over an array of `MAXPLY+1`. Harmless today (ply 255 returns eval immediately) — make the loop bound `<= MAXPLY`.

**m16. Uninitialised counter members in `Search`.**
`src/search.h:48-50`: `int64_t razoringEntryHit, razoringReturnHit = 0;` initialises only the second; `qsearchFutilityPruningHit`, `qsearchMoveCounterPruningHit` have no initialisers. All are reset in `search()` before use, so no live read today — add `= 0`.

**m17. `TranspositionTable::initKeys` uses 32-bit `mt19937` for 64-bit keys.**
`src/transpositiontable.h:26-28`. Works via `uniform_int_distribution<unsigned long long>`, but the distribution's output for a 32-bit engine is implementation-defined (different keys per stdlib — harmless, but means TT behaviour differs across platforms). `std::mt19937_64` is the obvious spelling.

**m18. `MoveGenerator::generateMoves` (static) builds a 16 KiB `HistoryTables` + full generator on the stack per call.**
`src/movegenerator.cpp:9-21`. Called per node in `Perft::perft` (`perft.h:38-41` builds its own per node too) and per move in `parseMove`. Not search-hot, but perft NPS pays for it. Also `Perft::perft(depth 0)` returns 0 rather than 1 (`perft.h:34-36`) — self-consistent with the test suite's cumulative sums, but nonstandard; worth a comment.

**m19. Debug output left in engine paths.**
`src/movegenerator.cpp:287, 356` (`info string TTMove not returned...`), `src/board.cpp:468-470` ("How did this happen"), `src/uci.cpp:20` ("Is this a problem in the UCI loop?"), `src/search.cpp:163` (fallback info string — keep this one, it is genuinely informative, but consider rewording). Legal UCI (`info string`) but noise; the board one prints bare text mid-`position` which is not.

**m20. `README.MD` claims "Made tt thread safe" — it is not (and does not need to be).**
`src/ttable.h` has no atomics; fine single-threaded, but the claim will bite whoever adds SMP. Correct the README (or note "single-thread only").

**m21. UCI `id name Zaphod 1.9` vs CMake `project(Zaphod VERSION 0.1.0)`.**
`src/uci.cpp:153`, `CMakeLists.txt:2`. Wire the version through `configure_file`/compile definition before tagging 2.0.

### Nits

- `quinesence` → `quiescence` (`search.h/.cpp`, several counters); `alphaOrginal` → `alphaOriginal` (`search.cpp:179`); `northWestccludedMoves` (`board.h:205`); `PertTestResult` (`perfttest.h:7`); "Converst", "Ripler", "prematulery" comments/typos.
- `#define BitBoard uint64_t` (`bitboard.h:4`) — a macro, not a type alias; `using BitBoard = uint64_t;` respects scoping and debuggers.
- `using namespace NNUE;` at namespace scope in a header (`nnueq.h:35`) leaks `H`, `IN`, `OUT` into every TU that includes `board.h`. Remove; qualify inside the class.
- Commented-out experiment blocks: `search.cpp:281-285, 318-330, 341-348, 350-363, 580-587`, `move.h:7-18`, `uci.cpp:219-246`, `main.cpp` unused includes. Delete or convert to tracked issues.
- Magic numbers in search: `20000` (razoring mate guard, `search.cpp:263`), `3` (check-extension cap, `:379`), `20 + i*5` (aspiration, `:100`), `1000` (time-check interval, `:842`), TT replacement `depth + 4 + pv*2` (`ttable.cpp:30`), score bands `80000/70000/30000` in move ordering. Name them alongside the existing `params.h` registry (as constants, not tunables, to keep bench identical).
- Dead code inventory (all safe to delete): `Search::see` (130-line drifted duplicate of `See::see` — the copy in search.cpp retains a `if (score[ply] < 0) { int x = 0; }` debug fossil at `:693-695`), `Search::getPinned`, `Search::equal`, `Search::clearTTOnSearch`, `Board::generatePawnHashKey`, `TranspositionEntry`/`TEType` (`transpositiontable.h:8-15`), `sideToMoveKey` (until M4 uses it), `Perft::dperft`/`dperftLeafNodeCounter`/`invalid*Move` statics (declared, never defined — `dperft(Board board)` takes `Board` by value, which is non-copyable since NNUE, so it cannot even be called), `perftWithStats`'s empty `if` block (`perft.h:122-128`, which also indexes `sqBB` by *piece enum*, a latent OOB if ever revived), `initSqToBitMapping`/`initInvertedSqToBitMapping` (`board.cpp:9-33`), `Material::pieceSquareScoreArrayMG`/`passedPawnArray`/`getPassPawnScore`/`getMaterialScore(Board&)`/`gamePhaseArray` (NNUE replaced all of it; keep `pieceMaterialScoreArray` for SEE/ordering), unused params `seeMarginQsearch`, `rfpQuadratic` (referenced only from a commented line), `Board::setBit(BitBoard&, bool, int)` (ignores its `highLow` argument — two callers pass `true` expecting something), `fixedSearchTime` (`uci.cpp:82`), `MoveGenerator::getLegalMoves`, `Board::getGamePhase/setGamePhase`, `Score`-unused `enPassantIncrement`/promo-enum locals flagged by the compiler (full list in the warning appendix).
- `Material::pieceMaterialScoreArray` is `std::array<int, 14>` indexed by `BitBoardEnum` — `All` (14) is out of bounds. Every current call site guards against `All`, but `See::see`'s `score[0] = ...[board.getPieceOnSquare(toSq)]` (`see.h:20`) is one unguarded EP-shaped call away from OOB. Either size the array 15 with a 0 for `All`, or `assert(piece != All)`.
- `See::see` has no explicit king-recapture illegality handling (the `100000000` line is commented out, `see.h:96`): sequences where the king "recaptures" into a defended square are scored as if legal. Accuracy nit, standard simplification.
- `MoveUndoInfo` is `alignas(64)` → `moveHistory` is 64 KiB per `Board`. Data fits in 48 B; alignment doubles the footprint. Perf nit (see P-section).
- `Board` leaks `magicMovesRook/Bishop` (`new`, no delete, no destructor) — process-lifetime objects, harmless, but a `unique_ptr` costs nothing.
- `hasPositionRepeated` counter name `moveCounter` shadows the concept used in search; and `movegenerator.cpp` shadow warnings (`board`, `checkMask`, `kingSquare` parameters shadowing members) are the top compiler-warning cluster — see appendix.

### Performance observations (report only)

- **P1.** TT probe (`ttable.h:65-85`) copies the bucket entry field-by-field into a 24-byte `TTEntry` and is not prefetched. A `__builtin_prefetch(&table[index(key)])` immediately after `makeMove` (hash is final there) is the standard cheap win. Bucket `alignas(64)` (m2) is a prerequisite to make it one line, one cache line.
- **P2.** `evaluate()` is called through `Board::evaluate` → `NNUEQ::forward` → accumulators held in `std::vector<Accumulator>` → `std::vector<int16_t>` (`nnueq.h:17-20, 42`): two pointer chases and heap storage for what should be two flat `alignas(32) int16_t[256]` arrays inside `NNUEQ`. Also removes the `accumulator[0]/[1]` bounds-checking-hostile indirection in `addPiece`/`removePiece` (hottest function in the engine after movegen).
- **P3.** `MoveUndoInfo` `alignas(64)` (see nit) makes `moveHistory` 64 KiB — a make/unmake pair touches a fresh cache line each ply where 48-byte packing would share lines; measure before changing.
- **P4.** `isSearchStopped` calls `steady_clock::now` via `duration_cast` on every 1000th node — fine; but `search()` *also* calls `high_resolution_clock::now` per iteration and mixes the two clocks (`search.cpp:19, 82, 86` vs `818, 847`). Unify on `steady_clock` (correctness-adjacent: `high_resolution_clock` may be non-monotonic on MSVC-adjacent platforms).
- **P5.** `givesCheck` is computed by `isSquareAttacked` full scan after every `makeMove` (`search.cpp:374-377`) even though `makeMove` already computed `checkers` for the new position in `calculateCheckersSnipersPins` — `board.getCheckers() != 0` is free and identical. (Behaviour-neutral swap, but verify with bench.)
- **P6.** Staged movegen generates *all* noisy or quiet moves then sorts (`std::stable_sort`) even when the first move cuts off; selection-sort-on-demand (pick max per `next()`) is the usual pattern. Report-only; touches ordering timing, not results.
- **P7.** Bishop magic re-derivation at startup (see inventory): ~tens of ms of nondeterministic `Board` construction cost, per `Board` object (three exist in a `GenerateData` run per thread: worker board + any temporaries). Precompute like the rook table.

---

## Warning appendix (compilers available here)

Clang 18 `-Wall -Wextra -Wpedantic -Wshadow`: **81 warnings**; GCC 13 same flags: **86** (superset shape). Zero warnings is achievable with the quick-wins batch. Clusters:

| Cluster | Count (clang) | Representative sites |
|---|---|---|
| Unused variable / set-but-unused | ~45 | `movegenerator.cpp` promo enums ×8, `enemyBoard`/`allPieces`/`sideToMove` locals, `see.h:12,13,64` ×2 each (header included twice), `search.cpp:75,125,422,634-694`, `uci.cpp:82` |
| Shadow declarations | ~14 | `board.h:251` (`gamePhase` param, ×6 across TUs), `movegenerator.cpp:655,1188` (`checkMask`, `kingSquare`, `board` params shadow members), `see.h:108` (`magic`), `search.cpp:734` |
| Unused parameter | 7 | `movegenerator.cpp:144` (`pieceBoard`), `:1188` (`checkers`,`pinned`,`snipers`,`kingSquare`), `board.cpp:783` (`highLow`), `search.cpp` |
| Sign-compare | 6 | `uci.cpp:160,204` (`int i < vec.size()`), `search.cpp:825,854` (`evaluatedNodes > nodeLimit`), `movegenerator.cpp:160` (`move.from() == fromSq`, `uint32_t` vs `int`) |
| Unused function / private field | 2 | `board.cpp:9` (`initSqToBitMapping`), `search.h:94` (`clearTTOnSearch`) |
| GCC-only | 5 | `board.cpp:115,117` ×4 (magic constants `> INT64_MAX` need `ull` suffix), `ttable.h:61` (`memset` on non-trivial `Bucket`) |
| Pedantic | 1 | `ttable.h:19` (`unsigned __int128` is a GNU extension — expected, keep behind the existing `#if`) |

`-Wconversion` adds **340** (316 `-Wsign-conversion`, 15 `-Wimplicit-int-conversion`, 4 `-Wfloat-conversion`, 4 `-Wimplicitly-unsigned-literal`), concentrated in `movegenerator.cpp` (137), `board.cpp`/`board.h` (104), `see.h` (32), `search.cpp` (27). Root causes: `BitBoardEnum` arithmetic (`P + sideToMove`), `Move::from()/to()` returning `uint32_t` mixed with `int` squares, and `size_t` vs `int` loop indices. Recommendation: do **not** chase all 340 in the hardening pass; fix the 15+4+4 genuine narrowing/float ones and the sign-compare subset above, and standardise new code on `int` squares / explicit casts at enum arithmetic. (MSVC `/W4` could not be run in this environment; the clang-cl preset in `CMakeSettings.json` will substantially mirror the clang list.)

---

## Verification baseline (captured during this audit)

Built with Clang 18 `-O2 -march=x86-64-v3 -std=c++23` (local build used a byte-array
shim in place of `#embed`, same net bytes; repo files untouched):

- **Perft**: all 6 suite positions pass (`Position 1` d7 … `Position 6` d6).
- **Bench**: `8380018 nodes`, identical across two consecutive runs — bench is
  deterministic and usable as the behaviour-neutrality gate for Phase 4.
  (Bishop magics being randomised per startup does not affect node counts, as
  the resulting attack tables are identical.)

---

## Phase 3.3 — Proposed commit plan

Smallest first; one logical change per commit. "Bench-neutral" = `bench` must print the identical node count; verified after every commit along with a clean build and the 6-position perft suite.

| # | Commit | Contents | Can change search behaviour? |
|---|---|---|---|
| 1 | `chore: remove dead code` | `Search::see`/`getPinned`/`equal`, `TranspositionEntry`, `dperft` family + undefined statics, `initSqToBitMapping`×2, unused Material tables, unused params, commented-out blocks, debug fossils (m19 board/uci strings), `fixedSearchTime`, `clearTTOnSearch` | No (bench-neutral by construction) |
| 2 | `chore: fix all -Wall -Wextra -Wshadow warnings` | unused locals/params, shadow renames, sign-compare casts, `ull` suffixes, `memset`→value-init in `TTable::clear` | No |
| 3 | `chore: header hygiene` | drop `using namespace NNUE;` from header, `#define BitBoard`→`using`, add missing direct includes (`<cstdint>`, `<memory>` in `search.h`, etc.) | No |
| 4 | `fix: UCI robustness` | C5 EOF exit, m12 `stoi` guards, m16 initialisers, m15 loop bound, m21 version string | No |
| 5 | `fix: time management floor and cap` | C1 + C2 (clamp budget to `[1, time−overhead]`, side-to-move-only clocks, `movestogo`) | Time-based decisions only; fixed-depth bench identical |
| 6 | `fix: never abort depth-1; legal fallback move` | C4 | Only in stopped-search endgames; bench (fixed depth) identical |
| 7 | `fix: make/unmake symmetry for clocks` | M5 (`int16_t` clocks, null-move fullMoveClock) | No (values only feed FEN/tool paths) |
| 8 | `fix: GenerateData tool` | M6, M9, M10 (tool-only files + `hasPositionRepeated` bound m5 optional) | Engine untouched |
| 9 | `build: CMake release defaults + compiler checks` | M7, M8 | No (same flags as intended release) |
| 10 | `fix: TT mate-score ply adjustment` | M1 (+ m1 depth type, m7 mate display) | **Yes — bench nodes will change; isolated commit** |
| 11 | `fix: proper side-to-move Zobrist key` | M4 | **Yes — bench nodes will change; isolated commit** |
| 12 | `fix: widen quiet history to int32` | M2 | **Yes (ordering) — isolated commit** |
| 13 | `fix: NNUE wrapping accumulator ops` | M3 | Expected no (verify bench identical; if it changes, saturation was occurring — stop and report) |
| 14 | `fix: stop/infinite support` (search thread + atomic stop) | C3 | Timing-dependent only; bench identical |

Recommended checkpoint after commit 4 (pure hygiene), and after commit 9 (all bench-neutral fixes) before starting the behaviour-changing tail (10-14). Items m4, m6, m8, m10, M11, P5, P6 and the "score first repetition as draw" proposal are **deliberately left out** — they change behaviour for debatable benefit and belong in a post-2.0 strength pass with SPRT testing.
