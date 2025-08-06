import chess
import chess.engine
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import statistics
import time
import sys

def load_epd_positions(epd_path):
    positions = []
    with open(epd_path, "r") as f:
        for line in f:
            try:
                board = chess.Board()
                ops = board.set_epd(line)
                if "bm" in ops:
                    # Take first best move only for now
                    bm_move = ops["bm"][0]
                    positions.append((board, bm_move))
            except Exception as e:
                print(f"Skipping invalid EPD line: {line.strip()} — {e}")
    return positions

def run_tests(engine_path, positions, depth=None, movetime=None):
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    passed = 0
    failed = 0
    nps_list = []
    nodes_list = []

    for idx, (board, expected_move) in enumerate(positions):
        try:
            if depth:
                info = engine.analyse(board, chess.engine.Limit(depth=depth), info=chess.engine.INFO_ALL)
                actual_move = info.get("pv", [None])[0]
            else:
                info = engine.analyse(board, chess.engine.Limit(time=movetime / 1000.0), info=chess.engine.INFO_ALL)
                actual_move = info.get("pv", [None])[0]

            nodes = info.get("nodes", 0)
            nps = info.get("nps", 0)

            passed_flag = actual_move == expected_move
            result_str = "PASS" if passed_flag else "FAIL"
            print(f"{idx+1:03d}: {result_str} — Expected: {expected_move.uci()}, Got: {actual_move.uci()} | Nodes: {nodes}, NPS: {nps}")

            if passed_flag:
                passed += 1
            else:
                failed += 1

            if nodes:
                nodes_list.append(nodes)
            if nps:
                nps_list.append(nps)

        except Exception as e:
            print(f"{idx+1:03d}: ERROR — {e}")
            failed += 1

    engine.quit()

    print("\n=== Summary ===")
    print(f"Total positions: {len(positions)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Avg NPS: {int(statistics.mean(nps_list)) if nps_list else 0}")
    print(f"Avg Nodes: {int(statistics.mean(nodes_list)) if nodes_list else 0}")


def run_tests_multithreaded(engine_path, positions, depth=None, movetime=None, threads=4):
    start = time.time()

    chunk_size = len(positions) // threads + 1
    print(f"Chunk size {chunk_size}")
    chunks = [positions[i:i + chunk_size] for i in range(0, len(positions), chunk_size)]

    def run_chunk(chunk, thread_id):
        results = []
        try:
            engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        except Exception as e:
            print(f"[Thread {thread_id}] Failed to launch engine: {e}")
            return results

        for idx, (board, expected_move) in enumerate(chunk):
            try:
                if depth:
                    result = engine.play(board, chess.engine.Limit(depth=depth))
                    info = engine.analyse(board, chess.engine.Limit(depth=depth), info=chess.engine.INFO_ALL)
                else:
                    result = engine.play(board, chess.engine.Limit(time=movetime / 1000.0))
                    info = engine.analyse(board, chess.engine.Limit(time=movetime / 1000.0), info=chess.engine.INFO_ALL)

                actual_move = result.move
                nodes = info.get("nodes", 0)
                nps = info.get("nps", 0)
                passed = actual_move == expected_move

                #result_str = "PASS" if passed else "FAIL"
                #print(f"{idx+1:03d}: {result_str} — Expected: {expected_move.uci()}, Got: {actual_move.uci()} | Nodes: {nodes}, NPS: {nps}")
                results.append({
                    "passed": passed,
                    "nodes": nodes,
                    "nps": nps,
                    "expected": expected_move.uci(),
                    "actual": actual_move.uci(),
                })

            except Exception as e:
                print(f"[Thread {thread_id}] Error on position: {e}")
                results.append({
                    "passed": False,
                    "nodes": 0,
                    "nps": 0,
                    "expected": expected_move.uci(),
                    "actual": "error",
                })

        engine.quit()
        return results

    # Start threads
    all_results = []
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(run_chunk, chunk, i) for i, chunk in enumerate(chunks)]
        for f in as_completed(futures):
            all_results.extend(f.result())

    # Aggregate stats
    passed = sum(1 for r in all_results if r["passed"])
    failed = len(all_results) - passed
    avg_nps = int(statistics.mean([r["nps"] for r in all_results if r["nps"]])) if all_results else 0
    avg_nodes = int(statistics.mean([r["nodes"] for r in all_results if r["nodes"]])) if all_results else 0

    for i, r in enumerate(all_results):
        status = "PASS" if r["passed"] else "FAIL"
        print(f"{i+1:03d}: {status} — Expected: {r['expected']}, Got: {r['actual']}, Nodes: {r['nodes']}, NPS: {r['nps']}")

    print("\n=== Summary ===")
    print(f"Total positions: {len(all_results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Avg NPS: {avg_nps}")
    print(f"Avg Nodes: {avg_nodes}")
    end = time.time()
    print(f"\n⏱️ Total elapsed time: {end - start:.2f} seconds")

# ------------------------------
# CLI
# ------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Zaphod on EPD file using python-chess")
    parser.add_argument("--engine", required=True, help="Path to Zaphod UCI executable")
    parser.add_argument("--epd", required=True, help="Path to EPD file")
    parser.add_argument("--depth", type=int, help="Fixed search depth")
    parser.add_argument("--movetime", type=int, help="Fixed movetime in milliseconds")

    args = parser.parse_args()

    if not args.depth and not args.movetime:
        print("Error: You must specify either --depth or --movetime")
        sys.exit(1)

    print(f"Loading EPD positions from {args.epd}...")
    epd_positions = load_epd_positions(args.epd)
    print(f"Loaded {len(epd_positions)} positions.")

    run_tests_multithreaded(args.engine, epd_positions, depth=args.depth, movetime=args.movetime)
