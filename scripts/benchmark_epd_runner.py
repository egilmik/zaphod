import chess
import chess.engine
import chess.pgn
import argparse
import statistics
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

    run_tests(args.engine, epd_positions, depth=args.depth, movetime=args.movetime)
