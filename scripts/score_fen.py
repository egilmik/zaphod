import argparse, os, sys, gzip, shutil, signal
from multiprocessing import Process, Queue, Value
from pathlib import Path
import chess
import chess.engine as uci

SENTINEL = None

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen_file", required=True, help="input .txt(.gz) with one FEN per line")
    ap.add_argument("--engine", required=True, help="path to UCI engine (e.g., stockfish)")
    ap.add_argument("--out", required=True, help="output file (aggregated).gz ok")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--hash", type=int, default=512, help="MB per engine")
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--nodes", type=int, default=50000)
    ap.add_argument("--movetime_ms", type=int, default=None)
    ap.add_argument("--threads_per_engine", type=int, default=1)
    ap.add_argument("--cp_cap", type=int, default=3200)
    ap.add_argument("--drop_mates", action="store_true")
    return ap.parse_args()

def open_read(path):
    return gzip.open(path, "rt") #if path.endswith(".gz") else open(path, "r")

def open_write(path):
    return gzip.open(path, "at") #if path.endswith(".gz") else open(path, "a")

def score_to_cp(score, stm_white: bool, cp_cap: int, drop_mates: bool):
    # score is a chess.engine.PovScore or Score; normalize to stm POV cp
    if score.is_mate():
        if drop_mates:
            return None
        # Positive = mate for the side to move sooner is better
        ply = score.mate()  # positive if mate for side to move, negative otherwise
        # Map mate distance to large cp, closer is larger magnitude
        cp = 30000 if ply is None else max(10000, 10000 - abs(ply)*10)
        cp = cp if ply and ply > 0 else -cp
    else:
        cp = score.pov(chess.WHITE).score()  # cp from White POV
        # convert to STM POV
        cp = cp if stm_white else -cp
    # clamp
    if cp is not None:
        if cp >  cp_cap: cp =  cp_cap
        if cp < -cp_cap: cp = -cp_cap
    return cp

def worker(proc_id, q: Queue, done: Value, args, tmp_path):
    # start engine
    engine = uci.SimpleEngine.popen_uci(args.engine)
    try:
        engine.configure({
            "Threads": args.threads_per_engine,
            "Hash": args.hash,
            "SyzygyPath": "",       # ensure off
            "Use NNUE": "true",     # or "false" if HCE; depends on your engine
            "Ponder": "false",
        })
    except Exception:
        pass

    counter = 0
    with open_write(tmp_path) as out:
        while True:
            item = q.get()
            if item is SENTINEL:
                break
            idx, fen = item
            fen = fen.strip()
            if not fen: 
                continue
            try:
                board = chess.Board(fen)
            except Exception as e:
                continue
            try:
                
                info = engine.analyse(board, chess.engine.Limit(depth=args.depth))
                # Normalize to STM POV
                stm_white = (board.turn == chess.WHITE)
                cp = score_to_cp(info["score"], stm_white, args.cp_cap, args.drop_mates)
                if cp is None:
                    continue
                out.write(f"{fen} ; {cp}\n")
                counter += 1
                if counter % 10000 == 0:
                    print(counter)
                    print(len(q))

            except Exception as e:
                continue
            
    engine.quit()
    with done.get_lock():
        done.value += 1

def main():
    args = parse_args()
    fen_path = Path(args.fen_file)
    out_path = Path(args.out)

    # temp shard files
    tmp_dir = out_path.parent / (out_path.name + ".shards")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ext = ".gz" if str(args.out).endswith(".gz") else ""
    shard_paths = [tmp_dir / f"part_{i}.txt{ext}" for i in range(args.workers)]

    q = Queue(maxsize=args.workers * 4)
    done = Value('i', 0)

    procs = []
    for i in range(args.workers):
        p = Process(target=worker, args=(i, q, done, args, shard_paths[i]))
        p.daemon = True
        p.start()
        procs.append(p)

    # graceful Ctrl+C: stop feeding, wait workers to drain
    stop = {"flag": False}
    def _sigint(_sig, _frm): stop["flag"] = True
    signal.signal(signal.SIGINT, _sigint)

    # producer
    with open_read(str(fen_path)) as f:
        for idx, line in enumerate(f):
            if stop["flag"]:
                break
            q.put((idx, line))
    # tell workers to stop
    for _ in procs:
        q.put(SENTINEL)
    for p in procs:
        p.join()

    # concatenate shards
    if out_path.exists():
        out_path.unlink()
    mode = "wb" if str(out_path).endswith(".gz") else "wb"
    with open(out_path, mode) as final_out:
        for sp in shard_paths:
            if not Path(sp).exists():
                continue
            with open(sp, "rb") as shard:
                shutil.copyfileobj(shard, final_out)
            os.remove(sp)
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

if __name__ == "__main__":
    main()
