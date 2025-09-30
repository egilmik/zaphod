# convert_txt_to_bin.py
import math, sys
import numpy as np

IN_FEATS   = 768
MAX_IDXS   = 48
PAD        = 0xFFFF
SCALE      = 600.0

REC_DTYPE = np.dtype([('idx','<u2',(MAX_IDXS,)), ('k','<u2'), ('y','<f4')])

def parse(line: str):
    s = line.strip()
    if not s or s.startswith('#') or ';' not in s: return None
    left, right = s.split(';', 1)
    try:
        cp = float(right.strip())
        idxs = [int(t) for t in left.split() if t]
    except ValueError:
        return None
    # filter + (optionally) dedup/sort
    idxs = [i for i in idxs if 0 <= i < IN_FEATS]
    if not idxs or len(idxs) > MAX_IDXS:  # drop oversize rows
        return None
    y = math.tanh(cp / SCALE)
    return idxs, y

def convert(txt_path: str, bin_path: str, chunk=1_000_000):
    buf = np.empty(chunk, dtype=REC_DTYPE)
    n = 0
    with open(txt_path, 'r', encoding='utf-8') as f, open(bin_path, 'wb') as out:
        for line in f:
            p = parse(line)
            if not p: continue
            idxs, y = p
            rec = buf[n]
            rec['k'] = len(idxs)
            rec['idx'].fill(PAD)
            rec['idx'][:len(idxs)] = np.asarray(idxs, dtype=np.uint16)
            rec['y'] = np.float32(y)
            n += 1
            if n == chunk:
                buf.tofile(out); n = 0
        if n:
            buf[:n].tofile(out)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_txt_to_bin.py <in.txt> <out.bin>")
        sys.exit(1)

    print(sys.argv[1])
    convert(sys.argv[1], sys.argv[2])
