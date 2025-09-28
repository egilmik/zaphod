# nnue_768x32x1_trainer_idx_stream.py
import math, random
from typing import List, Tuple, Iterable, Optional
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

IN_FEATS   = 768
HIDDEN     = 128

# -------------------------
# Config
# -------------------------
TRAIN_PATHS       = ["D:/source/zaphod_nnue/Data/1.9_data/1.9_11M_depth6_STM.txt","D:/source/zaphod_nnue/Data/1.9_data/1.9_113M_depth4_STM.txt","D:/source/zaphod_nnue/Data/1.9_data/1.9_108M_depth4_STM.txt","D:/source/zaphod_nnue/Data/1.9_data/1.9_88M_depth4_STM.txt"]  # add more shards here
VALIDATION_PATHS  = ["D:/source/zaphod_nnue/Data/1.9_data/1.9_5M_depth4_STM.txt"]  # add more shards here
EPOCHS            = 20
BATCH_SIZE        = 4096
LR                = 1e-3
TARGET_CP_SCALE   = 600.0              # tanh(cp/scale)
NUM_WORKERS       = 10                  # streaming works best with workers
PREFETCH_FACTOR   = 2                  # keep low to reduce RAM
PIN_MEMORY        = True
SEED              = 7
SAVE_PATH         = "nnue_768x32x1.pt"
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
SHUFFLE_BUFFER    = 100_000            # memory-bounded shuffle (≈ lines in RAM)

# -------------------------
# Parser
# -------------------------
def parse_indices_score(line: str) -> Optional[Tuple[list, float]]:
    s = line.strip()
    if not s or s.startswith("#") or ";" not in s:
        return None
    left, right = s.split(";", 1)
    try:
        cp = float(right.strip())
    except ValueError:
        return None
    try:
        idxs = [int(tok) for tok in left.split() if tok]
    except ValueError:
        return None
    # validate + dedup + sort
    #idxs = sorted({i for i in idxs if 0 <= i < IN_FEATS})
    if not idxs:
        return None
    return idxs, math.tanh(cp / TARGET_CP_SCALE)  # store target already transformed

# -------------------------
# Iterable Dataset with bounded shuffling
# -------------------------
class IdxScoreStreamDataset(IterableDataset):
    def __init__(self, paths: List[str], seed: int, shuffle_buffer: int):
        super().__init__()
        self.paths = list(paths)
        self.base_seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.epoch = 0

    def set_epoch(self, e: int):
        self.epoch = e

    def _line_iter(self, rng: random.Random) -> Iterable[str]:
        paths = self.paths[:]
        rng.shuffle(paths)
        for p in paths:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    yield line

    def __iter__(self):
        wi = get_worker_info()
        wid = wi.id if wi is not None else 0
        workers = wi.num_workers if wi is not None else 1

        # distinct RNG per worker+epoch
        seed = (self.base_seed ^ (self.epoch * 0x9E3779B97F4A7C15) ^ (wid * 0x85ebca6b)) & 0xFFFFFFFF
        rng = random.Random(seed)

        # bounded shuffle buffer of parsed samples
        buf: list[Tuple[list, float]] = []

        it = self._line_iter(rng)
        # Pre-fill buffer
        while len(buf) < self.shuffle_buffer:
            try:
                line = next(it)
            except StopIteration:
                break
            item = parse_indices_score(line)
            if item is None:
                continue
            # shard by worker (cheap modulo to split the stream)
            if (hash(line) & 0x7FFFFFFF) % workers != wid:
                continue
            buf.append(item)

        # Reservoir-style streaming with random pops
        for line in it:
            item = parse_indices_score(line)
            if item is None:
                continue
            if (hash(line) & 0x7FFFFFFF) % workers != wid:
                continue
            if buf:
                j = rng.randrange(len(buf))
                yield buf[j]
                buf[j] = item
            else:
                buf.append(item)

        # drain buffer
        while buf:
            j = rng.randrange(len(buf))
            yield buf.pop(j)

# -------------------------
# Collate: indices -> dense batch
# -------------------------
def collate_indices_to_dense(batch: list[Tuple[list, float]]):
    B = len(batch)
    X = torch.zeros((B, IN_FEATS), dtype=torch.float32)
    y = torch.empty((B, 1), dtype=torch.float32)
    for i, (idxs, tgt) in enumerate(batch):
        if idxs:
            X[i, idxs] = 1.0
        y[i, 0] = tgt
    return X, y

# -------------------------
# Model
# -------------------------
class NNUE_768x32x1(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(IN_FEATS, HIDDEN, bias=True)
        self.l2 = nn.Linear(HIDDEN, 1, bias=True)
        # Clipped Relu
        self.act = nn.Hardtanh(min_val=0.0, max_val=1.0, inplace=True)
        self._init()
    def _init(self):
        nn.init.kaiming_uniform_(self.l1.weight, a=0.0)
        nn.init.zeros_(self.l1.bias)
        nn.init.uniform_(self.l2.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.l2.bias)
    def forward(self, x):
        return self.l2(self.act(self.l1(x)))

# -------------------------
# Train
# -------------------------
def main():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    print(f"Device: {DEVICE}")

    # Streaming train dataset
    tr_ds = IdxScoreStreamDataset(TRAIN_PATHS, seed=SEED, shuffle_buffer=SHUFFLE_BUFFER)
    va_ds = IdxScoreStreamDataset(VALIDATION_PATHS, seed=SEED, shuffle_buffer=SHUFFLE_BUFFER)

    tr_loader = DataLoader(
        tr_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR,
        pin_memory=PIN_MEMORY, persistent_workers=(NUM_WORKERS>0),
        collate_fn=collate_indices_to_dense,
    )
    va_loader = DataLoader(
        va_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=PIN_MEMORY,
        collate_fn=collate_indices_to_dense,
    )

    print("Starting training")
    model = NNUE_768x32x1().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_va = float('inf')
    for ep in range(1, EPOCHS+1):
        tr_ds.set_epoch(ep)
        print(f"Epoch {ep:02d}")
        # Train
        model.train()
        tr_loss_sum = tr_cnt = 0
       
        for xb, yb in tqdm(tr_loader, desc=f"Train {ep:02d}", leave=False):
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tr_loss_sum += loss.item() * yb.size(0)
            tr_cnt      += yb.size(0)

        # Validation
        va_loss_sum = va_cnt = 0
        with torch.no_grad():
            for xb, yb in tqdm(va_loader, desc=f"Val {ep:02d}", leave=False):
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                loss = loss_fn(model(xb), yb)
                va_loss_sum += loss.item() * yb.size(0)
                va_cnt      += yb.size(0)

        tr_loss = tr_loss_sum / max(tr_cnt, 1)
        va_loss = va_loss_sum / max(va_cnt, 1)
        print(f"Epoch {ep:02d} | train MSE={tr_loss:.6f} | val MSE={va_loss:.6f}")
        
        if va_loss < best_va:
            best_va = va_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "score_from_stm": True,           # indices are white-POV
                    "target_cp_scale": TARGET_CP_SCALE,
                    "arch": "768x32x1",
                }
            }, SAVE_PATH)
            print(f"  -> saved: {SAVE_PATH}")

    print("Done.")

if __name__ == "__main__":
    main()
