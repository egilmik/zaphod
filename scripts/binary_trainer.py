

import sys
from typing import List
from tqdm import tqdm
import numpy as np
import struct
import torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import os

IN_FEATS   = 768
HIDDEN     = 256

# -------------------------
# Config
# -------------------------
TRAIN_PATHS       = [
    "D:\\source\\zaphod_nnue\\Data\\1.9_113M_depth4_STM.bin",
    "D:\\source\\zaphod_nnue\\Data\\1.9_11M_depth6_STM.bin",
    "D:\\source\\zaphod_nnue\\Data\\1.9_88M_depth4_STM.bin",
    "D:\\source\\zaphod_nnue\\Data\\1.9_108M_depth4_STM.bin",
    "D:\\source\\zaphod_nnue\\Data\\2.0_dev_83M_depth4.bin",
    "D:\\source\\zaphod_nnue\\Data\\2.0_dev_3e54e67d_74M_depth6.bin",
    "D:\\source\\zaphod_nnue\\Data\\2.0_dev_3e54e67d_12M_depth6.bin",
    "D:\\source\\zaphod_nnue\\Data\\2.0_dev_52a7cc16_215M_depth4.bin",
    
]
VALIDATION_PATHS  = ["D:\\source\\zaphod_nnue\\Data\\2.0_dev_50M_depth4.bin"]

EPOCHS            = 30
BATCH_SIZE        = 8192
LR                = 1e-3
NUM_WORKERS       = 10
PREFETCH_FACTOR   = 2
PIN_MEMORY        = True
SEED              = 7
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
SHUFFLE           = True  # shuffle within each shard via index permutation

# AMP
AMP_DTYPE = (torch.bfloat16
             if (DEVICE == "cuda" and torch.cuda.is_bf16_supported())
             else torch.float16)
USE_AMP = (DEVICE == "cuda")
torch.backends.cuda.matmul.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# -------------------------
# Binary record (must match converter)
# -------------------------
PAD      = 0xFFFF
MAX_IDXS = 48
DTYPE_102 = np.dtype([('idx','<u2',(MAX_IDXS,)), ('k','<u2'), ('y','<f4')])               # 102 bytes
DTYPE_104 = np.dtype([('idx','<u2',(MAX_IDXS,)), ('k','<u2'), ('pad','<u2'), ('y','<f4')]) # 104 bytes

def infer_rec_dtype(path: str):
    sz = os.path.getsize(path)
    if sz % DTYPE_104.itemsize == 0:
        return DTYPE_104
    if sz % DTYPE_102.itemsize == 0:
        return DTYPE_102
    raise ValueError(
        f"{path}: file size {sz} is not a multiple of 102 or 104. "
        "The shard may be truncated or written with a different layout."
    )

# -------------------------
# Dataset: memmap + pre-batched yields for EmbeddingBag
# Yields: (flat_idx[int64], offsets[int64], y[float32 Bx1])
# -------------------------
class BinIdxStream(IterableDataset):
    def __init__(self, paths: List[str], batch_records: int, shuffle: bool, seed: int):
        super().__init__()
        self.paths = list(paths)
        self.batch_records = batch_records
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, e: int): self.epoch = e

    def __iter__(self):
        wi = get_worker_info()
        wid = wi.id if wi else 0
        nw  = wi.num_workers if wi else 1

        base_seed = (self.seed ^ (self.epoch * 0x9E3779B97F4A7C15) ^ (wid * 0x85ebca6b)) & 0xFFFFFFFF

        for p_i, p in enumerate(self.paths):
            #prec_dtype = infer_rec_dtype(p)              # handles 102B vs 104B
            mm = np.memmap(p, mode='r', dtype=DTYPE_102)
            n  = mm.shape[0]
            if n == 0:
                continue

            # stride sharding
            idx = np.arange(wid, n, nw, dtype=np.int64)
            if idx.size == 0:
                continue

            if self.shuffle:
                rng = np.random.default_rng(base_seed ^ (p_i * 0x9E3779B97F4A7C15))
                rng.shuffle(idx)

            # pre-batched emission
            for s in range(0, idx.size, self.batch_records):
                sel = idx[s:s + self.batch_records]     # <-- define sel
                batch = mm[sel]                         # <-- now valid

                B = batch.shape[0]
                k = batch['k'].astype(np.int64)

                # offsets
                off = np.empty(B + 1, dtype=np.int64); off[0] = 0
                np.cumsum(k, out=off[1:])
                nnz = int(off[-1])

                # flat indices
                flat = np.empty(nnz, dtype=np.int64)
                src  = batch['idx']
                pos = 0
                for r in range(B):
                    cnt = int(k[r])
                    if cnt:
                        flat[pos:pos+cnt] = src[r, :cnt]
                        pos += cnt

                # targets (handle 102B vs 104B stride)
                y_view = batch['y']                     # shape (B,)
                #if rec_dtype is DTYPE_102:
                y = np.ascontiguousarray(y_view).reshape(B, 1)
                #else:
                #    y = y_view.reshape(B, 1)

                yield (torch.from_numpy(flat),torch.from_numpy(off),torch.from_numpy(y))


# -------------------------
# Model: EmbeddingBag + clipped ReLU + head
# -------------------------
class NNUE_EB_768x128x1(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.EmbeddingBag(IN_FEATS, HIDDEN, mode="sum", include_last_offset=True, sparse=True)
        nn.init.kaiming_uniform_(self.emb.weight, a=0.0)
        self.b1 = nn.Parameter(torch.zeros(HIDDEN))
        self.act = nn.Hardtanh(min_val=0.0, max_val=1.0, inplace=True)
        self.l2  = nn.Linear(HIDDEN, 1, bias=True)
        nn.init.uniform_(self.l2.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.l2.bias)

    def forward(self, idxs: torch.Tensor, offsets: torch.Tensor):
        h = self.emb(idxs, offsets) + self.b1
        h = self.act(h)
        return self.l2(h)

def quantizeModel(model, outputFile: str):
    
    sd   = model["model_state_dict"]
    cfg  = model.get("config", {})
    eval_scale = float(cfg.get("target_cp_scale", 600.0))

    # Float weights
    w1 = sd["emb.weight"].cpu().numpy().astype(np.float32).T   # (H, IN)
    b1 = sd["b1"  ].cpu().numpy().astype(np.float32)   # (H,)
    w2 = sd["l2.weight"].cpu().numpy().astype(np.float32).reshape(HIDDEN)  # (H,)
    b2 = sd["l2.bias"  ].cpu().numpy().astype(np.float32)[0]

    # ---------- L1: per-channel int8 ----------
    absmax_w1 = np.maximum(np.max(np.abs(w1), axis=1), 1e-12).astype(np.float32)
    s1 = absmax_w1 / 127.0                                  # (H,)
    W1_q = np.clip(np.rint(w1 / s1[:, None]), -127, 127).astype(np.int8)       # (H, IN)
    B1_q = np.rint(b1 / s1).astype(np.int32)                                        # (H,)

    # ---------- Hidden activation quant (clipped ReLU to [0,1]) ----------
    # Use full 7-bit grid over [0,1]:
    a1 = 1.0 / 127.0
    Q_CAP = int(np.floor(1.0 / a1))  # = 127

    # ---------- L2: symmetric int8 (single scale) ----------
    absmax_w2 = max(float(np.max(np.abs(w2))), 1e-12)
    s2 = absmax_w2 / 127.0
    W2_q = np.clip(np.rint(w2 / s2), -127, 127).astype(np.int8)

    B2_f = float(b2)  # keep in float

    # ---------- Write compact binary ----------
    # Layout (unchanged except hidden=32 and Q_CAP appended):
    # magic(8)="NNUEQ1\0\0"
    # dims: int32 (in=768, hidden=32, out=1)
    # eval_scale: float32
    # L1: s1[H] float32, W1_q int8[H*IN], B1_q int32[H]
    # Act: a1 float32, Q_CAP uint8
    # L2: s2 float32, W2_q int8[H], B2_f float32
    with open(outputFile, "wb") as f:
        f.write(b"NNUEQ1\0\0")
        f.write(struct.pack("<iii", IN_FEATS, HIDDEN, 1))   # FIXED: 32 not 128
        f.write(struct.pack("<f", eval_scale))

        f.write(s1.astype("<f4").tobytes(order="C"))
        f.write(W1_q.tobytes(order="C"))
        f.write(B1_q.astype("<i4").tobytes(order="C"))

        f.write(struct.pack("<f", float(a1)))
        f.write(struct.pack("<B", Q_CAP))                   # new: integer cap for q

        f.write(struct.pack("<f", float(s2)))
        f.write(W2_q.tobytes(order="C"))
        f.write(struct.pack("<f", B2_f))

    print(f"wrote {outputFile}  (H={HIDDEN}, a1={a1:.6g}, Q_CAP={Q_CAP}, s2={s2:.6g}, eval_scale={eval_scale})")

# -------------------------
# Train
# -------------------------
def main(output_dir: str):
    torch.manual_seed(SEED)
    if DEVICE == "cuda": torch.cuda.manual_seed_all(SEED)
    print(f"Device: {DEVICE}")

    tr_ds = BinIdxStream(TRAIN_PATHS, batch_records=BATCH_SIZE, shuffle=SHUFFLE, seed=SEED)
    va_ds = BinIdxStream(VALIDATION_PATHS, batch_records=BATCH_SIZE, shuffle=False, seed=SEED)

    # pre-batched dataset => trivial collate (take item)
    collate = (lambda x: x[0])

    tr_loader = DataLoader(
    tr_ds, batch_size=None, shuffle=False,
    num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR,
    pin_memory=PIN_MEMORY, persistent_workers=(NUM_WORKERS > 0))

    va_loader = DataLoader(
        va_ds, batch_size=None, shuffle=False,
        num_workers=max(1, NUM_WORKERS // 2), pin_memory=PIN_MEMORY)
    model = NNUE_EB_768x128x1().to(DEVICE)
    #try:
    #    model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    #except Exception:
    #    pass

    sparse_params = list(model.emb.parameters())
    dense_params  = [p for n, p in model.named_parameters() if not n.startswith("emb.")]
    opt_sparse = torch.optim.Adagrad(sparse_params, lr=2e-2)
    opt_dense  = torch.optim.Adam(dense_params,  lr=LR)
    loss_fn = nn.MSELoss()

    scaler = torch.amp.GradScaler("cuda",enabled=(USE_AMP and AMP_DTYPE == torch.float16))
    best_va = float("inf")

    for ep in range(1, EPOCHS + 1):
        tr_ds.set_epoch(ep); va_ds.set_epoch(ep)

        # ---- Train ----
        model.train()
        tr_loss_sum = tr_cnt = 0
        pbar = tqdm(tr_loader, desc=f"Train {ep:02d}", leave=False)

        for idxs, offs, yb in pbar:
            idxs = idxs.to(DEVICE, non_blocking=True)
            offs = offs.to(DEVICE, non_blocking=True)
            yb   = yb.to(DEVICE, non_blocking=True)

            opt_sparse.zero_grad(set_to_none=True)
            opt_dense.zero_grad(set_to_none=True)

            with torch.autocast(device_type=("cuda" if DEVICE=="cuda" else "cpu"),
                                dtype=AMP_DTYPE, enabled=USE_AMP):
                pred = model(idxs, offs)
                loss = loss_fn(pred, yb)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt_sparse)
                scaler.step(opt_dense)
                scaler.update()
            else:
                loss.backward()
                opt_sparse.step()
                opt_dense.step()

            tr_loss_sum += loss.item() * yb.size(0)
            tr_cnt      += yb.size(0)

        # ---- Validate ----
        model.eval()
        va_loss_sum = va_cnt = 0
        with torch.no_grad(), torch.autocast(device_type=("cuda" if DEVICE=="cuda" else "cpu"),
                                             dtype=AMP_DTYPE, enabled=USE_AMP):
            for idxs, offs, yb in tqdm(va_loader, desc=f"Val {ep:02d}", leave=False):
                idxs = idxs.to(DEVICE, non_blocking=True)
                offs = offs.to(DEVICE, non_blocking=True)
                yb   = yb.to(DEVICE, non_blocking=True)
                pred = model(idxs, offs)
                loss = loss_fn(pred, yb)
                va_loss_sum += loss.item() * yb.size(0)
                va_cnt      += yb.size(0)

        tr_loss = tr_loss_sum / max(tr_cnt, 1)
        va_loss = va_loss_sum / max(va_cnt, 1)
        elapsed_time = pbar.format_dict["elapsed"]
        print(f"Epoch {ep:02d} | train MSE={tr_loss:.6f} | val MSE={va_loss:.6f} | elapsed time{elapsed_time:.2f}s")
        
        

        if va_loss < best_va:
            outputFile = output_dir + str(ep) + ".pt"
            quantizedOutput = output_dir + str(ep) + ".nnue"
            best_va = va_loss
            ckpt ={
                "model_state_dict": (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict(),
                "config": {
                    "score_from_stm": True,
                    "target_cp_scale": 600.0,
                    "arch": "768x128x1-eb-bin",
                    "record": {"idx": MAX_IDXS, "pad": PAD, "dtype": "u16,u16,f32"},
                }
            }
            torch.save(ckpt,outputFile)
            print(f"  -> saved: {outputFile}")
            quantizeModel(ckpt, quantizedOutput)

    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: py binary_trainer.py <output directory>")
        sys.exit(1)
    print(sys.argv[1])
    main(sys.argv[1])
