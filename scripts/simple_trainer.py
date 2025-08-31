# nnue_768x32x1_trainer_chess.py
import math, os, random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import chess  # <-- python-chess

# -------------------------
# Config
# -------------------------
DATA_PATH         = "data.txt"   # each line: "<FEN> ; <score_cp>"
EPOCHS            = 3
BATCH_SIZE        = 4096
LR                = 1e-3
VAL_SPLIT         = 0.05
SCORE_FROM_STM    = True    # flip target if side-to-move is black
TARGET_CP_SCALE   = 600.0   # tanh(cp / scale)
NUM_WORKERS       = 0
SEED              = 7
SAVE_PATH         = "nnue_768x32x1.pt"
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Encoding via python-chess
# Planes: [P,N,B,R,Q,K,p,n,b,r,q,k]; squares A1=0..H8=63 (python-chess default)
# -------------------------
def encode_board_12x64_flat(board: chess.Board) -> np.ndarray:
    planes = np.zeros((12, 64), dtype=np.float32)
    # piece_type: 1..6 (PAWN..KING); color: True=WHITE, False=BLACK
    for sq, pc in board.piece_map().items():
        base = 0 if pc.color == chess.WHITE else 6
        plane = base + (pc.piece_type - 1)
        planes[plane, sq] = 1.0
    return planes.reshape(-1)  # (768,)

# -------------------------
# Dataset
# -------------------------
class FenScoreDataset(Dataset):
    def __init__(self, path: str, score_from_stm: bool = True, cp_scale: float = 600.0):
        self.items: List[Tuple[str, float]] = []
        self.score_from_stm = score_from_stm
        self.cp_scale = cp_scale

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#') or ';' not in s:
                    continue
                fen, sc = s.split(';', 1)
                fen = fen.strip()
                try:
                    cp = float(sc.strip())
                except ValueError:
                    continue
                self.items.append((fen, cp))

        if not self.items:
            raise RuntimeError("No samples loaded. Expect lines like: 'FEN ; score'")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fen, cp = self.items[idx]
        try:
            board = chess.Board(fen)  # validates FEN, sets turn/etc.
        except ValueError as e:
            raise ValueError(f"Bad FEN at line {idx}: {e}")

        if self.score_from_stm and board.turn == chess.BLACK:
            cp = -cp

        x = encode_board_12x64_flat(board)           # (768,)
        y = math.tanh(cp / self.cp_scale)            # scalar in (-1,1)
        return torch.from_numpy(x), torch.tensor([y], dtype=torch.float32)

# -------------------------
# Model 768x32x1
# -------------------------
class NNUE_768x32x1(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(768, 32, bias=True)
        self.l2 = nn.Linear(32, 1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self._init()

    def _init(self):
        nn.init.kaiming_uniform_(self.l1.weight, a=0.0)
        nn.init.zeros_(self.l1.bias)
        nn.init.uniform_(self.l2.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.l2.bias)

    def forward(self, x):
        h = self.act(self.l1(x))
        return self.l2(h)

# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -------------------------
# Train
# -------------------------
def main():
    set_seed(SEED)
    print(f"Device: {DEVICE}")

    ds = FenScoreDataset(DATA_PATH, score_from_stm=SCORE_FROM_STM, cp_scale=TARGET_CP_SCALE)
    n_total = len(ds)
    n_val = max(1, int(n_total * VAL_SPLIT))
    n_train = n_total - n_val
    tr_ds, va_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))

    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = NNUE_768x32x1().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print("Starting training")
    best_va = float('inf')
    for ep in range(1, EPOCHS + 1):
        model.train()
        tr_loss_sum = tr_cnt = 0
        for xb, yb in tr_loader:
            xb = xb.to(DEVICE, non_blocking=True).float()
            yb = yb.to(DEVICE, non_blocking=True).float()
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_loss_sum += loss.item() * yb.size(0)
            tr_cnt += yb.size(0)

        model.eval()
        va_loss_sum = va_cnt = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(DEVICE, non_blocking=True).float()
                yb = yb.to(DEVICE, non_blocking=True).float()
                loss = loss_fn(model(xb), yb)
                va_loss_sum += loss.item() * yb.size(0)
                va_cnt += yb.size(0)

        tr_loss = tr_loss_sum / max(1, tr_cnt)
        va_loss = va_loss_sum / max(1, va_cnt)
        print(f"Epoch {ep:02d} | train MSE={tr_loss:.6f} | val MSE={va_loss:.6f}")

        if va_loss < best_va:
            best_va = va_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "score_from_stm": SCORE_FROM_STM,
                    "target_cp_scale": TARGET_CP_SCALE,
                    "arch": "768x32x1",
                }
            }, SAVE_PATH)
            print(f"  -> saved: {SAVE_PATH}")

    print("Done.")

if __name__ == "__main__":
    main()
