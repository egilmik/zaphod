# nn764x64x1_trainer.py
# pip install torch numpy scikit-learn

import argparse, os, numpy as np, torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# ---------------- Model ----------------
class MLP764x64x1(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(764, 64)
        self.act1 = nn.ReLU(inplace=True)
        self.lin2 = nn.Linear(64, 1)
        # Kaiming init for ReLU
        nn.init.kaiming_uniform_(self.lin1.weight, a=0.0)
        nn.init.zeros_(self.lin1.bias)
        nn.init.kaiming_uniform_(self.lin2.weight, a=0.0)
        nn.init.zeros_(self.lin2.bias)

    def forward(self, x):
        x = self.act1(self.lin1(x))
        return self.lin2(x).squeeze(-1)  # (B,)

# ------------- Utils -------------------
def load_or_make_dummy(data_dir, n=10000):
    Xp = os.path.join(data_dir, "X.npy")
    yp = os.path.join(data_dir, "y.npy")
    if os.path.isfile(Xp) and os.path.isfile(yp):
        X = np.load(Xp).astype(np.float32)
        y = np.load(yp).astype(np.float32)
        assert X.shape[1] == 764, f"X must have 764 features, got {X.shape[1]}"
        return X, y
    # Dummy fallback: random features, linear target + noise
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, 764), dtype=np.float32)
    w = rng.standard_normal((764,), dtype=np.float32)
    y = X @ w * 10.0 + rng.standard_normal(n).astype(np.float32) * 50.0
    return X, y

def standardize(train_X, X):
    mu = train_X.mean(axis=0, keepdims=True)
    sd = train_X.std(axis=0, keepdims=True); sd[sd < 1e-6] = 1.0
    return (X - mu) / sd, mu.astype(np.float32), sd.astype(np.float32)

# ------------- Train -------------------
def train(
    data_dir: str,
    batch_size: int = 8192,
    epochs: int = 5,
    lr: float = 1e-3,
    huber_beta: float = 50.0,
    val_split: float = 0.05,
    seed: int = 42,
    out_ckpt: str = "nn764x64x1.pt",
    out_stats: str = "nn764x64x1_stats.npz",
    regression: bool = True,
):
    torch.manual_seed(seed); np.random.seed(seed)

    X, y = load_or_make_dummy(data_dir)

    # Optional: clip targets if centipawns
    if regression:
        y = np.clip(y, -2000.0, 2000.0)

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=val_split, random_state=seed)

    # Standardize using train stats only
    Xtr, mu, sd = standardize(Xtr, Xtr)
    Xva, _, _   = standardize(Xtr, Xva)  # apply same stats

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP764x64x1().to(device)

    if regression:
        criterion = nn.SmoothL1Loss(beta=huber_beta)  # Huber (good for cp)
    else:
        criterion = nn.BCEWithLogitsLoss()            # if you switch to win/draw/loss->binary

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=batch_size, shuffle=True, pin_memory=True,
    )
    va_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)),
        batch_size=batch_size, shuffle=False, pin_memory=True,
    )

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss, tot_cnt = 0.0, 0
        for xb, yb in tr_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * yb.numel()
            tot_cnt  += yb.numel()

        model.eval()
        with torch.no_grad():
            va_loss, va_cnt = 0.0, 0
            abs_err = 0.0
            for xb, yb in va_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(xb)
                loss = criterion(pred, yb)
                va_loss += loss.item() * yb.numel()
                va_cnt  += yb.numel()
                if regression:
                    abs_err += (pred - yb).abs().sum().item()
        sched.step()

        if regression:
            print(f"Epoch {ep:02d}: train_huber={tot_loss/tot_cnt:.3f}, "
                  f"val_huber={va_loss/va_cnt:.3f}, val_MAEcp={abs_err/va_cnt:.1f}")
        else:
            print(f"Epoch {ep:02d}: train_loss={tot_loss/tot_cnt:.3f}, val_loss={va_loss/va_cnt:.3f}")

    # Save weights + normalization stats
    torch.save(model.state_dict(), out_ckpt)
    np.savez(out_stats, mu=mu, sd=sd)
    print(f"Saved weights -> {out_ckpt}\nSaved stats -> {out_stats}")

# ------------- Inference (smoke test) -------------
def predict(ckpt: str, stats_npz: str, X: np.ndarray):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP764x64x1().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    st = np.load(stats_npz)
    mu, sd = st["mu"], st["sd"]
    Xn = (X.astype(np.float32) - mu) / sd
    with torch.no_grad():
        t = torch.from_numpy(Xn).to(device)
        out = model(t).cpu().numpy()
    return out

# ------------- CLI ---------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=".", help="folder containing X.npy and y.npy")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_split", type=float, default=0.05)
    ap.add_argument("--huber_beta", type=float, default=50.0)
    ap.add_argument("--ckpt", type=str, default="nn764x64x1.pt")
    ap.add_argument("--stats", type=str, default="nn764x64x1_stats.npz")
    ap.add_argument("--regression", action="store_true", help="use regression objective (default)")
    ap.add_argument("--classification", action="store_true", help="use binary classification (win/loss)")
    args = ap.parse_args()

    regression = True
    if args.classification: regression = False

    train(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        huber_beta=args.huber_beta,
        val_split=args.val_split,
        out_ckpt=args.ckpt,
        out_stats=args.stats,
        regression=regression,
    )
