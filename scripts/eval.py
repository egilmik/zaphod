# nnue_infer.py
import argparse, math
import numpy as np
import torch
from torch import nn
import chess  # pip install python-chess

# ---------- Encoding (identical to trainer) ----------
def encode_board_12x64_flat(board: chess.Board) -> np.ndarray:
    planes = np.zeros((12, 64), dtype=np.float32)
    for sq, pc in board.piece_map().items():
        base = 0 if pc.color == chess.WHITE else 6
        plane = base + (pc.piece_type - 1)  # PAWN..KING => 0..5 (+6 for black)
        planes[plane, sq] = 1.0
    return planes.reshape(-1)  # (768,)

# ---------- Model (identical to trainer) ----------
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
        return self.l2(h)  # y ≈ tanh(cp_white / scale)

# ---------- Utilities ----------
def atanh_clip(y: float, eps: float = 1e-6) -> float:
    # numeric-safe inverse tanh
    y = max(min(y, 1.0 - eps), -1.0 + eps)
    return 0.5 * math.log((1.0 + y) / (1.0 - y))

def load_model(weights_path: str, device: str = "cpu"):
    ckpt = torch.load(weights_path, map_location=device)
    model = NNUE_768x32x1()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    cfg = ckpt.get("config", {})
    scale = float(cfg.get("target_cp_scale", 600.0))  # must match training
    return model.to(device), scale, cfg

def infer_fen(fen: str, model: NNUE_768x32x1, scale: float, device: str = "cpu"):
    board = chess.Board(fen)  # validates FEN
    x = encode_board_12x64_flat(board)
    xb = torch.from_numpy(x).unsqueeze(0).to(device).float()  # (1,768)
    with torch.no_grad():
        y = model(xb).item()  # NN output ~ tanh(cp_white/scale), but unbounded slightly
    cp_white = atanh_clip(y) * scale  # convert back to centipawns (white-POV)
    cp_stm = cp_white if board.turn == chess.WHITE else -cp_white
    return y, cp_white, cp_stm, board.turn

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="nnue_768x32x1.pt", help="Path to .pt checkpoint")
    ap.add_argument("--fen", required=True, help="FEN string")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device")
    args = ap.parse_args()

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    model, scale, cfg = load_model(args.weights, device)
    y, cp_white, cp_stm, turn = infer_fen(args.fen, model, scale, device)

    print(f"Model output y        : {y:.6f}  (≈ tanh(cp_white/{scale:.1f}))")
    print(f"Eval (white-POV) [cp] : {cp_white:.2f}")
    print(f"Eval (STM-POV)   [cp] : {cp_stm:.2f}  (side to move: {'white' if turn == chess.WHITE else 'black'})")

if __name__ == "__main__":
    main()
