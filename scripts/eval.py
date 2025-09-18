# nnue_infer.py
import argparse, math
import numpy as np
import torch
from torch import nn
import chess  # pip install python-chess

# ---------- Encoding (white-POV 12x64, same as your trainer here) ----------
def encode_board_12x64_flat(board: chess.Board) -> np.ndarray:
    planes = np.zeros((12, 64), dtype=np.float32)
    for sq, pc in board.piece_map().items():
        base = 0 if pc.color == chess.WHITE else 6
        plane = base + (pc.piece_type - 1)  # PAWN..KING => 0..5 (+6 for black)
        planes[plane, sq] = 1.0
    return planes.reshape(-1)  # (768,)

def encode_board_stm(board: chess.Board) -> np.ndarray:
    # Normalize to STM=White by mirroring when Black to move
    b = board if board.turn == chess.WHITE else board.mirror()
    planes = np.zeros((12, 64), dtype=np.float32)
    for sq, pc in b.piece_map().items():
        base = 0 if pc.color == chess.WHITE else 6
        plane = base + (pc.piece_type - 1)
        planes[plane, sq] = 1.0
    return planes.reshape(-1)  # (768,)


# ---------- Model ----------
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
        return self.l2(h)  # ~ tanh(cp_white / scale)

# ---------- Utilities ----------
def atanh_clip(y: float, eps: float = 1e-6) -> float:
    y = max(min(float(y), 1.0 - eps), -1.0 + eps)
    return 0.5 * math.log((1.0 + y) / (1.0 - y))

def load_model(weights_path: str, device: str = "cpu"):
    ckpt = torch.load(weights_path, map_location=device)
    model = NNUE_768x32x1()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    cfg = ckpt.get("config", {})
    scale = float(cfg.get("target_cp_scale", 600.0))
    return model.to(device), scale, cfg

def infer_board(board: chess.Board, model: NNUE_768x32x1, scale: float, device: str = "cpu"):
    x = encode_board_stm(board)
    xb = torch.from_numpy(x).unsqueeze(0).to(device).float()
    with torch.no_grad():
        y = model(xb).item()  # approx tanh(cp_white/scale)
    cp_stm = atanh_clip(y) * scale
    cp_white = cp_stm if board.turn == chess.WHITE else -cp_stm
    return y, cp_white, cp_stm

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="nnue_768x32x1.pt", help="Path to .pt checkpoint")
    ap.add_argument("--fen", required=True, help="FEN string")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device")
    args = ap.parse_args()

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    model, scale, cfg = load_model(args.weights, device)

    # Original position P
    board = chess.Board(args.fen)
    y, cp_w, cp_stm = infer_board(board, model, scale, device)

    # Color-flipped twin P' (mirror swaps colors and side-to-move correctly)
    board_flip = board.mirror()
    y_f, cp_w_f, cp_stm_f = infer_board(board_flip, model, scale, device)

    # Print results
    print(f"Scale (cp): {scale:.1f}")
    print("\nPosition P")
    print(board)
    print(f"y (≈tanh(cp_white/scale)) : {y:.6f}")
    print(f"cp_white [cp]             : {cp_w:.2f}")
    print(f"cp_stm   [cp]             : {cp_stm:.2f} (STM: {'white' if board.turn else 'black'})")

    print("\nColor-flipped P'")
    print(board_flip)
    print(f"y' (≈tanh(cp_white/scale)): {y_f:.6f}")
    print(f"cp_white' [cp]            : {cp_w_f:.2f}")
    print(f"cp_stm'   [cp]            : {cp_stm_f:.2f} (STM: {'white' if board_flip.turn else 'black'})")

    # Consistency checks
    # For white-POV encoding: expect cp_white(P) ≈ -cp_white(P')
    # For STM-POV:           expect cp_stm(P)  ≈ -cp_stm(P')
    eps = 1e-3
    print("\nConsistency deltas (should be near zero):")
    print(f"cp_white(P) + cp_white(P')  = {cp_w + cp_w_f:+.3f} cp")
    print(f"cp_stm(P)   + cp_stm(P')    = {cp_stm - cp_stm_f:+.3f} cp")
    # Also show absolute diffs
    print(f"|cp_white(P) + cp_white(P')| = {abs(cp_w + cp_w_f):.3f} cp")
    print(f"|cp_stm(P)   + cp_stm(P')|   = {abs(cp_stm - cp_stm_f):.3f} cp")

if __name__ == "__main__":
    main()
