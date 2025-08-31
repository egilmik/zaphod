# export_to_nnuebin.py
import struct, torch, numpy as np

PT_PATH = "nnue_768x32x1.pt"   # from your trainer
BIN_OUT = "weights.nnue"       # binary for C++ loader

ckpt = torch.load(PT_PATH, map_location="cpu")
sd   = ckpt["model_state_dict"]
cfg  = ckpt.get("config", {})
scale = float(cfg.get("target_cp_scale", 600.0))

# PyTorch Linear shapes: weight [out, in], bias [out]
w1 = sd["l1.weight"].cpu().numpy().astype("<f4", copy=False)  # (32,768)
b1 = sd["l1.bias"  ].cpu().numpy().astype("<f4", copy=False)  # (32,)
w2 = sd["l2.weight"].cpu().numpy().astype("<f4", copy=False)  # (1,32)
b2 = sd["l2.bias"  ].cpu().numpy().astype("<f4", copy=False)  # (1,)

with open(BIN_OUT, "wb") as f:
    # Header: magic(8), in(int32), hidden(int32), out(int32), scale(float32)
    f.write(b"NNUEV1\0\0")                 # 8 bytes
    f.write(struct.pack("<iii", 768, 32, 1))
    f.write(struct.pack("<f", scale))
    # Data: row-major
    f.write(w1.tobytes(order="C"))         # 32*768 float32
    f.write(b1.tobytes(order="C"))         # 32
    f.write(w2.tobytes(order="C"))         # 1*32
    f.write(b2.tobytes(order="C"))         # 1
print(f"wrote {BIN_OUT} with scale={scale}")
