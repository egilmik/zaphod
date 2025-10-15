# export_to_nnuebin_int8.py
import struct, torch, numpy as np
import sys

#PT_PATH = "nnue_768x256x1.pt"
#BIN_OUT = "768x256x1.nnueq"

def main(input: str, output:str):
    ckpt = torch.load(input, map_location="cpu")
    sd   = ckpt["model_state_dict"]
    cfg  = ckpt.get("config", {})
    eval_scale = float(cfg.get("target_cp_scale", 600.0))

    # Dims
    IN_FEATS = 768
    HIDDEN   = sd["emb.weight"].shape[1]  # should multiple of 64/32

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
    with open(output, "wb") as f:
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

    print(f"wrote {output}  (H={HIDDEN}, a1={a1:.6g}, Q_CAP={Q_CAP}, s2={s2:.6g}, eval_scale={eval_scale})")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: py export_binary_model_qunatized.py <input file> <output file>")
        sys.exit(1)
    print(sys.argv[1])
    print(sys.argv[2])
    main(sys.argv[1],sys.argv[2])