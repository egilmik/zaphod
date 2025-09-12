# export_to_nnuebin_int8.py
import struct, torch, numpy as np
rng = np.random.default_rng(7)

PT_PATH = "nnue_768x32x1.pt"
BIN_OUT = "weights_int8.nnueq"

ckpt = torch.load(PT_PATH, map_location="cpu")
sd   = ckpt["model_state_dict"]
cfg  = ckpt.get("config", {})
scale = float(cfg.get("target_cp_scale", 600.0))

# Float weights
w1 = sd["l1.weight"].cpu().numpy().astype(np.float32)   # (32, 768)  out,in
b1 = sd["l1.bias"  ].cpu().numpy().astype(np.float32)   # (32,)
w2 = sd["l2.weight"].cpu().numpy().astype(np.float32)   # (1, 32)
b2 = sd["l2.bias"  ].cpu().numpy().astype(np.float32)   # (1,)

# ---------- L1: per-output-channel symmetric int8 ----------
# W1_q[o, i] ~ W1[o, i] / s1[o], with s1[o] = max|W1[o,:]|/127
absmax_w1 = np.maximum(np.max(np.abs(w1), axis=1), 1e-12)
s1 = absmax_w1 / 127.0                                # (32,)
W1_q = np.clip(np.round(w1 / s1[:, None]), -127, 127).astype(np.int8)  # (32,768)
B1_q = np.round(b1 / s1).astype(np.int32)                               # (32,)

# ---------- Hidden activation quantization (ReLU -> uint8) ----------
# We need a layer-wide a1 so that h_q = clip(round(h / a1), 0..127)
# Calibrate a1 with sparse binary inputs approximating ~30 active features.
def calibrate_a1(num_samples=20000, k_active=30):
    # simulate sparse 0/1 inputs with exactly k_active ones
    n_in = w1.shape[1]
    zs = []
    for _ in range(num_samples):
        idx = rng.choice(n_in, size=k_active, replace=False)
        # integer preact: p[o] = B1_q[o] + sum_i W1_q[o, idx_i]
        p = B1_q.astype(np.int32).copy()
        p += W1_q[:, idx].sum(axis=1, dtype=np.int32)
        # dequantize to float pre-activation: z = s1[o] * p[o]
        z = s1 * p
        zs.append(z)
    Z = np.maximum(0.0, np.vstack(zs))      # ReLU
    # pick 99.9th percentile across all hidden units/samples
    hi = np.percentile(Z, 99.9)
    return max(hi / 127.0, 1e-8)

a1 = calibrate_a1()

# ---------- L2: symmetric int8 (single scale) ----------
absmax_w2 = max(float(np.max(np.abs(w2))), 1e-12)
s2 = absmax_w2 / 127.0
W2_q = np.clip(np.round(w2 / s2), -127, 127).astype(np.int8)  # (1,32)
# keep b2 in float (final domain is float cp after tanh/scale on your side)
B2_f = b2.astype(np.float32)  # (1,)

# ---------- Write compact binary ----------
# Layout:
# magic(8) = "NNUEQ1\0\0"
# dims: int32 in, hidden, out
# eval_scale: float32   (your cp->tanh scale, passthrough)
# L1: s1[32] float32, W1_q int8[32*768], B1_q int32[32]
# Act: a1 float32
# L2: s2 float32, W2_q int8[1*32], B2_f float32[1]
with open(BIN_OUT, "wb") as f:
    f.write(b"NNUEQ1\0\0")
    f.write(struct.pack("<iii", 768, 32, 1))
    f.write(struct.pack("<f", scale))

    f.write(s1.astype("<f4").tobytes(order="C"))
    f.write(W1_q.tobytes(order="C"))
    f.write(B1_q.astype("<i4").tobytes(order="C"))

    f.write(struct.pack("<f", float(a1)))

    f.write(struct.pack("<f", float(s2)))
    f.write(W2_q.tobytes(order="C"))
    f.write(B2_f.astype("<f4").tobytes(order="C"))

print(f"wrote {BIN_OUT}  (a1={a1:.6g}, s2={s2:.6g}, eval_scale={scale})")
