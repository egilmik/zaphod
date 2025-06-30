import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# PST dimensions: 6 pieces × 64 squares
initial_pst = np.zeros(6 * 64)

def load_data(csv_path, max_samples=10000):
    df = pd.read_csv(csv_path)
    df = df.sample(n=min(max_samples, len(df)))  # shuffle and trim
    return df

def logistic_loss(pst, pst_weights, static_evals, results):
    # Adjust the static evaluation by adding PST contributions
    # Each static_eval is the original score; we add PST adjustments
    adjusted_evals = static_evals + np.dot(pst_weights, pst)
    
    preds = adjusted_evals / 400.0  # scale to logistic input
    logits = 1 / (1 + np.exp(-preds))
    loss = -np.mean(results * np.log(logits + 1e-9) + (1 - results) * np.log(1 - logits + 1e-9))
    return loss

def compute_pst_weights(fens):
    """
    Convert each FEN into a 384-dim PST vector (6×64), one per position.
    Each vector has 1 for the square where a piece is located, 0 elsewhere.
    White adds +1, black adds -1.
    """
    import chess
    weights = []
    for fen in fens:
        board = chess.Board(fen)
        vec = np.zeros(6 * 64)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                idx = piece.piece_type - 1  # 0-based index
                sign = 1 if piece.color == chess.WHITE else -1
                vec[idx * 64 + square] = sign
        weights.append(vec)
    return np.array(weights)

def tune(csv_path="enriched_data.csv"):
    df = load_data(csv_path)
    fens = df["fen"].tolist()
    results = df["result"].astype(float).values
    static_evals = 0# df["static_eval"].astype(float).values
    pst_weights = compute_pst_weights(fens)

    losses = []

    def loss_fn(pst):
        return logistic_loss(pst, pst_weights, static_evals, results)

    def callback(x):
        loss = loss_fn(x)
        losses.append(loss)
        print(f"Step {len(losses)}: loss = {loss:.4f}")

    result = minimize(
        loss_fn,
        initial_pst,
        method='L-BFGS-B',
        callback=callback,
        options={'maxiter': 50, 'disp': True, 'gtol': 1e-5}
    )
    
    print("Static eval range:", np.min(static_evals), np.max(static_evals))
    print("Mean result:", np.mean(results))

    return result.x, losses

def save_pst(pst, filename="pst.h"):
    with open(filename, "w") as f:
        for i in range(6):
            f.write(f"int pst_{i}[64] = {{\n")
            for j in range(64):
                f.write(f"{int(pst[i*64 + j])}, ")
                if (j + 1) % 8 == 0:
                    f.write("\n")
            f.write("};\n\n")

def save_pst_as_std_array(pst, filename="pst_mid.h"):
    with open(filename, "w") as f:
        f.write("#pragma once\n")
        f.write("#include <array>\n\n")
        f.write(f"constexpr std::array<std::array<int, 64>, 7> pieceSquareScoreArray = {{\n")
        f.write("    {},\n")  # index 0 is unused

        for i in range(6):
            f.write("    {\n        ")
            for j in range(64):
                f.write(f"{int(round(pst[i][j]))}, ")
                if (j + 1) % 8 == 0 and j != 63:
                    f.write("\n        ")
            f.write("\n    },\n")
        f.write("};\n")
    print(f"Wrote std::array PST to {filename}")
    

def plot_losses(losses):
    plt.plot(losses)
    plt.title("Texel Tuning Loss (using static evals)")
    plt.xlabel("Iteration")
    plt.ylabel("Log Loss")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    tuned_pst, losses = tune("midgame_data_enriched.csv")
    save_pst(tuned_pst)
    plot_losses(losses)
