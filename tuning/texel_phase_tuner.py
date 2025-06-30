import numpy as np
import pandas as pd
import chess
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Constants
NUM_MATERIAL = 6
NUM_PST = 6 * 64
TOTAL_PARAMS = NUM_MATERIAL + NUM_PST

def load_data(csv_path, max_samples=10000000):
    df = pd.read_csv(csv_path)
    df = df.sample(n=min(max_samples, len(df)))
    return df

def compute_combined_features(fens):
    features = []
    for fen in fens:
        board = chess.Board(fen)
        vec = np.zeros(TOTAL_PARAMS)

        # Material features
        for pt in range(1, 7):
            vec[pt - 1] += len(board.pieces(pt, chess.WHITE))
            vec[pt - 1] -= len(board.pieces(pt, chess.BLACK))

        # PST features
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                idx = piece.piece_type - 1
                sign = 1 if piece.color == chess.WHITE else -1
                vec[NUM_MATERIAL + idx * 64 + square] = sign

        features.append(vec)
    return np.array(features)

def logistic_loss(params, features, results, static_evals=None,l2_reg=1e-8):
    if static_evals is None:
        static_evals = np.zeros(len(results))
    logits = static_evals + np.dot(features, params)
    preds = logits / 400.0
    prob = 1 / (1 + np.exp(-preds))
    loss = -np.mean(results * np.log(prob + 1e-9) + (1 - results) * np.log(1 - prob + 1e-9))
    regularized_params = params[:-1]  # exclude log(k)
    l2_penalty = l2_reg * np.sum(regularized_params ** 2)
    return loss + l2_penalty

def tune_with_material(csv_path="quiet_positions.csv", max_iter=50):
    df = load_data(csv_path)
    fens = df["fen"].tolist()
    results = df["result"].astype(float).values
    #static_evals = np.zeros(len(results))
    static_evals = df["static_eval"].astype(float).values

    features = compute_combined_features(fens)
    initial_params = np.zeros(TOTAL_PARAMS)
    losses = []

    def loss_fn(params):
        return logistic_loss(params, features, results, static_evals)

    def callback(params):
        loss = loss_fn(params)
        losses.append(loss)
        print(f"Step {len(losses)}: loss = {loss:.4f}")

    result = minimize(
        loss_fn,
        initial_params,
        method='L-BFGS-B',
        callback=callback,
        options={'maxiter': max_iter, 'disp': True, 'gtol': 1e-6}
    )

    material = result.x[:NUM_MATERIAL]
    pst = result.x[NUM_MATERIAL:].reshape((6, 64))
    return material, pst, losses

def save_material(material, filename="material.h"):
    # Reorder to match Zaphod: [empty, rook, knight, bishop, queen, king, pawn]
    reordered_indices = [3, 1, 2, 4, 5, 0]  # from [pawn, knight, bishop, rook, queen, king]
    white = [0]  # index 0 = empty
    for idx in reordered_indices:
        white.append(int(round(material[idx])))

    black = [-v for v in white]

    combined = white + black  # 14 values: white[0..6], black[0..6]

    with open(filename, "w") as f:
        f.write("#pragma once\n")
        f.write("#include <array>\n\n")
        f.write("// Zaphod-style material values: white [0..6], black [7..13]\n")
        f.write("constexpr std::array<int, 14> materialValue = { ")
        f.write(", ".join(str(x) for x in combined))
        f.write(" };\n")

    print(f"Wrote Zaphod-format material values to {filename}")


def save_pst_as_std_array(pst, filename="pst_mid.h"):
    # Reorder to match: [empty, rook, knight, bishop, queen, king, pawn]
    piece_order = ["", "rook", "knight", "bishop", "queen", "king", "pawn"]
    reordered_indices = [3, 1, 2, 4, 5, 0]  # from [pawn, knight, bishop, rook, queen, king]

    pst_reordered = [np.zeros(64)]
    for idx in reordered_indices:
        pst_reordered.append(pst[idx])

    with open(filename, "w") as f:
        f.write("#pragma once\n")
        f.write("#include <array>\n\n")
        f.write("constexpr std::array<std::array<int, 64>, 7> pieceSquareScoreArray = {\n")
        for i, arr in enumerate(pst_reordered):
            f.write(f"    /* {piece_order[i]} */ {{\n        ")
            for j in range(64):
                f.write(f"{int(round(arr[j]))}, ")
                if (j + 1) % 8 == 0 and j != 63:
                    f.write("\n        ")
            f.write("\n    },\n")
        f.write("};\n")
    print(f"Wrote PST to {filename}")

def plot_losses(losses):
    plt.plot(losses)
    plt.title("Texel Tuning Loss (material + PST)")
    plt.xlabel("Iteration")
    plt.ylabel("Log Loss")
    plt.grid(True)
    plt.show()

# Entry point
if __name__ == "__main__":
    material, pst, losses = tune_with_material("midgame_data_enriched.csv")
    save_material(material,"material_mid.h")
    save_pst_as_std_array(pst, "pst_mid.h")
    

    material, pst, losses = tune_with_material("endgame_data_enriched.csv")
    save_material(material,"material_end.h")
    save_pst_as_std_array(pst, "pst_end.h")


