import subprocess
import pandas as pd
import time

ENGINE_PATH = "D:\\source\\zaphod\\out\\build\\x64-Release\\Release\\Zaphod.exe"  # path to your engine binary

def start_engine():
    print("Starting engine...")
    p = subprocess.Popen(
        [ENGINE_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1
    )

    # Initialize UCI
    p.stdin.write("uci\n")
    p.stdin.flush()

    while True:
        line = p.stdout.readline()
        if "uciok" in line:
            break

    return p

def get_static_eval(p, fen):
    p.stdin.write(f"position fen {fen}\n")
    p.stdin.write("eval\n")
    p.stdin.flush()

    while True:
        line = p.stdout.readline().strip()
        if line.startswith("eval "):
            try:
                return int(line.split()[1])
            except:
                return 0  # fallback on parse failure

def enrich_data(input_csv="eval_data.csv", output_csv="enriched_data.csv", limit=None):
    df = pd.read_csv(input_csv)
    if limit:
        df = df.iloc[:limit]

    p = start_engine()
    evals = []

    print("Evaluating positions...")
    for i, row in df.iterrows():
        fen = row["fen"]
        try:
            score = get_static_eval(p, fen)
        except Exception as e:
            print(f"Error at row {i}: {e}")
            score = 0
        evals.append(score)

        if i % 100 == 0:
            print(f"{i}/{len(df)} evaluated")

    df["static_eval"] = evals

    # Stop engine
    p.stdin.write("quit\n")
    p.stdin.flush()

    df.to_csv(output_csv, index=False)
    print(f"Saved enriched data to {output_csv}")

if __name__ == "__main__":
    enrich_data("midgame_data.csv", "midgame_data_enriched.csv", limit=None)
    enrich_data("endgame_data.csv", "endgame_data_enriched.csv", limit=None)
