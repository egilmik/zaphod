import chess
import chess.engine
import random

def generate_random_positions(batch_size):
    positions = []
    while len(positions) < batch_size:
        board = chess.Board()
        num_moves = random.randint(1, 100)
        for _ in range(num_moves):
            if board.is_game_over():
                break
            moves = list(board.legal_moves)
            move = random.choice(moves)
            board.push(move)
            if random.random() < 0.1:
                positions.append(board.fen())
                if len(positions) >= batch_size:
                    break
    return positions

def evaluate_position(fen, engine, time_limit=0.1):
    board = chess.Board(fen)
    try:
        info = engine.analyse(board, chess.engine.Limit(time=time_limit))
        score = info['score'].white().score(mate_score=100000)
    except Exception as e:
        print(f"Error evaluating position: {e}")
        score = None
    return score

def process_positions(total_positions, batch_size=1000):
    with chess.engine.SimpleEngine.popen_uci('d:\\source\\Stockfish\\stockfish-windows-x86-64-avx2.exe') as engine:
        for i in range(0, total_positions, batch_size):
            batch_positions = generate_random_positions(batch_size)
            evaluations = []
            for fen in batch_positions:
                score = evaluate_position(fen, engine)
                if score is not None:
                    evaluations.append((fen, score))
            # Save evaluations to file
            with open('evaluated_positions5.txt', 'a') as f:
                for fen, score in evaluations:
                    f.write(f"{fen};{score}\n")
            print(f"Processed batch {i // batch_size + 1} / {total_positions // batch_size}")

if __name__ == '__main__':
    total_positions = 100000  # One million positions
    batch_size = 1000  # Adjust based on memory and performance considerations
    process_positions(total_positions, batch_size)
