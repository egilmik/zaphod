import chess
import chess.engine
import random
from multiprocessing import Pool, cpu_count
import time
import threading
from queue import Queue

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

def evaluate_position_wrapper(args):
    fen, engine_path, time_limit, batch_size = args
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    batch_positions = generate_random_positions(batch_size)
    for fen in batch_positions:
        score = evaluate_position(fen, engine, time_limit)
    engine.close()
    return fen, score

def evaluate_position_wrapper(batch_positions, engine_path, time_limit, result_queue):
    """
    Evaluates a batch of positions using the given engine.
    """

    print(f"Evaluate_poisition_wrappper, nr of positions to evaluate {len(batch_positions)}")
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    batch_results = []
    for fen in batch_positions:
        score = evaluate_position(fen, engine, time_limit)
        batch_results.append((fen, score))
    engine.close()
    # Put the batch results into the queue
    result_queue.put(batch_results)

def process_positions_parallel(total_positions, max_threads=4):
    
    batch_size = int(total_positions/max_threads)
    
    # Generate all positions to be evaluated
    start_time = time.time()
    all_positions = generate_random_positions(total_positions)
    total_time = time.time() - start_time
    print(f"Generating random positions in {total_time:.2f} seconds")

    # Split positions into batches
    batches = [all_positions[i:i + batch_size] for i in range(0, total_positions, batch_size)]
    print(f"Number of batches {len(batches)}, batch size {batch_size}")

    # Create a queue to store the results
    result_queue = Queue()

    # Start processing batches
    start_time = time.time()
    process_batches(batches,result_queue,max_threads)
    total_time = time.time() - start_time
    print(f"Evaluation completed in {total_time:.2f} seconds")

    # Collect all results from the queue

    start_time = time.time()
    all_results = []
    while not result_queue.empty():
        batch_results = result_queue.get()
        with open('training_data_05102024.txt', 'a') as f:
                for fen, score in batch_results:
                    f.write(f"{fen};{score}\n")
    total_time = time.time() - start_time
    print(f"Writing to file completed in {total_time:.2f} seconds")

    

def process_batches(batches,result_queue,max_threads):
    engine_path = 'd:\\source\\Stockfish\\stockfish-windows-x86-64-avx2.exe'
    time_limit = 0.1  # Time limit per move in seconds
    threads = []
    active_threads = []

    for batch_positions in batches:
        # Wait if the maximum number of threads is reached
        while len(active_threads) >= max_threads:  # +1 for the main thread
            time.sleep(0.1)
            active_threads = [t for t in active_threads if t.is_alive()]

        # Create a new thread for each batch
        thread = threading.Thread(
            target=evaluate_position_wrapper,
            args=(batch_positions, engine_path, time_limit, result_queue)
        )
        threads.append(thread)
        thread.start()
        active_threads.append(thread)        

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

def process_positions(total_positions, batch_size=1000):
    with chess.engine.SimpleEngine.popen_uci('d:\\source\\Stockfish\\stockfish-windows-x86-64-avx2.exe') as engine:
        for i in range(0, total_positions, batch_size):
            start_time = time.time()  # Start timing the generation
            batch_positions = generate_random_positions(batch_size)
            evaluations = []
            for fen in batch_positions:
                score = evaluate_position(fen, engine)
                if score is not None:
                    evaluations.append((fen, score))
            # Save evaluations to file
            with open('training_data_05102024.txt', 'a') as f:
                for fen, score in evaluations:
                    f.write(f"{fen};{score}\n")
            duration = time.time() - start_time  # Calculate duration
            print(f"Processed batch {i // batch_size + 1} / {total_positions // batch_size} Time {duration}")

if __name__ == '__main__':
    total_positions = 100000  # One million positions
    max_threads = 12  # Adjust based on memory and performance considerations
    process_positions_parallel(total_positions, max_threads)
