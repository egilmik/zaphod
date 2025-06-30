import chess.pgn
import csv

def result_to_score(result_str):
    if result_str == "1-0":
        return 1.0
    elif result_str == "0-1":
        return 0.0
    elif result_str == "1/2-1/2":
        return 0.5
    return None

def material_score(board):
    """Returns total non-pawn material for both sides"""
    material = 0
    values = {
        chess.PAWN: 0,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }
    for piece_type in values:
        material += values[piece_type] * len(board.pieces(piece_type, chess.WHITE))
        material += values[piece_type] * len(board.pieces(piece_type, chess.BLACK))
    return material

def is_quiet_position(board):
    for move in board.legal_moves:
        if board.is_capture(move):
            return False
        if board.is_en_passant(move):
            return False
        if board.gives_check(move):
            return False
        if board.is_pseudo_legal(move) and board.piece_at(move.from_square).piece_type == chess.PAWN:
            if chess.square_rank(move.to_square) in [0, 7]:
                return False  # promotion
    return True

def extract_positions(pgn_file, mid_csv, end_csv,
                      min_ply=10, sample_every=4,
                      mid_threshold=20, end_threshold=12,
                      max_games=5000):
    mid_rows = []
    end_rows = []
    count = 0

    with open(pgn_file, "r") as pgn:
        while count < max_games:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            result = result_to_score(game.headers.get("Result", ""))
            if result is None:
                continue

            board = game.board()
            ply = 0

            for move in game.mainline_moves():
                board.push(move)
                ply += 1
                if ply < min_ply or ply % sample_every != 0:
                    continue

                if not is_quiet_position(board):
                    continue

                mat = material_score(board)
                fen = board.fen()

                if mat >= mid_threshold:
                    mid_rows.append([fen, result])
                elif mat <= end_threshold:
                    end_rows.append([fen, result])

            count += 1
            if count % 100 == 0:
                print(f"Processed {count} games...")

    print(f"Extracted {len(mid_rows)} midgame and {len(end_rows)} endgame positions.")

    with open(mid_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fen", "result"])
        writer.writerows(mid_rows)

    with open(end_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fen", "result"])
        writer.writerows(end_rows)

    print(f"Wrote midgame to {mid_csv}, endgame to {end_csv}")

if __name__ == "__main__":
    extract_positions(
        pgn_file="games.pgn",
        mid_csv="midgame_data.csv",
        end_csv="endgame_data.csv",
        min_ply=10,
        sample_every=4,
        mid_threshold=20,
        end_threshold=12,
        max_games=50000000
    )
