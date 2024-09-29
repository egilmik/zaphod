# feature_extractor.py

from pieces import Piece

def extract_features(board):
    # We'll create a feature vector of size 768
    # 12 pieces * 64 squares = 768
    features = [0] * 768

    for sq in range(64):
        piece = board.squares[sq]
        if piece != Piece.EMPTY.value:
            piece_type = abs(piece) - 1  # 0-indexed
            color_offset = 0 if piece > 0 else 6  # White: 0-5, Black: 6-11
            index = (color_offset + piece_type) * 64 + sq
            features[index] = 1  # One-hot encoding

    return features
