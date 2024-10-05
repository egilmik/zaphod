# evaluate_position.py

import torch
import torch.nn as nn
from nnue_model import NNUEModel
from feature_extractor import extract_features
from board import Board

def load_model(model_path):
    # Initialize the model architecture
    model = NNUEModel()
    # Load the saved model parameters
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

def evaluate_position(fen, model):
    # Initialize the board and set the position from FEN
    board = Board()
    board.set_position_from_fen(fen)
    # Extract features from the position
    features = extract_features(board)
    # Convert features to a PyTorch tensor
    features = torch.tensor(features, dtype=torch.float32)
    # Evaluate the position using the model
    with torch.no_grad():
        evaluation = model(features).item()
    return evaluation

def main():
    # Load the model
    model = load_model('nnue_model.pth')
    # Input the FEN of the position to evaluate
    print("Enter the FEN of the chess position:")
    fen = input().strip()
    # Evaluate the position
    evaluation = evaluate_position(fen, model)
    print(f"Evaluation of the position: {evaluation:.4f}")

if __name__ == '__main__':
    main()
