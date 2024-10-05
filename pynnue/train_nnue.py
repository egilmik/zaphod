# train_nnue.py

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from nnue_model import NNUEModel
from board import Board
from feature_extractor import extract_features
import time

class ChessDataset(Dataset):
    def __init__(self, epd_list, targets):
        self.positions = []
        self.targets = targets

        for epd in epd_list:
            board = Board()
            board.set_position_from_fen(epd)
            self.positions.append(board)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        board = self.positions[idx]
        features = extract_features(board)
        features = torch.tensor(features, dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return features, target

def load_epd_positions(epd_file):
    epd_list = []
    targets = []
    with open(epd_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Assume the target evaluation is included after a semicolon
            if ';' in line:
                epd_str, target_str = line.split(';')
                target = int(target_str.strip())
            else:
                epd_str = line
                target = 0.0  # Default target if not specified
            if abs(target) < 1000:
                epd_list.append(epd_str.strip())
                targets.append(target)
    return epd_list, targets

def load_epd_positions_pct(epd_file):
    epd_list = []
    targets = []
    with open(epd_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split the line into FEN and operations
            parts = line.split(';')
            fen_and_misc = parts[0].strip()
            fen_parts = fen_and_misc.split()
            if len(fen_parts) < 4:
                print(f"Invalid EPD line: {line}")
                continue
            fen = ' '.join(fen_parts[:4])
            operations = parts[1:]  # Remaining parts are operations
            target = 0.0  # Default target
            for op in operations:
                op = op.strip()
                if op.startswith('c1 '):
                    # Extract the evaluation score
                    if 'score:' in op:
                        try:
                            score_part = op.split('score:')[1]
                            score_str = score_part.strip().strip('%').strip()
                            percentage = float(score_str)
                            # Map percentage to [-1.0, 1.0]
                            target = (percentage - 50.0) / 50.0
                        except ValueError:
                            print(f"Invalid score format in operation: {op}")
            epd_list.append(fen)
            targets.append(percentage)
    return epd_list, targets

def main():
    # Load positions and targets from a file
    print("Loading training data")
    epd_list, targets = load_epd_positions('training_positions.txt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    

    # Create dataset and dataloader
    dataset = ChessDataset(epd_list, targets)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = NNUEModel()
    
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    print("Starting training")
    epochs = 90
    for epoch in range(epochs):
        start_time = time.time()  # Start timing the epoch
        running_loss = 0.0
        for features, target in dataloader:
            features = features.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, target.squeeze())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_time = time.time() - start_time  # Calculate epoch duration
        average_loss = running_loss/len(targets)
        print(f"Epoch {epoch+1}/{epochs}, "f"Train Loss: {average_loss:.4f},"f"Time: {epoch_time:.2f}s")

    # Save the trained model
    torch.save(model.state_dict(), 'nnue_model.pth')

    # Test the model with an EPD string
    test_epd = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"
    test_board = Board()
    test_board.set_position_from_fen(test_epd)
    test_features = extract_features(test_board)
    test_features = torch.tensor(test_features, dtype=torch.float32)
    model.eval()
    #with torch.no_grad():
    #    evaluation = model(test_features).item()
    #print(f"Evaluation for test position: {evaluation}")

    # Additional testing with more positions
    test_epd_list = [
        # White has a strong attack -3.1
        "r1bqkbnr/pppp1ppp/2n5/4p3/1b1P4/5N2/PPPN1PPP/R1BQKB1R w KQkq - 2 5",
        # Endgame position
        "8/5k2/8/8/8/8/5K2/8 w - - 0 1",
        # Expected +6.6
        "2bqkbn1/rpp1pppN/n2p4/p6p/3PP3/2N5/PPP2PPP/R1BQKB1R w KQ - 1 7"
    ]
    for test_epd in test_epd_list:
        test_board = Board()
        test_board.set_position_from_fen(test_epd)
        test_features = extract_features(test_board)
        test_features = torch.tensor(test_features, dtype=torch.float32)
        #with torch.no_grad():
        #    evaluation = model(test_features).item()
        #print(f"Evaluation for position '{test_epd}': {evaluation}")

if __name__ == '__main__':
    main()
