# nnue_model.py

import torch
import torch.nn as nn

class NNUEModel(nn.Module):
    def __init__(self, input_size=768, hidden_size=256):
        super(NNUEModel, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x.squeeze()  # Remove unnecessary dimensions
