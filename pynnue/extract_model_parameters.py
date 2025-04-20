import torch
from nnue_model import NNUEModel
import numpy as np

# Assuming your model is defined as NNUEModel
model = NNUEModel()
model.load_state_dict(torch.load('nnue_model.pth'))  # Load your trained model

# Accessing the layers
input_layer = model.input_layer
output_layer = model.output_layer

# Extract weights and biases
input_weights = input_layer.weight.data.numpy()    # Shape: [hidden_size, input_size]
input_biases = input_layer.bias.data.numpy()       # Shape: [hidden_size]

output_weights = output_layer.weight.data.numpy()  # Shape: [1, hidden_size]
output_bias = output_layer.bias.data.numpy()       # Shape: [1]

scale_factor = 1024  # Example scale factor

# Scale and convert to integers
input_weights_int = (input_weights * scale_factor).astype(np.int32)
input_biases_int = (input_biases * scale_factor).astype(np.int32)
output_weights_int = (output_weights * scale_factor).astype(np.int32)
output_bias_int = int(output_bias[0] * scale_factor)

# Save scaled integer weights
np.savetxt('input_weights_int.txt', input_weights_int, fmt='%d')
np.savetxt('input_biases_int.txt', input_biases_int, fmt='%d')
np.savetxt('output_weights_int.txt', output_weights_int, fmt='%d')
with open('output_bias_int.txt', 'w') as f:
    f.write(f"{output_bias_int}")