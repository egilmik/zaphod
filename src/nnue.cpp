// nnue.cpp

#include "nnue.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>


NNUE::NNUE()
    : input_weights_(INPUT_SIZE),
      hidden_biases_{},
      output_weights_{},
      output_bias_(0) {
    // Initialize weights and biases to zero or random values as needed
    // For simplicity, we'll initialize them to zero here
    for (auto& weights : input_weights_) {
        weights.fill(0);
    }
    hidden_biases_.fill(0);
    output_weights_.fill(0);
}

int NNUE::relu(int x) {
    return std::max(0, x);
}

int NNUE::forward(const std::array<int, INPUT_SIZE>& input) {
    // Hidden layer activations
    std::array<int, HIDDEN_SIZE> hidden_activations{};
    hidden_activations.fill(0);

    // Compute activations for the hidden layer
    for (int i = 0; i < INPUT_SIZE; ++i) {
        if (input[i] != 0) {
            for (int j = 0; j < HIDDEN_SIZE; ++j) {
                hidden_activations[j] += input[i] * input_weights_[i][j];
            }
        }
    }

    // Add biases and apply activation function
    for (int j = 0; j < HIDDEN_SIZE; ++j) {
        hidden_activations[j] += hidden_biases_[j];
        hidden_activations[j] = relu(hidden_activations[j]);
    }

    // Compute the output
    int output = output_bias_;
    for (int j = 0; j < HIDDEN_SIZE; ++j) {
        output += hidden_activations[j] * output_weights_[j];
    }

    // Optionally apply an activation function to the output
    return output;
}

void NNUE::load_weights(const std::vector<std::array<int, HIDDEN_SIZE>>& input_weights,
                        const std::array<int, HIDDEN_SIZE>& hidden_biases,
                        const std::array<int, HIDDEN_SIZE>& output_weights,
                        int output_bias) {
    input_weights_ = input_weights;
    hidden_biases_ = hidden_biases;
    output_weights_ = output_weights;
    output_bias_ = output_bias;
}

void NNUE::load_input_weights(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Failed to open " << filename << std::endl;
        return;
    }
    for (int i = 0; i < INPUT_SIZE; ++i) {
        std::string line;
        if (!std::getline(infile, line)) {
            std::cerr << "Error reading line " << i << " from " << filename << std::endl;
            break;
        }
        std::istringstream iss(line);
        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            if (!(iss >> input_weights_[i][j])) {
                std::cerr << "Error reading value at (" << i << "," << j << ") from " << filename << std::endl;
                break;
            }
        }
    }
    infile.close();
}

void NNUE::load_input_biases(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Failed to open " << filename << std::endl;
        return;
    }
    for (int j = 0; j < HIDDEN_SIZE; ++j) {
        if (!(infile >> hidden_biases_[j])) {
            std::cerr << "Error reading bias at index " << j << " from " << filename << std::endl;
            break;
        }
    }
    infile.close();
}

void NNUE::load_output_weights(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Failed to open " << filename << std::endl;
        return;
    }
    for (int j = 0; j < HIDDEN_SIZE; ++j) {
        if (!(infile >> output_weights_[j])) {
            std::cerr << "Error reading output weight at index " << j << " from " << filename << std::endl;
            break;
        }
    }
    infile.close();
}

void NNUE::load_output_bias(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Failed to open " << filename << std::endl;
        return;
    }
    if (!(infile >> output_bias_)) {
        std::cerr << "Error reading output bias from " << filename << std::endl;
    }
    infile.close();
}
