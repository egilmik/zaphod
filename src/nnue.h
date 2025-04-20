// nnue.h

#ifndef NNUE_H
#define NNUE_H

#include <array>
#include <vector>

class NNUE {
public:
    static const int INPUT_SIZE = 768;
    static const int HIDDEN_SIZE = 256;

    NNUE();

    // Forward pass to compute the evaluation score
    int forward(const std::array<int, INPUT_SIZE>& input);

    // Functions to load weights and biases
    void load_weights(const std::vector<std::array<int, HIDDEN_SIZE>>& input_weights,
                      const std::array<int, HIDDEN_SIZE>& hidden_biases,
                      const std::array<int, HIDDEN_SIZE>& output_weights,
                      int output_bias);

    void load_input_weights(const std::string& filename);
    void load_input_biases(const std::string& filename);
    void load_output_weights(const std::string& filename);
    void load_output_bias(const std::string& filename);

private:
    // Weights and biases
    std::vector<std::array<int, HIDDEN_SIZE>> input_weights_; // [INPUT_SIZE][HIDDEN_SIZE]
    std::array<int, HIDDEN_SIZE> hidden_biases_;              // [HIDDEN_SIZE]
    std::array<int, HIDDEN_SIZE> output_weights_;             // [HIDDEN_SIZE]
    int output_bias_;

    // Activation function
    int relu(int x);
};

#endif // NNUE_H
