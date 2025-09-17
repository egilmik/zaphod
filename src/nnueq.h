#ifndef NNUEQ_H
#define NNUEQ_H

#include <string>
#include <array>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "bitboard.h"
#include "board.h"

class NNUEQ {
public:
    static constexpr int IN = 768;
    static constexpr int H = 32;
    static constexpr int OUT = 1;

    // --- Common ---
    float scale_cp = 600.f; // cp scale used by training head

    // --- Float path (NNUEV1) ---
    std::vector<float> W1, B1, W2;
    float B2 = 0.f;

    // --- Quantized path (NNUEQ1) ---
    // L1
    std::vector<int8_t>  W1_q;   // H*IN
    std::vector<int32_t> B1_q;   // H
    std::vector<float>   s1;     // H (per-channel scales)
    float a1 = 1.f;              // hidden activation scale (layer-wide)
    // L2
    std::vector<int8_t>  W2_q;   // H
    float s2 = 1.f;
    float B2_f = 0.f;

    // State
    bool quantized = false;

    bool load(const std::string& path);
    float forward(Board& board);

};

#endif