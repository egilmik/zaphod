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

#include "bitboard.h";

struct Accumulator {
    explicit Accumulator(std::size_t n) : pre(n) {}
    std::vector<int16_t> pre;
};

class NNUEQ {
public:
    int IN = 768;
    int H = 128;
    int OUT = 1;

    // --- Common ---
    float scale_cp = 600.f; // cp scale used by training head

    // --- Float path (NNUEV1) ---
    std::vector<float> W1, B1, W2;
    float B2 = 0.f;

    // --- Quantized path (NNUEQ1) ---
    // L1
    std::vector<int16_t>  W1_q;   // H*IN
    std::vector<int32_t> B1_q;   // H
    std::vector<float>   s1;     // H (per-channel scales)
    float a1 = 1.f;              // hidden activation scale (layer-wide)
    // L2
    std::vector<int8_t>  W2_q;   // H
    std::vector<float> W2_f;   // W2_f[i] = W2_q[i] * (s2 * a1)
    float inv_a1 = 1.0f / 127;
    float s2 = 1.f;
    float B2_f = 0.f;

    //For Clipped relu
    int qCap = 0;

    // 0 White, 1 Black
    std::vector<Accumulator> accumulator;

    // State
    bool quantized = false;

    bool load(const std::string& path);
    float forward(BitBoardEnum stm);

    void addPiece(BitBoardEnum piece, int sq);
    void removePiece(BitBoardEnum piece, int sq);

    void clear();

    void add_row_i16_avx2(const int16_t* __restrict w, int16_t* __restrict acc);
    void sub_row_i16_avx2(const int16_t* __restrict w, int16_t* __restrict acc);
    void build_feature_major_rows(const std::vector<int8_t>& src, std::vector<int16_t>& dst);

    static int plane_index_from_piece(BitBoardEnum piece);
    static int encodeFeature(int piece, int sq, BitBoardEnum color);

};

#endif