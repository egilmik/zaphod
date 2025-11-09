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

struct Accumulator {
    explicit Accumulator(std::size_t n) : pre(n) {}
    std::vector<int16_t> pre;
};

namespace NNUE{
    constexpr int IN = 768;
    constexpr int H = 256;
    constexpr int OUT = 1;

    constexpr int16_t SCALE = 400;
    constexpr int16_t QA = 255;
    constexpr int16_t QB = 64;

    struct Network {
        alignas(32) std::array<int16_t, IN*H>  l0w;
        alignas(32) std::array<int16_t, H>  l0b;
        alignas(32) std::array<int16_t, H*2>  l1w;
        int16_t l1b;
    };
}

using namespace NNUE;

class NNUEQ {
public:
    
    NNUEQ() : net(std::make_unique<Network>()) {}
  
    std::vector<Accumulator> accumulator;

    bool load(const std::string& path);
    int forward(BitBoardEnum stm);

    void addPiece(BitBoardEnum piece, int sq);
    void removePiece(BitBoardEnum piece, int sq);

    void clear();

    void add_row_i16_avx2(const int16_t* __restrict w, int16_t* __restrict acc);
    void sub_row_i16_avx2(const int16_t* __restrict w, int16_t* __restrict acc);

    int64_t VectorizedSCReLU_AVX2(const int16_t* __restrict stmAcc, const int16_t* __restrict nstmAcc, const int16_t* __restrict wStm, const int16_t* __restrict wNstm);

    static int plane_index_from_piece(BitBoardEnum piece);
    static int encodeFeature(int piece, int sq, BitBoardEnum color);


private:
    // Set to true when network is loaded.
    bool isInitialized = false;
    std::unique_ptr<Network> net;
    

};

#endif
