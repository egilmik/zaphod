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

class NNUEQ {
public:
    const int IN = 768;
    const int H = 64;
    const int OUT = 1;

    

    const int16_t SCALE = 400;
    const int16_t QA = 255;
    const int16_t QB = 64;

    
    // L1
    std::vector<int16_t>  l0w;   // H*IN
    std::vector<int16_t> l0b;   // H
    
    // L2
    std::vector<int16_t>  l1w;   // H
    int16_t l1b;
    

    // 0 White, 1 Black
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

};

#endif
