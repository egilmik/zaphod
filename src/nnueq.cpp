#include "nnueq.h"
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <immintrin.h>
#include <iostream>
#include <cassert>

using namespace NNUE;

static inline int32_t hsum_epi32_avx2(__m256i v) {
    __m128i vlow = _mm256_castsi256_si128(v);
    __m128i vhigh = _mm256_extracti128_si256(v, 1);
    __m128i sum = _mm_add_epi32(vlow, vhigh);
    __m128i shuf = _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1));
    sum = _mm_add_epi32(sum, shuf);
    shuf = _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2));
    sum = _mm_add_epi32(sum, shuf);
    return _mm_cvtsi128_si32(sum);
}

int64_t NNUEQ::VectorizedSCReLU_AVX2(const int16_t* __restrict stmAcc,const int16_t* __restrict nstmAcc,const int16_t* __restrict wStm,const int16_t* __restrict wNstm) {
#ifndef __AVX2__
#  error "Build Zaphod with AVX2 enabled for this NNUE path."
#endif
    //static_assert((H % 16) == 0, "HL_SIZE must be a multiple of 16 for AVX2 path.");

    const __m256i V_ZERO = _mm256_setzero_si256();
    const __m256i V_QA = _mm256_set1_epi16(static_cast<int16_t>(QA));

    int64_t sum64 = 0;

    for (int i = 0; i < H; i += 16) {
        // load
        __m256i stmAccValues = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(stmAcc + i));
        __m256i nstmAccValues = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(nstmAcc + i));
        __m256i stmWeights = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wStm + i));
        __m256i nstmWeights = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wNstm + i));

        //Clamp
        __m256i stmAccClamped = _mm256_min_epi16(V_QA, _mm256_max_epi16(stmAccValues, V_ZERO));
        __m256i nstmAccClamped = _mm256_min_epi16(V_QA, _mm256_max_epi16(nstmAccValues, V_ZERO));

        // SCreLU
        __m256i stmActivated = _mm256_madd_epi16(stmAccClamped, _mm256_mullo_epi16(stmAccClamped, stmWeights));
        __m256i nstmActivated = _mm256_madd_epi16(nstmAccClamped, _mm256_mullo_epi16(nstmAccClamped, nstmWeights));

        sum64 += static_cast<int64_t>(hsum_epi32_avx2(stmActivated));
        sum64 += static_cast<int64_t>(hsum_epi32_avx2(nstmActivated));

    }
    return sum64; // units: QB * QA^2
}

static inline int64_t div_round_i64(int64_t num, int64_t den) {
    // symmetric round-to-nearest
    return (num >= 0) ? (num + den / 2) / den : -((-num + den / 2) / den);
}

int NNUEQ::forward(BitBoardEnum stm) {
    const auto& stmAcc = (stm == White) ? accumulator[0] : accumulator[1];
    const auto& nstmAcc = (stm == White) ? accumulator[1] : accumulator[0];

    const int16_t* a = stmAcc.pre.data();
    const int16_t* b = nstmAcc.pre.data();
    const int16_t* w0 = net.l1w.data();        // first H  = stm
    const int16_t* w1 = w0 + H;            // next  H  = ntm

    int64_t sum = VectorizedSCReLU_AVX2(a, b, w0, w1);        // QB*QA^2
    int64_t sum_qaqb = div_round_i64(sum, QA);                // QA*QB
    int64_t acc = sum_qaqb + net.l1b;                             // QA*QB
    int64_t out = div_round_i64(acc * SCALE, (int64_t)QA * QB);

    // optional clamp to engine range
    if (out > 30000) out = 30000;
    if (out < -30000) out = -30000;
    return (int)out;
}

void NNUEQ::removePiece(BitBoardEnum piece, int sq) {
    if (!isInitialized) {
        return;
    }
    int plane = plane_index_from_piece(piece);
    int featureWhite = encodeFeature(plane, sq, White);
    int featureBlack = encodeFeature(plane, sq, Black);

    const int16_t* weightW = net.l0w.data() + featureWhite * H;
    const int16_t* weightB = net.l0w.data() + featureBlack * H;
    int16_t* accW = accumulator[0].pre.data();
    int16_t* accB = accumulator[1].pre.data();


    #if defined(__AVX2__)
    sub_row_i16_avx2(weightW, accW);
    sub_row_i16_avx2(weightB, accB);

    #else
    for (int i = 0; i < H; ++i) { accW[i] -= weightW[i]; accB[i] -= weightB[i]; }
    #endif
}

void NNUEQ::addPiece(BitBoardEnum piece, int sq) {
    if (!isInitialized) {
        return;
    }
    int plane = plane_index_from_piece(piece);
    int featureWhite = encodeFeature(plane, sq, White);
    int featureBlack = encodeFeature(plane, sq, Black);
    
    const int16_t* weightW = net.l0w.data() + featureWhite * H;
    const int16_t* weightB = net.l0w.data() + featureBlack * H;
    int16_t* accW = accumulator[0].pre.data();
    int16_t* accB = accumulator[1].pre.data();
    

    #if defined(__AVX2__)
        

        add_row_i16_avx2(weightW, accW);
        add_row_i16_avx2(weightB, accB);

    #else
    for (int i = 0; i < H; ++i) { accW[i] += weightW[i]; accB[i] += weightB[i]; }
    #endif
}

void NNUEQ::clear() {
    if (!isInitialized) {
        return;
    }

    for (int h = 0; h < H; ++h) {
        accumulator[0].pre[h] = net.l0b[h];
        accumulator[1].pre[h] = net.l0b[h];
    }
}

void NNUEQ::add_row_i16_avx2(const int16_t* __restrict w, int16_t* __restrict acc) {
    for (int i = 0; i < H; i += 32) {
        __m256i a0 = _mm256_loadu_si256((const __m256i*)(acc + i));
        __m256i w0 = _mm256_loadu_si256((const __m256i*)(w + i));
        __m256i a1 = _mm256_loadu_si256((const __m256i*)(acc + i + 16));
        __m256i w1 = _mm256_loadu_si256((const __m256i*)(w + i + 16));
        _mm256_storeu_si256((__m256i*)(acc + i), _mm256_adds_epi16(a0, w0)); // sat add
        _mm256_storeu_si256((__m256i*)(acc + i + 16), _mm256_adds_epi16(a1, w1));
    }
}

void NNUEQ::sub_row_i16_avx2(const int16_t* __restrict w,int16_t* __restrict acc) {
    for (int i = 0; i < H; i += 32) {
        __m256i a0 = _mm256_loadu_si256((const __m256i*)(acc + i));
        __m256i w0 = _mm256_loadu_si256((const __m256i*)(w + i));
        __m256i a1 = _mm256_loadu_si256((const __m256i*)(acc + i + 16));
        __m256i w1 = _mm256_loadu_si256((const __m256i*)(w + i + 16));
        _mm256_storeu_si256((__m256i*)(acc + i), _mm256_subs_epi16(a0, w0)); // sat sub
        _mm256_storeu_si256((__m256i*)(acc + i + 16), _mm256_subs_epi16(a1, w1));
    }
}

int NNUEQ::plane_index_from_piece(BitBoardEnum piece) {
    // [P,N,B,R,Q,K,p,n,b,r,q,k] -> 0..11, else -1
    switch (piece) {
    case P: return 0; case N: return 1; case B: return 2;
    case R: return 3; case Q: return 4; case K: return 5;
    case p: return 6; case n: return 7; case b: return 8;
    case r: return 9; case q: return 10; case k: return 11;
    default:  return -1;
    }
}

int NNUEQ::encodeFeature(int piece, int sq, BitBoardEnum color) {
    if (color == Black) {
        sq ^= 56;
        piece = (piece < 6) ? (piece + 6) : (piece - 6);
    }
    return piece * 64 + sq;
}


bool NNUEQ::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    

    accumulator.push_back(Accumulator(H));
    accumulator.push_back(Accumulator(H));

    
    f.read((char*)net.l0w.data(), net.l0w.size() * sizeof(int16_t));
    f.read((char*)net.l0b.data(), net.l0b.size() * sizeof(int16_t));    
    f.read((char*)net.l1w.data(), net.l1w.size() * sizeof(int16_t));
    f.read(reinterpret_cast<char*>(&net.l1b), sizeof(int16_t));


    clear();
    isInitialized = true;
    return true;
}

