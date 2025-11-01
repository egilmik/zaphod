#include "nnueq.h"
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <immintrin.h>
#include <iostream>
#include <cassert>

float NNUEQ::forward(BitBoardEnum stm) {
    const int side = (stm == White) ? 0 : 1;

const int accH = (int)accumulator[side].pre.size();
const int s1H  = (int)s1.size();
const int w2H  = (int)W2_q.size();



    const int16_t* __restrict acc16 = accumulator[side].pre.data();
    const float* __restrict S1 = s1.data();
    const int8_t* __restrict W2Q = W2_q.data();

    int32_t acc2 = 0;
#if defined(__AVX512F__)
    // ===== your existing AVX-512 code (unchanged) =====
    const __m512  zero_ps = _mm512_setzero_ps();
    const __m512  a1_ps = _mm512_set1_ps(a1);
    const __m512i zero_i = _mm512_setzero_si512();
    const __m512i qcap32 = _mm512_set1_epi32((int)qCap);

    alignas(64) static thread_local int32_t qbuf[32];

    for (int i = 0; i < H; i += 32) {
        __m512i pre_i16 = _mm512_loadu_si512((const void*)(acc16 + i));
        __m256i lo16 = _mm512_castsi512_si256(pre_i16);
        __m256i hi16 = _mm512_extracti64x4_epi64(pre_i16, 1);
        __m512i lo32 = _mm512_cvtepi16_epi32(lo16);
        __m512i hi32 = _mm512_cvtepi16_epi32(hi16);

        __m512 z0 = _mm512_mul_ps(_mm512_cvtepi32_ps(lo32), _mm512_loadu_ps(S1 + i));
        __m512 z1 = _mm512_mul_ps(_mm512_cvtepi32_ps(hi32), _mm512_loadu_ps(S1 + i + 16));
        z0 = _mm512_max_ps(z0, zero_ps);
        z1 = _mm512_max_ps(z1, zero_ps);

        __m512 q0f = _mm512_floor_ps(_mm512_div_ps(z0, a1_ps));
        __m512 q1f = _mm512_floor_ps(_mm512_div_ps(z1, a1_ps));

        __m512i q0i = _mm512_cvttps_epi32(q0f);
        __m512i q1i = _mm512_cvttps_epi32(q1f);
        q0i = _mm512_min_epi32(_mm512_max_epi32(q0i, zero_i), qcap32);
        q1i = _mm512_min_epi32(_mm512_max_epi32(q1i, zero_i), qcap32);

        auto acc_end = acc16 + H;
        auto s1_end = S1 + H;
        auto w2_end = W2Q + H;
        assert(acc16 + i + 32 <= acc_end);
        assert(S1 + i + 32 <= s1_end);
        assert(W2Q + i + 32 <= w2_end);

        _mm512_storeu_si512((void*)qbuf, q0i);
        _mm512_storeu_si512((void*)(qbuf + 16), q1i);

        const int8_t* w2 = W2Q + i;
        for (int k = 0; k < 32; ++k)
            acc2 += (int32_t)w2[k] * qbuf[k];
    }
    // ===== end of your AVX-512 block =====

#elif defined(__AVX2__)
    
    const int32_t* __restrict B1 = B1_q.data();    
    const float* __restrict W2F = W2_f.data();   // prebuilt W2_q*(s2*a1)
    __m256 sum = _mm256_setzero_ps();
    const __m256 zero = _mm256_setzero_ps();
    const __m256 a1ps = _mm256_set1_ps(a1);
    const __m256i max127 = _mm256_set1_epi32(127);

    alignas(64) static thread_local int32_t qbuf[32];
    int i = 0;
    // Process 16 hidden units per iteration
    for (; i + 16 <= H; i += 16) {
        // --- widen pre = B1_q + acc16 ---
        __m256i x16 = _mm256_loadu_si256((const __m256i*)(acc16 + i));
        __m128i lo16 = _mm256_castsi256_si128(x16);
        __m128i hi16 = _mm256_extracti128_si256(x16, 1);
        __m256i lo32 = _mm256_cvtepi16_epi32(lo16);
        __m256i hi32 = _mm256_cvtepi16_epi32(hi16);

        // --- z = ReLU(s1 * pre) ---
        __m256 z0 = _mm256_mul_ps(_mm256_cvtepi32_ps(lo32), _mm256_loadu_ps(S1 + i));
        __m256 z1 = _mm256_mul_ps(_mm256_cvtepi32_ps(hi32), _mm256_loadu_ps(S1 + i + 8));
        z0 = _mm256_max_ps(z0, zero);
        z1 = _mm256_max_ps(z1, zero);

        // --- q = floor(z / a1), not z * inv_a1 ---
        __m256 q0f = _mm256_div_ps(z0, a1ps);
        __m256 q1f = _mm256_div_ps(z1, a1ps);
        q0f = _mm256_floor_ps(q0f);
        q1f = _mm256_floor_ps(q1f);

        // convert to int32 and clamp [0,127]
        __m256i q0i = _mm256_cvttps_epi32(q0f);
        __m256i q1i = _mm256_cvttps_epi32(q1f);
        __m256i zero32 = _mm256_setzero_si256();
        
        q0i = _mm256_min_epi32(_mm256_max_epi32(q0i, zero32), max127);
        q1i = _mm256_min_epi32(_mm256_max_epi32(q1i, zero32), max127);

        const int8_t* w2 = W2Q + i;

        __m128i w8 = _mm_loadu_si128((const __m128i*)w2);
        __m128i w8hi = _mm_srli_si128(w8, 8);

        // sign-extend to 8×int32 twice
        __m256i wlo32 = _mm256_cvtepi8_epi32(w8);    // weights 0..7
        __m256i whi32 = _mm256_cvtepi8_epi32(w8hi);  // weights 8..15

        // q0i, q1i are your clamped 8×int32 blocks (0..127), already computed
        __m256i mul0 = _mm256_mullo_epi32(q0i, wlo32);
        __m256i mul1 = _mm256_mullo_epi32(q1i, whi32);
        __m256i sum = _mm256_add_epi32(mul0, mul1);

        // horizontal reduce to scalar
        __m128i s128 = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
        s128 = _mm_hadd_epi32(s128, s128);
        s128 = _mm_hadd_epi32(s128, s128);

        acc2 += _mm_cvtsi128_si32(s128);
    }
#else
    // ===== Scalar fallback =====
    for (int i = 0; i < H; ++i) {
        float z = std::max(0.0f, (float)acc16[i] * S1[i]);
        int q = (int)std::floor(z / a1);
        if (q < 0) q = 0; else if (q > qCap) q = qCap;
        acc2 += (int)W2Q[i] * q;
    }
#endif

    float y = B2_f + (s2 * a1) * (float)acc2;
    const float one_minus = 1.f - 1e-6f;
    if (y > one_minus) y = one_minus; else if (y < -one_minus) y = -one_minus;
    y = 0.5f * std::log((1.f + y) / (1.f - y));
    return y * scale_cp;
}


void NNUEQ::removePiece(BitBoardEnum piece, int sq) {
    int plane = plane_index_from_piece(piece);
    int featureWhite = encodeFeature(plane, sq, White);
    int featureBlack = encodeFeature(plane, sq, Black);

    const int16_t* weightW = W1_q.data() + featureWhite * H;
    const int16_t* weightB = W1_q.data() + featureBlack * H;
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
    int plane = plane_index_from_piece(piece);
    int featureWhite = encodeFeature(plane, sq, White);
    int featureBlack = encodeFeature(plane, sq, Black);
    
    const int16_t* weightW = W1_q.data() + featureWhite * H;
    const int16_t* weightB = W1_q.data() + featureBlack * H;
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
    for (int h = 0; h < H; ++h) {
        accumulator[0].pre[h] = (int16_t)B1_q[h];
        accumulator[1].pre[h] = (int16_t)B1_q[h];
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

    auto rd_i32 = [&](int32_t& x) { f.read((char*)&x, 4); };
    auto rd_f32 = [&](float& x) { f.read((char*)&x, 4); };
    auto ensure = [&]() { if (!f) throw std::runtime_error("Weights truncated"); };

    char magic[8] = {};
    f.read(magic, 8); ensure();
    const std::string m(magic, magic + 8);

    int32_t in = 0, h = 0, out = 0;
    rd_i32(in); rd_i32(h); rd_i32(out); ensure();
    
    IN = in;
    H = h;
    OUT = out;

    accumulator.push_back(Accumulator(H));
    accumulator.push_back(Accumulator(H));

    rd_f32(scale_cp); ensure();

    
    // L1
    s1.resize(H);
    f.read((char*)s1.data(), H * sizeof(float)); ensure();

    std::vector<int8_t> w1_hidden_major(H * IN);
    f.read((char*)w1_hidden_major.data(), w1_hidden_major.size()); ensure();

    // Default: build feature-major i16 at load (fast row ops)
    build_feature_major_rows(w1_hidden_major, W1_q);

    B1_q.resize(H);
    f.read((char*)B1_q.data(), H * sizeof(int32_t)); ensure();

    // Act: a1 then q_cap (uint8)
    rd_f32(a1); ensure();

    // Try to read Q_CAP (uint8). If older file w/o it, default to floor(1/a1).
    uint8_t qcap_file = 0;
    if (f.peek() != std::char_traits<char>::eof()) {
        f.read((char*)&qcap_file, 1);
        if (!f) throw std::runtime_error("Weights truncated at Q_CAP");
        qCap = qcap_file;
    }
    else {
        qCap = (uint8_t)std::min(127, (int)std::floor(1.0f / a1));
    }

    // L2
    rd_f32(s2); ensure();

    W2_q.resize(H);
    f.read((char*)W2_q.data(), H * sizeof(int8_t)); ensure();

    rd_f32(B2_f); ensure();

    // precompute W2_f for float FMA path 
    W2_f.resize(H);
    for (int i = 0; i < H; ++i) {
        W2_f[i] = float(W2_q[i]) * (s2 * a1);
    }

    
    return true;
}

// src: [H_REAL][IN_FEATS] in hidden-major
// dst: [IN_FEATS][H_PAD] in feature-major
void NNUEQ::build_feature_major_rows(const std::vector<int8_t>& src,std::vector<int16_t>& dst) {
    dst.assign(size_t(IN) * size_t(H), 0);

    for (int h = 0; h < H; ++h) {
        const int8_t* src_row = &src[size_t(h) * IN];
        for (int f = 0; f < IN; ++f) {
            // each feature f has its own contiguous row in dst
            dst[size_t(f) * H + h] = static_cast<int16_t>(src_row[f]);
        }
    }
}
