#include "nnueq.h"
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <immintrin.h>
#include "nnue.h"

float NNUEQ::forward(BitBoardEnum stm) {
    const int side = (stm == White) ? 0 : 1;
    const float eps = 1e-6f;

    const int16_t* __restrict acc16 = accumulator[side].pre.data();
    const int32_t* __restrict B1 = B1_q.data();
    const float* __restrict S1 = s1.data();
    const float* __restrict W2F = W2_f.data();   // prebuilt W2_q*(s2*a1)
    const int8_t* __restrict W2Q = W2_q.data();
    __m256 sum = _mm256_setzero_ps();
    int32_t acc2 = 0;
    const __m256 zero = _mm256_setzero_ps();
    const __m256 a1ps = _mm256_set1_ps(a1);

    int i = 0;
    // Process 16 hidden units per iteration
    for (; i + 16 <= H; i += 16) {
        // --- widen pre = B1_q + acc16 ---
        __m256i x16 = _mm256_load_si256((const __m256i*)(acc16 + i));
        __m128i lo16 = _mm256_castsi256_si128(x16);
        __m128i hi16 = _mm256_extracti128_si256(x16, 1);
        __m256i lo32 = _mm256_cvtepi16_epi32(lo16);
        __m256i hi32 = _mm256_cvtepi16_epi32(hi16);

        __m256i b0 = _mm256_load_si256((const __m256i*)(B1 + i));
        __m256i b1 = _mm256_load_si256((const __m256i*)(B1 + i + 8));
        lo32 = _mm256_add_epi32(lo32, b0);
        hi32 = _mm256_add_epi32(hi32, b1);

        // --- z = ReLU(s1 * pre) ---
        __m256 z0 = _mm256_mul_ps(_mm256_cvtepi32_ps(lo32), _mm256_load_ps(S1 + i));
        __m256 z1 = _mm256_mul_ps(_mm256_cvtepi32_ps(hi32), _mm256_load_ps(S1 + i + 8));
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
        __m256i max127 = _mm256_set1_epi32(127);
        q0i = _mm256_min_epi32(_mm256_max_epi32(q0i, zero32), max127);
        q1i = _mm256_min_epi32(_mm256_max_epi32(q1i, zero32), max127);

        // --- Finish L2 exactly like scalar: int32 acc2 += (int32)(W2_q[i+k] * q[k]) ---
        // Extract to scalars for exact per-lane multiply-add (still faster overall because q computation is vectorized)
        alignas(32) int32_t qbuf[16];
        _mm256_store_si256((__m256i*)qbuf, q0i);
        _mm256_store_si256((__m256i*)(qbuf + 8), q1i);

        // Scalar L2 dot over 16 lanes (int8*int32 -> int32)
        const int8_t* w2 = W2Q + i;
        for (int k = 0; k < 16; ++k)
            acc2 += (int32_t)w2[k] * qbuf[k];
    }

    /*
    // Horizontal add of 'sum'
    alignas(32) float buf[8];
    _mm256_store_ps(buf, sum);
    float acc2f = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
    */

    // Final float in model output domain (same as your code)
    float y = B2_f + (float)acc2*s2*a1;

    // inverse tanh to centipawns
    const float one_minus = 1.f - 1e-6f;
    if (y > one_minus) y = one_minus;
    if (y < -one_minus) y = -one_minus;
    y = 0.5f * std::log((1.f + y) / (1.f - y));
    return y * scale_cp;
}

void NNUEQ::removePiece(BitBoardEnum piece, int sq) {
    int plane = NNUE::plane_index_from_piece(piece);
    int featureWhite = NNUE::encodeFeature(plane, sq, White);
    int featureBlack = NNUE::encodeFeature(plane, sq, Black);

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
    int plane = NNUE::plane_index_from_piece(piece);
    int featureWhite = NNUE::encodeFeature(plane, sq, White);
    int featureBlack = NNUE::encodeFeature(plane, sq, Black);
    
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
    accumulator[0].pre.fill(0);
    accumulator[1].pre.fill(0);
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


bool NNUEQ::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    char magic[8] = {};
    f.read(reinterpret_cast<char*>(magic), 8);
    if (!f) throw std::runtime_error("Failed reading magic");

    int32_t in = 0, h = 0, out = 0;
    auto rd_i32 = [&](int32_t& x) { f.read(reinterpret_cast<char*>(&x), 4); };
    auto rd_f32 = [&](float& x) { f.read(reinterpret_cast<char*>(&x), 4); };
    auto ensure = [&]() { if (!f) throw std::runtime_error("Weights truncated"); };

    // Common dims + cp scale first (both formats)
    rd_i32(in); rd_i32(h); rd_i32(out); ensure();
    if (in != IN || h != H || out != OUT)
        throw std::runtime_error("Unexpected dimensions in weights");

    rd_f32(scale_cp); ensure();

    std::string m(magic, magic + 8);

    if (m.rfind("NNUEQ1", 0) == 0) {
        // -------- Quantized format --------
        

        // L1
        s1.resize(H);
        f.read(reinterpret_cast<char*>(s1.data()), H * sizeof(float)); ensure();

        std::vector<int8_t> w1Temp;

        w1Temp.resize(H * IN);
        f.read(reinterpret_cast<char*>(w1Temp.data()), w1Temp.size() * sizeof(int8_t)); ensure();

        build_feature_major_rows(w1Temp, W1_q);


        B1_q.resize(H);
        f.read(reinterpret_cast<char*>(B1_q.data()), H * sizeof(int32_t)); ensure();

        rd_f32(a1); ensure();

        // L2
        rd_f32(s2); ensure();

        W2_q.resize(H);
        f.read(reinterpret_cast<char*>(W2_q.data()), H * sizeof(int8_t)); ensure();

        W2_f.resize(H);
        for (int i = 0; i < H; ++i) {
            W2_f[i] = float(W2_q[i]) * (s2 * a1);
        }

        rd_f32(B2_f); ensure();

        // clear float vectors to save RAM
        W1.clear(); B1.clear(); W2.clear(); B2 = 0.f;
        return true;
    }

    throw std::runtime_error("Bad magic in weights");
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
