#include "nnueq.h"
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <immintrin.h>
#include "nnue.h"

float NNUEQ::forward(BitBoardEnum stm) {
    // Gather active features
    int active[48];
    int nActive = 0;

    int side = stm == White ? 0 : 1;

    

    const float eps = 1e-6f;
    // ----- quantized path -----
    // L1: int32 pre = B1_q + sum W1_q[o, i_active]
    // Reconstruct float pre-act: z = s1[o] * pre[o], ReLU, then quantize to uint8 via a1
    // L2: int32 dot of (uint8 hq) · (int8 W2_q)
    int32_t acc2 = 0;
    for (int i = 0; i < H; ++i) {
        int32_t acc = B1_q[i] +accumulator[side].pre[i];

        float z = s1[i] * (float)acc;
        z = z > 0.f ? z : 0.f;
        int32_t q = std::floor(z / a1);
        if (q < 0) q = 0; else if (q > 127) q = 127;

        acc2 += (int32_t)(W2_q[i] * q);
    }

    // Final float in model output domain
    float y = B2_f + (s2 * a1) * (float)acc2;

    // Map to cp using your head: y ~ tanh(cp / scale_cp)
    if (y > 1.f - eps) y = 1.f - eps;
    if (y < -1.f + eps) y = -1.f + eps;
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
