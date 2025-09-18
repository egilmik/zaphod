#include "nnueq.h"
#include <fstream>
#include <stdexcept>
#include <cmath>
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
    uint8_t hq[H];
    int32_t acc2 = 0;
    for (int i = 0; i < H; ++i) {
        int32_t acc = B1_q[i] +accumulator[side].pre[i];

        float z = s1[i] * (float)acc;
        z = z > 0.f ? z : 0.f;
        int q = (int)std::lrint(z / a1);
        if (q < 0) q = 0; else if (q > 127) q = 127;
        hq[i] = (uint8_t)q;

        acc2 += (int32_t)((int16_t)W2_q[i]) * (int32_t)hq[i];
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

    for (int i = 0; i < H; i++) {
        accumulator[0].pre[i] -= (int32_t)W1_q[featureWhite + (i * 768)];
        accumulator[1].pre[i] -= (int32_t)W1_q[featureBlack + (i * 768)];
    }
}

void NNUEQ::addPiece(BitBoardEnum piece, int sq) {
    int plane = NNUE::plane_index_from_piece(piece);
    int featureWhite = NNUE::encodeFeature(plane, sq, White);
    int featureBlack = NNUE::encodeFeature(plane, sq, Black);
    
    for (int i = 0; i < H; i++) {
        accumulator[0].pre[i] += (int32_t)W1_q[featureWhite + (i * 768)];
        accumulator[1].pre[i] += (int32_t)W1_q[featureBlack + (i * 768)];
    }
}

void NNUEQ::clear() {
    accumulator[0].pre.fill(0);
    accumulator[1].pre.fill(0);
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

        W1_q.resize(H * IN);
        f.read(reinterpret_cast<char*>(W1_q.data()), W1_q.size() * sizeof(int8_t)); ensure();

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