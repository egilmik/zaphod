#include "nnueq.h"
#include <fstream>
#include <stdexcept>
#include <cmath>

// ---- feature helpers (yours unchanged) ----
int NNUEQ::plane_index_from_piece(BitBoardEnum piece) {
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
        piece ^= 6;
    }
    return piece * 64 + sq;
}

// ---- loader ----
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
        quantized = true;

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

    if (m.rfind("NNUEV1", 0) == 0) {
        // -------- Float format (legacy) --------
        quantized = false;

        W1.resize(H * IN);
        B1.resize(H);
        W2.resize(OUT * H);

        f.read(reinterpret_cast<char*>(W1.data()), W1.size() * sizeof(float));
        f.read(reinterpret_cast<char*>(B1.data()), B1.size() * sizeof(float));
        f.read(reinterpret_cast<char*>(W2.data()), W2.size() * sizeof(float));
        f.read(reinterpret_cast<char*>(&B2), sizeof(float));
        ensure();

        // clear quantized vectors
        W1_q.clear(); B1_q.clear(); s1.clear(); W2_q.clear();
        s2 = 1.f; a1 = 1.f; B2_f = 0.f;
        return true;
    }

    throw std::runtime_error("Bad magic in weights");
}

// ---- forward ----
float NNUEQ::forward(Board& board) {
    // Gather active features
    int active[48];
    int nActive = 0;
    BitBoard allPieces = board.getBitboard(All);
    while (allPieces) {
        int sq = board.popLsb(allPieces);
        BitBoardEnum piece = board.getPieceOnSquare(sq);
        int plane = plane_index_from_piece(piece);
        if (plane >= 0) active[nActive++] = encodeFeature(plane, sq, board.getSideToMove());
    }

    const float eps = 1e-6f;
    // ----- quantized path -----
    // L1: int32 pre = B1_q + sum W1_q[o, i_active]
    int32_t pre[H];
    for (int o = 0; o < H; ++o) {
        int32_t acc = B1_q[o];
        const int base = o * IN;
        for (int k = 0; k < nActive; ++k)
            acc += (int32_t)W1_q[base + active[k]];
        pre[o] = acc;
    }

    // Reconstruct float pre-act: z = s1[o] * pre[o], ReLU, then quantize to uint8 via a1
    uint8_t hq[H];
    for (int o = 0; o < H; ++o) {
        float z = s1[o] * (float)pre[o];
        float r = z > 0.f ? z : 0.f;
        int q = (int)std::lrint(r / a1);
        if (q < 0) q = 0; else if (q > 127) q = 127;
        hq[o] = (uint8_t)q;
    }

    // L2: int32 dot of (uint8 hq) · (int8 W2_q)
    int32_t acc2 = 0;
    for (int i = 0; i < H; ++i)
        acc2 += (int32_t)((int16_t)W2_q[i]) * (int32_t)hq[i];

    // Final float in model output domain
    float y = B2_f + (s2 * a1) * (float)acc2;

    // Map to cp using your head: y ~ tanh(cp / scale_cp)
    if (y > 1.f - eps) y = 1.f - eps;
    if (y < -1.f + eps) y = -1.f + eps;
    y = 0.5f * std::log((1.f + y) / (1.f - y));
    return y * scale_cp;
}
