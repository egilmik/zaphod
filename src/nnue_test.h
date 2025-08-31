#pragma once
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct NNUE768x32x1 {
    // dims
    static constexpr int IN = 768;
    static constexpr int H = 32;
    static constexpr int OUT = 1;

    // parameters
    std::vector<float> W1; // [H][IN] row-major
    std::vector<float> B1; // [H]
    std::vector<float> W2; // [OUT][H] (i.e. 32)
    float B2 = 0.f;
    float scale_cp = 600.f;

    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;

        char magic[8];
        f.read(reinterpret_cast<char*>(magic), 8);
        if (!f || std::string(magic, magic + 6) != "NNUEV1")
            throw std::runtime_error("Bad magic in weights");

        int32_t in = 0, h = 0, out = 0;
        f.read(reinterpret_cast<char*>(&in), 4);
        f.read(reinterpret_cast<char*>(&h), 4);
        f.read(reinterpret_cast<char*>(&out), 4);
        if (!f || in != IN || h != H || out != OUT)
            throw std::runtime_error("Unexpected dimensions in weights");

        f.read(reinterpret_cast<char*>(&scale_cp), 4);
        if (!f) throw std::runtime_error("Failed reading scale");

        W1.resize(H * IN);
        B1.resize(H);
        W2.resize(OUT * H);

        f.read(reinterpret_cast<char*>(W1.data()), W1.size() * sizeof(float));
        f.read(reinterpret_cast<char*>(B1.data()), B1.size() * sizeof(float));
        f.read(reinterpret_cast<char*>(W2.data()), W2.size() * sizeof(float));
        f.read(reinterpret_cast<char*>(&B2), sizeof(float));
        if (!f) throw std::runtime_error("Weights truncated");
        return true;
    }

    // y ≈ tanh(cp_white / scale); convert to cp by atanh with clipping
    static inline float atanh_clip(float y, float eps = 1e-6f) {
        if (y > 1.f - eps) y = 1.f - eps;
        if (y < -1.f + eps) y = -1.f + eps;
        return 0.5f * std::log((1.f + y) / (1.f - y));
    }

    // Forward: x[768] -> scalar
    inline float forward(const float* x) const {
        float hbuf[H];
        // h = ReLU(W1 * x + B1)
        for (int i = 0; i < H; ++i) {
            const float* w = &W1[i * IN];
            float s = B1[i];
            // unrolled dot is possible; keep simple
            for (int j = 0; j < IN; ++j) s += w[j] * x[j];
            hbuf[i] = s > 0.f ? s : 0.f;
        }
        // y = W2 * h + B2
        float y = B2;
        for (int i = 0; i < H; ++i) y += W2[i] * hbuf[i];
        return y;
    }

    // Convert NN output to centipawns (white POV)
    inline float output_to_cp_white(float y) const {
        return atanh_clip(y) * scale_cp;
    }
};

// --------- FEN -> 12x64 one-hot (A1=0..H8=63) ---------
inline int plane_index_from_piece(char p) {
    // [P,N,B,R,Q,K,p,n,b,r,q,k] -> 0..11, else -1
    switch (p) {
    case 'P': return 0; case 'N': return 1; case 'B': return 2;
    case 'R': return 3; case 'Q': return 4; case 'K': return 5;
    case 'p': return 6; case 'n': return 7; case 'b': return 8;
    case 'r': return 9; case 'q': return 10; case 'k': return 11;
    default:  return -1;
    }
}

// Returns side-to-move (true if white)
inline bool encode_fen_12x64(const std::string& fen, float out[768]) {
    std::fill(out, out + 768, 0.f);

    // split fields
    size_t sp1 = fen.find(' ');
    if (sp1 == std::string::npos) throw std::runtime_error("Bad FEN (no space)");
    size_t sp2 = fen.find(' ', sp1 + 1);
    if (sp2 == std::string::npos) throw std::runtime_error("Bad FEN (no side)");

    std::string board = fen.substr(0, sp1);
    std::string stm = fen.substr(sp1 + 1, sp2 - (sp1 + 1));

    // ranks 8..1
    int rank = 7; // 0-based index for rank, 7==rank8,...,0==rank1
    size_t pos = 0;
    for (int r = 0; r < 8; ++r) {
        if (pos >= board.size()) throw std::runtime_error("Bad FEN board");
        int file = 0;
        while (pos < board.size() && board[pos] != '/') {
            char c = board[pos++];
            if (std::isdigit(static_cast<unsigned char>(c))) {
                file += (c - '0');
            }
            else {
                int pl = plane_index_from_piece(c);
                if (pl >= 0) {
                    if (file < 0 || file > 7) throw std::runtime_error("File overflow");
                    int sq = (rank) * 8 + file; // A1=0 -> rank1=0, so rank==0 is bottom
                    // But rank currently 7..0 where 7==rank8=>OK because A1=0, A8=56
                    out[pl * 64 + sq] = 1.f;
                }
                ++file;
            }
        }
        if (file != 8) throw std::runtime_error("Bad FEN rank width");
        if (pos < board.size() && board[pos] == '/') ++pos;
        --rank;
    }
    if (rank != -1) throw std::runtime_error("Bad FEN (rank count)");

    bool white_to_move = (stm == "w");
    return white_to_move;
}
