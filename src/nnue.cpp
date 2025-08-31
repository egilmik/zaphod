#include "nnue.h"

float NNUE::forward(Board& board) {
	
	float input[768] = {};

    BitBoard allPieces = board.getBitboard(All);
    int square = 0;
    while (allPieces) {
        square = board.popLsb(allPieces);
        BitBoardEnum piece = board.getPieceOnSquare(square);
        int plane = plane_index_from_piece(piece);

        input[plane * 64 + square] = 1.f;
    }

    float hidden[32] = {};

    // Calculate first hidden
    for (int i = 0; i < 32; i++) {

        // Init hidden node with bias
        float value = B1[i]; 

        for (int j = 0; j < 768; j++) {
            value += W1[i * 768+j] * input[j];
        }
        hidden[i] = value > 0.f ? value : 0.f;
    }

    //Calculate output
    float output = B2;
    for (int i = 0; i < 32; i++) {
        output += W2[i] * hidden[i];
    }

    float eps = 1e-6f;

    // y = tanh(cp / scale )
    if (output > 1.f - eps) output = 1.f - eps;
    if (output < -1.f + eps) output = -1.f + eps;
    output =  0.5f * std::log((1.f + output) / (1.f - output));

    output *= scale_cp;
    return output;

}

 int NNUE::plane_index_from_piece(BitBoardEnum piece) {
    // [P,N,B,R,Q,K,p,n,b,r,q,k] -> 0..11, else -1
    switch (piece) {
    case P: return 0; case N: return 1; case B: return 2;
    case R: return 3; case Q: return 4; case K: return 5;
    case p: return 6; case n: return 7; case b: return 8;
    case r: return 9; case q: return 10; case k: return 11;
    default:  return -1;
    }
}

bool NNUE::load(const std::string& path) {
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