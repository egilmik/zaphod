#include <iostream>
#include "nnue_test.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " weights.nnue \"<FEN>\"\n";
        return 1;
    }
    const std::string wpath = argv[1];
    const std::string fen = argv[2];

    NNUE768x32x1 net;
    if (!net.load(wpath)) {
        std::cerr << "Failed to load weights\n";
        return 2;
    }

    float x[NNUE768x32x1::IN];
    bool stm_white = encode_fen_12x64(fen, x);

    float y = net.forward(x);                       // ~tanh(cp_white/scale)
    float cp_white = net.output_to_cp_white(y);     // white POV
    float cp_stm = stm_white ? cp_white : -cp_white;

    std::cout << "y (raw)          : " << y << "\n";
    std::cout << "eval white [cp]  : " << cp_white << "\n";
    std::cout << "eval  stm  [cp]  : " << cp_stm
        << "  (stm=" << (stm_white ? "w" : "b") << ")\n";
    return 0;
}
