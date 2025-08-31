#include <iostream>
#include "nnue_test.h"

int main(int argc, char** argv) {
    
    const std::string wpath = "D:\\weights.nnue";
    const std::string fen = "rn1qkb1r/pp2pppp/5n2/3p1b2/3P4/2N1P3/PP3PPP/R1BQKBNR w KQkq - 0 1";

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
