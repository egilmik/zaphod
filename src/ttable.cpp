#include "ttable.h"

void TTable::put(uint64_t key, int score, int staticEval, int depth, Move move, TType type) {
    int tableIdx = index(key);

    auto& entries = table[tableIdx];
    for (auto& tte : entries.entries) {
        if (tte.key == key || tte.type == TType::NO_TYPE) {
            if (tte.depth < depth) {
                tte.key = key;
                tte.bestMove = move;
                tte.score = score;
                tte.depth = depth;
                tte.staticEval = staticEval;
                tte.type = type;
                break;
            }
        }
    }
    
}