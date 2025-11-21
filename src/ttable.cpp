#include "ttable.h"

void TTable::put(uint64_t key, int score, int staticEval, int depth, Move move, TType type) {
    TTEntry& tte = table[index(key)];
    uint64_t existingKey = tte.key;
    if (existingKey == key && depth <= tte.depth) {
        return;
    }
    tte.key = key;
    tte.bestMove = move;
    tte.score = score;
    tte.depth = depth;
    tte.staticEval = staticEval;
    tte.type = type;
}