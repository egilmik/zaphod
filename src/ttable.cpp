#include "ttable.h"
#include <cassert>

void TTable::put(uint64_t key, int score, int staticEval, int depth, Move move, TType type) {
    int tableIdx = index(key);

    auto& entries = table[tableIdx];
    TTEntry* entryPtr = nullptr;
    auto minAge = std::numeric_limits<int32_t>::max();

    for (auto& candidate : entries.entries) {
        if (candidate.key == key || candidate.type == TType::NO_TYPE) {
            entryPtr = &candidate;
            break;
        }

        const auto relativeAge = tableAge - candidate.age;

        if (relativeAge < minAge) {
            entryPtr = &candidate;
            minAge = relativeAge;
        }
    }

    assert(entryPtr != nullptr);

    auto tte = *entryPtr;

    if (!(type == TType::EXACT || depth + 2 > tte.depth || key != tte.key)) {
        return;
    }


    tte.key = key;
    tte.bestMove = move;
    tte.score = score;
    tte.depth = depth;
    tte.staticEval = staticEval;
    tte.type = type;
    tte.age = tableAge;
    
    *entryPtr = tte;

}