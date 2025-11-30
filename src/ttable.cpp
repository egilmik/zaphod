#include "ttable.h"
#include <cassert>
#include <limits>

void TTable::put(uint64_t key, int score, int staticEval, int depth, Move move, TType type, bool pv) {
    auto packedKey = packKey(key);
    auto& entries = table[index(key)];
    InternalEntry* entryPtr = nullptr;
    auto minAge = std::numeric_limits<int32_t>::max();

    for (auto& candidate : entries.entries) {
        if (candidate.shortKey == packedKey || candidate.type() == TType::NO_TYPE) {
            entryPtr = &candidate;
            break;
        }

        const int32_t relativeAge = (InternalEntry::ageCycle + tableAge - candidate.age()) & InternalEntry::ageMask;
        const int32_t entryValue = candidate.depth - relativeAge * 2;

        if (entryValue < minAge) {
            entryPtr = &candidate;
            minAge = entryValue;
        }
    }

    assert(entryPtr != nullptr);

    auto tte = *entryPtr;

    if(!(type == TType::EXACT || depth + 4 + pv*2 > tte.depth || packedKey != tte.shortKey || tableAge != tte.age())){
        return;
    }

    tte.shortKey = packedKey;
    tte.move = move;
    tte.score = score;
    tte.depth = depth;
    tte.staticEval = staticEval;
    tte.setAgePVType(tableAge, pv, type);

    *entryPtr = tte;

}