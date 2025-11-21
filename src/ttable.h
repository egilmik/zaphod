#ifndef TTABLE_H
#define TTABLE_H

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>
#include "move.h"

// ---------- portable bit_floor ----------

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
static inline uint64_t bit_floor_64(uint64_t x) {
    if (!x) return 0;
    unsigned long idx;
    _BitScanReverse64(&idx, x);
    return 1ull << idx;
}
#else
static inline uint64_t bit_floor_64(uint64_t x) {
    if (!x) return 0;
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    x |= (x >> 16);
    x |= (x >> 32);
    return x - (x >> 1);
}
#endif
// ----------------------------------------

enum TType : uint8_t { EXACT, UPPER, LOWER, NO_TYPE };

struct TTEntry {
    uint64_t key = 0;
    int16_t  score = 0;
    int16_t staticEval = 0;
    int8_t depth = 0;
    TType    type = NO_TYPE;
    Move     bestMove{};
};

class TTable {
public:
    explicit TTable(size_t sizeMB) {
        setSize(sizeMB);
    }

    void setSize(size_t sizeMB) {
        // choose power-of-two bucket count
        uint64_t bytes = uint64_t(sizeMB) * (1ull << 20);
        uint64_t size = bytes / sizeof(TTEntry);
        if (size < 1024) size = 1024;
        nrOfEntries = bit_floor_64(size);
        if (nrOfEntries == 0) nrOfEntries = 1024;   // safety in case sizeMB==0
        keyMask = nrOfEntries - 1;
        table.reset(new TTEntry[nrOfEntries]);
    }

    TTable(const TTable&) = delete;
    TTable& operator=(const TTable&) = delete;

    void clear() noexcept {
        for (uint64_t i = 0; i < nrOfEntries; ++i)
            table[i].key = 0;
    }

    TTEntry probe(uint64_t key) const noexcept {
        int idx = index(key);
        const TTEntry& tte = table[index(key)];
        if (tte.key != key) {
            return TTEntry{};
        }

        return tte;
    }

    void put(uint64_t key, int score, int staticEval, int depth, Move move, TType type);

private:
    inline uint64_t index(uint64_t key) const noexcept { return key & keyMask; }



    std::unique_ptr<TTEntry[]> table;
    uint64_t nrOfEntries = 0;
    uint64_t keyMask = 0;
};

#endif // TTABLE_H
