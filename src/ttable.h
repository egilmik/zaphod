#ifndef TTABLE_H
#define TTABLE_H

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>
#include <array>
#include <bit>
#include "move.h"

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
        uint64_t size = bytes / sizeof(Bucket);
        if (size < 1024) size = 1024;
        nrOfEntries = std::bit_floor(size);
        if (nrOfEntries == 0) nrOfEntries = 1024;   // safety in case sizeMB==0
        keyMask = nrOfEntries - 1;
        table.reset(new Bucket[nrOfEntries]);
    }

    TTable(const TTable&) = delete;
    TTable& operator=(const TTable&) = delete;

    void clear() noexcept {
        for (uint64_t i = 0; i < nrOfEntries; ++i){
            for (uint64_t z = 0; z < Bucket::size; z++) {
                table[i].entries[z].key = 0;
            }
        
        }
    }

    TTEntry probe(uint64_t key) const noexcept {
        int idx = index(key);
        const Bucket& entries = table[index(key)];
        for (const auto tte : entries.entries) {
            if (tte.key == key) {
                return tte;
            }
        }

        return TTEntry{};
    }

    void put(uint64_t key, int score, int staticEval, int depth, Move move, TType type);

private:
    inline uint64_t index(uint64_t key) const noexcept { return key & keyMask; }

    struct Bucket {
        static constexpr uint8_t size = 3;
        std::array<TTEntry, size> entries{};
    };


    std::unique_ptr<Bucket[]> table;
    uint64_t nrOfEntries = 0;
    uint64_t keyMask = 0;
};

#endif // TTABLE_H
