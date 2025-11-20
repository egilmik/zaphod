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
    int32_t  score = 0;
    uint16_t depth = 0;
    TType    type = NO_TYPE;
    Move     bestMove{};
};

struct alignas(64) Bucket {          // 64 avoids false sharing
    std::atomic<uint64_t> key{ 0 };    // publish
    std::atomic<uint64_t> lo{ 0 };     // [score:int32][move:uint32]
    std::atomic<uint64_t> hi{ 0 };     // [depth:uint16][type:uint8][pad:40]
};

class TTable {
public:
    explicit TTable(size_t sizeMB) {
        setSize(sizeMB);
    }

    void setSize(size_t sizeMB) {
        // choose power-of-two bucket count
        uint64_t bytes = uint64_t(sizeMB) * (1ull << 20);
        uint64_t buckets = bytes / sizeof(Bucket);
        if (buckets < 1024) buckets = 1024;
        nrOfBuckets = bit_floor_64(buckets);
        if (nrOfBuckets == 0) nrOfBuckets = 1024;   // safety in case sizeMB==0
        keyMask = nrOfBuckets - 1;
        table.reset(new Bucket[nrOfBuckets]);
    }

    TTable(const TTable&) = delete;
    TTable& operator=(const TTable&) = delete;

    void clear() noexcept {
        for (uint64_t i = 0; i < nrOfBuckets; ++i)
            table[i].key.store(0, std::memory_order_relaxed);
    }

    std::optional<TTEntry> probe(uint64_t key) const noexcept {
        const Bucket& b = table[index(key)];
        uint64_t k = b.key.load(std::memory_order_acquire);
        if (k != key) return TTEntry{};
        uint64_t lo = b.lo.load(std::memory_order_relaxed);
        uint64_t hi = b.hi.load(std::memory_order_relaxed);
        TTEntry e;
        e.key = k;
        e.score = static_cast<int32_t>(lo >> 32);
        e.bestMove = Move(static_cast<uint32_t>(lo));
        e.depth = static_cast<uint16_t>(hi >> 8);
        e.type = static_cast<TType>(hi & 0xFF);
        return e;
    }

    void put(uint64_t key, int score, int depth, Move move, TType type) noexcept {
        Bucket& b = table[index(key)];
        uint64_t existingKey = b.key.load(std::memory_order_acquire);
        if (existingKey == key) {
            uint64_t oldHi = b.hi.load(std::memory_order_relaxed);
            uint16_t oldDepth = static_cast<uint16_t>(oldHi >> 8);
            if (depth <= oldDepth) return;
        }
        uint64_t lo = (uint64_t(uint32_t(score)) << 32) | uint64_t(move.value);
        uint64_t hi = (uint64_t(uint16_t(depth)) << 8) | uint64_t(uint8_t(type));
        b.lo.store(lo, std::memory_order_relaxed);
        b.hi.store(hi, std::memory_order_relaxed);
        b.key.store(key, std::memory_order_release);
    }

private:
    inline uint64_t index(uint64_t key) const noexcept { return key & keyMask; }



    std::unique_ptr<Bucket[]> table;
    uint64_t nrOfBuckets = 0;
    uint64_t keyMask = 0;
};

#endif // TTABLE_H
