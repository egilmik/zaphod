#ifndef TTABLE_H
#define TTABLE_H

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>
#include <array>
#include <bit>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <new>
#include <immintrin.h>
#include "move.h"

#if defined(__linux__)
    #include <sys/mman.h>
#elif defined(_WIN32)
    #include <malloc.h>
#endif

#if defined _MSC_VER
    #include <__msvc_int128.hpp>
    using u128 = std::_Unsigned128;
#else
    using u128 = unsigned __int128;
#endif

enum TType : uint8_t { 
    NO_TYPE = 0,
    EXACT = 1,
    UPPER = 2,
    LOWER = 3 };

struct TTEntry {
    uint64_t key = 0;
    int16_t  score = 0;
    int16_t staticEval = 0;
    int8_t depth = 0;
    bool pv = false;
    TType    type = NO_TYPE;
    Move     move{};
    uint16_t age;
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
        nrOfBuckets = size;
        if (nrOfBuckets == 0) nrOfBuckets = 1024;   // safety in case sizeMB==0
        table.reset(allocateBuckets(nrOfBuckets));
        generation = 0;
    }

    TTable(const TTable&) = delete;
    TTable& operator=(const TTable&) = delete;

    // Entries written before the last clear() carry an older generation and
    // are treated as empty, which makes clearing O(1). The table is only
    // really wiped when the 16-bit generation wraps around.
    void clear() noexcept {
        if (++generation == 0) {
            std::memset(table.get(), 0, nrOfBuckets * sizeof(Bucket));
        }
        tableAge = 0;
    }

    // Pull the bucket of key into the cache, ahead of the probe for it.
    void prefetch(uint64_t key) const noexcept {
        _mm_prefetch(reinterpret_cast<const char*>(&table[index(key)]), _MM_HINT_T0);
    }

    TTEntry probe(uint64_t key) const noexcept {
        auto packedKey = packKey(key);
        const Bucket& entries = table[index(key)];
        TTEntry entry{};

        for (const auto internal : entries.entries) {
            if (internal.generation == generation && internal.shortKey == packedKey) {
                entry.key = key;
                entry.score = internal.score;
                entry.staticEval = internal.staticEval;
                entry.move = internal.move;
                entry.depth = internal.depth;
                entry.pv = internal.pv();
                entry.type = internal.type();

                break;
            }
        }

        return entry;
    }

    void put(uint64_t key, int score, int staticEval, int depth, Move move, TType type, bool pv);

    void age() {
        tableAge = (tableAge + 1) % (1 << InternalEntry::ageBits); 
    }

    uint32_t packKey(uint64_t key) const {
        return static_cast<uint32_t>(key);
    }

private:
    

    inline uint64_t index(uint64_t key) const noexcept { 
        return static_cast<uint64_t>((static_cast<u128>(key) * static_cast<u128>(nrOfBuckets)) >> 64);
    }

    struct InternalEntry {
        static constexpr uint32_t ageBits = 5;
        static constexpr uint32_t ageCycle = 1 << ageBits;
        static constexpr uint32_t ageMask = ageCycle - 1;

        uint32_t shortKey;
        int16_t score;
        int16_t staticEval;
        Move move;
        uint8_t depth;
        uint8_t agePVType;
        // Fills the padding at the end of the entry, so the size stays 16 bytes
        uint16_t generation;

        uint32_t age() const {
            return static_cast<uint32_t>(agePVType >> 3);
        }
        
        bool pv() const {
            return (static_cast<uint32_t>(agePVType >> 2) & 1) != 0;
        }

        TType type() const {
            return static_cast<TType>(agePVType & 0x3);
        }

        void setAgePVType(uint32_t age, bool pv, TType type) {
            agePVType = (age << 3) | (static_cast<uint32_t>(pv) << 2) | static_cast<uint32_t>(type);
        }

    };

    static_assert(sizeof(InternalEntry) == 16, "TT entry should stay 16 bytes");

    struct alignas(32) Bucket {
        static constexpr uint8_t size = 3;
        std::array<InternalEntry, size> entries{};
        std::array < uint8_t, std::bit_ceil(sizeof(InternalEntry)* size) - sizeof(InternalEntry) * size> padding{};

    };


    // The table is backed by 2 MB pages where the OS allows it, every probe
    // is a random access into it and would otherwise mostly miss the TLB.
    static constexpr size_t TABLE_ALIGNMENT = 2 * 1024 * 1024;

    static Bucket* allocateBuckets(size_t count) {
        size_t bytes = (count * sizeof(Bucket) + TABLE_ALIGNMENT - 1) / TABLE_ALIGNMENT * TABLE_ALIGNMENT;
#if defined(_WIN32)
        void* mem = _aligned_malloc(bytes, TABLE_ALIGNMENT);
#else
        void* mem = std::aligned_alloc(TABLE_ALIGNMENT, bytes);
#endif
        if (!mem) throw std::bad_alloc();
#if defined(__linux__)
        madvise(mem, bytes, MADV_HUGEPAGE);
#endif
        Bucket* buckets = static_cast<Bucket*>(mem);
        std::uninitialized_value_construct_n(buckets, count);
        return buckets;
    }

    struct BucketDeleter {
        void operator()(Bucket* buckets) const noexcept {
#if defined(_WIN32)
            _aligned_free(buckets);
#else
            std::free(buckets);
#endif
        }
    };

    std::unique_ptr<Bucket[], BucketDeleter> table;
    uint64_t nrOfBuckets = 0;
    uint32_t tableAge = 0;
    uint16_t generation = 0;
};

#endif // TTABLE_H
