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
#include "move.h"

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
        table.reset(new Bucket[nrOfBuckets]);
    }

    TTable(const TTable&) = delete;
    TTable& operator=(const TTable&) = delete;

    void clear() noexcept {
        table.reset(new Bucket[nrOfBuckets]);
        tableAge = 0;
    }

    TTEntry probe(uint64_t key) const noexcept {
        auto packedKey = packKey(key);
        const Bucket& entries = table[index(key)];
        TTEntry entry{};

        for (const auto internal : entries.entries) {
            if (internal.shortKey == packedKey) {
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

    uint16_t packKey(uint64_t key) const {
        return static_cast<uint16_t>(key);
    }

private:
    

    inline uint64_t index(uint64_t key) const noexcept { 
        return static_cast<uint64_t>((static_cast<u128>(key) * static_cast<u128>(nrOfBuckets)) >> 64);
    }

    struct InternalEntry {
        static constexpr uint32_t ageBits = 5;
        static constexpr uint32_t ageCycle = 1 << ageBits;
        static constexpr uint32_t ageMask = ageCycle - 1;

        uint16_t shortKey;
        int16_t score;
        int16_t staticEval;
        Move move;
        uint8_t depth;
        uint8_t agePVType;

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

    struct alignas(32) Bucket {
        static constexpr uint8_t size = 3;
        std::array<InternalEntry, size> entries{};
        std::array < uint8_t, std::bit_ceil(sizeof(InternalEntry)* size) - sizeof(InternalEntry) * size> padding{};

    };


    std::unique_ptr<Bucket[]> table;
    uint64_t nrOfBuckets = 0;
    uint32_t tableAge = 0;
};

#endif // TTABLE_H
