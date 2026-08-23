#ifndef HISTORY_H
#define HISTORY_H

#include <cstdint>
#include <cstring>

class History {
public:

    void updateQuietHistory();

    inline void age() {
        for (int stm = 0; stm < 2; stm++) {
            for (int from = 0; from < 64; from++) {
                for (int to = 0; to < 64; to++) {
                    for (int fromThreat = 0; fromThreat < 2; fromThreat++) {
                        for (int toThreat = 0; toThreat < 2; toThreat++) {
                            int value = butterfly[stm][from][to][fromThreat][toThreat];
                            butterfly[stm][from][to][fromThreat][toThreat] = value * 750 / 1000;
                        }
                    }
                }
            }
        }
    }

    inline void updateButterflyScore(BitBoardEnum color, Move move, BitBoard threats, int32_t bonus) {
        int stm = (color == Black);
        int toAttacked = ((1ULL << move.to()) & threats) != 0;
        int fromAttacked = ((1ULL << move.from()) & threats) != 0;
        int32_t value = butterfly[stm][move.from()][move.to()][fromAttacked][toAttacked];
        value += bonus - value * std::abs(bonus) / 16000;
        butterfly[stm][move.from()][move.to()][fromAttacked][toAttacked] = value;
    }

    [[nodiscard]] inline int32_t butterflyScore(BitBoardEnum color, Move move, BitBoard threats) {
        int stm =  (color == Black);        
        
        int toAttacked = ((1ULL << move.to()) & threats) != 0;
        int fromAttacked = ((1ULL << move.from()) & threats) != 0;
        
        return butterfly[stm][move.from()][move.to()][fromAttacked][toAttacked];
    }

    void clear() {
        std::memset(&butterfly, 0, sizeof(butterfly));
    }

private:
    // [stm][from][to][from attacked][to attacked]
    int32_t butterfly[2][64][64][2][2] = {};
};

#endif