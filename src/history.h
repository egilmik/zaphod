#ifndef HISTORY_H
#define HISTORY_H

#include <cstdint>
#include <cstring>
#include "params.h"

using namespace zaphod::params;

class History {
public:

	static constexpr int CONT_PLIES = 4;
	static constexpr int contOffset[CONT_PLIES] = {1,2,4,6};

	using ContSlice = int16_t[14][64];

	History() : continuation(std::make_unique<ContTable>()) {}

    void updateQuietHistory();

    inline void age() {
        for (int stm = 0; stm < 2; stm++) {
            for (int from = 0; from < 64; from++) {
                for (int to = 0; to < 64; to++) {
                    for (int fromThreat = 0; fromThreat < 2; fromThreat++) {
                        for (int toThreat = 0; toThreat < 2; toThreat++) {
                            int value = butterfly[stm][from][to][fromThreat][toThreat];
                            butterfly[stm][from][to][fromThreat][toThreat] = value * butterflyAging() / 1000;
                        }
                    }
                }
            }
        }

    }

    [[nodiscard]] inline ContSlice* contSlice(BitBoardEnum prevPiece, uint32_t prevTo) {
        return &continuation->data[prevPiece][prevTo];
    }

    [[nodiscard]] inline int32_t contScore(ContSlice* const* slices, BitBoardEnum piece, uint32_t to, int ply) {
	    if(slices[ply]){
		    return (*slices[ply])[piece][to];
	    }
	    return 0;
    }

    inline void updateContScore(ContSlice* const* slices, BitBoardEnum piece, uint32_t to, int32_t bonus) {
        for (int i = 0; i < CONT_PLIES; i++) {
            if (!slices[i]) {
                continue;
            }
            int32_t value = (*slices[i])[piece][to];
            value += bonus - value * std::abs(bonus) / maxContHistory();
            (*slices[i])[piece][to] = static_cast<int16_t>(value);
        }
    }

    inline void updateCapturedPiece(BitBoardEnum piece, uint32_t to, BitBoardEnum capturedPiece, int32_t bonus) {
        int32_t value = capturedPieceHistory[piece][to][capturedPiece];
        value += bonus - value * std::abs(bonus) / maxCapturePieceHistoryBonus();
        capturedPieceHistory[piece][to][capturedPiece] = value;
    }

    inline int32_t capturedPieceScore(BitBoardEnum movedPiece, uint32_t to, BitBoardEnum capturedPiece) {
        return capturedPieceHistory[movedPiece][to][capturedPiece];
    }


    inline void updateButterflyScore(BitBoardEnum color, Move move, BitBoard threats, int32_t bonus) {
        int stm = (color == Black);
        int toAttacked = ((1ULL << move.to()) & threats) != 0;
        int fromAttacked = ((1ULL << move.from()) & threats) != 0;
        int32_t value = butterfly[stm][move.from()][move.to()][fromAttacked][toAttacked];
        value += bonus - value * std::abs(bonus) / maxButterflyHistory();
        butterfly[stm][move.from()][move.to()][fromAttacked][toAttacked] = value;
    }

    [[nodiscard]] inline int32_t butterflyScore(BitBoardEnum color, Move move, BitBoard threats) {
        int stm =  (color == Black);        
        
        int toAttacked = ((1ULL << move.to()) & threats) != 0;
        int fromAttacked = ((1ULL << move.from()) & threats) != 0;
        
        return butterfly[stm][move.from()][move.to()][fromAttacked][toAttacked];
    }

    [[nodiscard]] inline int32_t pieceToScore(BitBoardEnum piece, Move move, BitBoard threats) {
        int toAttacked = ((1ULL << move.to()) & threats) != 0;
        int fromAttacked = ((1ULL << move.from()) & threats) != 0;

        return pieceTo[piece][move.to()][fromAttacked][toAttacked];
    }

    inline void updatePieceToScore(BitBoardEnum piece, Move move, BitBoard threats, int32_t bonus) {
        int toAttacked = ((1ULL << move.to()) & threats) != 0;
        int fromAttacked = ((1ULL << move.from()) & threats) != 0;
        int32_t value = pieceTo[piece][move.to()][fromAttacked][toAttacked];
        value += bonus - value * std::abs(bonus) / maxPieceToHistory();
        pieceTo[piece][move.to()][fromAttacked][toAttacked] = value;
    }

    void clear() {
        std::memset(&butterfly, 0, sizeof(butterfly));
		std::memset(continuation.get(), 0, sizeof(ContTable));
        std::memset(&capturedPieceHistory, 0, sizeof(capturedPieceHistory));
        std::memset(&pieceTo, 0, sizeof(pieceTo));
    }

private:

    // [stm][from][to][from attacked][to attacked]
    int32_t butterfly[2][64][64][2][2] = {};
    int32_t capturedPieceHistory[14][64][14] = {};
    int32_t pieceTo[14][64][2][2] = {};

	struct ContTable {
			int16_t data[14][64][14][64] = {};
	};

	std::unique_ptr<ContTable> continuation;
};

#endif
