#ifndef HISTORY_H
#define HISTORY_H

#include <cstdint>
#include <cstring>
#include <span>
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

    [[nodiscard]] inline int corrIndex(BitBoard key) const {
        return static_cast<int>(key & (CORRECTION_SIZE - 1));
    }

    // Returns the correction in centipawns, already de-scaled.
    [[nodiscard]] inline int correction(BitBoardEnum stm, std::span<HashKeys> keyHistory, int ply) const {
        int side = (stm == Black);
        int sum = pawnCorrection[side][corrIndex(keyHistory[ply].pawnHash)] * pawnCorrectionWeight();
        sum += nonPawnCorrection[0][side][corrIndex(keyHistory[ply].nonPawnKey[0])] * 60;
        sum += nonPawnCorrection[1][side][corrIndex(keyHistory[ply].nonPawnKey[1])] * 60;
        sum += minorPieceCorrection[side][corrIndex(keyHistory[ply].minorPieceKey)] * 50;
        sum += majorPieceCorrection[side][corrIndex(keyHistory[ply].majorPieceKey)] * 50;
        return sum/CORRECTION_LIMIT;
        
    }

    // diff = bestScore - rawStaticEval, in centipawns
    inline void updateCorrection(BitBoardEnum stm, std::span<HashKeys> keyHistory, int ply,
        int diff, int depth) {
        int side = (stm == Black);

        int bonus = std::clamp(diff * depth / 8,-CORRECTION_BONUS_MAX,CORRECTION_BONUS_MAX);
        int16_t& entry = pawnCorrection[side][corrIndex(keyHistory[ply].pawnHash)];
        entry += bonus - entry * std::abs(bonus) / CORRECTION_LIMIT;

        int16_t& nonPawnWhiteEntry = nonPawnCorrection[0][side][corrIndex(keyHistory[ply].nonPawnKey[0])];
        nonPawnWhiteEntry += bonus - nonPawnWhiteEntry * std::abs(bonus) / CORRECTION_LIMIT;

        int16_t& nonPawnBlackEntry = nonPawnCorrection[1][side][corrIndex(keyHistory[ply].nonPawnKey[1])];
        nonPawnBlackEntry += bonus - nonPawnBlackEntry * std::abs(bonus) / CORRECTION_LIMIT;

        int16_t& minorEntry = minorPieceCorrection[side][corrIndex(keyHistory[ply].minorPieceKey)];
        minorEntry += bonus - minorEntry * std::abs(bonus) / CORRECTION_LIMIT;

        int16_t& majorEntry = majorPieceCorrection[side][corrIndex(keyHistory[ply].majorPieceKey)];
        majorEntry += bonus - majorEntry * std::abs(bonus) / CORRECTION_LIMIT;
    }

    void clear() {
        std::memset(&butterfly, 0, sizeof(butterfly));
		std::memset(continuation.get(), 0, sizeof(ContTable));
        std::memset(&capturedPieceHistory, 0, sizeof(capturedPieceHistory));
        std::memset(&pieceTo, 0, sizeof(pieceTo));
        std::memset(&pawnCorrection, 0, sizeof(pawnCorrection));
        std::memset(&nonPawnCorrection, 0, sizeof(nonPawnCorrection));
        std::memset(&minorPieceCorrection, 0, sizeof(minorPieceCorrection));
        std::memset(&majorPieceCorrection, 0, sizeof(majorPieceCorrection));
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

    static constexpr int CORRECTION_SIZE = 16384;
    static constexpr int CORRECTION_BONUS_MAX = 256;
    static constexpr int CORRECTION_LIMIT = 1024;

    // [stm][pawn key]
    int16_t pawnCorrection[2][CORRECTION_SIZE] = {};
    int16_t nonPawnCorrection[2][2][CORRECTION_SIZE] = {};
    int16_t minorPieceCorrection[2][CORRECTION_SIZE] = {};
    int16_t majorPieceCorrection[2][CORRECTION_SIZE] = {};
    
};

#endif
