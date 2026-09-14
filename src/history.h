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
    [[nodiscard]] inline int correction(BitBoardEnum stm, HashKeys keys, std::span<HashKeys> keyHistory, int historyPly) const {
        int side = (stm == Black);
        int sum = corrHist->pawnCorrection[side][corrIndex(keys.pawnHash)] * pawnCorrectionWeight();
        sum += corrHist->nonPawnCorrection[0][side][corrIndex(keys.nonPawnKey[0])] * 60;
        sum += corrHist->nonPawnCorrection[1][side][corrIndex(keys.nonPawnKey[1])] * 60;
        sum += corrHist->minorPieceCorrection[side][corrIndex(keys.minorPieceKey)] * 50;
        sum += corrHist->majorPieceCorrection[side][corrIndex(keys.majorPieceKey)] * 50;
        
        //History ply is the previous ply, so this is current - 1
        // Cont correction for 1,2,4
        sum += corrHist->contCorrection[side][corrIndex(keyHistory[historyPly].hashKey)]*50;
        sum += corrHist->contCorrection[side][corrIndex(keyHistory[historyPly - 1].hashKey)]*50;
        sum += corrHist->contCorrection[side][corrIndex(keyHistory[historyPly - 3].hashKey)]*50;
        

        return sum/CORRECTION_LIMIT;
        
    }

    // diff = bestScore - rawStaticEval, in centipawns
    inline void updateCorrection(BitBoardEnum stm, HashKeys keys, std::span<HashKeys> keyHistory, int historyPly,
        int diff, int depth) {
        int side = (stm == Black);

        int bonus = std::clamp(diff * depth / 8,-CORRECTION_BONUS_MAX,CORRECTION_BONUS_MAX);
        corrHist->pawnCorrection[side][corrIndex(keys.pawnHash)].update(bonus);
        corrHist->nonPawnCorrection[0][side][corrIndex(keys.nonPawnKey[0])].update(bonus);
        corrHist->nonPawnCorrection[1][side][corrIndex(keys.nonPawnKey[1])].update(bonus);
        corrHist->minorPieceCorrection[side][corrIndex(keys.minorPieceKey)].update(bonus);
        corrHist->majorPieceCorrection[side][corrIndex(keys.majorPieceKey)].update(bonus);

        //History ply is the previous ply, so this is current - 1
        // Cont correction for 1,2,4
        corrHist->contCorrection[side][corrIndex(keyHistory[historyPly].hashKey)].update(bonus);
        corrHist->contCorrection[side][corrIndex(keyHistory[historyPly-1].hashKey)].update(bonus);
        corrHist->contCorrection[side][corrIndex(keyHistory[historyPly-3].hashKey)].update(bonus);
        
    }

    void clear() {
        std::memset(&butterfly, 0, sizeof(butterfly));
		std::memset(continuation.get(), 0, sizeof(ContTable));
        std::memset(&capturedPieceHistory, 0, sizeof(capturedPieceHistory));
        std::memset(&pieceTo, 0, sizeof(pieceTo));
        std::memset(&corrHist->pawnCorrection, 0, sizeof(corrHist->pawnCorrection));
        std::memset(&corrHist->nonPawnCorrection, 0, sizeof(corrHist->nonPawnCorrection));
        std::memset(&corrHist->minorPieceCorrection, 0, sizeof(corrHist->minorPieceCorrection));
        std::memset(&corrHist->majorPieceCorrection, 0, sizeof(corrHist->majorPieceCorrection));
        std::memset(&corrHist->contCorrection, 0, sizeof(corrHist->contCorrection));
    }

private:

    

    // [stm][from][to][from attacked][to attacked]
    int32_t butterfly[2][64][64][2][2] = {};
    int32_t capturedPieceHistory[14][64][14] = {};
    int32_t pieceTo[14][64][2][2] = {};

	struct ContTable {
			int16_t data[14][64][14][64] = {};
	};

	std::unique_ptr<ContTable> continuation = std::make_unique<ContTable>();


    // Corrections

    static constexpr int CORRECTION_SIZE = 16384;
    static constexpr int CORRECTION_BONUS_MAX = 256;
    static constexpr int CORRECTION_LIMIT = 1024;

    struct CorrectionEntry {
        int32_t value = 0;

        inline void update(int32_t bonus) {
            value += bonus - value * std::abs(bonus) / CORRECTION_LIMIT;
        }

        [[nodiscard]] inline operator int32_t() const {
            return value;
        }
    };
    
    struct CorrectionHistory {
        // [stm][pawn key]
        CorrectionEntry pawnCorrection[2][CORRECTION_SIZE] = {};
        CorrectionEntry nonPawnCorrection[2][2][CORRECTION_SIZE] = {};
        CorrectionEntry minorPieceCorrection[2][CORRECTION_SIZE] = {};
        CorrectionEntry majorPieceCorrection[2][CORRECTION_SIZE] = {};
        CorrectionEntry contCorrection[2][CORRECTION_SIZE * 2] = {};
    };

    std::unique_ptr<CorrectionHistory> corrHist = std::make_unique<CorrectionHistory>();

};

#endif
