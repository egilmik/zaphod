#ifndef EVALUATION_H
#define EVALUATION_H
#include "board.h"
#include "material.h"


class Evaluation
{
public:

    static int evaluatePassedPawn(Board& board, BitBoardEnum side) {
        BitBoard pawns = board.getBitboard(P + side);
        int modifier = 1;
        BitBoardEnum otherSide = Black;
        BitBoard aFile = board.FileAMask;
        BitBoard hFile = board.FileHMask;


        if (side == Black) {
            modifier = -1;
            otherSide = White;
            aFile = board.FileHMask;
            hFile = board.FileAMask;
        }
        int score = 0;

        int sq = 0;
        while (pawns) {
            bool isPassed = true;
            sq = board.popLsb(pawns);
            BitBoard file = board.fileArray[sq % 8];
            //
            BitBoard sqNorthWest = board.sqBB[sq + 9 * modifier];
            BitBoard sqNorthEast = board.sqBB[sq + 7 * modifier];

            // Check if there is our own pawn in front
            int sqNorth = sq + 8 * modifier;
            while (sqNorth <= 63 && sqNorth >= 0 ) {
                BitBoard bbSqNorth = board.sqBB[sqNorth];
                if ((bbSqNorth & aFile) == 0) {
                    BitBoard bbSqWest = board.sqBB[sqNorth + (1 * modifier)];
                    if (bbSqWest == (board.getBitboard(otherSide + P) & bbSqWest)) {
                        isPassed = false;
                    }
                }
                if ((bbSqNorth & hFile) == 0) {
                    BitBoard bbSqEast = board.sqBB[sqNorth - (1 * modifier)];
                    if (bbSqEast == (board.getBitboard(otherSide + P) & bbSqEast)) {
                        isPassed = false;
                    }
                }
                
                if (bbSqNorth == (board.getBitboard(P + side) & bbSqNorth) || bbSqNorth == (board.getBitboard(P + otherSide) & bbSqNorth)) {
                    isPassed = false;
                }
                sqNorth += 8*modifier;
            }

            if (isPassed) {
                score += Material::getPassPawnScore(side, sq);
            }
        }
        return score;
    }
};

#endif