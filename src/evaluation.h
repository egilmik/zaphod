#ifndef EVALUATION_H
#define EVALUATION_H
#include "board.h"
#include "material.h"


class Evaluation
{
public:

    static const int doublePawnScore = 50;
    static const int BISHOP_PAIR = 30;
    static const int KNIGHT_PAIR = 8;
    static const int ROOK_PAIR = 16;
    
    static int evaluatePiecePairs(Board& board) {
        int score = 0;

        if (board.countSetBits(board.getBitboard(B)) > 1) score += BISHOP_PAIR;
        if (board.countSetBits(board.getBitboard(b)) > 1) score -= BISHOP_PAIR;
        if (board.countSetBits(board.getBitboard(N)) > 1) score += KNIGHT_PAIR;
        if (board.countSetBits(board.getBitboard(n)) > 1) score -= KNIGHT_PAIR;
        if (board.countSetBits(board.getBitboard(R)) > 1) score += ROOK_PAIR;
        if (board.countSetBits(board.getBitboard(r)) > 1) score -= ROOK_PAIR;

        return score;
    }

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

            
            int sqNorth = sq + 8 * modifier;
            BitBoard bbSqNorth = board.sqBB[sqNorth];

            // Check if there is our own pawn in front
            if (bbSqNorth == (board.getBitboard(P + side) & bbSqNorth)) {
                if (side == White) {
                    score -= doublePawnScore;
                }
                else {
                    score += doublePawnScore;
                }
                
            }

            //Check passed pawn
            while (sqNorth <= 63 && sqNorth >= 0 ) {                
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
                bbSqNorth = board.sqBB[sqNorth];
            }

            if (isPassed) {
                score += Material::getPassPawnScore(side, sq);
            }
        }
        return score;
    }
};

#endif