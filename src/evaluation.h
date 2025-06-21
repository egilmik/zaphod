#ifndef EVALUATION_H
#define EVALUATION_H
#include "board.h"
#include "material.h"


class Evaluation
{
public:

    // TODO This is a penalty, should be negative for clarity.  Code handles it correctly, so no bug
    static const int doublePawnScore = 50;
    static const int BISHOP_PAIR = 30;
    static const int KNIGHT_PAIR = 8;
    static const int ROOK_PAIR = 16;

    static const int PAWNSHIELD_RANK2_SCORE = 20;
    static const int PAWNSHIELD_RANK3_SCORE = 10;
    static const int PAWNSHIELD_MISSING_SCORE = -20;
    
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

    static int evaluatePawnShield(Board& board) {
        BitBoard whitePawns = board.getBitboard(P);
        BitBoard whiteKing = board.getBitboard(K);
        BitBoard blackPawns = board.getBitboard(p);
        BitBoard blackKing = board.getBitboard(k);

        int score = 0;
        
        if ((whiteKing & (board.FileAMask | board.FileBMask | board.FileCMask)) > 0) {

            score += fileRankPawnShieldScore(board.Rank2Mask, board.Rank3Mask, board.FileAMask, whitePawns);
            score += fileRankPawnShieldScore(board.Rank2Mask, board.Rank3Mask, board.FileBMask, whitePawns);
            score += fileRankPawnShieldScore(board.Rank2Mask, board.Rank3Mask, board.FileCMask, whitePawns);

        }
        else if ((whiteKing & (board.FileHMask | board.FileGMask | board.FileFMask)) > 0) {

            score += fileRankPawnShieldScore(board.Rank2Mask, board.Rank3Mask, board.FileHMask, whitePawns);
            score += fileRankPawnShieldScore(board.Rank2Mask, board.Rank3Mask, board.FileGMask, whitePawns);
            score += fileRankPawnShieldScore(board.Rank2Mask, board.Rank3Mask, board.FileFMask, whitePawns);
        }

        if ((blackKing & (board.FileAMask | board.FileBMask | board.FileCMask)) > 0) {
            score -= fileRankPawnShieldScore(board.Rank7Mask, board.Rank6Mask, board.FileAMask, blackPawns);
            score -= fileRankPawnShieldScore(board.Rank7Mask, board.Rank6Mask, board.FileBMask, blackPawns);
            score -= fileRankPawnShieldScore(board.Rank7Mask, board.Rank6Mask, board.FileCMask, blackPawns);
        }
        else if ((blackKing & (board.FileHMask | board.FileGMask | board.FileFMask)) > 0) {
            score -= fileRankPawnShieldScore(board.Rank7Mask, board.Rank6Mask, board.FileHMask, blackPawns);
            score -= fileRankPawnShieldScore(board.Rank7Mask, board.Rank6Mask, board.FileGMask, blackPawns);
            score -= fileRankPawnShieldScore(board.Rank7Mask, board.Rank6Mask, board.FileFMask, blackPawns);
        }

        return score;
    }

    // Rank2 and rank3 are rank6 and rank7 for black
    static int fileRankPawnShieldScore(BitBoard rank2, BitBoard rank3, BitBoard file, BitBoard pawns) {
        int score = 0;
        BitBoard filePawns = file & pawns;

        if ((filePawns & rank2) > 0) {
            score += PAWNSHIELD_RANK2_SCORE;
        }
        else if ((filePawns & rank3) > 0) {
            score += PAWNSHIELD_RANK3_SCORE;
        }
        else {
            score += PAWNSHIELD_MISSING_SCORE;
        }
        return score;
    }

};

#endif
