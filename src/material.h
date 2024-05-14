#ifndef MATERIAL_H
#define MATERIAL_H

#include "board.h"
#include <array>

class Material {

  public:
    static int getPieceSquareScore(Board &board)
    {
        int score = 0;
        score = getScoreForSpecificPiece(board,BitBoardEnum::P);
        score += getScoreForSpecificPiece(board,BitBoardEnum::p);
        score += getScoreForSpecificPiece(board,BitBoardEnum::K); 
        score += getScoreForSpecificPiece(board,BitBoardEnum::k);
        score += getScoreForSpecificPiece(board,BitBoardEnum::Q);
        score += getScoreForSpecificPiece(board,BitBoardEnum::q);
        score += getScoreForSpecificPiece(board,BitBoardEnum::R);
        score += getScoreForSpecificPiece(board,BitBoardEnum::r);
        score += getScoreForSpecificPiece(board,BitBoardEnum::N); 
        score += getScoreForSpecificPiece(board,BitBoardEnum::n);
        score += getScoreForSpecificPiece(board,BitBoardEnum::B);
        score += getScoreForSpecificPiece(board,BitBoardEnum::b);    
        
        return score;
    }

    static int getScoreForSpecificPiece(Board &board,BitBoardEnum piece)
    {
        BitBoard pieceBoard = board.getBitboard(piece);
        int score = 0;

        int pieceSquare = 0;
        while (pieceBoard != 0)    {
            pieceSquare = board.popLsb(pieceBoard);
            score += Material::getPieceSquareScore(piece,pieceSquare);
        }
        return score;
    }

    static int getMaterialScore(Board &board)
    {
      int score = 2000*(board.countSetBits(BitBoardEnum::K) - board.countSetBits(BitBoardEnum::k))
                  + 900*(board.countSetBits(BitBoardEnum::Q) - board.countSetBits(BitBoardEnum::q))
                  + 500*(board.countSetBits(BitBoardEnum::R) - board.countSetBits(BitBoardEnum::r))
                  + 330*(board.countSetBits(BitBoardEnum::B) - board.countSetBits(BitBoardEnum::b))
                  + 320*(board.countSetBits(BitBoardEnum::N) - board.countSetBits(BitBoardEnum::n))
                  + 100*(board.countSetBits(BitBoardEnum::P) - board.countSetBits(BitBoardEnum::p));
      return score;
    }

    static int getMaterialScore(BitBoardEnum piece) {        
        return pieceMaterialScoreArray[piece];
    }

    static int getPieceSquareScore(BitBoardEnum piece, int square) {
        int pieceIndex = piece;
        int modifier = 1;
        if (piece > Black) {
            square = flip[square];
            pieceIndex -= Black;
            modifier = -1;
        }
        return pieceSquareScoreArray[pieceIndex][square]*modifier;
    }

    inline static const std::array<int, 14> pieceMaterialScoreArray = { 0,500,320,330,900,2000,100,0,500,320,330,900,2000,100 };

private:
      //https://www.talkchess.com/forum3/viewtopic.php?f=2&t=68311&start=19
    inline static const int flip[64] = {
     56,  57,  58,  59,  60,  61,  62,  63,
     48,  49,  50,  51,  52,  53,  54,  55,
     40,  41,  42,  43,  44,  45,  46,  47,
     32,  33,  34,  35,  36,  37,  38,  39,
     24,  25,  26,  27,  28,  29,  30,  31,
     16,  17,  18,  19,  20,  21,  22,  23,
      8,   9,  10,  11,  12,  13,  14,  15,
      0,   1,   2,   3,   4,   5,   6,   7
    };

    inline static const std::array<std::array<int, 64>, 7> pieceSquareScoreArray = { {
            //Empty Black
            {-5,0,0,0,0,0,0,-5,
            -5,0,0,0,0,0,0,-5,
            -5,0,0,0,0,0,0,-5,
            -5,0,0,0,0,0,0,-5,
            -5,0,0,0,0,0,0,-5,
            -5,0,0,0,0,0,0,-5,
            -5,0,0,0,0,0,0,-5,
            -5,0,0,0,0,0,0,-5, },
            // Black Rook
            {-19, -13,   1,  17, 16,  7, -37, -26,
             -44, -16, -20,  -9, -1, 11,  -6, -71,
             -45, -25, -16, -17,  3,  0,  -5, -33,
             -36, -26, -12,  -1,  9, -7,   6, -23,
             -24, -11,   7,  26, 24, 35,  -8, -20,
              -5,  19,  26,  36, 17, 45,  61,  16,
              27,  32,  58,  62, 80, 67,  26,  44,
              32,  42,  32,  51, 63,  9,  31,  43,},
              // Black Knight
            {-105, -21, -58, -33, -17, -28, -19,  -23,
              -29, -53, -12,  -3,  -1,  18, -14,  -19,
              -23,  -9,  12,  10,  19,  17,  25,  -16,
              -13,   4,  16,  13,  28,  19,  21,   -8,
               -9,  17,  19,  53,  37,  69,  18,   22,
              -47,  60,  37,  65,  84, 129,  73,   44,
              -73, -41,  72,  36,  23,  62,   7,  -17,
             -167, -89, -34, -49,  61, -97, -15, -107,},
               // Black Bishop
            {-33,  -3, -14, -21, -13, -12, -39, -21,
               4,  15,  16,   0,   7,  21,  33,   1,
               0,  15,  15,  15,  14,  27,  18,  10,
              -6,  13,  13,  26,  34,  12,  10,   4,
              -4,   5,  19,  50,  37,  37,   7,  -2,
             -16,  37,  43,  40,  35,  50,  37,  -2,
             -26,  16, -18, -13,  30,  59,  18, -47,
             -29,   4, -82, -37, -25, -42,   7,  -8,},
               // Black Queen
            { -1, -18,  -9,  10, -15, -25, -31, -50,
             -35,  -8,  11,   2,   8,  15,  -3,   1,
             -14,   2, -11,  -2,  -5,   2,  14,   5,
              -9, -26,  -9, -10,  -2,  -4,   3,  -3,
             -27, -27, -16, -16,  -1,  17,  -2,   1,
             -13, -17,   7,   8,  29,  56,  47,  57,
             -24, -39,  -5,   1, -16,  57,  28,  54,
             -28,   0,  29,  12,  59,  44,  43,  45, },
            // Black King
            { -15,  36,  12, -54,   8, -28,  24,  14,
                1,   7,  -8, -64, -43, -16,   9,   8,
              -14, -14, -22, -46, -44, -30, -15, -27,
              -49,  -1, -27, -39, -46, -44, -33, -51,
              -17, -20, -12, -27, -30, -25, -14, -36,
               -9,  24,   2, -16, -20,   6,  22, -22,
               29,  -1, -20,  -7,  -8,  -4, -38, -29,
              -65,  23,  16, -15, -56, -34,   2,  13,},
            // Black Pawn
            {   0,   0,   0,   0,   0,   0,  0,   0,
              -35,  -1, -20, -23, -15,  24, 38, -22,
              -26,  -4,  -4, -10,   3,   3, 33, -12,
              -27,  -2,  -5,  12,  17,   6, 10, -25,
              -14,  13,   6,  21,  23,  12, 17, -23,
               -6,   7,  26,  31,  65,  56, 25, -20,
               98, 134,  61,  95,  68, 126, 34, -11,
                0,   0,   0,   0,   0,   0,  0,   0, },
        } };

};

#endif