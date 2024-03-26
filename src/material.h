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
        std::array<int,64> scoreArray = Material::pieceSquareScoreArray[piece]; 
        int score = 0;

        int pieceSquare = 0;
        while (pieceBoard != 0)    {
            pieceSquare = board.popLsb(pieceBoard);
            score += scoreArray[pieceSquare];
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

    inline static const std::array<int, 14> pieceMaterialScoreArray = { 0,500,320,330,900,2000,100,0,500,320,330,900,2000,100 };

      inline static const std::array<std::array<int,64>,14> pieceSquareScoreArray = {{
            // Empty white
            {-5,0,0,0,0,0,0,-5,
            -5,0,0,0,0,0,0,-5,
            -5,0,0,0,0,0,0,-5,
            -5,0,0,0,0,0,0,-5,
            -5,0,0,0,0,0,0,-5,
            -5,0,0,0,0,0,0,-5,
            -5,0,0,0,0,0,0,-5,
            -5,0,0,0,0,0,0,-5, },
            // White Rook
              {0,0,0,5,5,0,0,0,
            -5,0,0,0,0,0,0,-5,
            -5,0,0,0,0,0,0,-5,
            -5,0,0,0,0,0,0,-5,
            -5,0,0,0,0,0,0,-5,
            -5,0,0,0,0,0,0,-5,
            5,10,10,10,10,10,10,5,
            0,0,0,0,0,0,0,0 },
            // White Knight
            {-50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,0,0,0,0,-20,-40,
            -30,5,10,15,15,10,5,-30,
            -30,0,15,20,20,15,0,-30,
            -30,5,15,20,20,15,5,-30,
            -30,0,10,15,15,10,0,-30,
            -40,-20,0,0,0,0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50},
            // White Bishop
            {-20,-10,-10,-10,-10,-10,-10,-20,
            -10,5,0,0,0,0,5,-10,
            -10,10,10,10,10,10,10,-10,
            -10,0,10,10,10,10,0,-10,
            -10,5,5,10,10,5,5,-10,
            -10,0,5,10,10,5,0,-10,
            -10,0,0,0,0,0,0,-10,
            -20,-10,-10,-10,-10,-10,-10,-20,},
            // White Queen
            {-20,-10,-10,-5,-5,-10,-10,-20,
            -10,0,5,0,0,0,0,-10,
            -10,0,5,5,5,5,0,-10,
            -5,0,5,5,5,5,0,-5,
            0,0,5,5,5,5,0,0,
            -10,0,5,5,5,5,0,-10,
            -10,0,0,0,0,0,0,-10,
            -20,-10,-10,-5,-5,-10,10,-20, },
            // White King
            {20,30,10,0,0,10,30,20,
            20,20,0,0,0,0,20,20
            -10,-20,-20,-20,-20,-20,-20,-10
            -20,-30,-30,-40,-40,-30,-30,-20
            -30,-40,-40,-50,-50,-40,-40,-30
            -30,-40,-40,-50,-50,-40,-40,-30
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30},
            // White Pawn
            {0 ,0 ,0  ,0  ,0   ,0 ,0 ,0,           // Rank 1
             5 ,10,10 ,-20,-20 ,10,10,5,
             5 ,-5,-10,0  ,0  ,-10,-5,5,
             0 ,0 ,0  ,20 ,20 ,0  ,0 ,0,
             5 ,5 ,10 ,25 ,25 ,10 ,5 ,5,            
             10,10,20,30,30,20,10,10,
             50,50,50,50,50,50,50,50,
             0 ,0 ,0,0,0,0,0,0, },
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
            {0,0,0,0,0,0,0,0,
            -5,-10,-10,-10,-10,-10,-10,-5,
            5,0,0,0,0,0,0,5,
            5,0,0,0,0,0,0,5,
            5,0,0,0,0,0,0,5,
            5,0,0,0,0,0,0,5,
            5,0,0,0,0,0,0,5,
            0,0,0,-5,-5,0,0,0 },
            // Black Knight
            {50,40,30,30,30,30,40,50,
            40,20,0,0,0,0,20,40,
            30,0,-10,-15,-15,-10,0,30,
            30,-5,-15,-20,-20,-15,-5,30,
            30,0,-15,-20,-20,-15,0,30,
            30,-5,-10,-15,-15,-10,-5,30,
            40,20,0,0,0,0,20,40,
            50,40,30,30,30,30,40,50},
            // Black Bishop
            {20,10,10,10,10,10,10,20,
            10,0,0,0,0,0,0,10,
            10,0,-5,-10,-10,-5,0,10,
            10,-5,-5,-10,-10,-5,-5,10,
            10,0,-10,-10,-10,-10,0,10,            
            10,-10,-10,-10,-10,-10,-10,10,
            10,-5,0,0,0,0,-5,10,
            20,10,10,10,10,10,10,20,},
            // Black Queen
            {20,10,10,5,5,10,10,20,
            10,0,0,0,0,0,0,10,
            10,0,-5,-5,-5,-5,0,10,            
            0,0,-5,-5,-5,-5,0,0,
            5,0,-5,-5,-5,-5,0,5,
            10,0,-5,-5,-5,-5,0,10,
            10,0,-5,0,0,0,0,10,
            20,10,10,5,5,10,10,20, },
            // Black King
            {30,40,40,50,50,40,40,30,
            30,40,40,50,50,40,40,30,
            30,40,40,50,50,40,40,30,
            30,40,40,50,50,40,40,30,              
            20,30,30,40,40,30,30,20,
            10,20,20,20,20,20,20,10,            
            -20,-20,0,0,0,0,-20,-20,
            -20,-30,-10,0,0,-10,-30,-20,
            },
            // Black Pawn
            {0,0,0,0,0,0,0,0,
            -50,-50,-50,-50,-50,-50,-50,-50,
            -10,-10,-20,-30,-30,-20,-10,-10,
            -5,-5,-10,-25,-25,-10,-5,-5,
            0,0,0,-20,-20,0,0,0,  
            -5 ,5,10,0,0,10,5,-5,
            -5,-10,-10,20,20,-10,-10,-5,
            0,0,0,0,0,0,0,0, }}};

};

#endif