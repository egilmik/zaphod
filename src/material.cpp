#include "material.h"

static std::array<std::array<int,64>,14> initPieceSquareScore()
{  
  enum BitBoardEnum {White,R,N,B,Q,K,P,Black,r,n,b,q,k,p,All};
  std::array<std::array<int,64>,14> pieceSquareScore;
  pieceSquareScore[8] = Material::mg_rook_table;
  pieceSquareScore[9] = Material::mg_knight_table;
  pieceSquareScore[10] = Material::mg_bishop_table;
  pieceSquareScore[11] = Material::mg_queen_table;
  pieceSquareScore[12] = Material::mg_king_table;
  pieceSquareScore[13] = Material::mg_pawn_table;

  for(int i = 0; i < 64; i++){
    pieceSquareScore[1][i] = Material::mg_rook_table[i]^56;
    pieceSquareScore[2][i] = Material::mg_knight_table[i]^56;
    pieceSquareScore[3][i] = Material::mg_bishop_table[i]^56;
    pieceSquareScore[4][i] = Material::mg_queen_table[i]^56;
    pieceSquareScore[5][i] = Material::mg_king_table[i]^56;
    pieceSquareScore[6][i] = Material::mg_pawn_table[i]^56;
  }

  return pieceSquareScore;

}


const std::array<std::array<int,64>,14> Material::pieceSquareScoreArray = initPieceSquareScore();