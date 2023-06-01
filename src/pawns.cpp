#include "pawns.h"

Pawns::Pawns()
{
}

int Pawns::evaluate(Board &board, BitBoardEnum color)
{
    BitBoard pawns = board.getBitboard(BitBoardEnum::P+color);
    int score = 0;

    // Check for doubled pawns
    for(int i = 0; i < 8; i++){
        BitBoard line = Board::getLineMask(i);
        int pawnCount = Board::countSetBits(Board::getLineMask(i) & pawns);
        score -= (pawnCount-1)*0.75;        
    }
    return 0;
}
