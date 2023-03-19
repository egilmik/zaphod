#include "movegenerator.h"
#include <iostream>

std::vector<Move> MoveGenerator::generateMoves(Board board)
{
    std::vector<Move> moveVector;
    //board.printBoard();
    generatePawnPush(board,moveVector);
    board.changeSideToMove();
    return moveVector;
}

void MoveGenerator::generatePawnPush(Board board, std::vector<Move> &moveVector)
{
    BitBoard pawns;

    BitBoard allPieces = board.getBitboard(Board::All);
    Board::BitBoardEnum movedPiece;    

    if(board.getSideToMove() == board.White){
        pawns = board.getBitboard(Board::P);
        movedPiece = Board::P;
    } else {
        pawns = board.getBitboard(Board::p);
        movedPiece == Board::p;
    }

    int pawnIncrement = 8;
    int pawnDoubleIncrement = 16;

    if(board.getSideToMove() == board.Black){
        pawnDoubleIncrement = -16;
        pawnIncrement = -8;
    }
    
    
    int fromSq = board.popLsb(pawns);
    while (fromSq != 0)
    {

        
        int toSq = fromSq+8;

        if(!board.checkBit(allPieces,toSq)){
            Move move = {fromSq,toSq, movedPiece};
            moveVector.push_back(move);
        }
        
        if(fromSq > 7 && fromSq < 16){
            //Two squares forward from starting position
            toSq+=8;

            if(!board.checkBit(allPieces,toSq)){
                Move move = {fromSq,toSq, movedPiece};
                moveVector.push_back(move);
            }
        }
        fromSq = board.popLsb(pawns);

    }
}

void MoveGenerator::generatePawnCaptures(Board board)
{
}

void MoveGenerator::generateRookMove(Board board)
{
}

