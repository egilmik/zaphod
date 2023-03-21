#include "movegenerator.h"
#include <iostream>

std::vector<Move> MoveGenerator::generateMoves(Board &board)
{
    std::vector<Move> moveVector;
    generatePawnPush(board,moveVector);
    generateKnightMoves(board, moveVector);
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
        movedPiece = Board::p;
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

        
        int toSq = fromSq+pawnIncrement;

        if(!board.checkBit(allPieces,toSq)){
            Move move = {fromSq,toSq, movedPiece};
            moveVector.push_back(move);
        }
        
        if((fromSq > 7 && fromSq < 16) || 
            (fromSq > 47 && fromSq< 56)){
            //Two squares forward from starting position
            toSq =fromSq+pawnDoubleIncrement;

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

void MoveGenerator::generateKnightMoves(Board board, std::vector<Move> &moveVector)
{
    BitBoard knights;

    BitBoard allPieces = board.getBitboard(Board::All);
    Board::BitBoardEnum movedPiece;    

    if(board.getSideToMove() == board.White){
        knights = board.getBitboard(Board::N);
        movedPiece = Board::N;
    } else {
        knights = board.getBitboard(Board::n);
        movedPiece = Board::n;
    }

    int fromSq = board.popLsb(knights);
    while (fromSq != 0)
    {
        BitBoard knightMoves = board.getKnightMask(fromSq);

        
        int toSq = board.popLsb(knightMoves);
        while(toSq != 0){
            if(!board.checkBit(allPieces,toSq)){
                Move move = {fromSq,toSq, movedPiece};
                moveVector.push_back(move);
            }
            toSq = board.popLsb(knightMoves);
        }
        fromSq = board.popLsb(knights);
    }

}

void MoveGenerator::generateRookMove(Board board)
{
}

