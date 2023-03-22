#include "movegenerator.h"
#include <iostream>

std::vector<Move> MoveGenerator::generateMoves(Board &board)
{
    std::vector<Move> moveVector;
    generatePawnMoves(board,moveVector);
    generateKnightMoves(board, moveVector);
    return moveVector;
}

void MoveGenerator::generatePawnMoves(Board board, std::vector<Move> &moveVector)
{
    BitBoard pawns;

    BitBoard allPieces = board.getBitboard(Board::All);
    Board::BitBoardEnum movedPiece;
    Board::BitBoardEnum sideToMove = board.getSideToMove();

    if(sideToMove == board.White){
        pawns = board.getBitboard(Board::P);
        movedPiece = Board::P;
    } else {
        pawns = board.getBitboard(Board::p);
        movedPiece = Board::p;
    }

    int pawnIncrement = 8;
    int pawnDoubleIncrement = 16;

    if(sideToMove == board.Black){
        pawnDoubleIncrement = -16;
        pawnIncrement = -8;
    }
    
    
    int fromSq = board.popLsb(pawns);
    while (fromSq != 0)
    {

        
        int toSq = fromSq+pawnIncrement;

        if(!board.checkBit(allPieces,toSq)){
            Move move = {fromSq,toSq, false, movedPiece};
            moveVector.push_back(move);
        }
        
        if((fromSq > 7 && fromSq < 16) || 
            (fromSq > 47 && fromSq< 56)){
            //Two squares forward from starting position
            toSq =fromSq+pawnDoubleIncrement;

            if(!board.checkBit(allPieces,toSq)){
                Move move = {fromSq,toSq, false, movedPiece};
                moveVector.push_back(move);
            }
        }
        if(sideToMove == Board::White){
            if(board.checkBit(Board::BitBoardEnum::Black,fromSq+7)){
                Move move = {fromSq,fromSq+7, true, movedPiece};
                moveVector.push_back(move);
            }
            if(board.checkBit(Board::BitBoardEnum::Black, fromSq+9)){
                Move move = {fromSq,fromSq+9, true, movedPiece};
                moveVector.push_back(move);
            }
        }

        if(sideToMove == Board::Black){
            if(board.checkBit(Board::BitBoardEnum::White,fromSq+7)){
                Move move = {fromSq,fromSq+7, true, movedPiece};
                moveVector.push_back(move);
            }
            if(board.checkBit(Board::BitBoardEnum::White, fromSq+9)){
                Move move = {fromSq,fromSq+9,true, movedPiece};
                moveVector.push_back(move);
            }
        }
        
        fromSq = board.popLsb(pawns);

    }
}


void MoveGenerator::generateKnightMoves(Board board, std::vector<Move> &moveVector)
{
    BitBoard knights;

    BitBoard allPieces = board.getBitboard(Board::All);
    Board::BitBoardEnum movedPiece;    
    Board::BitBoardEnum sideToMove = board.getSideToMove();
    BitBoard enemyBoard = board.getEnemyBoard();


    if(sideToMove == board.White){
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
                Move move = {fromSq,toSq, false, movedPiece};
                moveVector.push_back(move);
            }

            
            if(board.checkBit(enemyBoard,toSq)){
                Move move = {fromSq,toSq, true, movedPiece};
                moveVector.push_back(move);
            }
        

            toSq = board.popLsb(knightMoves);
        }
        fromSq = board.popLsb(knights);
    }

}

void MoveGenerator::generateRookMoves(Board board, std::vector<Move> &moveVector)
{
}

void MoveGenerator::generateBishopMoves(Board board, std::vector<Move> &moveVector)
{
}

void MoveGenerator::generateQueenMoves(Board board, std::vector<Move> &moveVector)
{
}

void MoveGenerator::generateKingMoves(Board board, std::vector<Move> &moveVector)
{
}


