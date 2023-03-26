#include "movegenerator.h"
#include <iostream>

std::vector<Move> MoveGenerator::generateMoves(Board &board)
{
    std::vector<Move> moveVector;
    generatePawnMoves(board,moveVector);
    generateKnightMoves(board, moveVector);
    generateRookMoves(board,moveVector);
    return moveVector;
}

void MoveGenerator::generatePawnMoves(Board board, std::vector<Move> &moveVector)
{
    BitBoard pawns;

    BitBoard allPieces = board.getBitboard(Board::All);
    Board::BitBoardEnum movedPiece;
    Board::BitBoardEnum sideToMove = board.getSideToMove();
    BitBoard enemyBoard = board.getEnemyBoard();

    if(sideToMove == board.White){
        pawns = board.getBitboard(Board::P);
        movedPiece = Board::P;
    } else {
        pawns = board.getBitboard(Board::p);
        movedPiece = Board::p;
    }

    int pawnIncrement = 8;
    int pawnDoubleIncrement = 16;
    int pawnCaptureLeftIncrement = 7;
    int pawnCaptureRightIncrement = 9;
    BitBoard rank2or7 = board.Rank2Mask;

    if(sideToMove == board.Black){
        pawnDoubleIncrement = -16;
        pawnIncrement = -8;
        pawnCaptureLeftIncrement = -7;
        pawnCaptureRightIncrement = -9;
        rank2or7 = board.Rank7Mask;
    }
    
    
    int fromSq = board.popLsb(pawns);
    while (fromSq != 0)
    {

        
        int toSq = fromSq+pawnIncrement;

        if(!board.checkBit(allPieces,toSq)){
            Move move = {fromSq,toSq, false, movedPiece};
            moveVector.push_back(move);
        }
        
        if(board.checkBit(rank2or7,fromSq)){
            //Two squares forward from starting position
            toSq =fromSq+pawnDoubleIncrement;

            if(!board.checkBit(allPieces,toSq)){
                Move move = {fromSq,toSq, false, movedPiece};
                moveVector.push_back(move);
            }
        }

        //Capture
        if(board.checkBit(enemyBoard,fromSq+pawnCaptureLeftIncrement)){
            Move move = {fromSq,fromSq+pawnCaptureLeftIncrement, true, movedPiece};
            moveVector.push_back(move);
        }
        if(board.checkBit(enemyBoard,fromSq+pawnCaptureRightIncrement)){
            Move move = {fromSq,fromSq+pawnCaptureRightIncrement, true, movedPiece};
            moveVector.push_back(move);
        }

        // En passant

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

    // TODO, this might cause a bug when the knight is at position 0
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
    BitBoard allPieces = board.getBitboard(Board::All);
    Board::BitBoardEnum movedPiece;
    Board::BitBoardEnum sideToMove = board.getSideToMove();
    BitBoard enemyBoard = board.getEnemyBoard();
    BitBoard rooks;

    if(sideToMove == board.White){
        rooks = board.getBitboard(Board::R);
        movedPiece = Board::R;
    } else {
        rooks = board.getBitboard(Board::r);
        movedPiece = Board::r;
    }

    int fromSq = 0;
    while(rooks != 0){
        fromSq = board.popLsb(rooks);
        BitBoard rookBoard = 0;
        board.setBit(rookBoard,fromSq);

        BitBoard moves = southOccludedMoves(rookBoard, ~allPieces);
        moves |= northOccludedMoves(rookBoard, ~allPieces);
        board.popBit(moves,fromSq);

        int toSq = 0;
        
        while(moves != 0){
            toSq = board.popLsb(moves);
            Move move = {fromSq,toSq, true, movedPiece};
            moveVector.push_back(move);            
        }
        
    }
}

BitBoard MoveGenerator::southAttacks(BitBoard rooks, BitBoard empty) {
   BitBoard flood = rooks;
   flood |= rooks = (rooks >> 8) & empty;
   flood |= rooks = (rooks >> 8) & empty;
   flood |= rooks = (rooks >> 8) & empty;
   flood |= rooks = (rooks >> 8) & empty;
   flood |= rooks = (rooks >> 8) & empty;
   flood |= (rooks >> 8) & empty;
   return flood >> 8;
}

BitBoard MoveGenerator::northAttacks(BitBoard rooks, BitBoard empty) {
   BitBoard flood = rooks;
   flood |= rooks = (rooks << 8) & empty;
   flood |= rooks = (rooks << 8) & empty;
   flood |= rooks = (rooks << 8) & empty;
   flood |= rooks = (rooks << 8) & empty;
   flood |= rooks = (rooks << 8) & empty;
   flood |= (rooks << 8) & empty;
   return flood << 8;
}

BitBoard MoveGenerator::southOccludedMoves(BitBoard pieces, BitBoard empty)
{
   BitBoard flood = 0;
   while (pieces) {
      flood |= pieces;
      pieces = (pieces >> 8) & empty;
   }
   return flood;
}

BitBoard MoveGenerator::northOccludedMoves(BitBoard pieces, BitBoard empty)
{
   BitBoard flood = 0;
   while (pieces) {
      flood |= pieces;
      pieces = (pieces << 8) & empty;
   }
   return flood;
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


