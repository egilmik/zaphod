#include "movegenerator.h"
#include <iostream>

std::vector<Move> MoveGenerator::generateMoves(Board &board)
{
    std::vector<Move> moveVector;
    generatePawnMoves(board,moveVector);
    generateKnightMoves(board, moveVector);
    generateRookMoves(board,moveVector);
    generateKingMoves(board,moveVector);
    generateBishopMoves(board,moveVector);
    generateQueenMoves(board, moveVector);
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
    BitBoard ownBoard = board.getOwnBoard();
    BitBoard rooks;

    if(sideToMove == board.White){
        rooks = board.getBitboard(Board::R);
        movedPiece = Board::R;
    } else {
        rooks = board.getBitboard(Board::r);
        movedPiece = Board::r;
    }

    // TODO Attack for rooks
    int fromSq = 0;
    while(rooks != 0){
        fromSq = board.popLsb(rooks);
        BitBoard rookBoard = 0;
        board.setBit(rookBoard,fromSq);

        BitBoard moves = southOccludedMoves(rookBoard, ~allPieces);
        moves |= northOccludedMoves(rookBoard, ~allPieces);
        moves |= westOccludedMoves(rookBoard, ~allPieces);
        moves |= eastOccludedMoves(rookBoard, ~allPieces);

        int toSq = 0;
        
        while(moves != 0){
            toSq = board.popLsb(moves);
            if(board.checkBit(~allPieces,toSq)){
                Move move = {fromSq,toSq, false, movedPiece};
                moveVector.push_back(move);            
            } else if(board.checkBit(enemyBoard,toSq)){
                Move move = {fromSq,toSq, true, movedPiece};
                moveVector.push_back(move);            
            }

            
        }
        
    }
}

void MoveGenerator::generateBishopMoves(Board board, std::vector<Move> &moveVector)
{
    BitBoard allPieces = board.getBitboard(Board::All);
    Board::BitBoardEnum movedPiece;
    Board::BitBoardEnum sideToMove = board.getSideToMove();
    BitBoard enemyBoard = board.getEnemyBoard();
    BitBoard ownBoard = board.getOwnBoard();
    BitBoard bishops;

    if(sideToMove == board.White){
        bishops = board.getBitboard(Board::B);
        movedPiece = Board::b;
    } else {
        bishops = board.getBitboard(Board::b);
        movedPiece = Board::b;
    }

    int fromSq = 0;
    while(bishops != 0){
        fromSq = board.popLsb(bishops);
        BitBoard bishopsBoard = 0;
        board.setBit(bishopsBoard,fromSq);

        BitBoard moves = northEastOccludedMoves(bishopsBoard, ~allPieces);
        moves |= northWestccludedMoves(bishopsBoard, ~allPieces);
        moves |= southEastOccludedMoves(bishopsBoard, ~allPieces);
        moves |= southWestOccludedMoves(bishopsBoard, ~allPieces);

        int toSq = 0;
        
        while(moves != 0){
            toSq = board.popLsb(moves);
            if(board.checkBit(~allPieces,toSq)){
                Move move = {fromSq,toSq, false, movedPiece};
                moveVector.push_back(move);            
            } else if(board.checkBit(enemyBoard,toSq)){
                Move move = {fromSq,toSq, true, movedPiece};
                moveVector.push_back(move);            
            }
        }
        
    }
}

void MoveGenerator::generateQueenMoves(Board board, std::vector<Move> &moveVector)
{
    BitBoard allPieces = board.getBitboard(Board::All);
    Board::BitBoardEnum movedPiece;
    Board::BitBoardEnum sideToMove = board.getSideToMove();
    BitBoard enemyBoard = board.getEnemyBoard();
    BitBoard ownBoard = board.getOwnBoard();
    BitBoard queens;

    if(sideToMove == board.White){
        queens = board.getBitboard(Board::Q);
        movedPiece = Board::Q;
    } else {
        queens = board.getBitboard(Board::q);
        movedPiece = Board::q;
    }

    // TODO Attack for rooks
    int fromSq = 0;
    while(queens != 0){
        fromSq = board.popLsb(queens);
        BitBoard queenBoard = 0;
        board.setBit(queenBoard,fromSq);

        BitBoard moves = southOccludedMoves(queenBoard, ~allPieces);
        moves |= northOccludedMoves(queenBoard, ~allPieces);
        moves |= westOccludedMoves(queenBoard, ~allPieces);
        moves |= eastOccludedMoves(queenBoard, ~allPieces);
        moves |= northEastOccludedMoves(queenBoard, ~allPieces);
        moves |= northWestccludedMoves(queenBoard, ~allPieces);
        moves |= southEastOccludedMoves(queenBoard, ~allPieces);
        moves |= southWestOccludedMoves(queenBoard, ~allPieces);

        int toSq = 0;
        
        while(moves != 0){
            toSq = board.popLsb(moves);
            if(board.checkBit(~allPieces,toSq)){
                Move move = {fromSq,toSq, false, movedPiece};
                moveVector.push_back(move);            
            } else if(board.checkBit(enemyBoard,toSq)){
                Move move = {fromSq,toSq, true, movedPiece};
                moveVector.push_back(move);            
            }
        }
        
    }
}

void MoveGenerator::generateKingMoves(Board board, std::vector<Move> &moveVector)
{
    BitBoard king;

    BitBoard allPieces = board.getBitboard(Board::All);
    Board::BitBoardEnum movedPiece;    
    Board::BitBoardEnum sideToMove = board.getSideToMove();
    BitBoard enemyBoard = board.getEnemyBoard();


    if(sideToMove == board.White){
        king = board.getBitboard(Board::K);
        movedPiece = Board::K;
    } else {
        king = board.getBitboard(Board::k);
        movedPiece = Board::k;
    }

    int fromSq = board.popLsb(king);
    BitBoard kingMove = board.getKingMask(fromSq);

    
    //TODO Might cause bug when king or tosq is 0?
    int toSq = board.popLsb(kingMove);
    while(toSq != 0){
        if(!board.checkBit(allPieces,toSq)){
            Move move = {fromSq,toSq, false, movedPiece};
            moveVector.push_back(move);
        }
            
        if(board.checkBit(enemyBoard,toSq)){
            Move move = {fromSq,toSq, true, movedPiece};
            moveVector.push_back(move);
        }
        

        toSq = board.popLsb(kingMove);
    }
    

}

BitBoard MoveGenerator::southOccludedMoves(BitBoard pieces, BitBoard empty)
{
   BitBoard flood = 0;
   while (pieces) {
      flood |= pieces;
      pieces = (pieces >> 8) & empty;
   }
   return (flood >> 8);
}

BitBoard MoveGenerator::northOccludedMoves(BitBoard pieces, BitBoard empty)
{
   BitBoard flood = 0;
   while (pieces) {
      flood |= pieces;
      pieces = (pieces << 8) & empty;
   }
   return (flood << 8);
}

BitBoard MoveGenerator::eastOccludedMoves(BitBoard pieces, BitBoard empty)
{
    BitBoard flood = 0;
    empty &= ~Board::FileAMask;
    while(pieces){
        flood |= pieces;
        pieces = (pieces >> 1) & empty;
    }
    return (flood >> 1) & ~Board::FileAMask;
}

BitBoard MoveGenerator::westOccludedMoves(BitBoard pieces, BitBoard empty)
{
   BitBoard flood = 0;
    empty &= ~Board::FileHMask;
    while(pieces){
        flood |= pieces;
        pieces = (pieces << 1) & empty;
    }
    return (flood << 1) & ~Board::FileHMask;
}

// TODO wrong bit shift?
BitBoard MoveGenerator::northEastOccludedMoves(BitBoard pieces, BitBoard empty)
{
    BitBoard flood = 0;
    empty &= ~Board::FileAMask;
    while(pieces){
        flood |= pieces = (pieces >> 9) & empty;
    }
    return (flood >> 9) & ~Board::FileAMask;
}

// TODO wrong bit shift?
BitBoard MoveGenerator::northWestccludedMoves(BitBoard pieces, BitBoard empty)
{
    BitBoard flood = 0;
    empty &= ~Board::FileHMask;
    while(pieces){
        flood |= pieces = (pieces >> 7) & empty;
    }
    return (flood >> 7) & ~Board::FileHMask;
}

// TODO wrong bit shift?
BitBoard MoveGenerator::southEastOccludedMoves(BitBoard pieces, BitBoard empty)
{
    BitBoard flood = 0;
    empty &= ~Board::FileAMask;
    while(pieces){
        flood |= pieces = (pieces << 9) & empty;
    }
    return (flood << 9) & ~Board::FileAMask;
}
// TODO wrong bit shift?
BitBoard MoveGenerator::southWestOccludedMoves(BitBoard pieces, BitBoard empty)
{
    BitBoard flood = 0;
    empty &= ~Board::FileAMask;
    while(pieces){
        flood |= pieces = (pieces << 9) & empty;
    }
    return (flood << 9) & ~Board::FileAMask;
}
