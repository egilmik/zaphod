#include "movegenerator.h"
#include <iostream>

void MoveGenerator::generateMoves(Board &board,std::vector<Move> &moveVector)
{
    generatePawnMoves(board,moveVector);
    generateKnightMoves(board, moveVector);
    generateRookMoves(board,moveVector);
    generateKingMoves(board,moveVector);
    generateBishopMoves(board,moveVector);
    generateQueenMoves(board, moveVector);
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


            if(!board.checkBit(allPieces,toSq) && !board.checkBit(allPieces,fromSq+pawnIncrement)){
                Move move = {fromSq,toSq, false, movedPiece};
                moveVector.push_back(move);
            }
        }

        //Capture
        
        BitBoard pawn = 0;
        BitBoard attack = 0;
        board.setBit(pawn,fromSq);

        if(sideToMove == board.White){
            attack = enemyBoard & board.southEastOne(pawn);
            attack |= enemyBoard & board.southWestOne(pawn);
        } else {            
            attack = enemyBoard & board.northEastOne(pawn);
            attack |= enemyBoard & board.northWestOne(pawn);
        }

        while(attack != 0){
            toSq = board.popLsb(attack);
            Move move = {fromSq,toSq, true, movedPiece};
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
    BitBoard emptySquares = ~allPieces;
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

    int fromSq = 0;
    while(rooks != 0){
        fromSq = board.popLsb(rooks);
        BitBoard rookBoard = 0;
        board.setBit(rookBoard,fromSq);

        BitBoard moves = board.southOccludedMoves(rookBoard, emptySquares);
        moves |= board.northOccludedMoves(rookBoard, emptySquares);
        moves |= board.westOccludedMoves(rookBoard, emptySquares);
        moves |= board.eastOccludedMoves(rookBoard, emptySquares);

        int toSq = 0;
        
        while(moves != 0){
            toSq = board.popLsb(moves);
            if(board.checkBit(emptySquares,toSq)){
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
    BitBoard emptySquares = ~allPieces;
    Board::BitBoardEnum sideToMove = board.getSideToMove();
    BitBoard enemyBoard = board.getEnemyBoard();
    BitBoard ownBoard = board.getOwnBoard();
    BitBoard bishops;

    if(sideToMove == board.White){
        bishops = board.getBitboard(Board::B);
        movedPiece = Board::B;
    } else {
        bishops = board.getBitboard(Board::b);
        movedPiece = Board::b;
    }

    int fromSq = 0;
    while(bishops != 0){
        fromSq = board.popLsb(bishops);
        BitBoard bishopsBoard = 0;
        board.setBit(bishopsBoard,fromSq);

        BitBoard moves = board.northEastOccludedMoves(bishopsBoard, emptySquares);
        moves |= board.northWestccludedMoves(bishopsBoard, emptySquares);
        moves |= board.southEastOccludedMoves(bishopsBoard, emptySquares);
        moves |= board.southWestOccludedMoves(bishopsBoard, emptySquares);

        int toSq = 0;
        
        while(moves != 0){
            toSq = board.popLsb(moves);
            if(board.checkBit(emptySquares,toSq)){
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
    BitBoard emptySquares = ~allPieces;
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

    int fromSq = 0;
    while(queens != 0){
        fromSq = board.popLsb(queens);
        BitBoard queenBoard = 0;
        board.setBit(queenBoard,fromSq);

        BitBoard moves = board.southOccludedMoves(queenBoard, emptySquares);
        moves |= board.northOccludedMoves(queenBoard, emptySquares);
        moves |= board.westOccludedMoves(queenBoard, emptySquares);
        moves |= board.eastOccludedMoves(queenBoard, emptySquares);
        moves |= board.northEastOccludedMoves(queenBoard, emptySquares);
        moves |= board.northWestccludedMoves(queenBoard, emptySquares);
        moves |= board.southEastOccludedMoves(queenBoard, emptySquares);
        moves |= board.southWestOccludedMoves(queenBoard, emptySquares);

        int toSq = 0;
        
        while(moves != 0){
            toSq = board.popLsb(moves);
            if(board.checkBit(emptySquares,toSq)){
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

