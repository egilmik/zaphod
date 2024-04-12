#include "movegenerator.h"
#include <iostream>

void MoveGenerator::generateMoves(Board &board,MoveList &moveList)
{
    
    generateKnightMoves(board, moveList);
    generateRookMoves(board,moveList);
    generateKingMoves(board,moveList);
    generateBishopMoves(board,moveList);
    generateQueenMoves(board, moveList);
    //Pawns last, to prevent promotions to move twice
    generatePawnMoves(board,moveList);
}

void MoveGenerator::generatePawnMoves(Board &board, MoveList &moveList)
{
    BitBoard pawns;

    BitBoard allPieces = board.getBitboard(BitBoardEnum::All);
    BitBoardEnum movedPiece;
    BitBoardEnum sideToMove = board.getSideToMove();
    BitBoard enemyBoard = board.getEnemyBoard();

    int pawnIncrement = 8;
    int pawnDoubleIncrement = 16;
    int pawnCaptureLeftIncrement = 7;
    int pawnCaptureRightIncrement = 9;
    int enPassantIncrement = -8;
    BitBoard doublePushRank = board.Rank2Mask;
    BitBoard promotionRank = board.Rank8Mask;


    BitBoardEnum queenPromo = BitBoardEnum::Q;
    BitBoardEnum bishopPromo = BitBoardEnum::B;
    BitBoardEnum knightPromo = BitBoardEnum::N;
    BitBoardEnum rookPromo = BitBoardEnum::R;


    if(sideToMove == BitBoardEnum::White){
        pawns = board.getBitboard(BitBoardEnum::P);
        movedPiece = BitBoardEnum::P;
    } else {
        pawns = board.getBitboard(BitBoardEnum::p);
        movedPiece = BitBoardEnum::p;

        pawnDoubleIncrement = -16;
        pawnIncrement = -8;
        pawnCaptureLeftIncrement = -7;
        pawnCaptureRightIncrement = -9;
        
        doublePushRank = board.Rank7Mask;
        promotionRank = board.Rank1Mask;

        queenPromo = BitBoardEnum::q;
        bishopPromo = BitBoardEnum::b;
        knightPromo = BitBoardEnum::n;
        rookPromo = BitBoardEnum::r;
    }    
    
    int fromSq = board.popLsb(pawns);
    while (fromSq != 0)
    {        
        int toSq = fromSq+pawnIncrement;
        BitBoard fromSqBoard = board.sqBB[fromSq];

        if((board.sqBB[toSq] & allPieces) == 0){            
            if((board.sqBB[toSq] & promotionRank) != 0){
                moveList.moves[moveList.counter++] = {fromSq,toSq,false,queenPromo,false,false,false,movedPiece};
                moveList.moves[moveList.counter++] = {fromSq,toSq,false,rookPromo,false,false,false,movedPiece};
                moveList.moves[moveList.counter++] = {fromSq,toSq,false,knightPromo,false,false,false,movedPiece};
                moveList.moves[moveList.counter++] = {fromSq,toSq,false,bishopPromo,false,false,false,movedPiece};
            } else {
                moveList.moves[moveList.counter++] = {fromSq,toSq, false, BitBoardEnum::All,false, false,false,movedPiece};
            }
            
        }
        
        if((fromSqBoard & doublePushRank) != 0){
            //Two squares forward from starting position
            toSq =fromSq+pawnDoubleIncrement;

            if((board.sqBB[toSq] & allPieces) == 0 && (board.sqBB[fromSq+pawnIncrement] & allPieces) == 0){
                moveList.moves[moveList.counter++] = {fromSq,toSq, false,BitBoardEnum::All,true,false,false,movedPiece};
            }
        }

        //Capture
        
        BitBoard attack = 0;   

        if(sideToMove == BitBoardEnum::White){
            attack = enemyBoard & (board.southEastOne(fromSqBoard) | board.southWestOne(fromSqBoard));
        } else {            
            attack = enemyBoard & (board.northEastOne(fromSqBoard) | board.northWestOne(fromSqBoard));
        }

        while(attack != 0){
            toSq = board.popLsb(attack);

            if((board.sqBB[toSq] & promotionRank) != 0){
                moveList.moves[moveList.counter++] = {fromSq,toSq,true,queenPromo,false,false,false,movedPiece};
                moveList.moves[moveList.counter++] = {fromSq,toSq,true,rookPromo,false,false,false,movedPiece};
                moveList.moves[moveList.counter++] = {fromSq,toSq,true,knightPromo,false,false,false,movedPiece};
                moveList.moves[moveList.counter++] = {fromSq,toSq,true,bishopPromo,false,false,false,movedPiece};
            } else {
                moveList.moves[moveList.counter++] = {fromSq,toSq, true,BitBoardEnum::All,false,false,false,movedPiece};
            }
        }

        if(board.getEnPassantSq() != Board::noSq){
            BitBoard attack = 0;

            if(sideToMove == BitBoardEnum::White){
                attack = board.sqBB[board.getEnPassantSq()] & (board.southEastOne(fromSqBoard) | board.southWestOne(fromSqBoard));
            } else {            
                attack = board.sqBB[board.getEnPassantSq()] & (board.northEastOne(fromSqBoard) | board.northWestOne(fromSqBoard));
            }

            while(attack != 0){
                toSq = board.popLsb(attack);
                moveList.moves[moveList.counter++] = {fromSq,toSq, true,BitBoardEnum::All,false,true, false,movedPiece};
            }
        }

        fromSq = board.popLsb(pawns);
    }
}


void MoveGenerator::generateKnightMoves(Board &board, MoveList &moveList)
{
    BitBoard knights;

    BitBoard allPieces = board.getBitboard(BitBoardEnum::All);
    BitBoardEnum movedPiece;    
    BitBoardEnum sideToMove = board.getSideToMove();
    BitBoard enemyBoard = board.getEnemyBoard();


    if(sideToMove == BitBoardEnum::White){
        knights = board.getBitboard(BitBoardEnum::N);
        movedPiece = BitBoardEnum::N;
    } else {
        knights = board.getBitboard(BitBoardEnum::n);
        movedPiece = BitBoardEnum::n;
    }

    
    int fromSq = 0;
    while (knights!= 0)
    {
        fromSq = board.popLsb(knights);
        BitBoard knightMoves = board.getKnightMask(fromSq);

        int toSq = 0;

        while(knightMoves != 0){
            toSq = board.popLsb(knightMoves);
            if(!board.checkBit(allPieces,toSq)){
                moveList.moves[moveList.counter++] = {fromSq,toSq, false,BitBoardEnum::All,false, false,false,movedPiece};
            }

            
            if(board.checkBit(enemyBoard,toSq)){
                moveList.moves[moveList.counter++] = {fromSq,toSq, true,BitBoardEnum::All,false, false,false,movedPiece};
            }
           
        }
    }
}

void MoveGenerator::generateRookMoves(Board &board, MoveList &moveList)
{
    BitBoard allPieces = board.getBitboard(BitBoardEnum::All);
    BitBoard emptySquares = ~allPieces;
    BitBoardEnum movedPiece;
    BitBoardEnum sideToMove = board.getSideToMove();
    BitBoard enemyBoard = board.getEnemyBoard();
    BitBoard ownBoard = board.getOwnBoard();
    BitBoard rooks;

    if(sideToMove == BitBoardEnum::White){
        rooks = board.getBitboard(BitBoardEnum::R);
        movedPiece = BitBoardEnum::R;
    } else {
        rooks = board.getBitboard(BitBoardEnum::r);
        movedPiece = BitBoardEnum::r;
    }

    int fromSq = 0;
    while(rooks != 0){
        fromSq = board.popLsb(rooks);
        /*
        BitBoard rookBoard = 0;
        board.setBit(rookBoard,fromSq);

        BitBoard moves = board.southOccludedMoves(rookBoard, emptySquares);
        moves |= board.northOccludedMoves(rookBoard, emptySquares);
        moves |= board.westOccludedMoves(rookBoard, emptySquares);
        moves |= board.eastOccludedMoves(rookBoard, emptySquares);
        */

        uint64_t magic = ((board.getBitboard(All) & board.rookMask[fromSq]) * board.magicNumberRook[fromSq]) >> board.magicNumberShiftsRook[fromSq];
        BitBoard magicBoard = (*board.magicMovesRook)[fromSq][magic];

        int toSq = 0;

        //MoveList old;
        //MoveList magicList;

        while (magicBoard != 0) {
            toSq = board.popLsb(magicBoard);
            if (board.checkBit(emptySquares, toSq)) {
                moveList.moves[moveList.counter++] = { fromSq,toSq, false,BitBoardEnum::All,false, false,false,movedPiece };
                
            }
            else if (board.checkBit(enemyBoard, toSq)) {
                moveList.moves[moveList.counter++] = { fromSq,toSq, true,BitBoardEnum::All,false, false,false,movedPiece };
                
            }
        }
        
        /*
        while(moves != 0){
            toSq = board.popLsb(moves);
            if(board.checkBit(emptySquares,toSq)){
                moveList.moves[moveList.counter++] = {fromSq,toSq, false,BitBoardEnum::All,false, false,false,movedPiece};
                old.moves[old.counter++] = { fromSq,toSq, false,BitBoardEnum::All,false, false,false,movedPiece };
            } else if(board.checkBit(enemyBoard,toSq)){
                moveList.moves[moveList.counter++] = {fromSq,toSq, true,BitBoardEnum::All,false, false,false,movedPiece};
                old.moves[old.counter++] = { fromSq,toSq, true,BitBoardEnum::All,false, false,false,movedPiece };
            }

            
        }
        

        if (magicList.counter != old.counter) {
            board.printBoard();
            board.printBoard(board.rookMask[fromSq], fromSq);

            uint64_t magic = ((board.getBitboard(All) & board.rookMask[fromSq]) * board.magicNumberRook[fromSq]) >> board.magicNumberShiftsRook[fromSq];
            BitBoard magicBoard = (*board.magicMovesRook)[fromSq][magic];
            board.printBoard(magicBoard, fromSq);

            int x = 0;
        }
        */
        
    }
}

void MoveGenerator::generateBishopMoves(Board &board, MoveList &moveList)
{
    BitBoard allPieces = board.getBitboard(BitBoardEnum::All);
    BitBoardEnum movedPiece;
    BitBoard emptySquares = ~allPieces;
    BitBoardEnum sideToMove = board.getSideToMove();
    BitBoard enemyBoard = board.getEnemyBoard();
    BitBoard ownBoard = board.getOwnBoard();
    BitBoard bishops;

    if(sideToMove == BitBoardEnum::White){
        bishops = board.getBitboard(BitBoardEnum::B);
        movedPiece = BitBoardEnum::B;
    } else {
        bishops = board.getBitboard(BitBoardEnum::b);
        movedPiece = BitBoardEnum::b;
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
                moveList.moves[moveList.counter++] = {fromSq,toSq, false,BitBoardEnum::All,false, false,false,movedPiece};
            } else if(board.checkBit(enemyBoard,toSq)){
                moveList.moves[moveList.counter++] = {fromSq,toSq, true,BitBoardEnum::All,false, false,false,movedPiece};
            }
        }
        
    }
}

void MoveGenerator::generateQueenMoves(Board &board, MoveList &moveList)
{
    BitBoardEnum movedPiece;
    BitBoard emptySquares = ~board.getBitboard(BitBoardEnum::All);
    BitBoardEnum sideToMove = board.getSideToMove();
    BitBoard enemyBoard = board.getEnemyBoard();
    BitBoard ownBoard = board.getOwnBoard();
    BitBoard queens;

    if(sideToMove == BitBoardEnum::White){
        queens = board.getBitboard(BitBoardEnum::Q);
        movedPiece = BitBoardEnum::Q;
    } else {
        queens = board.getBitboard(BitBoardEnum::q);
        movedPiece = BitBoardEnum::q;
    }

    int fromSq = 0;
    while(queens != 0){
        fromSq = board.popLsb(queens);
        BitBoard queenBoard = board.sqBB[fromSq];
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
                moveList.moves[moveList.counter++] = {fromSq,toSq, false,BitBoardEnum::All,false, false,false,movedPiece};
            } else if(board.checkBit(enemyBoard,toSq)){
                moveList.moves[moveList.counter++] = {fromSq,toSq, true,BitBoardEnum::All,false, false,false,movedPiece};
            }
        }
        
    }
}

void MoveGenerator::generateKingMoves(Board &board, MoveList &moveList)
{
    BitBoard king;

    BitBoard allPieces = board.getBitboard(BitBoardEnum::All);
    BitBoardEnum movedPiece;    
    BitBoardEnum sideToMove = board.getSideToMove();
    BitBoard enemyBoard = board.getEnemyBoard();


    if(sideToMove == BitBoardEnum::White){
        king = board.getBitboard(BitBoardEnum::K);
        movedPiece = BitBoardEnum::K;
    } else {
        king = board.getBitboard(BitBoardEnum::k);
        movedPiece = BitBoardEnum::k;
    }

    int fromSq = board.popLsb(king);
    BitBoard kingMove = board.getKingMask(fromSq);  
    
    int toSq = 0;
    while(kingMove != 0){
        toSq = board.popLsb(kingMove);
        
        if(!board.checkBit(allPieces,toSq)){
            moveList.moves[moveList.counter++] = {fromSq,toSq, false,BitBoardEnum::All,false, false,false,movedPiece};
        } else if(board.checkBit(enemyBoard,toSq)){
            moveList.moves[moveList.counter++] = {fromSq,toSq, true,BitBoardEnum::All,false, false,false,movedPiece};
        }
        
    }

    if(sideToMove == BitBoardEnum::White){
        if(board.getCastleRightsWK()){
            BitBoard castlineSquares = 0;
            board.setBit(castlineSquares,5);
            board.setBit(castlineSquares,6);
            if((allPieces & castlineSquares) == 0 && !board.isSquareAttacked(castlineSquares | board.sqBB[fromSq], BitBoardEnum::Black)){
                moveList.moves[moveList.counter++] = {fromSq,fromSq+2,false,BitBoardEnum::All,false,false,true,movedPiece};
            }  //f1,g1;
        }
        if(board.getCastleRightsWQ()){
            BitBoard checkSquaresWQ = 0;            
            board.setBit(checkSquaresWQ,2);
            board.setBit(checkSquaresWQ,3);

            BitBoard emptySquaresWQ = checkSquaresWQ;
            board.setBit(emptySquaresWQ,1);

            if((allPieces & emptySquaresWQ) == 0 && !board.isSquareAttacked(checkSquaresWQ| board.sqBB[fromSq], BitBoardEnum::Black)){
                moveList.moves[moveList.counter++] = {fromSq,fromSq-2,false,BitBoardEnum::All,false,false,true,movedPiece};
            }  //b1,c1,d1;
        }
    } else if(sideToMove == BitBoardEnum::Black){
        if(board.getCastleRightsBK()){
            BitBoard castlineSquares = 0;
            board.setBit(castlineSquares,61);
            board.setBit(castlineSquares,62);
            if((allPieces & castlineSquares) == 0 && !board.isSquareAttacked(castlineSquares | board.sqBB[fromSq], BitBoardEnum::White)){
                moveList.moves[moveList.counter++] = {fromSq,fromSq+2,false,BitBoardEnum::All,false,false,true,movedPiece};
            }
        }
        if(board.getCastleRightsBQ()){
            BitBoard checkSquaresBQ = 0;
            
            board.setBit(checkSquaresBQ,58);
            board.setBit(checkSquaresBQ,59);
            BitBoard emptySquaresBQ = checkSquaresBQ;
            board.setBit(emptySquaresBQ,57);
            if((allPieces & emptySquaresBQ) == 0 && !board.isSquareAttacked(checkSquaresBQ | board.sqBB[fromSq], BitBoardEnum::White)){
                moveList.moves[moveList.counter++] = {fromSq,fromSq-2,false,BitBoardEnum::All,false,false,true,movedPiece};
            }
        }
    }
    
}

