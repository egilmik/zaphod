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
    BitBoard allPieces = board.getBitboard(BitBoardEnum::All);
    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(BitBoardEnum::N + board.getSideToMove());
    BitBoard enemyBoard = board.getEnemyBoard();
    BitBoard knights = board.getBitboard(movedPiece);

    
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
    BitBoard emptySquares = ~board.getBitboard(BitBoardEnum::All);
    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(BitBoardEnum::R + board.getSideToMove());
    BitBoard enemyBoard = board.getEnemyBoard();
    BitBoard rooks = board.getBitboard(movedPiece);
    

    if(board.getSideToMove() == BitBoardEnum::White){
        rooks = board.getBitboard(BitBoardEnum::R);
        movedPiece = BitBoardEnum::R;
    } else {
        rooks = board.getBitboard(BitBoardEnum::r);
        movedPiece = BitBoardEnum::r;
    }

    int fromSq = 0;
    while(rooks != 0){
        fromSq = board.popLsb(rooks);

        
        BitBoard magicBoard = board.getRookMagics(fromSq);
        int toSq = 0;

        while (magicBoard != 0) {
            toSq = board.popLsb(magicBoard);
            if (board.checkBit(emptySquares, toSq)) {
                moveList.moves[moveList.counter++] = { fromSq,toSq, false,BitBoardEnum::All,false, false,false,movedPiece };
                
            }
            else if (board.checkBit(enemyBoard, toSq)) {
                moveList.moves[moveList.counter++] = { fromSq,toSq, true,BitBoardEnum::All,false, false,false,movedPiece };
                
            }
        }
    }
}

void MoveGenerator::generateBishopMoves(Board &board, MoveList &moveList)
{
    BitBoard emptySquares = ~board.getBitboard(BitBoardEnum::All);
    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(BitBoardEnum::B + board.getSideToMove());
    BitBoard enemyBoard = board.getEnemyBoard();
    BitBoard bishops = board.getBitboard(movedPiece);

    int fromSq = 0;
    while(bishops != 0){
        fromSq = board.popLsb(bishops);
        BitBoard moves = board.getBishopMagics(fromSq);

        BitBoard captures = moves & enemyBoard;
        BitBoard silentMoves = moves & emptySquares;

        int toSq = 0;
        
        while(silentMoves != 0){
            toSq = board.popLsb(silentMoves);
            moveList.moves[moveList.counter++] = {fromSq,toSq, false,BitBoardEnum::All,false, false,false,movedPiece};
        }

        while (captures!= 0) {
            toSq = board.popLsb(captures);
            moveList.moves[moveList.counter++] = { fromSq,toSq, true,BitBoardEnum::All,false, false,false,movedPiece };
        }
        
    }
}

void MoveGenerator::generateQueenMoves(Board &board, MoveList &moveList)
{
    BitBoard emptySquares = ~board.getBitboard(BitBoardEnum::All);
    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(BitBoardEnum::Q + board.getSideToMove());
    BitBoard enemyBoard = board.getEnemyBoard();
    BitBoard queens = board.getBitboard(movedPiece);

    int fromSq = 0;
    while(queens != 0){
        fromSq = board.popLsb(queens);

        BitBoard moves = (board.getBishopMagics(fromSq) | board.getRookMagics(fromSq));

        BitBoard captures = moves & enemyBoard;
        BitBoard silentMoves = moves & emptySquares;

        int toSq = 0;

        while (silentMoves != 0) {
            toSq = board.popLsb(silentMoves);
            moveList.moves[moveList.counter++] = { fromSq,toSq, false,BitBoardEnum::All,false, false,false,movedPiece };
        }

        while (captures != 0) {
            toSq = board.popLsb(captures);
            moveList.moves[moveList.counter++] = { fromSq,toSq, true,BitBoardEnum::All,false, false,false,movedPiece };
        }
        
    }
}

void MoveGenerator::generateKingMoves(Board &board, MoveList &moveList)
{
    BitBoard allPieces = board.getBitboard(BitBoardEnum::All);
    BitBoard emptySquares = ~allPieces;
    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(BitBoardEnum::K + board.getSideToMove());
    BitBoard enemyBoard = board.getEnemyBoard();
    BitBoard king = board.getBitboard(movedPiece);
    BitBoardEnum sideToMove = board.getSideToMove();

    int fromSq = board.popLsb(king);
    BitBoard kingMove = board.getKingMask(fromSq);

    BitBoard enemyKnightAttacks = 0;
    /*
    //Here we can remove at least knight moves
    BitBoard enemyKnights = board.getBitboard(static_cast<BitBoardEnum>(BitBoardEnum::N + board.getOtherSide()));
    int knightSquare = 0;
    BitBoard enemyKnightAttacks = 0;
    while (enemyKnights != 0) {
        knightSquare = board.popLsb(enemyKnights);
        enemyKnightAttacks |= board.getKnightMask(knightSquare);
    }

    kingMove &= ~enemyKnightAttacks;

    */
    
    
    
    int toSq = 0;
    while(kingMove != 0){
        toSq = board.popLsb(kingMove);


        
        if(!board.checkBit(allPieces,toSq)){
            moveList.moves[moveList.counter++] = {fromSq,toSq, false,BitBoardEnum::All,false, false,false,movedPiece};
        } else if(board.checkBit(enemyBoard,toSq)){
            moveList.moves[moveList.counter++] = {fromSq,toSq, true,BitBoardEnum::All,false, false,false,movedPiece};
        }
        
    }

    if (sideToMove == BitBoardEnum::White) {
        if (board.getCastleRightsWK()) {
            BitBoard castlineSquares = 0;
            board.setBit(castlineSquares, 5);
            board.setBit(castlineSquares, 6);
            if ((allPieces & castlineSquares) == 0 && !board.isSquareAttacked(castlineSquares | board.sqBB[fromSq], BitBoardEnum::Black)) {
                moveList.moves[moveList.counter++] = { fromSq,fromSq + 2,false,BitBoardEnum::All,false,false,true,movedPiece };
            }  //f1,g1;
        }
        if (board.getCastleRightsWQ()) {
            BitBoard checkSquaresWQ = 0;
            board.setBit(checkSquaresWQ, 2);
            board.setBit(checkSquaresWQ, 3);

            BitBoard emptySquaresWQ = checkSquaresWQ;
            board.setBit(emptySquaresWQ, 1);

            if ((allPieces & emptySquaresWQ) == 0 && !board.isSquareAttacked(checkSquaresWQ | board.sqBB[fromSq], BitBoardEnum::Black)) {
                moveList.moves[moveList.counter++] = { fromSq,fromSq - 2,false,BitBoardEnum::All,false,false,true,movedPiece };
            }  //b1,c1,d1;
        }
    }
    else if (sideToMove == BitBoardEnum::Black) {
        if (board.getCastleRightsBK()) {
            BitBoard castlineSquares = 0;
            board.setBit(castlineSquares, 61);
            board.setBit(castlineSquares, 62);
            if ((allPieces & castlineSquares) == 0 && !board.isSquareAttacked(castlineSquares | board.sqBB[fromSq], BitBoardEnum::White)) {
                moveList.moves[moveList.counter++] = { fromSq,fromSq + 2,false,BitBoardEnum::All,false,false,true,movedPiece };
            }
        }
        if (board.getCastleRightsBQ()) {
            BitBoard checkSquaresBQ = 0;

            board.setBit(checkSquaresBQ, 58);
            board.setBit(checkSquaresBQ, 59);
            BitBoard emptySquaresBQ = checkSquaresBQ;
            board.setBit(emptySquaresBQ, 57);
            if ((allPieces & emptySquaresBQ) == 0 && !board.isSquareAttacked(checkSquaresBQ | board.sqBB[fromSq], BitBoardEnum::White)) {
                moveList.moves[moveList.counter++] = { fromSq,fromSq - 2,false,BitBoardEnum::All,false,false,true,movedPiece };
            }
        }
    }
    
}

