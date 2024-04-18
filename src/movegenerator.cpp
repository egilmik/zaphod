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

void MoveGenerator::generatePawnMoves(Board& board, MoveList& moveList)
{
    BitBoard allPieces = board.getBitboard(BitBoardEnum::All);
    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(BitBoardEnum::P + board.getSideToMove());
    BitBoard pawns = board.getBitboard(movedPiece);
    BitBoardEnum sideToMove = board.getSideToMove();
    BitBoard enemyBoard = board.getEnemyBoard();

    int pawnIncrement = 8;
    int pawnDoubleIncrement = 16;
    int pawnCaptureLeftIncrement = 7;
    int pawnCaptureRightIncrement = 9;
    int enPassantIncrement = -8;
    BitBoard doublePushRank = board.Rank3Mask;
    BitBoard promotionRank = board.Rank8Mask;

    if (sideToMove == BitBoardEnum::Black) {
        pawnDoubleIncrement = -16;
        pawnIncrement = -8;
        pawnCaptureLeftIncrement = -7;
        pawnCaptureRightIncrement = -9;
        doublePushRank = board.Rank6Mask;
        promotionRank = board.Rank1Mask;
    }

    BitBoardEnum queenPromo = static_cast<BitBoardEnum>(BitBoardEnum::Q + board.getSideToMove());
    BitBoardEnum bishopPromo = static_cast<BitBoardEnum>(BitBoardEnum::B + board.getSideToMove());
    BitBoardEnum knightPromo = static_cast<BitBoardEnum>(BitBoardEnum::N + board.getSideToMove());
    BitBoardEnum rookPromo = static_cast<BitBoardEnum>(BitBoardEnum::R + board.getSideToMove());

    BitBoard singlePush = 0;
    BitBoard doublePush = 0;
    BitBoard promotions = 0;
    BitBoard nwAttacks = 0;
    BitBoard neAttacks = 0;
    BitBoard promoNWAttacks = 0;
    BitBoard promoNEAttacks = 0;


    if (board.getSideToMove() == White) {
        singlePush = (pawns << 8) & ~allPieces;
        doublePush = ((singlePush & doublePushRank) << 8) & ~allPieces;

        neAttacks = ((pawns & ~board.FileHMask) << 7) & enemyBoard;
        nwAttacks = ((pawns & ~board.FileAMask) << 9) & enemyBoard;
    }
    else {
        singlePush = (pawns >> 8) & ~allPieces;
        doublePush = ((singlePush & doublePushRank) >> 8) & ~allPieces;
        neAttacks = ((pawns & ~board.FileAMask) >> 7) & enemyBoard;
        nwAttacks = ((pawns & ~board.FileHMask) >> 9) & enemyBoard;
    }

    promotions = (singlePush & promotionRank);
    singlePush &= ~promotionRank;
    promoNEAttacks = (neAttacks & promotionRank);
    promoNWAttacks = (nwAttacks & promotionRank);
    neAttacks &= ~promotionRank;
    nwAttacks &= ~promotionRank;

    //Single push
    int square = 0;
    while (singlePush) {
        square = board.popLsb(singlePush);
        moveList.moves[moveList.counter++] = { square - pawnIncrement,square, false, BitBoardEnum::All,false, false,false,movedPiece };
    }

    while (doublePush) {
        square = board.popLsb(doublePush);
        moveList.moves[moveList.counter++] = { square - pawnDoubleIncrement,square, false, BitBoardEnum::All,true, false,false,movedPiece };
    }

    while (promotions) {
        square = board.popLsb(promotions);
        moveList.moves[moveList.counter++] = { square - pawnIncrement,square,false,queenPromo,false,false,false,movedPiece };
        moveList.moves[moveList.counter++] = { square - pawnIncrement,square,false,rookPromo,false,false,false,movedPiece };
        moveList.moves[moveList.counter++] = { square - pawnIncrement,square,false,knightPromo,false,false,false,movedPiece };
        moveList.moves[moveList.counter++] = { square - pawnIncrement,square,false,bishopPromo,false,false,false,movedPiece };
    }

    while (nwAttacks) {
        square = board.popLsb(nwAttacks);
        moveList.moves[moveList.counter++] = { square - pawnCaptureRightIncrement,square, true,BitBoardEnum::All,false,false,false,movedPiece };
    }

    while (neAttacks) {
        square = board.popLsb(neAttacks);
        moveList.moves[moveList.counter++] = { square - pawnCaptureLeftIncrement,square, true,BitBoardEnum::All,false,false,false,movedPiece };
    }

    while (promoNEAttacks) {
        square = board.popLsb(promoNEAttacks);
        moveList.moves[moveList.counter++] = { square - pawnCaptureLeftIncrement,square,true,queenPromo,false,false,false,movedPiece };
        moveList.moves[moveList.counter++] = { square - pawnCaptureLeftIncrement,square,true,rookPromo,false,false,false,movedPiece };
        moveList.moves[moveList.counter++] = { square - pawnCaptureLeftIncrement,square,true,knightPromo,false,false,false,movedPiece };
        moveList.moves[moveList.counter++] = { square - pawnCaptureLeftIncrement,square,true,bishopPromo,false,false,false,movedPiece };
    }

    while (promoNWAttacks) {
        square = board.popLsb(promoNWAttacks);
        moveList.moves[moveList.counter++] = { square - pawnCaptureRightIncrement,square,true,queenPromo,false,false,false,movedPiece };
        moveList.moves[moveList.counter++] = { square - pawnCaptureRightIncrement,square,true,rookPromo,false,false,false,movedPiece };
        moveList.moves[moveList.counter++] = { square - pawnCaptureRightIncrement,square,true,knightPromo,false,false,false,movedPiece };
        moveList.moves[moveList.counter++] = { square - pawnCaptureRightIncrement,square,true,bishopPromo,false,false,false,movedPiece };
    }

    if (board.getEnPassantSq() != Board::noSq) {
        int fromSq = board.popLsb(pawns);
        while (fromSq != 0)
        {
            int toSq = fromSq + pawnIncrement;
            BitBoard fromSqBoard = board.sqBB[fromSq];
            BitBoard attack = 0;

            if (sideToMove == BitBoardEnum::White) {
                attack = board.sqBB[board.getEnPassantSq()] & (board.southEastOne(fromSqBoard) | board.southWestOne(fromSqBoard));
            }
            else {
                attack = board.sqBB[board.getEnPassantSq()] & (board.northEastOne(fromSqBoard) | board.northWestOne(fromSqBoard));
            }

            while (attack != 0) {
                toSq = board.popLsb(attack);
                moveList.moves[moveList.counter++] = { fromSq,toSq, true,BitBoardEnum::All,false,true, false,movedPiece };
            }

            fromSq = board.popLsb(pawns);
        }
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
    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(BitBoardEnum::K + board.getSideToMove());
    BitBoard enemyBoard = board.getEnemyBoard();
    BitBoard king = board.getBitboard(movedPiece);
    BitBoardEnum sideToMove = board.getSideToMove();

    int fromSq = board.popLsb(king);
    BitBoard kingMove = board.getKingMask(fromSq);


    //Here we can remove at least knight moves
    BitBoard enemyKnights = board.getBitboard(static_cast<BitBoardEnum>(BitBoardEnum::N + board.getOtherSide()));
    int knightSquare = 0;
    BitBoard enemyKnightAttacks = 0;
    while (enemyKnights != 0) {
        knightSquare = board.popLsb(enemyKnights);
        enemyKnightAttacks |= board.getKnightMask(knightSquare);
    }

    kingMove &= ~enemyKnightAttacks;

    
    
    
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

