#include "movegenerator.h"
#include <iostream>

void MoveGenerator::generateMoves(Board &board,MoveList &moveList)
{
    
    BitBoard king = board.getBitboard(K + board.getSideToMove());
    int kingSquare = board.popLsb(king);
    king = board.getBitboard(K + board.getSideToMove());
    BitBoard snipers = board.getSnipers(kingSquare, board.getOtherSide());
    BitBoard sniperCopy = snipers;

    BitBoard pinned = 0;
    BitBoard potentialPinned = 0;
    

    while (sniperCopy) {
        int sniperSquare = board.popLsb(sniperCopy);
        potentialPinned = board.sqBetween[kingSquare][sniperSquare] & board.getBitboard(All);
        if (board.countSetBits(potentialPinned) == 1) {
            pinned |= potentialPinned & board.getBitboard(board.getSideToMove());
        }
    }

    //  Finding pieces giving check to the current side to move
    BitBoard checkers = 0;

    uint64_t magic = ((board.getBitboard(All) & board.rookMask[kingSquare]) * board.magicNumberRook[kingSquare]) >> board.magicNumberShiftsRook[kingSquare];
    checkers |= (*board.magicMovesRook)[kingSquare][magic] & (board.getBitboard(Q + board.getOtherSide()) | board.getBitboard(R + board.getOtherSide()));
    magic = ((board.getBitboard(All) & board.bishopMask[kingSquare]) * board.magicNumberBishop[kingSquare]) >> board.magicNumberShiftsBishop[kingSquare];
    checkers |= (*board.magicMovesBishop)[kingSquare][magic] & (board.getBitboard(Q + board.getOtherSide()) | board.getBitboard(B + board.getOtherSide()));

    checkers |= board.getKnightMask(kingSquare) & board.getBitboard(N + board.getOtherSide());

    if (board.getSideToMove() == White) {

        checkers |= ((king & ~board.FileHMask) << 7) & board.getBitboard(P + board.getOtherSide());
        checkers |= ((king & ~board.FileAMask) << 9) & board.getBitboard(P + board.getOtherSide());
    }
    else {
        checkers |= ((king & ~board.FileAMask) >> 7) & board.getBitboard(P + board.getOtherSide());
        checkers |= ((king & ~board.FileHMask) >> 9) & board.getBitboard(P + board.getOtherSide());
    }


    if (board.countSetBits(checkers) < 2) {
        generateKnightMoves(board, moveList, pinned);
        generateRookMoves(board, moveList, checkers, kingSquare);
        generateBishopMoves(board, moveList, checkers, kingSquare,pinned,snipers);
        generateQueenMoves(board, moveList, checkers, kingSquare, pinned,snipers);
        //Pawns last, to prevent promotions to move twice
        generatePawnMoves(board, moveList,checkers,kingSquare,pinned,snipers);
    }
    generateKingMoves(board, moveList);
}

void MoveGenerator::generatePawnMoves(Board& board, MoveList& moveList, BitBoard checkers, int kingSquare, BitBoard pinned, BitBoard snipers)
{
    BitBoard allPieces = board.getBitboard(BitBoardEnum::All);
    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(BitBoardEnum::P + board.getSideToMove());
    BitBoard pawns = board.getBitboard(movedPiece);
    BitBoardEnum sideToMove = board.getSideToMove();
    BitBoard enemyBoard = board.getEnemyBoard();

    BitBoard pinnedPawn = pawns & pinned;
    pawns &= ~pinnedPawn;

    if (pinnedPawn > 0) {
        //board.printBoard(pawns);
        //board.printBoard(pinnedPawn);
        int x = 0;
    }

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

    //Checking pinned pieces individually
    BitBoard pinnedPawnSinglePush = 0;
    BitBoard pinnedDoublePush = 0;
    BitBoard pinnedNEAttack = 0;
    BitBoard pinnedNWAttack = 0;

    BitBoard pinnedCopy = pinned;
    int pinnedSquare = board.popLsb(pinnedCopy);

    if (board.getSideToMove() == White) {
        singlePush = (pawns << 8) & ~allPieces;        
        doublePush = ((singlePush & doublePushRank) << 8) & ~allPieces;
        neAttacks = ((pawns & ~board.FileHMask) << 7) & enemyBoard;
        nwAttacks = ((pawns & ~board.FileAMask) << 9) & enemyBoard;

        pinnedPawnSinglePush = (pinnedPawn << 8) & ~allPieces;
        pinnedDoublePush = ((pinnedPawnSinglePush & doublePushRank) << 8) & ~allPieces;
        pinnedNEAttack = ((pinnedPawn & ~board.FileHMask) << 7) & enemyBoard;
        pinnedNWAttack = ((pinnedPawn & ~board.FileAMask) << 9) & enemyBoard;
    }
    else {
        singlePush = (pawns >> 8) & ~allPieces;
        doublePush = ((singlePush & doublePushRank) >> 8) & ~allPieces;
        neAttacks = ((pawns & ~board.FileAMask) >> 7) & enemyBoard;
        nwAttacks = ((pawns & ~board.FileHMask) >> 9) & enemyBoard;

        pinnedPawnSinglePush = (pinnedPawn >> 8) & ~allPieces;
        pinnedDoublePush = ((pinnedPawnSinglePush & doublePushRank) >> 8) & ~allPieces;
        pinnedNEAttack = ((pinnedPawn & ~board.FileAMask) >> 7) & enemyBoard;
        pinnedNWAttack = ((pinnedPawn & ~board.FileHMask) >> 9) & enemyBoard;
    }

    pinnedPawnSinglePush = makeLegalMoves(board, pinnedPawnSinglePush, pinned, checkers, snipers, pinnedSquare, kingSquare);
    pinnedDoublePush = makeLegalMoves(board, pinnedDoublePush, pinned, checkers, snipers, pinnedSquare, kingSquare);
    pinnedNEAttack = makeLegalMoves(board, pinnedNEAttack, pinned, checkers, snipers, pinnedSquare, kingSquare);
    pinnedNWAttack = makeLegalMoves(board, pinnedNWAttack, pinned, checkers, snipers, pinnedSquare, kingSquare);

    singlePush |= pinnedPawnSinglePush;
    doublePush |= pinnedDoublePush;
    neAttacks |= pinnedNEAttack;
    nwAttacks |= pinnedNWAttack;

    //If in check, only consider moves that capture checker or obstruct the check
    BitBoard inBetween = 0;
    BitBoard checks = checkers;
    while (checks) {
        inBetween |= board.sqBetween[kingSquare][board.popLsb(checks)];
    }
    if ((checkers | inBetween) > 0) {
        singlePush &= (inBetween | checkers);
        doublePush &= (inBetween | checkers);
        neAttacks &= (inBetween | checkers);
        nwAttacks &= (inBetween | checkers);
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
        
        while (pawns)
        {
            int fromSq = board.popLsb(pawns);
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
                BitBoard checkers = 0;
                BitBoard all = board.getBitboard(All);
                //Removing current attacking pawn and enpassant pawn from board to perform check check
                BitBoard toBeRemoved = (board.sqBB[fromSq] | board.sqBB[board.getEnPassantSq()- pawnIncrement]);

                all &= ~toBeRemoved;
                
                
                uint64_t magic = ((all & board.rookMask[kingSquare]) * board.magicNumberRook[kingSquare]) >> board.magicNumberShiftsRook[kingSquare];
                checkers |= (*board.magicMovesRook)[kingSquare][magic] & (board.getBitboard(Q + board.getOtherSide()) | board.getBitboard(R + board.getOtherSide()));
                magic = ((all & board.bishopMask[kingSquare]) * board.magicNumberBishop[kingSquare]) >> board.magicNumberShiftsBishop[kingSquare];
                checkers |= (*board.magicMovesBishop)[kingSquare][magic] & (board.getBitboard(Q + board.getOtherSide()) | board.getBitboard(B + board.getOtherSide()));

                if(checkers == 0){
                    moveList.moves[moveList.counter++] = { fromSq,toSq, true,BitBoardEnum::All,false,true, false,movedPiece };
                }
            }

        }
    }
}


void MoveGenerator::generateKnightMoves(Board &board, MoveList &moveList, BitBoard pinned)
{
    BitBoard allPieces = board.getBitboard(BitBoardEnum::All);
    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(BitBoardEnum::N + board.getSideToMove());
    BitBoard enemyBoard = board.getEnemyBoard();
    BitBoard knights = board.getBitboard(movedPiece) & ~pinned;


    
    int fromSq = 0;
    while (knights)
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

void MoveGenerator::generateRookMoves(Board &board, MoveList &moveList, BitBoard checkers, int kingSquare)
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


    BitBoard inBetween = 0;
    BitBoard checks = checkers;

    while (checks) {
        inBetween |= board.sqBetween[kingSquare][board.popLsb(checks)];
    }


    int fromSq = 0;
    while(rooks){
        fromSq = board.popLsb(rooks);

        
        BitBoard magicBoard = board.getRookMagics(fromSq);

        if ((checkers | inBetween) > 0) {
            magicBoard &= (inBetween | checkers);
        }
        int toSq = 0;

        while (magicBoard) {
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

BitBoard MoveGenerator::makeLegalMoves(Board &board, BitBoard moves, BitBoard pinned, BitBoard checkers, BitBoard snipers, int fromSq, int kingSquare) {
    // Inbetween king and checker
    BitBoard inBetweenKChecker = 0;
    BitBoard checks = checkers;

    while (checks) {
        inBetweenKChecker |= board.sqBetween[kingSquare][board.popLsb(checks)];
    }

    if ((checkers | inBetweenKChecker) > 0) {
        moves &= (inBetweenKChecker | checkers);
    }


    // We are pinned
    if ((pinned & board.sqBB[fromSq]) > 0) {
        BitBoard sniperCopy = snipers;
        while (sniperCopy) {
            int sniperSquare = board.popLsb(sniperCopy);
            BitBoard inBetween = board.sqBetween[kingSquare][sniperSquare] & board.sqBB[fromSq];
            if (inBetween > 0) {
                inBetween = board.sqBetween[kingSquare][sniperSquare] | board.sqBB[sniperSquare];
                moves &= inBetween;
            }
        }
    }
    return moves;
}

void MoveGenerator::generateBishopMoves(Board &board, MoveList &moveList, BitBoard checkers, int kingSquare, BitBoard pinned, BitBoard snipers)
{
    BitBoard emptySquares = ~board.getBitboard(BitBoardEnum::All);
    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(BitBoardEnum::B + board.getSideToMove());
    BitBoard enemyBoard = board.getEnemyBoard();
    BitBoard bishops = board.getBitboard(movedPiece);

    int fromSq = 0;
    while(bishops){
        fromSq = board.popLsb(bishops);
        BitBoard moves = board.getBishopMagics(fromSq);

        moves = makeLegalMoves(board, moves, pinned, checkers, snipers, fromSq, kingSquare);

        BitBoard captures = moves & enemyBoard;
        BitBoard silentMoves = moves & emptySquares;



        int toSq = 0;
        
        while(silentMoves){
            toSq = board.popLsb(silentMoves);
            moveList.moves[moveList.counter++] = {fromSq,toSq, false,BitBoardEnum::All,false, false,false,movedPiece};
        }

        while (captures!= 0) {
            toSq = board.popLsb(captures);
            moveList.moves[moveList.counter++] = { fromSq,toSq, true,BitBoardEnum::All,false, false,false,movedPiece };
        }
        
    }
}

void MoveGenerator::generateQueenMoves(Board &board, MoveList &moveList, BitBoard checkers, int kingSquare, BitBoard pinned, BitBoard snipers)
{
    BitBoard emptySquares = ~board.getBitboard(BitBoardEnum::All);
    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(BitBoardEnum::Q + board.getSideToMove());
    BitBoard enemyBoard = board.getEnemyBoard();
    BitBoard queens = board.getBitboard(movedPiece);

    // Inbetween king and checker
    BitBoard inBetweenKChecker = 0;
    BitBoard checks = checkers;

    while (checks) {
        inBetweenKChecker |= board.sqBetween[kingSquare][board.popLsb(checks)];
    }

    int fromSq = 0;
    while(queens){
        fromSq = board.popLsb(queens);

        BitBoard moves = (board.getBishopMagics(fromSq) | board.getRookMagics(fromSq));

        if ((checkers | inBetweenKChecker) > 0) {
            moves &= (inBetweenKChecker | checkers);
        }


        // We are pinned
        if ((pinned & board.sqBB[fromSq]) > 0) {
            BitBoard sniperCopy = snipers;
            while (sniperCopy) {
                int sniperSquare = board.popLsb(sniperCopy);
                BitBoard inBetween = board.sqBetween[kingSquare][sniperSquare] & board.sqBB[fromSq];
                if (inBetween > 0) {
                    inBetween = board.sqBetween[kingSquare][sniperSquare] | board.sqBB[sniperSquare];
                    moves &= inBetween;
                }
            }
        }


        BitBoard captures = moves & enemyBoard;
        BitBoard silentMoves = moves & emptySquares;

        int toSq = 0;

        while (silentMoves) {
            toSq = board.popLsb(silentMoves);
            moveList.moves[moveList.counter++] = { fromSq,toSq, false,BitBoardEnum::All,false, false,false,movedPiece };
        }

        while (captures) {
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
    while(kingMove){
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

