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
        generateKnightMoves(board, moveList, checkers, kingSquare, pinned, snipers);
        generateRookMoves(board, moveList, checkers, kingSquare,pinned,snipers);
        generateBishopMoves(board, moveList, checkers, kingSquare,pinned,snipers);
        generateQueenMoves(board, moveList, checkers, kingSquare, pinned,snipers);
        //Pawns last, to prevent promotions to move twice
        generatePawnMoves(board, moveList,checkers,kingSquare,pinned,snipers);
    }
    generateKingMoves(board, moveList, checkers, kingSquare, pinned, snipers);
    moveList.checkers = checkers;
    board.setLegalMovesForSideToMove(moveList.counter);
}

void MoveGenerator::generatePawnMoves(Board& board, MoveList& moveList, BitBoard checkers, int kingSquare, BitBoard pinned, BitBoard snipers)
{
    BitBoard allPieces = board.getBitboard(BitBoardEnum::All);
    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(BitBoardEnum::P + board.getSideToMove());
    BitBoard pawns = board.getBitboard(movedPiece);
    BitBoardEnum sideToMove = board.getSideToMove();
    BitBoard enemyBoard = board.getEnemyBoard();

    BitBoard pinnedPawns = pawns & pinned;
    pawns &= ~pinnedPawns;


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

     
    //Checking pinned pieces individually
    BitBoard pinnedPawnSinglePush = 0;
    BitBoard pinnedDoublePush = 0;
    BitBoard pinnedNEAttack = 0;
    BitBoard pinnedNWAttack = 0;
    int pinnedSquare = 0;

    while (pinnedPawns) {
        pinnedSquare = board.popLsb(pinnedPawns);
        BitBoard pinnedPawnBB = board.sqBB[pinnedSquare];

        if (board.getSideToMove() == White) {
            pinnedPawnSinglePush = (pinnedPawnBB << 8) & ~allPieces;
            pinnedDoublePush = ((pinnedPawnSinglePush & doublePushRank) << 8) & ~allPieces;
            pinnedNEAttack = ((pinnedPawnBB & ~board.FileHMask) << 7) & enemyBoard;
            pinnedNWAttack = ((pinnedPawnBB & ~board.FileAMask) << 9) & enemyBoard;
        }
        else {
            pinnedPawnSinglePush = (pinnedPawnBB >> 8) & ~allPieces;
            pinnedDoublePush = ((pinnedPawnSinglePush & doublePushRank) >> 8) & ~allPieces;
            pinnedNEAttack = ((pinnedPawnBB & ~board.FileAMask) >> 7) & enemyBoard;
            pinnedNWAttack = ((pinnedPawnBB & ~board.FileHMask) >> 9) & enemyBoard;
        }
        singlePush |= makeLegalMoves(board, pinnedPawnSinglePush, pinned, checkers, snipers, pinnedSquare, kingSquare);
        doublePush |= makeLegalMoves(board, pinnedDoublePush, pinned, checkers, snipers, pinnedSquare, kingSquare);
        neAttacks |= makeLegalMoves(board, pinnedNEAttack, pinned, checkers, snipers, pinnedSquare, kingSquare);
        nwAttacks |= makeLegalMoves(board, pinnedNWAttack, pinned, checkers, snipers, pinnedSquare, kingSquare);
    }

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
        moveList.moves[moveList.counter++] = Move::make<NORMAL>(square - pawnIncrement,square);
    }

    while (doublePush) {
        square = board.popLsb(doublePush);
        moveList.moves[moveList.counter++] = Move::make<NORMAL>(square - pawnDoubleIncrement, square);
    }

    while (promotions) {
        square = board.popLsb(promotions);
        moveList.moves[moveList.counter++] = Move::make<PROMOTION>(square - pawnIncrement, square, Q);
        moveList.moves[moveList.counter++] = Move::make<PROMOTION>(square - pawnIncrement, square, B);
        moveList.moves[moveList.counter++] = Move::make<PROMOTION>(square - pawnIncrement, square, R);
        moveList.moves[moveList.counter++] = Move::make<PROMOTION>(square - pawnIncrement, square, N);
    }

    while (nwAttacks) {
        square = board.popLsb(nwAttacks);
        moveList.moves[moveList.counter++] = Move::make<NORMAL>(square - pawnCaptureRightIncrement, square);
    };

    while (neAttacks) {
        square = board.popLsb(neAttacks);
        moveList.moves[moveList.counter++] = Move::make<NORMAL>(square - pawnCaptureLeftIncrement, square);
    }

    while (promoNEAttacks) {
        square = board.popLsb(promoNEAttacks);
        moveList.moves[moveList.counter++] = Move::make<PROMOTION>(square - pawnCaptureLeftIncrement, square, Q);
        moveList.moves[moveList.counter++] = Move::make<PROMOTION>(square - pawnCaptureLeftIncrement, square, B);
        moveList.moves[moveList.counter++] = Move::make<PROMOTION>(square - pawnCaptureLeftIncrement, square, R);
        moveList.moves[moveList.counter++] = Move::make<PROMOTION>(square - pawnCaptureLeftIncrement, square, N);
    }

    while (promoNWAttacks) {
        square = board.popLsb(promoNWAttacks);
        moveList.moves[moveList.counter++] = Move::make<PROMOTION>(square - pawnCaptureRightIncrement, square, Q);
        moveList.moves[moveList.counter++] = Move::make<PROMOTION>(square - pawnCaptureRightIncrement, square, B);
        moveList.moves[moveList.counter++] = Move::make<PROMOTION>(square - pawnCaptureRightIncrement, square, R);
        moveList.moves[moveList.counter++] = Move::make<PROMOTION>(square - pawnCaptureRightIncrement, square, N);
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

                //Remove both pawns from old pos, then add in new pos
                all &= ~toBeRemoved;
                all |= board.sqBB[toSq];
                
                
                uint64_t magic = ((all & board.rookMask[kingSquare]) * board.magicNumberRook[kingSquare]) >> board.magicNumberShiftsRook[kingSquare];
                checkers |= (*board.magicMovesRook)[kingSquare][magic] & (board.getBitboard(Q + board.getOtherSide()) | board.getBitboard(R + board.getOtherSide()));
                magic = ((all & board.bishopMask[kingSquare]) * board.magicNumberBishop[kingSquare]) >> board.magicNumberShiftsBishop[kingSquare];
                checkers |= (*board.magicMovesBishop)[kingSquare][magic] & (board.getBitboard(Q + board.getOtherSide()) | board.getBitboard(B + board.getOtherSide()));

                if(checkers == 0){
                    moveList.moves[moveList.counter++] = Move::make<MoveType::EN_PASSANT>(fromSq, toSq);
                }
            }

        }
    }
}

BitBoard MoveGenerator::pawnAttacks(Board &board, BitBoardEnum color) {
    BitBoard attacks = 0;
    BitBoard pawns = board.getBitboard(P + color);

    if (color == White) {
        attacks |= ((pawns & ~board.FileHMask) << 7);
        attacks |= ((pawns & ~board.FileAMask) << 9);
    }
    else {
        attacks |= ((pawns & ~board.FileAMask) >> 7);
        attacks |= ((pawns & ~board.FileHMask) >> 9);
    }
    return attacks;
}


void MoveGenerator::generateKnightMoves(Board &board, MoveList &moveList, BitBoard checkers, int kingSquare, BitBoard pinned, BitBoard snipers)
{
    BitBoard emptySquares = ~board.getBitboard(BitBoardEnum::All);
    BitBoard allPieces = board.getBitboard(BitBoardEnum::All);
    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(BitBoardEnum::N + board.getSideToMove());
    BitBoard enemyBoard = board.getEnemyBoard();
    BitBoard knights = board.getBitboard(movedPiece) & ~pinned;


    BitBoard inBetween = 0;
    BitBoard checks = checkers;

    while (checks) {
        inBetween |= board.sqBetween[kingSquare][board.popLsb(checks)];
    }

    
    int fromSq = 0;
    while (knights)
    {
        fromSq = board.popLsb(knights);
        BitBoard moves = board.getKnightMask(fromSq);

        if ((checkers | inBetween) > 0) {
            moves &= (inBetween | checkers);
        }

        BitBoard captures = moves & enemyBoard;
        BitBoard silentMoves = moves & emptySquares;

        int toSq = 0;
        while (silentMoves) {
            toSq = board.popLsb(silentMoves);
            moveList.moves[moveList.counter++] = Move::make<NORMAL>(fromSq, toSq);
        }

        while (captures != 0) {
            toSq = board.popLsb(captures);
            moveList.moves[moveList.counter++] = Move::make<NORMAL>(fromSq, toSq);
        }
    }
}

void MoveGenerator::generateRookMoves(Board &board, MoveList &moveList, BitBoard checkers, int kingSquare, BitBoard pinned, BitBoard snipers)
{
    BitBoard emptySquares = ~board.getBitboard(BitBoardEnum::All);
    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(BitBoardEnum::R + board.getSideToMove());
    BitBoard enemyBoard = board.getEnemyBoard();
    BitBoard rooks = board.getBitboard(movedPiece);
    


    int fromSq = 0;
    while(rooks){
        fromSq = board.popLsb(rooks);

        
        BitBoard moves = board.getRookMagics(fromSq);

        moves = makeLegalMoves(board, moves, pinned, checkers, snipers, fromSq, kingSquare);
        BitBoard captures = moves & enemyBoard;
        BitBoard silentMoves = moves & emptySquares;

        int toSq = 0;
        while (silentMoves) {
            toSq = board.popLsb(silentMoves);
            moveList.moves[moveList.counter++] = Move::make<NORMAL>(fromSq, toSq);
        }

        while (captures != 0) {
            toSq = board.popLsb(captures);
            moveList.moves[moveList.counter++] = Move::make<NORMAL>(fromSq, toSq);
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
            moveList.moves[moveList.counter++] = Move::make<NORMAL>(fromSq, toSq);
        }

        while (captures!= 0) {
            toSq = board.popLsb(captures);
            moveList.moves[moveList.counter++] = Move::make<NORMAL>(fromSq, toSq);
        }
        
    }
}

void MoveGenerator::generateQueenMoves(Board &board, MoveList &moveList, BitBoard checkers, int kingSquare, BitBoard pinned, BitBoard snipers)
{
    BitBoard emptySquares = ~board.getBitboard(BitBoardEnum::All);
    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(BitBoardEnum::Q + board.getSideToMove());
    BitBoard enemyBoard = board.getEnemyBoard();
    BitBoard queens = board.getBitboard(movedPiece);

    int fromSq = 0;
    while(queens){
        fromSq = board.popLsb(queens);

        BitBoard moves = (board.getBishopMagics(fromSq) | board.getRookMagics(fromSq));

        moves = makeLegalMoves(board, moves, pinned, checkers, snipers, fromSq, kingSquare);


        BitBoard captures = moves & enemyBoard;
        BitBoard silentMoves = moves & emptySquares;

        int toSq = 0;

        while (silentMoves) {
            toSq = board.popLsb(silentMoves);
            moveList.moves[moveList.counter++] = Move::make<NORMAL>(fromSq, toSq);
        }

        while (captures) {
            toSq = board.popLsb(captures);
            moveList.moves[moveList.counter++] = Move::make<NORMAL>(fromSq, toSq);
        }
        
    }
}

void MoveGenerator::generateKingMoves(Board &board, MoveList &moveList, BitBoard checkers, int kingSquare, BitBoard pinned, BitBoard snipers)
{
    BitBoard emptySquares = ~board.getBitboard(BitBoardEnum::All);
    BitBoard allPieces = board.getBitboard(BitBoardEnum::All);
    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(BitBoardEnum::K + board.getSideToMove());
    BitBoard enemyBoard = board.getEnemyBoard();
    BitBoard king = board.getBitboard(movedPiece);
    BitBoard otherKingBoard = board.getBitboard(K + board.getOtherSide());
    int otherKingSq = board.popLsb(otherKingBoard);
    BitBoardEnum sideToMove = board.getSideToMove();

    int fromSq = board.popLsb(king);
    BitBoard moves = board.getKingMask(fromSq);


    
    //Here we can remove at least knight moves
    BitBoard enemyKnights = board.getBitboard(static_cast<BitBoardEnum>(BitBoardEnum::N + board.getOtherSide()));
    int knightSquare = 0;
    BitBoard enemyKnightAttacks = 0;
    while (enemyKnights != 0) {
        knightSquare = board.popLsb(enemyKnights);
        enemyKnightAttacks |= board.getKnightMask(knightSquare);
    }

    moves &= ~enemyKnightAttacks;
    moves &= ~board.getKingMask(otherKingSq);
    
    BitBoard all = board.getBitboard(All) & ~board.getBitboard(K + board.getSideToMove());

    BitBoard attacks = 0;
    BitBoard enemyRooks = board.getBitboard(Q + board.getOtherSide()) | board.getBitboard(R + board.getOtherSide());
    BitBoard enemyBishops = board.getBitboard(Q + board.getOtherSide()) | board.getBitboard(B + board.getOtherSide());

    while (enemyRooks) {
        int square = board.popLsb(enemyRooks);
        uint64_t magic = ((all & board.rookMask[square]) * board.magicNumberRook[square]) >> board.magicNumberShiftsRook[square];
        attacks |= (*board.magicMovesRook)[square][magic];
    }

    while (enemyBishops) {
        int square = board.popLsb(enemyBishops);
        uint64_t magic = ((all & board.bishopMask[square]) * board.magicNumberBishop[square]) >> board.magicNumberShiftsBishop[square];
        attacks |= (*board.magicMovesBishop)[square][magic];
    }
    
    attacks |= pawnAttacks(board, board.getOtherSide());

    moves &= ~attacks;

    BitBoard captures = moves & enemyBoard;
    BitBoard silentMoves = moves & emptySquares;

    int toSq = 0;
    while (silentMoves) {
        toSq = board.popLsb(silentMoves);
        moveList.moves[moveList.counter++] = Move::make<NORMAL>(fromSq, toSq);
    }

    while (captures != 0) {
        toSq = board.popLsb(captures);
        moveList.moves[moveList.counter++] = Move::make<NORMAL>(fromSq, toSq);
    }

    if (sideToMove == BitBoardEnum::White) {
        if (board.getCastleRightsWK()) {
            BitBoard castlineSquares = 0;
            board.setBit(castlineSquares, 5);
            board.setBit(castlineSquares, 6);
            if ((allPieces & castlineSquares) == 0 && !board.isSquareAttacked(castlineSquares | board.sqBB[fromSq], BitBoardEnum::Black)) {
                moveList.moves[moveList.counter++] = Move::make<CASTLING>(fromSq, fromSq+2);
            }  //f1,g1;
        }
        if (board.getCastleRightsWQ()) {
            BitBoard checkSquaresWQ = 0;
            board.setBit(checkSquaresWQ, 2);
            board.setBit(checkSquaresWQ, 3);

            BitBoard emptySquaresWQ = checkSquaresWQ;
            board.setBit(emptySquaresWQ, 1);

            if ((allPieces & emptySquaresWQ) == 0 && !board.isSquareAttacked(checkSquaresWQ | board.sqBB[fromSq], BitBoardEnum::Black)) {
                moveList.moves[moveList.counter++] = Move::make<CASTLING>(fromSq, fromSq - 2);
            }  //b1,c1,d1;
        }
    }
    else if (sideToMove == BitBoardEnum::Black) {
        if (board.getCastleRightsBK()) {
            BitBoard castlineSquares = 0;
            board.setBit(castlineSquares, 61);
            board.setBit(castlineSquares, 62);
            if ((allPieces & castlineSquares) == 0 && !board.isSquareAttacked(castlineSquares | board.sqBB[fromSq], BitBoardEnum::White)) {
                moveList.moves[moveList.counter++] = Move::make<CASTLING>(fromSq, fromSq + 2);
            }
        }
        if (board.getCastleRightsBQ()) {
            BitBoard checkSquaresBQ = 0;

            board.setBit(checkSquaresBQ, 58);
            board.setBit(checkSquaresBQ, 59);
            BitBoard emptySquaresBQ = checkSquaresBQ;
            board.setBit(emptySquaresBQ, 57);
            if ((allPieces & emptySquaresBQ) == 0 && !board.isSquareAttacked(checkSquaresBQ | board.sqBB[fromSq], BitBoardEnum::White)) {
                moveList.moves[moveList.counter++] = Move::make<CASTLING>(fromSq, fromSq - 2);
            }
        }
    }
    
}

