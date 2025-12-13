#include "movegenerator.h"
#include <iostream>
#include "material.h"
#include <algorithm>
#include "see.h"

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

void MoveGenerator::init(Board& b, Move tt, bool onlyCaptures, Move killer[], HistoryTables* hist) {
    board = &b;
    onlyNoisy = onlyCaptures;
    killerMove = killer;
    histTable = hist;
    ttMove = tt;
    BitBoard king = board->getBitboard(K + board->getSideToMove());
    kingSquare = Board::popLsb(king);
    king = board->getBitboard(K + board->getSideToMove());
    snipers = board->getSnipers(kingSquare, board->getOtherSide());
    BitBoard sniperCopy = snipers;

    pinned = 0;
    BitBoard potentialPinned = 0;


    while (sniperCopy) {
        int sniperSquare = Board::popLsb(sniperCopy);
        potentialPinned = board->sqBetween[kingSquare][sniperSquare] & board->getBitboard(All);
        if (board->countSetBits(potentialPinned) == 1) {
            pinned |= potentialPinned & board->getBitboard(board->getSideToMove());
        }
    }

    //  Finding pieces giving check to the current side to move
    checkers = 0;

    uint64_t magic = ((board->getBitboard(All) & board->rookMask[kingSquare]) * board->magicNumberRook[kingSquare]) >> board->magicNumberShiftsRook[kingSquare];
    checkers |= (*board->magicMovesRook)[kingSquare][magic] & (board->getBitboard(Q + board->getOtherSide()) | board->getBitboard(R + board->getOtherSide()));
    magic = ((board->getBitboard(All) & board->bishopMask[kingSquare]) * board->magicNumberBishop[kingSquare]) >> board->magicNumberShiftsBishop[kingSquare];
    checkers |= (*board->magicMovesBishop)[kingSquare][magic] & (board->getBitboard(Q + board->getOtherSide()) | board->getBitboard(B + board->getOtherSide()));

    checkers |= board->getKnightMask(kingSquare) & board->getBitboard(N + board->getOtherSide());


    if (board->getSideToMove() == White) {

        checkers |= ((king & ~board->FileHMask) << 7) & board->getBitboard(P + board->getOtherSide());
        checkers |= ((king & ~board->FileAMask) << 9) & board->getBitboard(P + board->getOtherSide());
    }
    else {
        checkers |= ((king & ~board->FileAMask) >> 7) & board->getBitboard(P + board->getOtherSide());
        checkers |= ((king & ~board->FileHMask) >> 9) & board->getBitboard(P + board->getOtherSide());
    }
    
}

bool MoveGenerator::isMoveLegal(Move move) {
    BitBoardEnum piece = board->getPieceOnSquare(move.from());
    BitBoard pieceBoard = board->getBitboard(piece);
    BitBoard fromBB = board->sqBB[move.from()];
    BitBoard toBB = board->sqBB[move.to()];
    BitBoard emptySquares = ~board->getBitboard(BitBoardEnum::All);
    BitBoard ownBoard = board->getBitboard(board->getSideToMove());
    BitBoard enemyBoard = board->getBitboard(board->getOtherSide());
    bool isCapture = (enemyBoard & toBB) > 0;

    BitBoardEnum normPiece =static_cast<BitBoardEnum>( piece - board->getSideToMove());
    MoveList list;
    switch (normPiece) {
    case P:
        generatePawnMoves(*board, list, checkers, kingSquare, pinned, snipers);
        for (int i = 0; i < list.counter; i++) {
            if (list.moves[i].value == ttMove.value) return true;
        }
        break;
    case N:
        generateKnightMoves(*board, list, checkers, kingSquare, pinned, snipers);
        for (int i = 0; i < list.counter; i++) {
            if (list.moves[i].value == ttMove.value) return true;
        }
        break;
    case B:      
        if (isMoveLegalSliders(move, isCapture, board->getBishopMagics(move.from()), pieceBoard, enemyBoard, emptySquares)) return true;
        break;
    case R:
        if (isMoveLegalSliders(move, isCapture, board->getRookMagics(move.from()), pieceBoard, enemyBoard, emptySquares)) return true;
        break;
    case Q:
        if (isMoveLegalSliders(move, isCapture, board->getBishopMagics(move.from()) | board->getRookMagics(move.from()), pieceBoard, enemyBoard, emptySquares)) return true;
        break;
    case K:
        generateKingMoves(*board, list, checkers, kingSquare, pinned, snipers);
        for (int i = 0; i < list.counter; i++) {
            if (list.moves[i].value == ttMove.value) return true;
        }
        break;
    default:
        return false;
    }
        
    return false;

}

bool MoveGenerator::isMoveLegalSliders(Move move, bool isCapture, BitBoard moves, BitBoard pieceBoard, BitBoard enemyBoard, BitBoard emptySquares) {
    int fromSq = 0;
    BitBoard toBB = board->sqBB[move.to()];
    
    while (pieceBoard) {
        fromSq = Board::popLsb(pieceBoard);
        moves = makeLegalMoves(*board, moves, pinned, checkers, snipers, fromSq, kingSquare);
        int toSq = 0;

        if (isCapture) {
            BitBoard captures = moves & enemyBoard;                  
            if ((captures & toBB)> 0) return true;
        }
        else {
            BitBoard silentMoves = moves & emptySquares;
            while (silentMoves) {
                toSq = Board::popLsb(silentMoves);
                if (move.from() == fromSq && move.to() == toSq) return true;
            }
        }

    }
    return false;
}

Move MoveGenerator::next() {

  

    switch (currentStage) {

        case TT_MOVE:
            currentStage = GEN_NOISY;
            
            if (ttMove) {
                if (isMoveLegal(ttMove)) {
                    return ttMove;
                }
            }
            
            [[fallthrough]];
        case GEN_NOISY:
            currentStage = GOOD_NOISY;
            if (board->countSetBits(checkers) < 2) {
                generateKnightNoisy();
                generateBishopNoisy();
                generateRookNoisy();
                generateQueenNoisy();
                generatePawnNoisy();
            }
            generateKingNoisy();

            sortNoisyMoves();
            [[fallthrough]];
        case GOOD_NOISY:
            
            //Return good noisy
            if (noisyIdx < noisyMoves.size() && noisyMoves[noisyIdx].score > 0) {
                return noisyMoves[noisyIdx++].move;
            } 

            // if no moves left, next stage
            currentStage = KILLER;
            [[fallthrough]];
        case KILLER:
            currentStage = GEN_QUIET;
            //Check if killer moves are valid
            //If no killer, next stage

            [[fallthrough]];
        case GEN_QUIET:
            currentStage = QUIET;
            //Generate quite
            //Score
            if(!onlyNoisy || checkers != 0){

                if (board->countSetBits(checkers) < 2) {
                    generateKnightQuiet();
                    generateBishopQuiet();
                    generateRookQuiet();
                    generateQueenQuiet();
                    generatePawnQuiet();
                }

                generateKingQuiet();
                scoreQuietMoves();
            }
            
            
            //Remove killer moves
            [[fallthrough]];
        case QUIET:
            //Return quiet
            // if no more moves left, next stage
            if (quietIdx < quietMoves.size()) {
                return quietMoves[quietIdx++].move;
            }

            currentStage = BAD_NOISY;
            [[fallthrough]];

        case BAD_NOISY:
            //Return bad noisy
            // if no more moves left, next stage
            if (noisyIdx < noisyMoves.size()) {
                return noisyMoves[noisyIdx++].move;
            }
            currentStage = END;
            [[fallthrough]];
        case END:
            return Move();

    };

}

struct SortStruct {
    int score;
    Move move;
};

void MoveGenerator::sortNoisyMoves() {
    int side = 0;
    if (board->getSideToMove() == Black) {
        side = 1;
    }

    SortStruct sortArray[256];
    for (int i = 0; i < noisyMoves.size(); i++) {
        SortStruct entry{};

        entry.move = noisyMoves[i].move;
        
        if (entry.move.value == ttMove.value) {
            noisyMoves.erase(noisyMoves.begin() + i);
            i--;
            continue;
        }
        


        if (entry.move.getMoveType() == PROMOTION) {
            //TODO: Promotion capture
            entry.score = 80000;
        }
        else if (board->getPieceOnSquare(entry.move.to()) != All) {

            BitBoardEnum capturedPiece = board->getPieceOnSquare(entry.move.to());
            BitBoardEnum attacker = board->getPieceOnSquare(entry.move.from());

            if (entry.move.getMoveType() == EN_PASSANT) {
                capturedPiece = P;
                entry.score = 70000;
            }
            else {
                int Mvv = Material::pieceMaterialScoreArray[capturedPiece];
                int lva = Material::pieceMaterialScoreArray[attacker];

                int mvvlva = (Mvv - lva) / 100;
                entry.score = 70000 + mvvlva;

                if (Mvv > lva) {
                    entry.score = 70000 + mvvlva;
                }
                else {
                    using namespace See;
                    int seeScore = see(*board, entry.move.from(), entry.move.to(), board->getSideToMove());
                    if (seeScore >= 0) {
                        entry.score = 70000 + seeScore;
                    }
                    else {
                        entry.score = -70000 - seeScore;
                    }
                    
                }
            }
        }
        sortArray[i] = entry;
    }

    std::stable_sort(sortArray, sortArray + noisyMoves.size(),
        [](const SortStruct& a, const SortStruct& b) {
            if (a.score != b.score) return a.score > b.score;     // strict order
            return a.move.value < b.move.value;                   // total tie-break
        });

    for (int i = 0; i < noisyMoves.size(); i++) {
        noisyMoves[i].move = sortArray[i].move;
        noisyMoves[i].score = sortArray[i].score;
    }
}

void MoveGenerator::scoreQuietMoves() {

    int side = 0;
    if (board->getSideToMove() == Black) {
        side = 1;
    }


    SortStruct sortArray[256];
    for (int i = 0; i < quietMoves.size(); i++) {
        SortStruct entry{};
        entry.move = quietMoves[i].move;
        
        if (entry.move.value == ttMove.value){ 
            quietMoves.erase(quietMoves.begin() + i);
            i--;
            continue;
        }
        
        
        if (killerMove[0].value == entry.move.value ||
            killerMove[1].value == entry.move.value) {
            entry.score = 60000;
        }
        else if (histTable->quiet[side][entry.move.from()][entry.move.to()] != 0) {
            entry.score = 30000 + histTable->quiet[side][entry.move.from()][entry.move.to()];
        }
        else {
            //quiet move
            entry.score = 0;

        }
        sortArray[i] = entry;

        
    }

    std::stable_sort(sortArray, sortArray + quietMoves.size(),
        [](const SortStruct& a, const SortStruct& b) {
            if (a.score != b.score) return a.score > b.score;     // strict order
            return a.move.value < b.move.value;                   // total tie-break
        });

    for (int i = 0; i < quietMoves.size(); i++) {
        quietMoves[i].move = sortArray[i].move;
        quietMoves[i].score = sortArray[i].score;
    }
}

void MoveGenerator::generatePawnNoisy() {
    BitBoard allPieces = board->getBitboard(BitBoardEnum::All);
    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(BitBoardEnum::P + board->getSideToMove());
    BitBoard pawns = board->getBitboard(movedPiece);
    BitBoardEnum sideToMove = board->getSideToMove();
    BitBoard enemyBoard = board->getEnemyBoard();

    BitBoard pinnedPawns = pawns & pinned;
    pawns &= ~pinnedPawns;


    int pawnIncrement = 8;
    int pawnDoubleIncrement = 16;
    int pawnCaptureLeftIncrement = 7;
    int pawnCaptureRightIncrement = 9;
    int enPassantIncrement = -8;
    BitBoard doublePushRank = Board::Rank3Mask;
    BitBoard promotionRank = Board::Rank8Mask;

    if (sideToMove == BitBoardEnum::Black) {
        pawnDoubleIncrement = -16;
        pawnIncrement = -8;
        pawnCaptureLeftIncrement = -7;
        pawnCaptureRightIncrement = -9;
        doublePushRank = Board::Rank6Mask;
        promotionRank = Board::Rank1Mask;
    }

    BitBoardEnum queenPromo = static_cast<BitBoardEnum>(BitBoardEnum::Q + sideToMove);
    BitBoardEnum bishopPromo = static_cast<BitBoardEnum>(BitBoardEnum::B + sideToMove);
    BitBoardEnum knightPromo = static_cast<BitBoardEnum>(BitBoardEnum::N + sideToMove);
    BitBoardEnum rookPromo = static_cast<BitBoardEnum>(BitBoardEnum::R + sideToMove);

    BitBoard singlePush = 0;
    BitBoard doublePush = 0;
    BitBoard promotions = 0;
    BitBoard nwAttacks = 0;
    BitBoard neAttacks = 0;
    BitBoard promoNWAttacks = 0;
    BitBoard promoNEAttacks = 0;



    if (sideToMove == White) {
        singlePush = (pawns << 8) & ~allPieces;
        doublePush = ((singlePush & doublePushRank) << 8) & ~allPieces;
        neAttacks = ((pawns & ~Board::FileHMask) << 7) & enemyBoard;
        nwAttacks = ((pawns & ~Board::FileAMask) << 9) & enemyBoard;
    }
    else {
        singlePush = (pawns >> 8) & ~allPieces;
        doublePush = ((singlePush & doublePushRank) >> 8) & ~allPieces;
        neAttacks = ((pawns & ~Board::FileAMask) >> 7) & enemyBoard;
        nwAttacks = ((pawns & ~Board::FileHMask) >> 9) & enemyBoard;
    }


    //Checking pinned pieces individually
    BitBoard pinnedPawnSinglePush = 0;
    BitBoard pinnedDoublePush = 0;
    BitBoard pinnedNEAttack = 0;
    BitBoard pinnedNWAttack = 0;
    int pinnedSquare = 0;

    while (pinnedPawns) {
        pinnedSquare = Board::popLsb(pinnedPawns);
        BitBoard pinnedPawnBB = Board::sqBB[pinnedSquare];

        if (sideToMove == White) {
            pinnedPawnSinglePush = (pinnedPawnBB << 8) & ~allPieces;            
            pinnedNEAttack = ((pinnedPawnBB & ~Board::FileHMask) << 7) & enemyBoard;
            pinnedNWAttack = ((pinnedPawnBB & ~Board::FileAMask) << 9) & enemyBoard;
        }
        else {
            pinnedPawnSinglePush = (pinnedPawnBB >> 8) & ~allPieces;
            pinnedNEAttack = ((pinnedPawnBB & ~Board::FileAMask) >> 7) & enemyBoard;
            pinnedNWAttack = ((pinnedPawnBB & ~Board::FileHMask) >> 9) & enemyBoard;
        }
        singlePush |= makeLegalMoves(*board, pinnedPawnSinglePush, pinned, checkers, snipers, pinnedSquare, kingSquare);
        doublePush |= makeLegalMoves(*board, pinnedDoublePush, pinned, checkers, snipers, pinnedSquare, kingSquare);
        neAttacks |= makeLegalMoves(*board, pinnedNEAttack, pinned, checkers, snipers, pinnedSquare, kingSquare);
        nwAttacks |= makeLegalMoves(*board, pinnedNWAttack, pinned, checkers, snipers, pinnedSquare, kingSquare);
    }

    //If in check, only consider moves that capture checker or obstruct the check
    BitBoard inBetween = 0;
    BitBoard checks = checkers;
    while (checks) {
        inBetween |= board->sqBetween[kingSquare][Board::popLsb(checks)];
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

    int square = 0;

    while (promotions) {
        square = Board::popLsb(promotions);
        noisyMoves.push_back({ 0, Move::make<PROMOTION>(square - pawnIncrement, square, Q) });
        noisyMoves.push_back({ 0, Move::make<PROMOTION>(square - pawnIncrement, square, B) });
        noisyMoves.push_back({ 0, Move::make<PROMOTION>(square - pawnIncrement, square, R) });
        noisyMoves.push_back({ 0, Move::make<PROMOTION>(square - pawnIncrement, square, N) });
    }

    while (nwAttacks) {
        square = Board::popLsb(nwAttacks);
        noisyMoves.push_back({ 0, Move::make<NORMAL>(square - pawnCaptureRightIncrement, square) });
    };

    while (neAttacks) {
        square = Board::popLsb(neAttacks);
        noisyMoves.push_back({ 0, Move::make<NORMAL>(square - pawnCaptureLeftIncrement, square) });
    }

    while (promoNEAttacks) {
        square = Board::popLsb(promoNEAttacks);
        noisyMoves.push_back({ 0, Move::make<PROMOTION>(square - pawnCaptureLeftIncrement, square, Q) });
        noisyMoves.push_back({ 0, Move::make<PROMOTION>(square - pawnCaptureLeftIncrement, square, B) });
        noisyMoves.push_back({ 0, Move::make<PROMOTION>(square - pawnCaptureLeftIncrement, square, R) });
        noisyMoves.push_back({ 0, Move::make<PROMOTION>(square - pawnCaptureLeftIncrement, square, N) });
    }

    while (promoNWAttacks) {
        square = Board::popLsb(promoNWAttacks);
        noisyMoves.push_back({ 0, Move::make<PROMOTION>(square - pawnCaptureRightIncrement, square, Q) });
        noisyMoves.push_back({ 0, Move::make<PROMOTION>(square - pawnCaptureRightIncrement, square, B) });
        noisyMoves.push_back({ 0, Move::make<PROMOTION>(square - pawnCaptureRightIncrement, square, R) });
        noisyMoves.push_back({ 0, Move::make<PROMOTION>(square - pawnCaptureRightIncrement, square, N) });
    }

    if (board->getEnPassantSq() != Board::noSq) {

        while (pawns)
        {
            int fromSq = Board::popLsb(pawns);
            int toSq = fromSq + pawnIncrement;
            BitBoard fromSqBoard = Board::sqBB[fromSq];
            BitBoard attack = 0;

            if (sideToMove == BitBoardEnum::White) {
                attack = Board::sqBB[board->getEnPassantSq()] & (board->southEastOne(fromSqBoard) | board->southWestOne(fromSqBoard));
            }
            else {
                attack = Board::sqBB[board->getEnPassantSq()] & (board->northEastOne(fromSqBoard) | board->northWestOne(fromSqBoard));
            }

            while (attack != 0) {

                toSq = Board::popLsb(attack);
                BitBoard checkers = 0;
                BitBoard all = board->getBitboard(All);
                //Removing current attacking pawn and enpassant pawn from board to perform check check
                BitBoard toBeRemoved = (Board::sqBB[fromSq] | Board::sqBB[board->getEnPassantSq() - pawnIncrement]);

                //Remove both pawns from old pos, then add in new pos
                all &= ~toBeRemoved;
                all |= Board::sqBB[toSq];


                uint64_t magic = ((all & board->rookMask[kingSquare]) * board->magicNumberRook[kingSquare]) >> board->magicNumberShiftsRook[kingSquare];
                checkers |= (*board->magicMovesRook)[kingSquare][magic] & (board->getBitboard(Q + board->getOtherSide()) | board->getBitboard(R + board->getOtherSide()));
                magic = ((all & board->bishopMask[kingSquare]) * board->magicNumberBishop[kingSquare]) >> board->magicNumberShiftsBishop[kingSquare];
                checkers |= (*board->magicMovesBishop)[kingSquare][magic] & (board->getBitboard(Q + board->getOtherSide()) | board->getBitboard(B + board->getOtherSide()));

                if (checkers == 0) {
                    noisyMoves.push_back({ 0, Move::make<MoveType::EN_PASSANT>(fromSq, toSq) });
                }
            }

        }
    }
}

void MoveGenerator::generatePawnQuiet() {
    BitBoard allPieces = board->getBitboard(BitBoardEnum::All);
    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(BitBoardEnum::P + board->getSideToMove());
    BitBoard pawns = board->getBitboard(movedPiece);
    BitBoardEnum sideToMove = board->getSideToMove();
    BitBoard enemyBoard = board->getEnemyBoard();

    BitBoard pinnedPawns = pawns & pinned;
    pawns &= ~pinnedPawns;


    int pawnIncrement = 8;
    int pawnDoubleIncrement = 16;
    BitBoard doublePushRank = Board::Rank3Mask;
    BitBoard promotionRank = Board::Rank8Mask;

    if (sideToMove == BitBoardEnum::Black) {
        pawnDoubleIncrement = -16;
        pawnIncrement = -8;
        doublePushRank = Board::Rank6Mask;
        promotionRank = Board::Rank1Mask;
    }

    BitBoard singlePush = 0;
    BitBoard doublePush = 0;
    BitBoard promotions = 0;




    if (sideToMove == White) {
        singlePush = (pawns << 8) & ~allPieces;
        doublePush = ((singlePush & doublePushRank) << 8) & ~allPieces;
    }
    else {
        singlePush = (pawns >> 8) & ~allPieces;
        doublePush = ((singlePush & doublePushRank) >> 8) & ~allPieces;
    }


    //Checking pinned pieces individually
    BitBoard pinnedPawnSinglePush = 0;
    BitBoard pinnedDoublePush = 0;
    int pinnedSquare = 0;

    while (pinnedPawns) {
        pinnedSquare = Board::popLsb(pinnedPawns);
        BitBoard pinnedPawnBB = Board::sqBB[pinnedSquare];

        if (board->getSideToMove() == White) {
            pinnedPawnSinglePush = (pinnedPawnBB << 8) & ~allPieces;
            pinnedDoublePush = ((pinnedPawnSinglePush & doublePushRank) << 8) & ~allPieces;
        }
        else {
            pinnedPawnSinglePush = (pinnedPawnBB >> 8) & ~allPieces;
            pinnedDoublePush = ((pinnedPawnSinglePush & doublePushRank) >> 8) & ~allPieces;
            
        }
        singlePush |= makeLegalMoves(*board, pinnedPawnSinglePush, pinned, checkers, snipers, pinnedSquare, kingSquare);
        doublePush |= makeLegalMoves(*board, pinnedDoublePush, pinned, checkers, snipers, pinnedSquare, kingSquare);
        
    }

    //If in check, only consider moves that capture checker or obstruct the check
    BitBoard inBetween = 0;
    BitBoard checks = checkers;
    while (checks) {
        inBetween |= board->sqBetween[kingSquare][Board::popLsb(checks)];
    }
    if ((checkers | inBetween) > 0) {
        singlePush &= (inBetween | checkers);
        doublePush &= (inBetween | checkers);
    }


    promotions = (singlePush & promotionRank);
    singlePush &= ~promotionRank;

    //Single push
    int square = 0;
    while (singlePush) {
        square = board->popLsb(singlePush);
        quietMoves.push_back({ 0, Move::make<NORMAL>(square - pawnIncrement, square)});
    }

    while (doublePush) {
        square = board->popLsb(doublePush);
        quietMoves.push_back({ 0, Move::make<NORMAL>(square - pawnDoubleIncrement, square) });
    }

    
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

void MoveGenerator::generateKnightNoisy() {
    BitBoard emptySquares = ~board->getBitboard(BitBoardEnum::All);
    BitBoard allPieces = board->getBitboard(BitBoardEnum::All);
    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(BitBoardEnum::N + board->getSideToMove());
    BitBoard enemyBoard = board->getEnemyBoard();
    BitBoard knights = board->getBitboard(movedPiece) & ~pinned;


    BitBoard inBetween = 0;
    BitBoard checks = checkers;

    while (checks) {
        inBetween |= board->sqBetween[kingSquare][Board::popLsb(checks)];
    }


    int fromSq = 0;
    while (knights)
    {
        fromSq = Board::popLsb(knights);
        BitBoard moves = board->getKnightMask(fromSq);

        if ((checkers | inBetween) > 0) {
            moves &= (inBetween | checkers);
        }

        BitBoard captures = moves & enemyBoard;
        BitBoard silentMoves = moves & emptySquares;

        int toSq = 0;

        while (captures != 0) {
            toSq = Board::popLsb(captures);
            noisyMoves.push_back({ 0, Move::make<NORMAL>(fromSq, toSq) });
        }
    }
}

void MoveGenerator::generateKnightQuiet() {
    BitBoard emptySquares = ~board->getBitboard(BitBoardEnum::All);
    BitBoard allPieces = board->getBitboard(BitBoardEnum::All);
    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(BitBoardEnum::N + board->getSideToMove());
    BitBoard enemyBoard = board->getEnemyBoard();
    BitBoard knights = board->getBitboard(movedPiece) & ~pinned;


    BitBoard inBetween = 0;
    BitBoard checks = checkers;

    while (checks) {
        inBetween |= board->sqBetween[kingSquare][Board::popLsb(checks)];
    }

    while (checks) {
        inBetween |= board->sqBetween[kingSquare][Board::popLsb(checks)];
    }


    int fromSq = 0;
    while (knights)
    {
        fromSq = Board::popLsb(knights);
        BitBoard moves = board->getKnightMask(fromSq);

        if ((checkers | inBetween) > 0) {
            moves &= (inBetween | checkers);
        }

        BitBoard captures = moves & enemyBoard;
        BitBoard silentMoves = moves & emptySquares;

        int toSq = 0;
        while (silentMoves) {
            toSq = Board::popLsb(silentMoves);
            quietMoves.push_back({ 0, Move::make<NORMAL>(fromSq, toSq) });
        }

    }
}

void MoveGenerator::generateBishopNoisy() {
    BitBoard emptySquares = ~board->getBitboard(BitBoardEnum::All);
    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(BitBoardEnum::B + board->getSideToMove());
    BitBoard enemyBoard = board->getEnemyBoard();
    BitBoard bishops = board->getBitboard(movedPiece);

    int fromSq = 0;
    while (bishops) {
        fromSq = Board::popLsb(bishops);
        BitBoard moves = board->getBishopMagics(fromSq);

        moves = makeLegalMoves(*board, moves, pinned, checkers, snipers, fromSq, kingSquare);

        BitBoard captures = moves & enemyBoard;        

        int toSq = 0;

        while (captures != 0) {
            toSq = Board::popLsb(captures);
            noisyMoves.push_back({ 0, Move::make<NORMAL>(fromSq, toSq) });
        }

    }
}

void MoveGenerator::generateBishopQuiet() {
    BitBoard emptySquares = ~board->getBitboard(BitBoardEnum::All);
    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(BitBoardEnum::B + board->getSideToMove());
    BitBoard enemyBoard = board->getEnemyBoard();
    BitBoard bishops = board->getBitboard(movedPiece);

    int fromSq = 0;
    while (bishops) {
        fromSq = Board::popLsb(bishops);
        BitBoard moves = board->getBishopMagics(fromSq);

        moves = makeLegalMoves(*board, moves, pinned, checkers, snipers, fromSq, kingSquare);

        BitBoard silentMoves = moves & emptySquares;
        int toSq = 0;

        while (silentMoves) {
            toSq = Board::popLsb(silentMoves);
            quietMoves.push_back({ 0, Move::make<NORMAL>(fromSq, toSq) });
        }

    }
}

void MoveGenerator::generateRookNoisy() {
    BitBoard emptySquares = ~board->getBitboard(BitBoardEnum::All);
    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(BitBoardEnum::R + board->getSideToMove());
    BitBoard enemyBoard = board->getEnemyBoard();
    BitBoard rooks = board->getBitboard(movedPiece);

    int fromSq = 0;
    while (rooks) {
        fromSq = Board::popLsb(rooks);
        BitBoard moves = board->getRookMagics(fromSq);
        moves = makeLegalMoves(*board, moves, pinned, checkers, snipers, fromSq, kingSquare);
        BitBoard captures = moves & enemyBoard;

        int toSq = 0;
        while (captures != 0) {
            toSq = Board::popLsb(captures);
            noisyMoves.push_back({0, Move::make<NORMAL>(fromSq, toSq) });
        }
    }
}

void MoveGenerator::generateRookQuiet() {
    BitBoard emptySquares = ~board->getBitboard(BitBoardEnum::All);
    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(BitBoardEnum::R + board->getSideToMove());
    BitBoard enemyBoard = board->getEnemyBoard();
    BitBoard rooks = board->getBitboard(movedPiece);

    int fromSq = 0;
    while (rooks) {
        fromSq = Board::popLsb(rooks);
        BitBoard moves = board->getRookMagics(fromSq);
        moves = makeLegalMoves(*board, moves, pinned, checkers, snipers, fromSq, kingSquare);
        BitBoard silentMoves = moves & emptySquares;

        int toSq = 0;
        while (silentMoves) {
            toSq = Board::popLsb(silentMoves);
            quietMoves.push_back({ 0, Move::make<NORMAL>(fromSq, toSq) });
        }

    }
}

void MoveGenerator::generateQueenNoisy() {
    BitBoard emptySquares = ~board->getBitboard(BitBoardEnum::All);
    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(BitBoardEnum::Q + board->getSideToMove());
    BitBoard enemyBoard = board->getEnemyBoard();
    BitBoard queens = board->getBitboard(movedPiece);

    int fromSq = 0;
    while (queens) {
        fromSq = Board::popLsb(queens);
        BitBoard moves = (board->getBishopMagics(fromSq) | board->getRookMagics(fromSq));
        moves = makeLegalMoves(*board, moves, pinned, checkers, snipers, fromSq, kingSquare);
        BitBoard captures = moves & enemyBoard;
        

        int toSq = 0;
        while (captures) {
            toSq = Board::popLsb(captures);
            noisyMoves.push_back({ 0, Move::make<NORMAL>(fromSq, toSq) });
        }

    }
}

void MoveGenerator::generateQueenQuiet() {
    BitBoard emptySquares = ~board->getBitboard(BitBoardEnum::All);
    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(BitBoardEnum::Q + board->getSideToMove());
    BitBoard enemyBoard = board->getEnemyBoard();
    BitBoard queens = board->getBitboard(movedPiece);

    int fromSq = 0;
    while (queens) {
        fromSq = Board::popLsb(queens);
        BitBoard moves = (board->getBishopMagics(fromSq) | board->getRookMagics(fromSq));
        moves = makeLegalMoves(*board, moves, pinned, checkers, snipers, fromSq, kingSquare);
        BitBoard silentMoves = moves & emptySquares;

        int toSq = 0;
        while (silentMoves) {
            toSq = Board::popLsb(silentMoves);
            quietMoves.push_back({ 0, Move::make<NORMAL>(fromSq, toSq) });
        }

    }
}

void MoveGenerator::generateKingNoisy() {
    BitBoard emptySquares = ~board->getBitboard(BitBoardEnum::All);
    BitBoard allPieces = board->getBitboard(BitBoardEnum::All);
    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(BitBoardEnum::K + board->getSideToMove());
    BitBoard enemyBoard = board->getEnemyBoard();
    BitBoard king = board->getBitboard(movedPiece);
    BitBoard otherKingBoard = board->getBitboard(K + board->getOtherSide());
    int otherKingSq = board->popLsb(otherKingBoard);
    BitBoardEnum sideToMove = board->getSideToMove();

    int fromSq = board->popLsb(king);
    BitBoard moves = board->getKingMask(fromSq);



    //Here we can remove at least knight moves
    BitBoard enemyKnights = board->getBitboard(static_cast<BitBoardEnum>(BitBoardEnum::N + board->getOtherSide()));
    int knightSquare = 0;
    BitBoard enemyKnightAttacks = 0;
    while (enemyKnights != 0) {
        knightSquare = board->popLsb(enemyKnights);
        enemyKnightAttacks |= board->getKnightMask(knightSquare);
    }

    moves &= ~enemyKnightAttacks;
    moves &= ~board->getKingMask(otherKingSq);

    BitBoard all = board->getBitboard(All) & ~board->getBitboard(K + board->getSideToMove());

    BitBoard attacks = 0;
    BitBoard enemyRooks = board->getBitboard(Q + board->getOtherSide()) | board->getBitboard(R + board->getOtherSide());
    BitBoard enemyBishops = board->getBitboard(Q + board->getOtherSide()) | board->getBitboard(B + board->getOtherSide());

    while (enemyRooks) {
        int square = board->popLsb(enemyRooks);
        uint64_t magic = ((all & board->rookMask[square]) * board->magicNumberRook[square]) >> board->magicNumberShiftsRook[square];
        attacks |= (*board->magicMovesRook)[square][magic];
    }

    while (enemyBishops) {
        int square = board->popLsb(enemyBishops);
        uint64_t magic = ((all & board->bishopMask[square]) * board->magicNumberBishop[square]) >> board->magicNumberShiftsBishop[square];
        attacks |= (*board->magicMovesBishop)[square][magic];
    }

    attacks |= pawnAttacks(*board, board->getOtherSide());

    moves &= ~attacks;

    BitBoard captures = moves & enemyBoard;

    int toSq = 0;
    while (captures != 0) {
        toSq = board->popLsb(captures);
        noisyMoves.push_back({ 0, Move::make<NORMAL>(fromSq, toSq) });
    }
}

void MoveGenerator::generateKingQuiet() {
    BitBoard emptySquares = ~board->getBitboard(BitBoardEnum::All);
    BitBoard allPieces = board->getBitboard(BitBoardEnum::All);
    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(BitBoardEnum::K + board->getSideToMove());
    BitBoard enemyBoard = board->getEnemyBoard();
    BitBoard king = board->getBitboard(movedPiece);
    BitBoard otherKingBoard = board->getBitboard(K + board->getOtherSide());
    int otherKingSq = board->popLsb(otherKingBoard);
    BitBoardEnum sideToMove = board->getSideToMove();

    int fromSq = board->popLsb(king);
    BitBoard moves = board->getKingMask(fromSq);



    //Here we can remove at least knight moves
    BitBoard enemyKnights = board->getBitboard(static_cast<BitBoardEnum>(BitBoardEnum::N + board->getOtherSide()));
    int knightSquare = 0;
    BitBoard enemyKnightAttacks = 0;
    while (enemyKnights != 0) {
        knightSquare = board->popLsb(enemyKnights);
        enemyKnightAttacks |= board->getKnightMask(knightSquare);
    }

    moves &= ~enemyKnightAttacks;
    moves &= ~board->getKingMask(otherKingSq);

    BitBoard all = board->getBitboard(All) & ~board->getBitboard(K + board->getSideToMove());

    BitBoard attacks = 0;
    BitBoard enemyRooks = board->getBitboard(Q + board->getOtherSide()) | board->getBitboard(R + board->getOtherSide());
    BitBoard enemyBishops = board->getBitboard(Q + board->getOtherSide()) | board->getBitboard(B + board->getOtherSide());

    while (enemyRooks) {
        int square = board->popLsb(enemyRooks);
        uint64_t magic = ((all & board->rookMask[square]) * board->magicNumberRook[square]) >> board->magicNumberShiftsRook[square];
        attacks |= (*board->magicMovesRook)[square][magic];
    }

    while (enemyBishops) {
        int square = board->popLsb(enemyBishops);
        uint64_t magic = ((all & board->bishopMask[square]) * board->magicNumberBishop[square]) >> board->magicNumberShiftsBishop[square];
        attacks |= (*board->magicMovesBishop)[square][magic];
    }

    attacks |= pawnAttacks(*board, board->getOtherSide());

    moves &= ~attacks;
    BitBoard silentMoves = moves & emptySquares;

    int toSq = 0;
    while (silentMoves) {
        toSq = board->popLsb(silentMoves);
        quietMoves.push_back({ 0, Move::make<NORMAL>(fromSq, toSq) });
    }

    if (sideToMove == BitBoardEnum::White) {
        if (board->getCastleRightsWK()) {
            BitBoard castlineSquares = 0;
            board->setBit(castlineSquares, 5);
            board->setBit(castlineSquares, 6);
            if ((allPieces & castlineSquares) == 0 && !board->isSquareAttacked(castlineSquares | board->sqBB[fromSq], BitBoardEnum::Black)) {
                quietMoves.push_back({ 0, Move::make<CASTLING>(fromSq, fromSq + 2) });
            }  //f1,g1;
        }
        if (board->getCastleRightsWQ()) {
            BitBoard checkSquaresWQ = 0;
            board->setBit(checkSquaresWQ, 2);
            board->setBit(checkSquaresWQ, 3);

            BitBoard emptySquaresWQ = checkSquaresWQ;
            board->setBit(emptySquaresWQ, 1);

            if ((allPieces & emptySquaresWQ) == 0 && !board->isSquareAttacked(checkSquaresWQ | board->sqBB[fromSq], BitBoardEnum::Black)) {
                quietMoves.push_back({ 0,  Move::make<CASTLING>(fromSq, fromSq - 2) });
            }  //b1,c1,d1;
        }
    }
    else if (sideToMove == BitBoardEnum::Black) {
        if (board->getCastleRightsBK()) {
            BitBoard castlineSquares = 0;
            board->setBit(castlineSquares, 61);
            board->setBit(castlineSquares, 62);
            if ((allPieces & castlineSquares) == 0 && !board->isSquareAttacked(castlineSquares | board->sqBB[fromSq], BitBoardEnum::White)) {
                quietMoves.push_back({ 0, Move::make<CASTLING>(fromSq, fromSq + 2) });
            }
        }
        if (board->getCastleRightsBQ()) {
            BitBoard checkSquaresBQ = 0;

            board->setBit(checkSquaresBQ, 58);
            board->setBit(checkSquaresBQ, 59);
            BitBoard emptySquaresBQ = checkSquaresBQ;
            board->setBit(emptySquaresBQ, 57);
            if ((allPieces & emptySquaresBQ) == 0 && !board->isSquareAttacked(checkSquaresBQ | board->sqBB[fromSq], BitBoardEnum::White)) {
                quietMoves.push_back({ 0, Move::make<CASTLING>(fromSq, fromSq - 2) });
            }
        }
    }
}

BitBoard MoveGenerator::pawnAttacks(BitBoard pawns,BitBoardEnum color) {
    BitBoard attacks = 0;
    if (color == White) {
        attacks |= ((pawns & ~Board::FileHMask) << 7);
        attacks |= ((pawns & ~Board::FileAMask) << 9);
    }
    else {
        attacks |= ((pawns & ~Board::FileAMask) >> 7);
        attacks |= ((pawns & ~Board::FileHMask) >> 9);
    }
    return attacks;
}

BitBoard MoveGenerator::pawnAttacks(Board &board, BitBoardEnum color) {
    
    BitBoard pawns = board.getBitboard(P + color);
    return pawnAttacks(pawns, color);

    
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

