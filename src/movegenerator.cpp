#include "movegenerator.h"
#include <iostream>
#include "material.h"
#include <algorithm>
#include "see.h"
#include <cassert>
#include "tools/fentools.h"

void MoveGenerator::generateMoves(Board &board,MoveList &moveList)
{
    MoveGenerator gen;
    Move killer[2] = {};
    History hist;
//    gen.init(board, Move{}, false, killer, &hist);
    
    Move move;
    while((move = gen.next())){
        moveList.moves[moveList.counter++] = move;
    }
    moveList.checkers = board.getCheckers();
}

void MoveGenerator::init(Board& b, Move tt, bool onlyCaptures, Move killer[], History *hist, History::ContSlice* const* cont) {
    currentStage = TT_MOVE;
    noisyIdx = 0;
    quietIdx = 0;
    noisyCount = 0;
    quietCount = 0;
    killerIdx = 0;
    ttMoveFound = false;
    board = &b;
    onlyNoisy = onlyCaptures;
    killerMove = killer;
    history = hist;
    contSlices = cont;
    ttMove = tt;

    BitBoard king = board->getBitboard(K + board->getSideToMove());
    assert(king != 0);  // king must exist
    kingSquare = Board::popLsb(king);
    assert(kingSquare < 64);

    checkMask = ~static_cast<BitBoard>(0);
    BitBoard checkers = board->getCheckers();
    if (checkers) {
        BitBoard inBetween = 0;
        BitBoard checks = checkers;
        while (checks) {
            inBetween |= board->sqBetween[kingSquare][Board::popLsb(checks)];
        }
        checkMask = inBetween | checkers;
    }

    computeKingDanger();
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

    //There is no piece to move
    if (piece == All) return false;

    // Cannot capture the king (guards against TT hash collisions producing king-capture moves)
    if (toBB & board->getBitboard(static_cast<BitBoardEnum>(K + board->getOtherSide()))) return false;

    // We cannot move enemy piece
    if (fromBB & enemyBoard) return false;

    // If we are not castling we cannot capture own piece
    if (move.getMoveType() != CASTLING && ownBoard & toBB) return false;

    // A CASTLING move must originate from the side's king. Without this check a TT castle move
    // retrieved via hash collision passes isMoveLegalSliders (which ignores move type) when a
    // rook happens to occupy the king's former square and can slide to the castle destination.
    if (move.getMoveType() == CASTLING &&
        piece != static_cast<BitBoardEnum>(K + board->getSideToMove())) return false;

    // A PROMOTION or EN_PASSANT move must originate from the side's pawn. Without this check a
    // TT move of either type retrieved via hash collision passes isMoveLegalSliders when a
    // slider occupies the pawn's former square. A promotion collision corrupts the board by
    // turning the slider into a phantom pawn (revert always restores P, not the original piece).
    // An EP collision removes the wrong captured pawn and misplaces the piece on the board.
    BitBoardEnum ownPawn = static_cast<BitBoardEnum>(P + board->getSideToMove());
    if ((move.getMoveType() == PROMOTION || move.getMoveType() == EN_PASSANT) &&
        piece != ownPawn) return false;

    BitBoardEnum normPiece = static_cast<BitBoardEnum>(piece - board->getSideToMove());

    // In double check only king moves are legal — a non-king move can capture one checker
    // but the second checker still gives check, so it doesn't resolve the position
    if (board->countSetBits(board->getCheckers()) >= 2 && normPiece != K) return false;
    MoveList list;
    switch (normPiece) {
    case P:
        generatePawnMoves(*board, list, kingSquare, board->getPins(), board->getSnipers());
        for (int i = 0; i < list.counter; i++) {
            if (list.moves[i].value == move.value) return true;
        }
        break;
    case N:{
        if (board->getPins() & fromBB) return false;
        BitBoard targets = board->getKnightMask(move.from()) & ~ownBoard & checkMask;
        if (targets & toBB) return true;
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
    case K: {
        // Guard against TT-collision moves targeting a pawn-attacked square.
        // generateKingMoves filters via the pawn bitboard, but an inconsistent
        // board state can cause it to miss this check.  A direct bitboard query
        // here is always safe (pawns are not blocked by intervening pieces).
        BitBoard enemyPawnAtks = pawnAttacks(
            board->getBitboard(static_cast<BitBoardEnum>(P + board->getOtherSide())),
            board->getOtherSide());
        if (enemyPawnAtks & toBB) return false;
        generateKingMoves(*board, list, board->getCheckers(), kingSquare, board->getPins(), board->getSnipers());
        for (int i = 0; i < list.counter; i++) {
            if (list.moves[i].value == move.value) return true;
        }
        break;
    }
    default:
        return false;
    }
        
    return false;

}

bool MoveGenerator::isMoveLegalSliders(Move move, bool isCapture, BitBoard moves, BitBoard pieceBoard, BitBoard enemyBoard, BitBoard emptySquares) {
    int fromSq = 0;
    BitBoard toBB = board->sqBB[move.to()];    
    
    fromSq = move.from();
    moves = makeLegalMoves(*board, moves, board->getPins(), checkMask, board->getSnipers(), fromSq, kingSquare);
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

    
    return false;
}

Move MoveGenerator::next() {

  

    switch (currentStage) {

        case TT_MOVE:
            currentStage = GEN_NOISY;
            
            if (ttMove) {
                if (isMoveLegal(ttMove)) {
                    ttMoveFound = true;
                    bool isCapture = board->getPieceOnSquare(ttMove.to()) != All;
                    if (onlyNoisy && isCapture) {
                        return ttMove;
                    }
                    if (!onlyNoisy) {
                        return ttMove;
                    }
                }
            }
            
            [[fallthrough]];
        case GEN_NOISY:
            currentStage = GOOD_NOISY;
            if (board->countSetBits(board->getCheckers()) < 2) {
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
            if (noisyIdx < noisyCount && noisyMoves[noisyIdx].score > 0) {
                return noisyMoves[noisyIdx++].move;
            } 

            // if no moves left, next stage
            currentStage = KILLER;
            [[fallthrough]];
        case KILLER:           
            if (!onlyNoisy || board->getCheckers() != 0) {
                while (killerIdx < 2) {
                    Move killer = killerMove[killerIdx++];
                    if (killer && killer.value != ttMove.value && isMoveLegal(killer)) {
                        return killer;
                    }
                }
            }           

            //Check if killer moves are valid
            //If no killer, next stage
            currentStage = GEN_QUIET;
            [[fallthrough]];
        case GEN_QUIET:
            currentStage = QUIET;
            //Generate quite
            //Score
            if(!onlyNoisy || board->getCheckers() != 0){

                if (board->countSetBits(board->getCheckers()) < 2) {
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
            if (quietIdx < quietCount) {
                return quietMoves[quietIdx++].move;
            }

            currentStage = BAD_NOISY;
            [[fallthrough]];

        case BAD_NOISY:
            //Return bad noisy
            // if no more moves left, next stage
            if (noisyIdx < noisyCount) {
                return noisyMoves[noisyIdx++].move;
            }
            currentStage = END;
            [[fallthrough]];
        default:
            return Move();

    };

}

void MoveGenerator::sortNoisyMoves() {
    int side = 0;
    if (board->getSideToMove() == Black) {
        side = 1;
    }

    for (int i = 0; i < noisyCount; i++) {
        

        
        
        if (noisyMoves[i].move.value == ttMove.value) {
            if (!ttMoveFound) {
                std::cout << "info string TTMove not returned, but still present int noisy" << std::endl;
            }
	    noisyMoves[i] = noisyMoves[--noisyCount];
            i--;
            continue;
        }
        


        if (noisyMoves[i].move.getMoveType() == PROMOTION) {
            //TODO: Promotion capture

            noisyMoves[i].score = 80000;
        }
        else if (board->getPieceOnSquare(noisyMoves[i].move.to()) != All) {

            BitBoardEnum capturedPiece = board->getPieceOnSquare(noisyMoves[i].move.to());
            BitBoardEnum attacker = board->getPieceOnSquare(noisyMoves[i].move.from());

            if (noisyMoves[i].move.getMoveType() == EN_PASSANT) {
                capturedPiece = P;
                noisyMoves[i].score = 70000;
            }
            else {
                assert(attacker != All);
                assert(capturedPiece != All);
                int Mvv = Material::pieceMaterialScoreArray[capturedPiece];
                int lva = Material::pieceMaterialScoreArray[attacker];

                int mvvlva = (Mvv - lva) / 100;
                noisyMoves[i].score = 70000 + mvvlva;

                if (Mvv > lva) {
                    noisyMoves[i].score = 70000 + mvvlva;
                }
                else {
                    using namespace See;
                    int seeScore = see(*board, noisyMoves[i].move.from(), noisyMoves[i].move.to(), board->getSideToMove());
                    if (seeScore >= 0) {
                        noisyMoves[i].score = 70000 + seeScore;
                    }
                    else {
                        noisyMoves[i].score = -70000 - seeScore;
                    }
                    
                }
            }
        }
    }

    std::stable_sort(noisyMoves, noisyMoves+noisyCount,
        [](const ScoredMove& a, const ScoredMove& b) {
            if (a.score != b.score) return a.score > b.score;     // strict order
            return a.move.value < b.move.value;                   // total tie-break
        });
}

void MoveGenerator::scoreQuietMoves() {

    int side =  (board->getSideToMove() == Black);

    for (int i = 0; i < quietCount; i++) {
        
        if (quietMoves[i].move.value == ttMove.value){
            if (!ttMoveFound) {
                //bool test = isMoveLegal(ttMove);
                std::cout << "info string TTMove not returned, but still present int quiet" << std::endl;
            }
            quietMoves[i] = quietMoves[--quietCount]; 
            i--;
            continue;
        }
        
        if (killerMove[0].value == quietMoves[i].move.value ||
            killerMove[1].value == quietMoves[i].move.value) {

            quietMoves[i] = quietMoves[--quietCount];
            i--;
            continue;
        }

        quietMoves[i].score = history->butterflyScore(board->getSideToMove(), quietMoves[i].move, board->getThreats());
       if(contSlices){
		BitBoardEnum piece = board->getPieceOnSquare(quietMoves[i].move.from());
	       quietMoves[i].score += history->contScore(contSlices, piece, quietMoves[i].move.to());
       }	       
    }

    std::stable_sort(quietMoves, quietMoves+quietCount,
        [](const ScoredMove& a, const ScoredMove& b) {
            if (a.score != b.score) return a.score > b.score;     // strict order
            return a.move.value < b.move.value;                   // total tie-break
        });

}

void MoveGenerator::generatePawnNoisy() {
    BitBoard allPieces = board->getBitboard(BitBoardEnum::All);
    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(BitBoardEnum::P + board->getSideToMove());
    BitBoard pawns = board->getBitboard(movedPiece);
    BitBoardEnum sideToMove = board->getSideToMove();
    BitBoard enemyBoard = board->getEnemyBoard();

    BitBoard pinnedPawns = pawns & board->getPins();
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
    BitBoard promotions = 0;
    BitBoard nwAttacks = 0;
    BitBoard neAttacks = 0;
    BitBoard promoNWAttacks = 0;
    BitBoard promoNEAttacks = 0;



    if (sideToMove == White) {
        singlePush = (pawns << 8) & ~allPieces;
        neAttacks = ((pawns & ~Board::FileHMask) << 7) & enemyBoard;
        nwAttacks = ((pawns & ~Board::FileAMask) << 9) & enemyBoard;
    }
    else {
        singlePush = (pawns >> 8) & ~allPieces;
        neAttacks = ((pawns & ~Board::FileAMask) >> 7) & enemyBoard;
        nwAttacks = ((pawns & ~Board::FileHMask) >> 9) & enemyBoard;
    }


    //Checking pinned pieces individually
    BitBoard pinnedPawnSinglePush = 0;
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
        singlePush |= makeLegalMoves(*board, pinnedPawnSinglePush, board->getPins(), checkMask, board->getSnipers(), pinnedSquare, kingSquare);
        neAttacks |= makeLegalMoves(*board, pinnedNEAttack, board->getPins(), checkMask, board->getSnipers(), pinnedSquare, kingSquare);
        nwAttacks |= makeLegalMoves(*board, pinnedNWAttack, board->getPins(), checkMask, board->getSnipers(), pinnedSquare, kingSquare);
    }


    
    singlePush &= checkMask;
    neAttacks &= checkMask;
    nwAttacks &= checkMask;
    


    promotions = (singlePush & promotionRank);
    singlePush &= ~promotionRank;
    promoNEAttacks = (neAttacks & promotionRank);
    promoNWAttacks = (nwAttacks & promotionRank);
    neAttacks &= ~promotionRank;
    nwAttacks &= ~promotionRank;

    int square = 0;

    while (promotions) {
        square = Board::popLsb(promotions);
        addNoisy(Move::make<PROMOTION>(square - pawnIncrement, square, Q));
        addNoisy(Move::make<PROMOTION>(square - pawnIncrement, square, B));
        addNoisy(Move::make<PROMOTION>(square - pawnIncrement, square, R));
        addNoisy(Move::make<PROMOTION>(square - pawnIncrement, square, N));
    }

    while (nwAttacks) {
        square = Board::popLsb(nwAttacks);
        assert(!(board->sqBB[square] & board->getBitboard(static_cast<BitBoardEnum>(K + board->getOtherSide()))));
        addNoisy(Move::make<NORMAL>(square - pawnCaptureRightIncrement, square));
    };

    while (neAttacks) {
        square = Board::popLsb(neAttacks);
        assert(!(board->sqBB[square] & board->getBitboard(static_cast<BitBoardEnum>(K + board->getOtherSide()))));
        addNoisy(Move::make<NORMAL>(square - pawnCaptureLeftIncrement, square));
    }

    while (promoNEAttacks) {
        square = Board::popLsb(promoNEAttacks);
        assert(!(board->sqBB[square] & board->getBitboard(static_cast<BitBoardEnum>(K + board->getOtherSide()))));
        addNoisy(Move::make<PROMOTION>(square - pawnCaptureLeftIncrement, square, Q));
        addNoisy(Move::make<PROMOTION>(square - pawnCaptureLeftIncrement, square, B));
        addNoisy(Move::make<PROMOTION>(square - pawnCaptureLeftIncrement, square, R));
        addNoisy(Move::make<PROMOTION>(square - pawnCaptureLeftIncrement, square, N));
    }

    while (promoNWAttacks) {
        square = Board::popLsb(promoNWAttacks);
        assert(!(board->sqBB[square] & board->getBitboard(static_cast<BitBoardEnum>(K + board->getOtherSide()))));
        addNoisy(Move::make<PROMOTION>(square - pawnCaptureRightIncrement, square, Q));
        addNoisy(Move::make<PROMOTION>(square - pawnCaptureRightIncrement, square, B));
        addNoisy(Move::make<PROMOTION>(square - pawnCaptureRightIncrement, square, R));
        addNoisy(Move::make<PROMOTION>(square - pawnCaptureRightIncrement, square, N));
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
                int capturedPawnSq = board->getEnPassantSq() - pawnIncrement;
                assert(capturedPawnSq >= 0 && capturedPawnSq < 64);
                assert(board->sqBB[capturedPawnSq] & board->getBitboard(static_cast<BitBoardEnum>(P + board->getOtherSide())));
                BitBoard checkers = 0;
                BitBoard all = board->getBitboard(All);
                //Removing current attacking pawn and enpassant pawn from board to perform check check
                BitBoard toBeRemoved = (Board::sqBB[fromSq] | Board::sqBB[capturedPawnSq]);

                //Remove both pawns from old pos, then add in new pos
                all &= ~toBeRemoved;
                all |= Board::sqBB[toSq];


                uint64_t magic = ((all & board->rookMask[kingSquare]) * board->magicNumberRook[kingSquare]) >> board->magicNumberShiftsRook[kingSquare];
                checkers |= (*board->magicMovesRook)[kingSquare][magic] & (board->getBitboard(Q + board->getOtherSide()) | board->getBitboard(R + board->getOtherSide()));
                magic = ((all & board->bishopMask[kingSquare]) * board->magicNumberBishop[kingSquare]) >> board->magicNumberShiftsBishop[kingSquare];
                checkers |= (*board->magicMovesBishop)[kingSquare][magic] & (board->getBitboard(Q + board->getOtherSide()) | board->getBitboard(B + board->getOtherSide()));

                if (checkers == 0) {
                    addNoisy(Move::make<MoveType::EN_PASSANT>(fromSq, toSq));
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

    BitBoard pinnedPawns = pawns & board->getPins();
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
        singlePush |= makeLegalMoves(*board, pinnedPawnSinglePush, board->getPins(), checkMask, board->getSnipers(), pinnedSquare, kingSquare);
        doublePush |= makeLegalMoves(*board, pinnedDoublePush, board->getPins(), checkMask, board->getSnipers(), pinnedSquare, kingSquare);
        
    }

    
    singlePush &= checkMask;
    doublePush &= checkMask;


    promotions = (singlePush & promotionRank);
    singlePush &= ~promotionRank;

    //Single push
    int square = 0;
    while (singlePush) {
        square = board->popLsb(singlePush);
        addQuiet(Move::make<NORMAL>(square - pawnIncrement, square));
    }

    while (doublePush) {
        square = board->popLsb(doublePush);
        addQuiet(Move::make<NORMAL>(square - pawnDoubleIncrement, square));
    }

    
}

void MoveGenerator::generatePawnMoves(Board& board, MoveList& moveList, int kingSquare, BitBoard pinned, BitBoard snipers)
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
        singlePush |= makeLegalMoves(board, pinnedPawnSinglePush, pinned, checkMask, snipers, pinnedSquare, kingSquare);
        doublePush |= makeLegalMoves(board, pinnedDoublePush, pinned, checkMask, snipers, pinnedSquare, kingSquare);
        neAttacks |= makeLegalMoves(board, pinnedNEAttack, pinned, checkMask, snipers, pinnedSquare, kingSquare);
        nwAttacks |= makeLegalMoves(board, pinnedNWAttack, pinned, checkMask, snipers, pinnedSquare, kingSquare);
    }

    
    singlePush &= checkMask;
    doublePush &= checkMask;
    neAttacks &= checkMask;
    nwAttacks &= checkMask;
    


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
                int capturedPawnSq = board.getEnPassantSq() - pawnIncrement;
                assert(capturedPawnSq >= 0 && capturedPawnSq < 64);
                assert(board.sqBB[capturedPawnSq] & board.getBitboard(static_cast<BitBoardEnum>(P + board.getOtherSide())));
                BitBoard checkers = 0;
                BitBoard all = board.getBitboard(All);
                //Removing current attacking pawn and enpassant pawn from board to perform check check
                BitBoard toBeRemoved = (board.sqBB[fromSq] | board.sqBB[capturedPawnSq]);

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
    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(BitBoardEnum::N + board->getSideToMove());
    BitBoard enemyBoard = board->getEnemyBoard();
    BitBoard knights = board->getBitboard(movedPiece) & ~board->getPins();

    int fromSq = 0;
    while (knights)
    {
        fromSq = Board::popLsb(knights);
        BitBoard moves = board->getKnightMask(fromSq);

        moves &= checkMask;

        BitBoard captures = moves & enemyBoard;

        int toSq = 0;

        while (captures != 0) {
            toSq = Board::popLsb(captures);
            assert(!(board->sqBB[toSq] & board->getBitboard(static_cast<BitBoardEnum>(K + board->getOtherSide()))));
            addNoisy(Move::make<NORMAL>(fromSq, toSq));
        }
    }
}

void MoveGenerator::generateKnightQuiet() {
    BitBoard emptySquares = ~board->getBitboard(BitBoardEnum::All);
    BitBoard allPieces = board->getBitboard(BitBoardEnum::All);
    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(BitBoardEnum::N + board->getSideToMove());
    BitBoard enemyBoard = board->getEnemyBoard();
    BitBoard knights = board->getBitboard(movedPiece) & ~board->getPins();

    int fromSq = 0;
    while (knights)
    {
        fromSq = Board::popLsb(knights);
        BitBoard moves = board->getKnightMask(fromSq);

        moves &= checkMask;

        BitBoard captures = moves & enemyBoard;
        BitBoard silentMoves = moves & emptySquares;

        int toSq = 0;
        while (silentMoves) {
            toSq = Board::popLsb(silentMoves);
            addQuiet(Move::make<NORMAL>(fromSq, toSq));
        }

    }
}

void MoveGenerator::generateBishopNoisy() {
    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(BitBoardEnum::B + board->getSideToMove());
    BitBoard enemyBoard = board->getEnemyBoard();
    BitBoard bishops = board->getBitboard(movedPiece);

    int fromSq = 0;
    while (bishops) {
        fromSq = Board::popLsb(bishops);
        BitBoard moves = board->getBishopMagics(fromSq);

        moves = makeLegalMoves(*board, moves, board->getPins(), checkMask, board->getSnipers(), fromSq, kingSquare);

        BitBoard captures = moves & enemyBoard;        

        int toSq = 0;

        while (captures != 0) {
            toSq = Board::popLsb(captures);
            assert(!(board->sqBB[toSq] & board->getBitboard(static_cast<BitBoardEnum>(K + board->getOtherSide()))));
            addNoisy(Move::make<NORMAL>(fromSq, toSq));
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

        moves = makeLegalMoves(*board, moves, board->getPins(), checkMask, board->getSnipers(), fromSq, kingSquare);

        BitBoard silentMoves = moves & emptySquares;
        int toSq = 0;

        while (silentMoves) {
            toSq = Board::popLsb(silentMoves);
            addQuiet(Move::make<NORMAL>(fromSq, toSq));
        }

    }
}

void MoveGenerator::generateRookNoisy() {
    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(BitBoardEnum::R + board->getSideToMove());
    BitBoard enemyBoard = board->getEnemyBoard();
    BitBoard rooks = board->getBitboard(movedPiece);

    int fromSq = 0;
    while (rooks) {
        fromSq = Board::popLsb(rooks);
        BitBoard moves = board->getRookMagics(fromSq);
        moves = makeLegalMoves(*board, moves, board->getPins(), checkMask, board->getSnipers(), fromSq, kingSquare);
        BitBoard captures = moves & enemyBoard;

        int toSq = 0;
        while (captures != 0) {
            toSq = Board::popLsb(captures);
            assert(!(board->sqBB[toSq] & board->getBitboard(static_cast<BitBoardEnum>(K + board->getOtherSide()))));
            addNoisy(Move::make<NORMAL>(fromSq, toSq));
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
        moves = makeLegalMoves(*board, moves, board->getPins(), checkMask, board->getSnipers(), fromSq, kingSquare);
        BitBoard silentMoves = moves & emptySquares;

        int toSq = 0;
        while (silentMoves) {
            toSq = Board::popLsb(silentMoves);
            addQuiet(Move::make<NORMAL>(fromSq, toSq));
        }

    }
}

void MoveGenerator::generateQueenNoisy() {
    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(BitBoardEnum::Q + board->getSideToMove());
    BitBoard enemyBoard = board->getEnemyBoard();
    BitBoard queens = board->getBitboard(movedPiece);

    int fromSq = 0;
    while (queens) {
        fromSq = Board::popLsb(queens);
        BitBoard moves = (board->getBishopMagics(fromSq) | board->getRookMagics(fromSq));
        moves = makeLegalMoves(*board, moves, board->getPins(), checkMask, board->getSnipers(), fromSq, kingSquare);
        BitBoard captures = moves & enemyBoard;
        

        int toSq = 0;
        while (captures) {
            toSq = Board::popLsb(captures);
            assert(!(board->sqBB[toSq] & board->getBitboard(static_cast<BitBoardEnum>(K + board->getOtherSide()))));
            addNoisy(Move::make<NORMAL>(fromSq, toSq));
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
        moves = makeLegalMoves(*board, moves, board->getPins(), checkMask, board->getSnipers(), fromSq, kingSquare);
        BitBoard silentMoves = moves & emptySquares;

        int toSq = 0;
        while (silentMoves) {
            toSq = Board::popLsb(silentMoves);
            addQuiet(Move::make<NORMAL>(fromSq, toSq));
        }

    }
}

void MoveGenerator::generateKingNoisy() {
    BitBoard king = board->getBitboard(static_cast<BitBoardEnum>(K + board->getSideToMove()));
    int fromSq = board->popLsb(king);
    BitBoard moves = board->getKingMask(fromSq) & ~kingDangerMask;
    BitBoard captures = moves & board->getEnemyBoard();

    int toSq = 0;
    while (captures != 0) {
        toSq = board->popLsb(captures);
        assert(!(board->sqBB[toSq] & board->getBitboard(static_cast<BitBoardEnum>(K + board->getOtherSide()))));
        addNoisy(Move::make<NORMAL>(fromSq, toSq));
    }
}

void MoveGenerator::generateKingQuiet() {
    BitBoard emptySquares = ~board->getBitboard(BitBoardEnum::All);
    BitBoard allPieces = board->getBitboard(All);
    BitBoardEnum sideToMove = board->getSideToMove();
    BitBoard king = board->getBitboard(static_cast<BitBoardEnum>(K + board->getSideToMove()));


    int fromSq = board->popLsb(king);
    BitBoard moves = board->getKingMask(fromSq) & ~kingDangerMask;

    BitBoard silentMoves = moves & emptySquares;

    int toSq = 0;
    while (silentMoves) {
        toSq = board->popLsb(silentMoves);
        addQuiet(Move::make<NORMAL>(fromSq, toSq));
    }

    Move castles[2];
    int nCastles = generateCastlingMoves(fromSq, castles);
    for (int i = 0; i < nCastles; i++) addQuiet(castles[i]);
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

BitBoard MoveGenerator::makeLegalMoves(Board &board, BitBoard moves, BitBoard pinned, BitBoard checkMask, BitBoard snipers, int fromSq, int kingSquare) {
    
    moves &= checkMask;

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

void MoveGenerator::computeKingDanger() {
    BitBoard danger = 0;

    // Enemy knights
    BitBoard enemyKnights = board->getBitboard(static_cast<BitBoardEnum>(N + board->getOtherSide()));
    while (enemyKnights) {
        danger |= board->getKnightMask(board->popLsb(enemyKnights));
    }

    // Enemy king
    BitBoard otherKing = board->getBitboard(K + board->getOtherSide());
    danger |= board->getKingMask(board->popLsb(otherKing));

    // Enemy sliders. Our own king is removed from the occupancy so a checking
    // slider also marks the square *behind* the king as unsafe (king can't retreat along the ray).
    BitBoard all = board->getBitboard(All) & ~board->getBitboard(K + board->getSideToMove());
    BitBoard enemyRooks = board->getBitboard(Q + board->getOtherSide()) | board->getBitboard(R + board->getOtherSide());
    BitBoard enemyBishops = board->getBitboard(Q + board->getOtherSide()) | board->getBitboard(B + board->getOtherSide());

    while (enemyRooks) {
        int sq = board->popLsb(enemyRooks);
        uint64_t magic = ((all & board->rookMask[sq]) * board->magicNumberRook[sq]) >> board->magicNumberShiftsRook[sq];
        danger |= (*board->magicMovesRook)[sq][magic];
    }
    while (enemyBishops) {
        int sq = board->popLsb(enemyBishops);
        uint64_t magic = ((all & board->bishopMask[sq]) * board->magicNumberBishop[sq]) >> board->magicNumberShiftsBishop[sq];
        danger |= (*board->magicMovesBishop)[sq][magic];
    }

    // Enemy pawns
    danger |= pawnAttacks(*board, board->getOtherSide());
    kingDangerMask = danger;
}

struct CastleSpec {
    BitBoard emptySquares;
    BitBoard safeSquares;
    int kingOffset;
};

constexpr BitBoard bit(int sq) { return BitBoard(1) << sq; }

constexpr CastleSpec CASTLES[4] = {
    {bit(5) | bit(6), bit(5) | bit(6), 2 },
    {bit(1) | bit(2) | bit(3), bit(2) | bit(3), -2},
    {bit(61) | bit(62), bit(61) | bit(62), 2},
    {bit(57) | bit(58) | bit(59), bit(58) | bit(59), -2}
};

int MoveGenerator::generateCastlingMoves(int kingSq, Move out[2]) {
    bool right[4] = { board->getCastleRightsWK(), board->getCastleRightsWQ(), board->getCastleRightsBK(), board->getCastleRightsBQ() };

    BitBoard allPieces = board->getBitboard(All);
    BitBoard enemyBoard = board->getEnemyBoard();
    BitBoard king = board->getBitboard(static_cast<BitBoardEnum>(K + board->getSideToMove()));

    //Check if it is black or white to select correct castle spec
    int firstIdx = (board->getSideToMove() == White) ? 0 : 2;

    int moveCount = 0;

    for (int i = firstIdx; i < firstIdx + 2; i++) {
        const CastleSpec& spec = CASTLES[i];
        if (!right[i]) continue;
        if (allPieces & spec.emptySquares) continue;
        if (board->getThreats() & (spec.safeSquares | king)) continue;
        out[moveCount++] = Move::make<CASTLING>(kingSq, kingSq + spec.kingOffset);
    }
    return moveCount;

}

void MoveGenerator::generateKingMoves(Board &board, MoveList &moveList, BitBoard checkers, int kingSquare, BitBoard pinned, BitBoard snipers)
{
    BitBoard emptySquares = ~board.getBitboard(BitBoardEnum::All);
    BitBoard allPieces = board.getBitboard(All);
    BitBoardEnum sideToMove = board.getSideToMove();
    BitBoard king = board.getBitboard(static_cast<BitBoardEnum>(K + board.getSideToMove()));
    

    int fromSq = board.popLsb(king);
    BitBoard moves = board.getKingMask(fromSq) & ~kingDangerMask;

    BitBoard captures = moves & board.getEnemyBoard();
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

    Move castles[2];
    int nCastles = generateCastlingMoves(fromSq, castles);
    for (int i = 0; i < nCastles; i++) moveList.moves[moveList.counter++] = castles[i];
    
}

