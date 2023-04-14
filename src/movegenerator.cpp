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

    BitBoard allPieces = board.getBitboard(Board::All);
    Board::BitBoardEnum movedPiece;
    Board::BitBoardEnum sideToMove = board.getSideToMove();
    BitBoard enemyBoard = board.getEnemyBoard();

    int pawnIncrement = 8;
    int pawnDoubleIncrement = 16;
    int pawnCaptureLeftIncrement = 7;
    int pawnCaptureRightIncrement = 9;
    int enPassantIncrement = -8;
    BitBoard doublePushRank = board.Rank2Mask;
    BitBoard promotionRank = board.Rank8Mask;


    Board::BitBoardEnum queenPromo = Board::Q;
    Board::BitBoardEnum bishopPromo = Board::B;
    Board::BitBoardEnum knightPromo = Board::N;
    Board::BitBoardEnum rookPromo = Board::R;


    if(sideToMove == board.White){
        pawns = board.getBitboard(Board::P);
        movedPiece = Board::P;
    } else {
        pawns = board.getBitboard(Board::p);
        movedPiece = Board::p;

        pawnDoubleIncrement = -16;
        pawnIncrement = -8;
        pawnCaptureLeftIncrement = -7;
        pawnCaptureRightIncrement = -9;
        
        doublePushRank = board.Rank7Mask;
        promotionRank = board.Rank1Mask;

        queenPromo = Board::q;
        bishopPromo = Board::b;
        knightPromo = Board::n;
        rookPromo = Board::r;
    }    
    
    int fromSq = board.popLsb(pawns);
    while (fromSq != 0)
    {        
        int toSq = fromSq+pawnIncrement;
        BitBoard fromSqBoard = board.sqToBitBoard[fromSq];

        if((board.sqToBitBoard[toSq] & allPieces) == 0){            
            if((board.sqToBitBoard[toSq] & promotionRank) != 0){
                moveList.moves[moveList.counter++] = {fromSq,toSq,false,queenPromo,false,false,false,movedPiece};
                moveList.moves[moveList.counter++] = {fromSq,toSq,false,rookPromo,false,false,false,movedPiece};
                moveList.moves[moveList.counter++] = {fromSq,toSq,false,knightPromo,false,false,false,movedPiece};
                moveList.moves[moveList.counter++] = {fromSq,toSq,false,bishopPromo,false,false,false,movedPiece};
            } else {
                moveList.moves[moveList.counter++] = {fromSq,toSq, false, Board::All,false, false,false,movedPiece};
            }
            
        }
        
        if((fromSqBoard & doublePushRank) != 0){
            //Two squares forward from starting position
            toSq =fromSq+pawnDoubleIncrement;

            if((board.sqToBitBoard[toSq] & allPieces) == 0 && (board.sqToBitBoard[fromSq+pawnIncrement] & allPieces) == 0){
                moveList.moves[moveList.counter++] = {fromSq,toSq, false,Board::All,true,false,false,movedPiece};
            }
        }

        //Capture
        
        BitBoard attack = 0;        

        if(sideToMove == board.White){
            attack = enemyBoard & board.southEastOne(fromSqBoard);
            attack |= enemyBoard & board.southWestOne(fromSqBoard);
        } else {            
            attack = enemyBoard & board.northEastOne(fromSqBoard);
            attack |= enemyBoard & board.northWestOne(fromSqBoard);
        }

        while(attack != 0){
            toSq = board.popLsb(attack);

            if((board.sqToBitBoard[toSq] & promotionRank) != 0){
                moveList.moves[moveList.counter++] = {fromSq,toSq,true,queenPromo,false,false,false,movedPiece};
                moveList.moves[moveList.counter++] = {fromSq,toSq,true,rookPromo,false,false,false,movedPiece};
                moveList.moves[moveList.counter++] = {fromSq,toSq,true,knightPromo,false,false,false,movedPiece};
                moveList.moves[moveList.counter++] = {fromSq,toSq,true,bishopPromo,false,false,false,movedPiece};
            } else {
                moveList.moves[moveList.counter++] = {fromSq,toSq, true,Board::All,false,false,false,movedPiece};
            }
        }

        if(board.getEnPassantSq() != board.getNoSq()){
            BitBoard attack = 0;

            if(sideToMove == board.White){
                attack = board.sqToBitBoard[board.getEnPassantSq()] & board.southEastOne(fromSqBoard);
                attack |= board.sqToBitBoard[board.getEnPassantSq()] & board.southWestOne(fromSqBoard);
            } else {            
                attack = board.sqToBitBoard[board.getEnPassantSq()] & board.northEastOne(fromSqBoard);
                attack |= board.sqToBitBoard[board.getEnPassantSq()] & board.northWestOne(fromSqBoard);
            }

            while(attack != 0){
                toSq = board.popLsb(attack);
                moveList.moves[moveList.counter++] = {fromSq,toSq, true,Board::All,false,true, false,movedPiece};
            }
        }

        fromSq = board.popLsb(pawns);
    }
}


void MoveGenerator::generateKnightMoves(Board &board, MoveList &moveList)
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

    
    int fromSq = 0;
    while (knights!= 0)
    {
        fromSq = board.popLsb(knights);
        BitBoard knightMoves = board.getKnightMask(fromSq);

        int toSq = 0;

        while(knightMoves != 0){
            toSq = board.popLsb(knightMoves);
            if(!board.checkBit(allPieces,toSq)){
                moveList.moves[moveList.counter++] = {fromSq,toSq, false,Board::All,false, false,false,movedPiece};
            }

            
            if(board.checkBit(enemyBoard,toSq)){
                moveList.moves[moveList.counter++] = {fromSq,toSq, true,Board::All,false, false,false,movedPiece};
            }
           
        }
    }
}

void MoveGenerator::generateRookMoves(Board &board, MoveList &moveList)
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
                moveList.moves[moveList.counter++] = {fromSq,toSq, false,Board::All,false, false,false,movedPiece};
            } else if(board.checkBit(enemyBoard,toSq)){
                moveList.moves[moveList.counter++] = {fromSq,toSq, true,Board::All,false, false,false,movedPiece};
            }

            
        }
        
    }
}

void MoveGenerator::generateBishopMoves(Board &board, MoveList &moveList)
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
                moveList.moves[moveList.counter++] = {fromSq,toSq, false,Board::All,false, false,false,movedPiece};
            } else if(board.checkBit(enemyBoard,toSq)){
                moveList.moves[moveList.counter++] = {fromSq,toSq, true,Board::All,false, false,false,movedPiece};
            }
        }
        
    }
}

void MoveGenerator::generateQueenMoves(Board &board, MoveList &moveList)
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
                moveList.moves[moveList.counter++] = {fromSq,toSq, false,Board::All,false, false,false,movedPiece};
            } else if(board.checkBit(enemyBoard,toSq)){
                moveList.moves[moveList.counter++] = {fromSq,toSq, true,Board::All,false, false,false,movedPiece};
            }
        }
        
    }
}

void MoveGenerator::generateKingMoves(Board &board, MoveList &moveList)
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
    
    int toSq = 0;
    while(kingMove != 0){
        toSq = board.popLsb(kingMove);
        
        if(!board.checkBit(allPieces,toSq)){
            moveList.moves[moveList.counter++] = {fromSq,toSq, false,Board::All,false, false,false,movedPiece};
        } else if(board.checkBit(enemyBoard,toSq)){
            moveList.moves[moveList.counter++] = {fromSq,toSq, true,Board::All,false, false,false,movedPiece};
        }
        
    }

    if(sideToMove == Board::White){
        if(board.getCastleRightsWK()){
            BitBoard castlineSquares = 0;
            board.setBit(castlineSquares,5);
            board.setBit(castlineSquares,6);
            if(!board.isSquareAttacked(castlineSquares | board.sqToBitBoard[fromSq], Board::White) && (allPieces & castlineSquares) == 0){
                moveList.moves[moveList.counter++] = {fromSq,fromSq+2,false,Board::All,false,false,true,movedPiece};
            }  //f1,g1;
        }
        if(board.getCastleRightsWQ()){
            BitBoard castlineSquares = 0;
            board.setBit(castlineSquares,2);
            board.setBit(castlineSquares,3);
            if(!board.isSquareAttacked(castlineSquares| board.sqToBitBoard[fromSq], Board::White) && (allPieces & castlineSquares) == 0){
                moveList.moves[moveList.counter++] = {fromSq,fromSq-2,false,Board::All,false,false,true,movedPiece};
            }  //b1,c1,d1;
        }
    } else if(sideToMove == Board::Black){
        if(board.getCastleRightsBK()){
            BitBoard castlineSquares = 0;
            board.setBit(castlineSquares,61);
            board.setBit(castlineSquares,62);
            if(!board.isSquareAttacked(castlineSquares | board.sqToBitBoard[fromSq], Board::Black) && (allPieces & castlineSquares) == 0){
                moveList.moves[moveList.counter++] = {fromSq,fromSq+2,false,Board::All,false,false,true,movedPiece};
            }
        }
        if(board.getCastleRightsBQ()){
            BitBoard castlineSquares = 0;
            board.setBit(castlineSquares,58);
            board.setBit(castlineSquares,59);
            if(!board.isSquareAttacked(castlineSquares | board.sqToBitBoard[fromSq], Board::Black) && (allPieces & castlineSquares) == 0){
                moveList.moves[moveList.counter++] = {fromSq,fromSq-2,false,Board::All,false,false,true,movedPiece};
            }
        }
    }
    
}

