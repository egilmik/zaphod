#include "movegenerator.h"
#include <iostream>

std::vector<Move> MoveGenerator::generateMoves(Board board)
{
    std::vector<Move> moveVector;
    generatePawnPush(board,moveVector);
    return moveVector;
}

void MoveGenerator::generatePawnPush(Board board, std::vector<Move> &moveVector)
{
    BitBoard pawnMoves;
    BitBoard pawns;

    BitBoard allPieces = board.getBitboard(Board::All);    

    if(board.sideToMove == board.White){
        pawns = board.getBitboard(Board::P);
    } else {
        pawns = board.getBitboard(Board::p);
    }
    
    int fromSq = board.popLsb(pawns);;
    while (fromSq != 0)
    {
        int toSq = fromSq+8;

        if(!board.checkBit(allPieces,toSq)){
            std::cout << "pseudo legal move from " << fromSq << " to " << toSq << std::endl;
            Move move;
            move.fromSq = fromSq;
            move.toSq = toSq;
            move.piece = Board::BitBoardEnum::P;

            moveVector.push_back(move);
        }
        
        if(fromSq > 7 && fromSq < 16){
            //Two squares forward from starting position
            toSq+=8;

            if(!board.checkBit(allPieces,toSq)){
                std::cout << "pseudo legal move from " << fromSq << " to " << toSq << std::endl;
                Move move;
                move.fromSq = fromSq;
                move.toSq = toSq;
                move.piece = Board::BitBoardEnum::P;
                moveVector.push_back(move);
            }
        }
        fromSq = board.popLsb(pawns);

    }
}

void MoveGenerator::generatePawnCaptures(Board board)
{
}

void MoveGenerator::generateRookMove(Board board)
{
}

