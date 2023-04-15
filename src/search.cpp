#include "search.h"

Move Search::searchAlphaBeta(Board board, int depth)
{
    MoveList moveList;
    MoveGenerator::generateMoves(board,moveList);
    int score = 0;
    int currentBestScore = -100000;
    Move bestMove;
    for(int i = 0; i < moveList.counter; i++){
        Move move = moveList.moves[i];
        bool valid = board.makeMove(move.fromSq,move.toSq,move.piece,move.capture,move.enpassant,move.doublePawnPush,move.castling,move.promotion);
        if(valid){
            score = -negaMax(board,-1000000000,1000000000,depth);
            if(score >= currentBestScore){
                currentBestScore = score;
                bestMove = move;
            }
        }

        board.revertLastMove();               
    }

    if(currentBestScore <= -100000){
        Move move;
        move.fromSq = 0;
        move.toSq = 0;
        return move;
    }
    std::cout << currentBestScore << std::endl;
    return bestMove;
    
}

int Search::negaMax(Board board, int alpha, int beta, int depthLeft)
{
    if(depthLeft == 0) return evaluate(board);
    
    MoveList moveList;
    MoveGenerator::generateMoves(board,moveList);
    int score = 0;
    for(int i = 0; i < moveList.counter; i++){
        Move move = moveList.moves[i];
        bool valid = board.makeMove(move.fromSq,move.toSq,move.piece,move.capture,move.enpassant,move.doublePawnPush,move.castling,move.promotion);
        if(valid){
            score = -negaMax(board,-beta,-alpha,depthLeft-1);
        }
        if(score >= beta){
            return beta;
        }
        if(score > alpha){
            alpha = score;
        }

        board.revertLastMove();               
    }
    return alpha;

}

int Search::getPieceSquareScore(Board &board)
{
    int score = getScoreForSpecificPiece(board,Board::P) -getScoreForSpecificPiece(board,Board::p);
    
    return score;
}

int Search::getScoreForSpecificPiece(Board &board,Board::BitBoardEnum piece)
{
    BitBoard pieceBoard = board.getBitboard(piece);
    std::array<int,64> scoreArray = pieceSquareScoreArray[piece]; 
    int score = 0;

    int pieceSquare = 0;
    while (pieceBoard != 0)    {
        pieceSquare = board.popLsb(pieceBoard);
        score = scoreArray[pieceSquare];
    }
    return score;
}

int Search::getMaterialScore(Board &board)
{
    int score = 2000*(board.countSetBits(Board::K) - board.countSetBits(Board::k))
                + 900*(board.countSetBits(Board::Q) - board.countSetBits(Board::q))
                + 500*(board.countSetBits(Board::R) - board.countSetBits(Board::r))
                + 330*(board.countSetBits(Board::B) - board.countSetBits(Board::b))
                + 320*(board.countSetBits(Board::N) - board.countSetBits(Board::n))
                + 100*(board.countSetBits(Board::P) - board.countSetBits(Board::p));
    return score;
}

int Search::evaluate(Board &board)
{
    int score = getPieceSquareScore(board);
    score += getMaterialScore(board);

    if(board.getSideToMove() == Board::Black){
        return -score;
    }
                   
    //std::cout << score << std::endl;
    return score;
}
