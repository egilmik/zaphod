#include "search.h"

Move Search::searchAlphaBeta(Board board, int depth)
{
    pseudoLegalNodeCounter = 0;
    MoveList moveList;
    MoveGenerator::generateMoves(board,moveList);
    int score = 0;
    int currentBestScore = -100000;
    Move bestMove;
    for(int i = 0; i < moveList.counter; i++){
        Move move = moveList.moves[i];
        bool valid = board.makeMove(move.fromSq,move.toSq,move.piece,move.capture,move.enpassant,move.doublePawnPush,move.castling,move.promotion);
        if(valid){
            pseudoLegalNodeCounter++;
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
    pseudoLegalNodeCounter+= moveList.counter;
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
    int score = getScoreForSpecificPiece(board,BitBoardEnum::P);
    score -= getScoreForSpecificPiece(board,BitBoardEnum::p);
    score += getScoreForSpecificPiece(board,BitBoardEnum::K); 
    score -= getScoreForSpecificPiece(board,BitBoardEnum::k);
    score += getScoreForSpecificPiece(board,BitBoardEnum::Q);
    score -= getScoreForSpecificPiece(board,BitBoardEnum::q);
    score += getScoreForSpecificPiece(board,BitBoardEnum::R);
    score -= getScoreForSpecificPiece(board,BitBoardEnum::r);
    score += getScoreForSpecificPiece(board,BitBoardEnum::N); 
    score -= getScoreForSpecificPiece(board,BitBoardEnum::n);
    score += getScoreForSpecificPiece(board,BitBoardEnum::B);
    score -= getScoreForSpecificPiece(board,BitBoardEnum::b);    
    
    return score;
}

int Search::getScoreForSpecificPiece(Board &board,BitBoardEnum piece)
{
    BitBoard pieceBoard = board.getBitboard(piece);
    std::array<int,64> scoreArray = pieceSquareScoreArray[piece]; 
    int score = 0;

    int pieceSquare = 0;
    while (pieceBoard != 0)    {
        pieceSquare = board.popLsb(pieceBoard);
        score += scoreArray[pieceSquare];
    }
    return score;
}

int Search::getMaterialScore(Board &board)
{
    int score = 2000*(board.countSetBits(BitBoardEnum::K) - board.countSetBits(BitBoardEnum::k))
                + 900*(board.countSetBits(BitBoardEnum::Q) - board.countSetBits(BitBoardEnum::q))
                + 500*(board.countSetBits(BitBoardEnum::R) - board.countSetBits(BitBoardEnum::r))
                + 330*(board.countSetBits(BitBoardEnum::B) - board.countSetBits(BitBoardEnum::b))
                + 320*(board.countSetBits(BitBoardEnum::N) - board.countSetBits(BitBoardEnum::n))
                + 100*(board.countSetBits(BitBoardEnum::P) - board.countSetBits(BitBoardEnum::p));
    return score;
}

int Search::evaluate(Board &board)
{
    evaluatedNodes++;
    int score = getPieceSquareScore(board);
    score += getMaterialScore(board);

    if(board.getSideToMove() == BitBoardEnum::Black){
        return -score;
    }
                   
    //std::cout << score << std::endl;
    return score;
}
