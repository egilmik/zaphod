#include "search.h"
#include "perft.h"
#include "material.h"
#include <algorithm>

Score Search::search(Board &board, int maxDepth)
{    
    int lowerBound = -20000;
    int upperBound = 20000;
    bool inIteration = true;

    if (board.getSideToMove() == BitBoardEnum::Black) {
        isBlackMaxPlayer = true;
    }

    for (int i = 1; i <= maxDepth; i++) {
        currentTargetDepth = i;
        int score = searchAlphaBeta(board, i, lowerBound, upperBound, true);
    }
    

    //std::cout << "Score " << score << std::endl;
    std::cout << "Evaluated nodes: " << evaluatedNodes << std::endl;
    
    return bestMove;
}
int Search::searchAlphaBeta(Board &board, int depth, int alpha, int beta, bool maximizingPlayer)
{
    if (depth == 0) return evaluate(board);

    MoveList moveList;
    MoveGenerator::generateMoves(board, moveList);
    int score = 0;

    if (depth == currentTargetDepth) {
        for (int i = 0; i < moveList.counter; i++){
            if (moveList.moves[i].fromSq == bestMove.bestMove.fromSq && moveList.moves[i].toSq == bestMove.bestMove.toSq) {
                Move moveZero = moveList.moves[0];
                moveList.moves[0] = moveList.moves[i];
                moveList.moves[i] = moveZero;
            }
        }
    }

    if (maximizingPlayer) {
        score = -INFINITY;
        for (int i = 0; i < moveList.counter; i++) {
            Move move = moveList.moves[i];
            if (board.makeMove(move)) {
                score = searchAlphaBeta(board, depth - 1, alpha, beta, false);

                if (score >= beta) {
                    board.revertLastMove();
                    return beta;
                }

                if (score > alpha) {
                    alpha = score;
                    if (depth == currentTargetDepth) {
                        bestMove.bestMove = move;
                        bestMove.score = alpha;
                        bestMove.depth = depth;
                    }
                }
            }
            board.revertLastMove();
        }
        return alpha;
    } else {
        score = INFINITY;
        for (int i = 0; i < moveList.counter; i++) {
            Move move = moveList.moves[i];
            if (board.makeMove(move)) {
                score = searchAlphaBeta(board, depth - 1, alpha, beta, true);
                if (score <= alpha) {
                    board.revertLastMove();
                    return alpha;
                }

                if (score < beta) {
                    beta = score;
                }
            }
            board.revertLastMove();
        }

        return beta;
        
    }   

    return 0;
}

int Search::quinesence(Board board, int alpha, int beta,int depth)
{
    if(depth == 0) return evaluate(board);
    
    MoveList moveList;
    MoveList moveListReduced;
    MoveGenerator::generateMoves(board,moveList);
    for(int i = 0; i < moveList.counter; i++){
        if(moveList.moves[i].capture || moveList.moves[i].promotion){
            moveListReduced.moves[moveListReduced.counter++] = moveList.moves[i];
        }
    }
    if(moveListReduced.counter == 0){
        return evaluate(board);
    }

    int score = 0;
    for(int i = 0; i < moveListReduced.counter; i++){
        Move move = moveListReduced.moves[i];
        bool valid = board.makeMove(move);
        if(valid){            
            score = -quinesence(board,-beta,-alpha,depth-1);
        }

        if(score > alpha){
            alpha = score;
        }
        if(alpha >= beta){            
            return beta;
        }
        

        board.revertLastMove();               
    }
    return alpha;
}



bool compare(SortStruct a, SortStruct b)
{
    return a.score > b.score;
}

void Search::sortMoveList(Board board, MoveList &list)
{
    // Sort best move from last iteration first
    SortStruct* sortArray = new SortStruct[list.counter];
    for(int i = 0; i< list.counter; i++){
        SortStruct entry;
        entry.move = list.moves[i];
        if(equal(list.moves[i],bestMove.bestMove)){
            entry.score = 10000;
        } else if(entry.move.promotion) {
            entry.score = 1000;
        } else if(entry.move.capture){
            entry.score = 100;
        } else{
            entry.score = -100;
        }
        sortArray[i] = entry;
    }
    // MVV-LVA sorting

    std::sort(sortArray, sortArray+list.counter, compare);
    for(int i = 0; i< list.counter; i++){
        list.moves[i] = sortArray[i].move;
    }
}

int Search::evaluate(Board &board)
{
    evaluatedNodes++;
    int score = board.getPieceSquareScore();
    score += board.getMaterialScore();
    
    if(isBlackMaxPlayer){
        return score*=-1;
    }
    
                   
    //std::cout << score << std::endl;
    return score;
}

bool Search::equal(Move a, Move b)
{
    return (a.fromSq == b.fromSq &&
            a.toSq == b.toSq);
}
