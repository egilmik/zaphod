#include "search.h"
#include "perft.h"
#include "material.h"
#include <algorithm>

Score Search::search(Board board, int maxDepth)
{    
    int lowerBound = -20000;
    int upperBound = 20000;
    bool inIteration = true;
    currentTargetDepth = maxDepth;
    int score = searchAlphaBeta(board, currentTargetDepth, lowerBound, upperBound, true);

    //for(int i = 1; i <= maxDepth; i++){
    //    currentTargetDepth = i;
        //int score = negaMax(board,lowerBound,upperBound,i);
        
            
        /*while(inIteration){
            
            Score score = searchAlphaBeta(board,i, lowerBound, upperBound);
            if(score.score > upperBound){
                std::cout << "Fail high, depth: " << i << std::endl;
                upperBound += 50;
            } else if( score.score < lowerBound){
                lowerBound -= 50;
                std::cout << "Fail low, depth: " << i << std::endl;
            } else {
                bestMove = score;
                inIteration = false;
            }

        }
        
        if(i > 4){
            lowerBound = bestMove.score-200;
            upperBound = bestMove.score+200;
        }
        inIteration = true;
        

        std::cout << Perft::getNotation(bestMove.bestMove) << " Score: " << (double)bestMove.score/100.0 << " Depth: "<< bestMove.depth << std::endl;
        std::cout << "TT hits " << ttHits << std::endl;
        std::cout << "TT size " << transpositionMap.size() << std::endl;
        std::cout << "Evaluated nodes: " << evaluatedNodes << std::endl;
        //for(int i = 0; i < currentTargetDepth; i++){
        //    std::cout << Perft::getNotation(pvMoves[i]) << " ";
        //}
        std::cout << std::endl;
        std::cout << "Iteration done: " << i << std::endl;
        std::cout << std::endl;
    }
    */

    //std::cout << "Score " << score << std::endl;
    //std::cout << "Evaluated nodes: " << evaluatedNodes << std::endl;
    
    return bestMove;
}
int Search::searchAlphaBeta(Board board, int depth, int alpha, int beta, bool maximizingPlayer)
{
    if (depth == 0) return evaluate(board);

    MoveList moveList;
    MoveGenerator::generateMoves(board, moveList);
    int score = 0;

    if (maximizingPlayer) {
        score = -INFINITY;
        for (int i = 0; i < moveList.counter; i++) {
            Move move = moveList.moves[i];
            if (board.makeMove(move)) {
                score = searchAlphaBeta(board, depth - 1, alpha, beta, false);

                if (score >= beta) {
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


int Search::negaMax(Board board, int alpha, int beta, int depth)
{
    int score = 0;
    
    //if(depth == 0) return quinesence(board,-beta,-alpha,2);
    if(depth == 0) return evaluate(board);

    //int alphaOrginal = alpha;
    
    MoveList moveList;
    MoveGenerator::generateMoves(board,moveList);
    /*if (currentTargetDepth == depth) {
        sortMoveList(board,moveList);
    }

    */
    
    
    BitBoard key = board.getHashKey();
    /*
    std::unordered_map<BitBoard,TranspositionEntry>::iterator it = transpositionMap.find(key);
    if(it != transpositionMap.end() && it->second.depth >= depth){
        TEType entryType = it->second.type;
        if(entryType == TEType::exact){
            return it->second.score;
        } else if(entryType == TEType::lower){
            alpha = std::max(it->second.score,alpha);
        } else if(entryType == TEType::upper){
            beta = std::max(it->second.score,beta);
        }

        if (alpha >= beta){ 
            return it->second.score;
        }
    }
    */

    Move alphaMove;

    int validMoves = moveList.counter;
    
    for(int i = 0; i < moveList.counter; i++){
        Move move = moveList.moves[i];
        if(board.makeMove(move)){
            score = -negaMax(board,-beta,-alpha,depth-1);
            if(score >= beta){            
                return beta;
            }           

            if(score > alpha){
                alpha = score;
                alphaMove = move;
                pvMoves[currentTargetDepth-depth] = move;
            }
            
        } else {
            validMoves--;
        }
        board.revertLastMove();
    }

    if(validMoves == 0 && board.isSquareAttacked(board.getSideToMove()+BitBoardEnum::K,board.getOtherSide())){
        alpha = 3000;
    }
    /*
    //Replace if depth is higher
    it = transpositionMap.find(key);
    if(it == transpositionMap.end() || it->second.depth < depth){
        TranspositionEntry entry;
        entry.depth = depth;
        entry.score = alpha;
        if(alpha <= alphaOrginal){
            entry.type = TEType::upper;
        } else if(alpha >= beta){
            entry.type = TEType::lower;
        } else {
            entry.type = TEType::exact;
        }
        transpositionMap[key] = entry;
    }
    */
    
    if(depth == currentTargetDepth){
        bestMove.bestMove = alphaMove;
        bestMove.score = alpha;
        bestMove.depth = depth;
    }
    
     
    return alpha;

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
    /*
    if(board.getSideToMove() == BitBoardEnum::Black){
        return score*=-1;
    }
    */
                   
    //std::cout << score << std::endl;
    return score;
}

bool Search::equal(Move a, Move b)
{
    return (a.fromSq == b.fromSq &&
            a.toSq == b.toSq);
}
