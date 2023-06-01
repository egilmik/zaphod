#include "search.h"
#include "perft.h"
#include "material.h"
#include <algorithm>

Score Search::search(Board board, int maxDepth)
{
    Score bestScore;
    int lowerBound = -10000000;
    int upperBound = 10000000;
    bool inIteration = true;
    for(int i = 1; i <= maxDepth; i+=1){

        while(inIteration){
            Score score = searchAlphaBeta(board,i, lowerBound, upperBound);
            if(score.score > upperBound){
                std::cout << "Fail high, depth: " << i << std::endl;
                upperBound += 50;
            } else if( score.score < lowerBound){
                lowerBound -= 50;
                std::cout << "Fail low, depth: " << i << std::endl;
            } else {
                bestScore = score;
                inIteration = false;
            }

        }
        //
        //if(i > 4){
        //    lowerBound = bestScore.score-100;
        //    upperBound = bestScore.score+100;
        //}
        inIteration = true;
        std::cout << "Iteration done: " << i << std::endl;
        std::cout << std::endl;
    }
    
    return bestScore;
}

Score Search::searchAlphaBeta(Board board, int depth, int alpha, int beta)
{
    bestMove.depth = 0;
    bestMove.score = -100000;
    evaluatedNodes = 0;
    ttHits = 0;
    MoveList moveList;
    MoveGenerator::generateMoves(board,moveList);
    sortMoveList(board,moveList);
    int score = 0;
    for(int i = 0; i < moveList.counter; i++){
        Move move = moveList.moves[i];
        bool valid = board.makeMove(move);

        if(valid){            
            score = -negaMax(board,alpha,beta,depth);            
            if(score >= bestMove.score){
                bestMove.score = score;
                bestMove.depth = depth;
                bestMove.bestMove = move;
            }

            TranspositionEntry entry;
            entry.depth = depth;
            entry.score = score;
            entry.type = TEType::exact;           

            transpositionMap[board.getHashKey()] = entry;           
        }

        board.revertLastMove();               
    }

    std::cout << Perft::getNotation(bestMove.bestMove) << " Score: " << bestMove.score << " Depth: "<< depth << std::endl;
    std::cout << "TT hits " << ttHits << std::endl;
    std::cout << "TT size " << transpositionMap.size() << std::endl;
    std::cout << "Evaluated nodes: " << evaluatedNodes << std::endl;
    return bestMove;
    
}

int Search::negaMax(Board board, int alpha, int beta, int depth)
{
    
    //if(depth == 0) return quinesence(board,-beta,-alpha,2);
    if(depth == 0) return evaluate(board);

    int alphaOrginal = alpha;
    
    MoveList moveList;
    MoveGenerator::generateMoves(board,moveList);
    //sortMoveList(board,moveList);
    int score = 0;
    BitBoard key = board.getHashKey();

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
    
    for(int i = 0; i < moveList.counter; i++){
        Move move = moveList.moves[i];
        bool valid = board.makeMove(move);
        if(valid){            
            score = -negaMax(board,-beta,-alpha,depth-1);
            if(score >= beta){            
                return beta;
            }           

            if(score > alpha){
                alpha = score;
            }
            
        }
        board.revertLastMove();
    }

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
    SortStruct sortArray[list.counter];
    for(int i = 0; i< list.counter; i++){
        board.makeMove(list.moves[i]);

        BitBoard key = board.getHashKey();
        std::unordered_map<BitBoard,TranspositionEntry>::iterator it = transpositionMap.find(key);
        SortStruct entry;
        entry.move = list.moves[i];
        entry.score = -100000;
        if(it != transpositionMap.end()){                
            entry.score = it->second.score;
        }
        sortArray[i] = entry;
        board.revertLastMove();
    }
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

    if(board.getSideToMove() == BitBoardEnum::Black){
        return -score;
    }
                   
    //std::cout << score << std::endl;
    return score;
}
