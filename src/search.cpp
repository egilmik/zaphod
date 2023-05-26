#include "search.h"
#include "perft.h"
#include "material.h"
#include <algorithm>


Score Search::searchAlphaBeta(Board board, int depth)
{
    bestMove.depth = 0;
    bestMove.score = -100000;
    evaluatedNodes = 0;
    ttHits = 0;
    int alpha = -1000000000;
    int beta = 1000000000;
    MoveList moveList;
    MoveGenerator::generateMoves(board,moveList);
    sortMoveList(board,moveList);
    int score = 0;
    int currentBestScore = -100000;
    for(int i = 0; i < moveList.counter; i++){
        Move move = moveList.moves[i];
        bool valid = board.makeMove(move);      

        if(valid){
            BitBoard key = board.generateHashKey();
            std::unordered_map<BitBoard,TranspositionEntry>::iterator it = transpositionMap.find(key);
            if(it != transpositionMap.end() && it->second.depth >= depth){
                TranspositionEntry entry = it->second;
                score = entry.score;
                ttHits++;
            } else {
                score = -negaMax(board,alpha,beta,depth);
                TranspositionEntry entry = {move,depth,score};
                transpositionMap[key] = entry;
            }
            
            if(score >= bestMove.score){
                bestMove.score = score;
                bestMove.depth = depth;
                bestMove.bestMove = move;
            }

            if(score >= beta){            
                return bestMove;
            }

            if(score > alpha){
                alpha = score;
            }

            
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
    
    MoveList moveList;
    MoveGenerator::generateMoves(board,moveList);
    int score = 0;
    for(int i = 0; i < moveList.counter; i++){
        Move move = moveList.moves[i];
        bool valid = board.makeMove(move);
        if(valid){
            BitBoard key = board.generateHashKey();
            BitBoard incKey = board.getHashKey();
            if(key != incKey){

                std::cout << Perft::getNotation(move) << std::endl;
                board.printBoard();
                int x = 0;
            }
            std::unordered_map<BitBoard,TranspositionEntry>::iterator it = transpositionMap.find(key);
            if(it != transpositionMap.end() && it->second.depth >= depth){                
                score = it->second.score;
                ttHits++;
            } else {
                score = -negaMax(board,-beta,-alpha,depth-1);
                
            }

            if(score >= beta){            
                 return beta;
            }
    
            if(score > alpha){
                alpha = score;
            }
            
        }
        
        

        board.revertLastMove();               
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

struct SortStruct {
    int score;
    Move move;
};

bool compare(SortStruct a, SortStruct b)
{
    return a.score > b.score;
}

void Search::sortMoveList(Board board, MoveList &list)
{
    SortStruct sortArray[list.counter];
    for(int i = 0; i< list.counter; i++){
        board.makeMove(list.moves[i]);

        BitBoard key = board.generateHashKey();
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
