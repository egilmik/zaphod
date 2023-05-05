#include "search.h"
#include "perft.h"
#include "material.h"
#include <algorithm>


Score Search::searchAlphaBeta(Board board, int depth)
{
    bestMove.depth = 0;
    bestMove.score = -100000;
    ttHits = 0;
    MoveList moveList;
    MoveGenerator::generateMoves(board,moveList);
    sortMoveList(board,moveList);
    int score = 0;
    int currentBestScore = -100000;
    for(int i = 0; i < moveList.counter; i++){
        Move move = moveList.moves[i];
        bool valid = board.makeMove(move);      

        if(valid){
            BitBoard key = ttable.generateKey(board);
            std::unordered_map<BitBoard,TranspositionEntry>::iterator it = ttable.transpositionMap.find(key);
            if(it != ttable.transpositionMap.end() && it->second.depth >= depth){
                TranspositionEntry entry = it->second;
                score = entry.score;
                ttHits++;
            } else {
                score = -negaMax(board,-1000000000,1000000000,depth);
                TranspositionEntry entry = {move,depth,score};
                ttable.transpositionMap[key] = entry;
            }
            
            if(score >= bestMove.score){
                bestMove.score = score;
                bestMove.depth = depth;
                bestMove.bestMove = move;
                
            }
        }

        board.revertLastMove();               
    }

    std::cout << Perft::getNotation(bestMove.bestMove) << " Score: " << bestMove.score << " Depth: "<< depth << std::endl;
    std::cout << "TT hits " << ttHits << std::endl;
    std::cout << "TT size " << ttable.transpositionMap.size() << std::endl;
    return bestMove;
    
}

int Search::negaMax(Board board, int alpha, int beta, int depth)
{
    
    if(depth == 0) return evaluate(board);
    
    MoveList moveList;
    MoveGenerator::generateMoves(board,moveList);
    pseudoLegalNodeCounter+= moveList.counter;
    int score = 0;
    for(int i = 0; i < moveList.counter; i++){
        Move move = moveList.moves[i];
        bool valid = board.makeMove(move);
        if(valid){
            BitBoard key = ttable.generateKey(board);
            std::unordered_map<BitBoard,TranspositionEntry>::iterator it = ttable.transpositionMap.find(key);
            if(it != ttable.transpositionMap.end() && it->second.depth >= depth){                
                score = it->second.score;
                ttHits++;
            } else {
                score = -negaMax(board,-beta,-alpha,depth-1);
                //TranspositionEntry entry = {move,depth,score};                
                //ttable.transpositionMap[key] = entry;
            }
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

        BitBoard key = ttable.generateKey(board);
        std::unordered_map<BitBoard,TranspositionEntry>::iterator it = ttable.transpositionMap.find(key);
        SortStruct entry;
        entry.move = list.moves[i];
        entry.score = -100000;
        if(it != ttable.transpositionMap.end()){                
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
