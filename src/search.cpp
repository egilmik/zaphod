#include "search.h"
#include "perft.h"
#include "material.h"


Move Search::searchAlphaBeta(Board board, int depth)
{
    targetDepth = depth;
    bestMove.depth = 0;
    bestMove.score = -100000;
    ttHits = 0;
    MoveList moveList;
    MoveGenerator::generateMoves(board,moveList);
    int score = 0;
    int currentBestScore = -100000;
    for(int i = 0; i < moveList.counter; i++){
        Move move = moveList.moves[i];
        bool valid = board.makeMove(move);      

        if(valid){
            BitBoard key = ttable.generateKey(board);
            std::unordered_map<BitBoard,TranspositionEntry>::iterator it = ttable.transpositionMap.find(key);
            if(it != ttable.transpositionMap.end() && it->second.depth >= targetDepth){
                TranspositionEntry entry = it->second;
                score = entry.score;
                ttHits++;
            } else {
                score = -negaMax(board,-1000000000,1000000000,0);
                TranspositionEntry entry = {move,targetDepth,score};
                ttable.transpositionMap[key] = entry;
            }
            
            if(score >= bestMove.score){
                bestMove.score = score;
                bestMove.depth = depth;
                bestMove.moves.moves[0] = move;
                bestMove.moves.counter = 1;
            }
        }

        board.revertLastMove();               
    }

    if(bestMove.score <= -100000){
        Move move;
        move.fromSq = 0;
        move.toSq = 0;
        return move;
    }
    std::cout << Perft::getNotation(bestMove.moves.moves[0]) << " Score: " << bestMove.score << " Depth: "<< targetDepth << std::endl;
    std::cout << "TT hits " << ttHits << std::endl;
    //for(int i = 0; i < depth; i++){
    //    std::cout << Perft::getNotation(bestMove.moves.moves[i]) << std::endl;
    //}   

    return bestMove.moves.moves[0];
    
}

int Search::negaMax(Board board, int alpha, int beta, int depth)
{
    depth +=1;
    if(targetDepth == depth) return evaluate(board);
    
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
            if(it != ttable.transpositionMap.end() && it->second.depth >= targetDepth-depth){                
                score = it->second.score;
                ttHits++;
            } else {
                score = -negaMax(board,-beta,-alpha,depth);
                TranspositionEntry entry = {move,targetDepth-depth,score};                
                ttable.transpositionMap[key] = entry;
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
