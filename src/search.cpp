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
    BitBoard key = board.getHashKey();
    for(int i = 0; i < moveList.counter; i++){
        Move move = moveList.moves[i];
        bool valid = board.makeMove(move);

        if(valid){            
            score = -negaMax(board,alpha,beta,depth);            
            if(score >= bestMove.score){
                bestMove.score = score;
                bestMove.depth = depth;
                bestMove.bestMove = move;
                TranspositionEntry entry = {move,depth,score};
                transpositionMap[key] = entry;
            }

            if(score >= beta){            
                TranspositionEntry entry = {move,depth,score};
                transpositionMap[key] = entry;
                return bestMove;
            }

            if(score > alpha){
                alpha = score;
                TranspositionEntry entry = {move,depth,score};
                transpositionMap[key] = entry;
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
    sortMoveList(board,moveList);
    int score = 0;
    BitBoard key = board.getHashKey();
    for(int i = 0; i < moveList.counter; i++){
        Move move = moveList.moves[i];
        bool valid = board.makeMove(move);
        if(valid){            
            score = -negaMax(board,-beta,-alpha,depth-1);
            if(score >= beta){            
                TranspositionEntry entry = {move,depth,score};
                transpositionMap[key] = entry;
                return beta;
            }
    
            if(score > alpha){
                alpha = score;
                TranspositionEntry entry = {move,depth,score};
                transpositionMap[key] = entry;
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



bool compare(SortStruct a, SortStruct b)
{
    return a.score > b.score;
}

void Search::sortMoveList(Board board, MoveList &list)
{
    BitBoard key = board.getHashKey();
    std::unordered_map<BitBoard,TranspositionEntry>::iterator it = transpositionMap.find(key);
    SortStruct sortArray[list.counter];

    //Init sort list
    for(int i = 0; i < list.counter; i++){
        SortStruct entry;
            entry.move = list.moves[i];
            entry.score = -100000;
            sortArray[i] = entry;
    }

    // Find best move in Transposition table
    if(it != transpositionMap.end()){
        Move bestMove = it->second.bestMove;
        for(int i = 0; i< list.counter; i++){        
            Move move = list.moves[i];
            /*if((move.capture && bestMove.capture) &&
                (move.castling && bestMove.castling) &&
                (move.doublePawnPush && bestMove.doublePawnPush) &&
                (move.enpassant && bestMove.enpassant) &&
                (move.fromSq && bestMove.fromSq) &&
                (move.piece && bestMove.piece) &&
                (move.promotion && bestMove.promotion) &&
                (move.toSq && bestMove.toSq)){
                    sortArray[i].score = it->second.score;
                    break;
                }
            */
           if((move.fromSq && bestMove.fromSq) &&
                (move.piece && bestMove.piece) &&
                (move.toSq && bestMove.toSq)){
                    sortArray[i].score = it->second.score;
                    break;
                }
        }
    }

    // Sort....
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
