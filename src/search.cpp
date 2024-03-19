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
        int score = negamax(board, i, lowerBound, upperBound);
    }
    

    //std::cout << "Score " << score << std::endl;
    std::cout << "Evaluated nodes: " << evaluatedNodes << std::endl;
    
    return bestMove;
}

int Search::negamax(Board& board, int depth, int alpha, int beta)
{
    if (depth == 0) return evaluate(board);

    MoveList moveList;
    MoveGenerator::generateMoves(board, moveList);
    int score = 0;

    int alphaOrginal = alpha;
    Move alphaMove;
    BitBoard key = board.getHashKey();

    std::unordered_map<BitBoard, TranspositionEntry>::iterator it = transpositionMap.find(key);
    /*if (it != transpositionMap.end() && it->second.depth >= depth) {
        TEType entryType = it->second.type;
        if (entryType == TEType::exact) {
            return it->second.score;
        }
        else if (entryType == TEType::lower) {
            alpha = std::max(it->second.score, alpha);
        }
        else if (entryType == TEType::upper) {
            beta = std::max(it->second.score, beta);
        }

        if (alpha >= beta) {
            return it->second.score;
        }
    }
    */

    if(it != transpositionMap.end() && it->second.type == TEType::exact){
        for (int i = 0; i < moveList.counter; i++) {
            if (moveList.moves[i].fromSq == it->second.bestMove.fromSq && moveList.moves[i].toSq == it->second.bestMove.toSq) {
                Move moveZero = moveList.moves[0];
                moveList.moves[0] = moveList.moves[i];
                moveList.moves[i] = moveZero;
            }
        }
    }
    
    int validMoves = moveList.counter;

    for (int i = 0; i < moveList.counter; i++) {
        Move move = moveList.moves[i];
        if (board.makeMove(move)) {
            score = -negamax(board, depth - 1, -beta, -alpha);
            if (score >= beta) {
                board.revertLastMove();
                return beta;
            }

            if (score > alpha) {
                alpha = score;
                alphaMove = move;
                if (depth == currentTargetDepth) {
                    bestMove.bestMove = move;
                    bestMove.score = alpha;
                    bestMove.depth = depth;
                }
            }

        }
        else {
            validMoves--;
        }
        board.revertLastMove();
    }

    if (validMoves == 0 && board.isSquareAttacked(board.getSideToMove() + BitBoardEnum::K, board.getOtherSide())) {
        alpha = 3000;
    }

    //Replace if depth is higher
    it = transpositionMap.find(key);
    if (it == transpositionMap.end() || it->second.depth <= depth) {
        
        if (alpha <= alphaOrginal) {
            //entry.type = TEType::upper;
        }
        else if (alpha >= beta) {
            //entry.type = TEType::lower;
        }
        else {
            TranspositionEntry entry;
            entry.depth = depth;
            entry.score = alpha;
            entry.bestMove = alphaMove;
            entry.type = TEType::exact;
            transpositionMap[key] = entry;
        }
    }

    return alpha;
}

int Search::quinesence(Board &board, int alpha, int beta,int depth)
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

    if (board.getSideToMove() == BitBoardEnum::Black) {
        return score *= -1;
    }
    return score;
}

bool Search::equal(Move a, Move b)
{
    return (a.fromSq == b.fromSq &&
            a.toSq == b.toSq);
}
