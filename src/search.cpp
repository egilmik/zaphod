#include "search.h"
#include "perft.h"
#include "material.h"
#include <algorithm>

Score Search::search(Board &board, int maxDepth)
{    
    int lowerBound = -20000;
    int upperBound = 20000;
    bool inIteration = true;

    for (int i = 1; i <= maxDepth; i++) {
        currentTargetDepth = i;
        int score = negamax(board, i, lowerBound, upperBound);
    }
 
    
    return bestMove;
}

int Search::negamax(Board& board, int depth, int alpha, int beta)
{
    //if (depth == 0) return quinesence(board, -beta, -alpha,1);

    if (depth == 0) return evaluate(board);

    MoveList moveList;
    MoveGenerator::generateMoves(board, moveList);
    int score = 0;

    int alphaOrginal = alpha;
    Move alphaMove{};
    
    sortMoveList(board, moveList);
    
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

    if (alpha > alphaOrginal) {

        BitBoard key = board.getHashKey();
        std::unordered_map<BitBoard, TranspositionEntry>::iterator it = transpositionMap.find(key);
        
        if (it == transpositionMap.end() || it->second.depth < depth) {
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

    int standPat = evaluate(board);
    
    if (standPat >= beta) {
        return beta;
    } else if(alpha < standPat) {
        alpha = standPat;
    }

    if (depth > 5) {
        return standPat;
    }

    MoveList moveList;
    MoveList moveListReduced;
    MoveGenerator::generateMoves(board,moveList);
    for(int i = 0; i < moveList.counter; i++){
        if(moveList.moves[i].capture /* || moveList.moves[i].promotion != BitBoardEnum::All*/) {
            moveListReduced.moves[moveListReduced.counter++] = moveList.moves[i];
        }
    }

    int score = 0;
    for(int i = 0; i < moveListReduced.counter; i++){
        Move move = moveListReduced.moves[i];
        bool valid = board.makeMove(move);
        if(valid){            
            score = -quinesence(board,-beta,-alpha,depth++);
        }

        if(score > alpha){
            alpha = score;
        }
        if(alpha >= beta){      
            board.revertLastMove();
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

void Search::sortMoveList(Board &board, MoveList &list)
{
    BitBoard key = board.getHashKey();
    std::unordered_map<BitBoard, TranspositionEntry>::iterator it = transpositionMap.find(key);

    SortStruct sortArray[256];
    for(int i = 0; i< list.counter; i++){
        SortStruct entry;
        entry.move = list.moves[i];
        if(it != transpositionMap.end() && equal(list.moves[i],it->second.bestMove)){
            entry.score = 10000;
        } else if(entry.move.promotion != BitBoardEnum::All) {
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

bool Search::equal(Move &a, Move &b)
{
    return (a.fromSq == b.fromSq &&
            a.toSq == b.toSq);
}
