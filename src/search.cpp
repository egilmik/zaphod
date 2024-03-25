#include "search.h"
#include "perft.h"
#include "material.h"
#include <algorithm>
#include <chrono>

Score Search::search(Board &board, int maxDepth, int maxTime)
{    
    maxSearchTime = maxTime;
    startTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();;
    int lowerBound = -20000;
    int upperBound = 20000;
    stopSearch = false;
    bool inIteration = true;

    for (int i = 1; i <= maxDepth; i++) {
        currentTargetDepth = i;
        int score = negamax(board, i, lowerBound, upperBound);
        if (stopSearch) {
            break;
        }
        std::cout << "info depth " << i << " score cp " << score << " pv " << Perft::getNotation(bestMove.bestMove) << std::endl;
        currentFinishedDepth = i;
    }
 
    
    return bestMove;
}

int Search::negamax(Board& board, int depth, int alpha, int beta)
{
    if (depth == 0) return quinesence(board, alpha, beta, 1);
    BitBoard key = board.getHashKey();


    // Check if max search time has been exhausted
    // Returns beta to prevent things going to shit
    if (evaluatedNodes % 1000 && isSearchStopped()) {
        return beta;
    }

    std::unordered_map<BitBoard, TranspositionEntry>::iterator it = transpositionMap.find(key);
    if (it != transpositionMap.end() && it->second.depth >= depth) {
        TEType entryType = it->second.type;
        if (entryType == TEType::exact) {
            exactHit++;
            return it->second.score;
        }
        else if (entryType == TEType::lower) {
            lowerBoundHit++;
            alpha = std::max(it->second.score, alpha);
        }
        else if (entryType == TEType::upper) {
            upperBoundHit++;
            beta = std::max(it->second.score, beta);
        }

        if (alpha >= beta) {
            return it->second.score;
        }
    }

    

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
                if (it == transpositionMap.end() || it->second.depth < depth) {
                    transpositionMap[key] = { move, TEType::lower, depth, beta };
                }
                
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

        if (board.getSideToMove() == BitBoardEnum::Black) {
            alpha *= -1;
        }
    }

    //Replace if depth is higher
    if (it == transpositionMap.end() || it->second.depth < depth) {
        if (alpha <= alphaOrginal) {
            transpositionMap[key] = { alphaMove, TEType::upper, depth, alpha };
        }
        else if (alpha >= beta) {
            transpositionMap[key] = { alphaMove, TEType::lower, depth, alpha };
        }
        else {
            transpositionMap[key] = { alphaMove, TEType::exact, depth, alpha};
        }
    }

    return alpha;
}



int Search::quinesence(Board &board, int alpha, int beta,int depth)
{

    int standPat = evaluate(board);

    // Check if max search time has been exhausted
    // Returns beta to prevent things going to shit
    if (evaluatedNodes % 1000 && isSearchStopped()) {
        return beta;
    }

    
    if (standPat >= beta) {
        return beta;
    } else if(alpha < standPat) {
        alpha = standPat;
    }

    if (depth > 50) {
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
    
    std::unordered_map<BitBoard, TranspositionEntry>::iterator it = transpositionMap.find(board.getHashKey());
    SortStruct sortArray[256];
    for(int i = 0; i< list.counter; i++){
        SortStruct entry;
        entry.move = list.moves[i];
        if(it != transpositionMap.end() && equal(list.moves[i], it->second.bestMove)){
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

MoveList Search::reconstructPV(Board& board, int depth)
{
    MoveList list;

    for (int i = 0; i < depth; i++) {
        BitBoard key = board.getHashKey();
        std::unordered_map<BitBoard, TranspositionEntry>::iterator it = transpositionMap.find(key);
        if (it != transpositionMap.end()) {
            board.makeMove(it->second.bestMove);
            list.moves[list.counter++] = it->second.bestMove;
        }
        else {
            return list;
        }

    }

    return list;
}

bool Search::isSearchStopped()
{
    auto end = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    auto diff = end - startTime;
    if (diff > maxSearchTime) {
        stopSearch = true;
    }
    return stopSearch;
}
