#include "search.h"
#include "perft.h"
#include "material.h"
#include <algorithm>
#include <chrono>

Score Search::search(Board &board, int maxDepth, int maxTime)
{    
    maxSearchTime = maxTime;
    maxQuinesenceDepthThisSearch = 0;
    startTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();;
    int lowerBound = -100000;
    int upperBound = 100000;
    stopSearch = false;
    bool inIteration = true;
    Score bestScore;

    auto start = std::chrono::high_resolution_clock::now();


    for (int i = 1; i <= maxDepth; i++) {
        currentTargetDepth = i;
        maxQuinesenceDepthThisSearch = 0;
        int score = negamax(board, i, lowerBound, upperBound);
        if (stopSearch) {
            break;
        }

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        int nps = (double)evaluatedNodes / ((double)duration.count() / (double)1000);

        std::cout << "info depth " << i << " seldepth " << i+maxQuinesenceDepthThisSearch << " score cp " << score << " nodes " << evaluatedNodes << " nps " << nps << " pv " << Perft::getNotation(bestMoveIteration.bestMove) << std::endl;
        currentFinishedDepth = i;
        bestScore = bestMoveIteration;
    }
 
    
    return bestScore;
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
            if (depth == currentTargetDepth) {
                bestMoveIteration.bestMove = it->second.bestMove;
                bestMoveIteration.score = alpha;
                bestMoveIteration.depth = depth;
            }
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
            if (depth == currentTargetDepth) {
                bestMoveIteration.bestMove = it->second.bestMove;
                bestMoveIteration.score = alpha;
                bestMoveIteration.depth = depth;
            }

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
        board.makeMove(move);
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
                bestMoveIteration.bestMove = move;
                bestMoveIteration.score = alpha;
                bestMoveIteration.depth = depth;
            }
        }

        board.revertLastMove();
    }

    if (validMoves == 0) {
        if (board.isSquareAttacked(board.getSideToMove() + BitBoardEnum::K, board.getOtherSide())) {
            // We are check mate
            alpha = -30000-(currentTargetDepth-depth);

            if (board.getSideToMove() == BitBoardEnum::Black) {
                alpha *= -1;
            }
        }
        else if (board.isSquareAttacked(board.getOtherSide() + BitBoardEnum::K, board.getSideToMove())) {
            // They are check mate
            alpha = -30000+ (currentTargetDepth - depth);

            if (board.getSideToMove() == BitBoardEnum::Black) {
                alpha *= -1;
            }
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
        if (alpha < beta && alpha > alphaOrginal) {
            transpositionMap[key] = { alphaMove, TEType::exact, depth, alpha};
        }
    }
    

    return alpha;
}



int Search::quinesence(Board &board, int alpha, int beta,int depth)
{

    int standPat = evaluate(board);

    if (maxQuinesenceDepthThisSearch < depth) {
        maxQuinesenceDepthThisSearch = depth;
    }

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

    if (depth > 5) {
        return standPat;
    }

    MoveList moveList;
    MoveList moveListReduced;
    MoveGenerator::generateMoves(board,moveList);
    for(int i = 0; i < moveList.counter; i++){
        if(board.getPieceOnSquare(moveList.moves[i].to()) != All || moveList.moves[i].getMoveType() == PROMOTION) {
            moveListReduced.moves[moveListReduced.counter++] = moveList.moves[i];
        }
    }

    //sortMoveList(board, moveListReduced);

    int score = 0;
    for(int i = 0; i < moveListReduced.counter; i++){
        Move move = moveListReduced.moves[i];
        bool valid = board.makeMove(move);
        score = -quinesence(board,-beta,-alpha,depth+1);

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
        } else if(entry.move.getMoveType() == PROMOTION) {
            entry.score = 1000;
        } else if(board.getPieceOnSquare(entry.move.to()) != All ){
            BitBoardEnum capturedPiece = board.getPieceOnSquare(entry.move.to());
            BitBoardEnum attacker = board.getPieceOnSquare(entry.move.from());
            if (entry.move.getMoveType() == EN_PASSANT) {
                capturedPiece = P;
            }
            int Mvv = Material::getMaterialScore(capturedPiece);
            int lva = Material::getMaterialScore(attacker);
            entry.score = 100 + (Mvv - lva)/100;
        } else{
            entry.score = 0;
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
    int score = 0;
    for (int i = 0; i < 64; i++) {
        BitBoardEnum piece = board.getPieceOnSquare(i);
        if (piece != All) {
            score += Material::getPieceSquareScore(piece, i, 1.0);
        }

    }

    //int score = board.getPieceSquareScore();


    score += board.getMaterialScore();
    //score += board.getMobilityDiff();

    if (board.getSideToMove() == BitBoardEnum::Black) {
        return score *= -1;
    }
    return score;
}


bool Search::equal(Move &a, Move &b)
{
    return (a.from() == b.from() &&
            a.to() == b.to());
}

MoveList Search::reconstructPV(Board& board, int depth)
{
    MoveList list;

    for (int i = 0; i < depth; i++) {
        BitBoard key = board.getHashKey();
        std::unordered_map<BitBoard, TranspositionEntry>::iterator it = transpositionMap.find(key);
        if (it != transpositionMap.end() && it->second.type == TEType::exact) {
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
