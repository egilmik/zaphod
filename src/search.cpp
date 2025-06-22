#include "search.h"
#include "perft.h"
#include "material.h"
#include <algorithm>
#include <chrono>
#include "evaluation.h"

Score Search::search(Board &board, int maxDepth, int maxTime)
{   
    tt.clear();
    pawnTable.clear();
    
    maxSearchTime = maxTime;
    startTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();;
    stopSearch = false;
    evaluatedNodes = 0;
    pawnTTHits = 0;
    lmrHit = 0;
    lmrResearchHit = 0;
    bestMoveIteration.depth = 0;



    bool inIteration = true;
    Score bestScore;
    int lowerBound = -100000;
    int upperBound = 100000;

    auto start = std::chrono::high_resolution_clock::now();
    

    for (int i = 1; i <= maxDepth; i++) {
        currentTargetDepth = i;
        maxQuinesenceDepthThisSearch = 0;
        maxPlyThisIteration = 0;

        /////////////////////////////////
        // Limit quiesence the first 2 iterations,
        // to make sure we have a decent score fast. 
        // Make sure it does not make illegal moves due to missing best move.
        if (i < 3) {
            currentQuiesenceTargetDepth = 2;
        }
        else {
            currentQuiesenceTargetDepth = 10;
        }

        //Reset search stack check extension
        ss[0].checkExt = 0;
        int score = negamax(board, i, lowerBound, upperBound,0);
        if (stopSearch) {
            break;
        }

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        int nps = (double)evaluatedNodes / ((double)duration.count() / (double)1000);

        std::cout << "info depth " << i << " seldepth " << maxPlyThisIteration << " score cp " << score << " nodes " << evaluatedNodes << " nps " << nps << " pv " << Perft::getNotation(bestMoveIteration.bestMove) << std::endl;
        currentFinishedDepth = i;
        bestScore = bestMoveIteration;
    } 


    ////////////////////////////
    // We might have canceled early and do not have a valid move.
    // We pick one.....  Lets see how that goes
    ////////////////////////////
    if (bestScore.depth == 0) {
        MoveList list;
        MoveGenerator::generateMoves(board, list);
        // Lets try sorting to perhaps hit something in TT
        sortMoveList(board, list);
        bestScore = { 0,0, list.moves[0] }; 
    }
    
    return bestScore;
}



int Search::negamax(Board& board, int depth, int alpha, int beta, int ply)
{
    if (depth == 0) return quinesence(board, alpha, beta, 1,ply);
    BitBoard key = board.getHashKey();
    bool isRoot = ply == 0;


    // Check if max search time has been exhausted
    // Returns beta to prevent things going to shit
    if ((evaluatedNodes % 100) == 0 && isSearchStopped()) {
        return beta;
    }
    //////////////////////////
    // Has repeated 3-fold
    //////////////////////////
    if (board.hasPositionRepeated()) {
        alpha = 0;
        if(alpha > beta){
            return alpha;
        }
    }
    
    bool isTTValid = false;
    TTEntry* tte = tt.probe(key, isTTValid);

    if (isTTValid && tte->depth >= depth) {
        TType entryType = tte->type;
        if (entryType == EXACT) {
            exactHit++;
            if (isRoot) {
                bestMoveIteration.bestMove = tte->bestMove;
                bestMoveIteration.score = alpha;
                bestMoveIteration.depth = depth;
            }
            return tte->score;
        }/*
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
        */
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

        bool moveIsCapture = board.getPieceOnSquare(move.to()) != All;

        board.makeMove(move);
        int plyCheckExtension = ss[ply].checkExt;
        int extension = 0;

        ////////////
        // Check extension
        ////////////
        BitBoard kingBB = board.getBitboard(board.getSideToMove() + BitBoardEnum::K);
        if (board.isSquareAttacked(kingBB, board.getOtherSide()) && plyCheckExtension < 3 && depth > 1) {
           extension++;
        }

        ss[ply + 1].checkExt = plyCheckExtension + extension;
        


        
        if (i < 4 || depth < 4 || extension > 0 || moveIsCapture) {
            score = -negamax(board, depth - 1 + extension, -beta, -alpha, ply + 1);
        }
        else {
            ////////////
            // LMR
            ////////////
            score = -negamax(board, depth - 2, -(alpha + 1), -alpha, ply + 1);
            lmrHit++;

            if (score > alpha) {
                lmrResearchHit++;
                score = -negamax(board, depth - 1, -(alpha + 1), -alpha, ply + 1);
                if (score > alpha && score < beta) {
                    // Full re-search
                    score = -negamax(board, depth - 1, -beta, -alpha, ply + 1);
                }
            } 
        }



        
        board.revertLastMove();

        if (score > alpha) {
            alpha = score;
            alphaMove = move;
            if (isRoot) {
                bestMoveIteration.bestMove = move;
                bestMoveIteration.score = alpha;
                bestMoveIteration.depth = depth;
            }
        }

        if (score >= beta) {
            
            /*if (it == transpositionMap.end() || it->second.depth < depth) {
                transpositionMap[key] = { move, TEType::lower, depth, beta };
            }*/
                
            break;
        }

        

    }

    if (validMoves == 0) {

        BitBoard kingBB = board.getBitboard(board.getSideToMove() + BitBoardEnum::K);
        if (board.isSquareAttacked(kingBB, board.getOtherSide())) {
            // We are check mate
            alpha = -300000+(currentTargetDepth-depth);
        }
        else {
            // Stalemate
            alpha = 0;
        }
    }

    
    //Replace if depth is higher
    if (!isTTValid || (isTTValid && tte->depth < depth)) {
        if (alpha <= alphaOrginal) {
            tt.put(key, alpha, depth, alphaMove, UPPER);
        }
        else if (alpha >= beta) {
            tt.put(key, alpha, depth, alphaMove, LOWER);
        } 
        if (alpha < beta && alpha > alphaOrginal) {
            tt.put(key, alpha, depth, alphaMove, EXACT);
        }
    }
    

    return alpha;
}



int Search::quinesence(Board &board, int alpha, int beta,int depth, int ply)
{

    //////////////////////////
    // Has repeated 3-fold
    //////////////////////////
    if (board.hasPositionRepeated()) {
        return 0;
    }

    int standPat = evaluate(board);
    
    if (maxQuinesenceDepthThisSearch < depth) {
        maxQuinesenceDepthThisSearch = depth;
    }

    if (maxPlyThisIteration < ply) {
        maxPlyThisIteration = ply;
    }

    // Check if max search time has been exhausted
    // Returns beta to prevent things going to shit
    if ((evaluatedNodes % 1000) > 0 && isSearchStopped()) {
        return beta;
    }

    
    if (standPat >= beta) {
        return beta;
    } else if(alpha < standPat) {
        alpha = standPat;
    }

    if (depth >= currentQuiesenceTargetDepth) {
        return standPat;
    }

    MoveList moveList;
    MoveList moveListReduced;
    MoveGenerator::generateMoves(board,moveList);
    for(int i = 0; i < moveList.counter; i++){
        if(moveList.checkers != 0 || board.getPieceOnSquare(moveList.moves[i].to()) != All || moveList.moves[i].getMoveType() == PROMOTION) {
            moveListReduced.moves[moveListReduced.counter++] = moveList.moves[i];
        }
    }

    //sortMoveList(board, moveListReduced);

    int score = 0;
    for(int i = 0; i < moveListReduced.counter; i++){
        Move move = moveListReduced.moves[i];
        bool valid = board.makeMove(move);
        score = -quinesence(board,-beta,-alpha,depth+1, ply+1);

        if(score > alpha){
            alpha = score;
        }
        if(alpha >= beta){      
            board.revertLastMove();
            return beta;
        }
        

        board.revertLastMove();               
    }

    if (moveList.counter == 0) {
        BitBoard kingBB = board.getBitboard(board.getSideToMove() + BitBoardEnum::K);
        if (board.isSquareAttacked(kingBB, board.getOtherSide())) {
            // We are check mate
            alpha = -300000 + (currentTargetDepth - depth);
        }
        else {
            // Stalemate
            alpha = 0;
        }
    }

    return alpha;
}



bool compare(SortStruct a, SortStruct b)
{
    return a.score > b.score;
}

void Search::sortMoveList(Board &board, MoveList &list)
{

    bool isTTValid = false;
    TTEntry* tte = tt.probe(board.getHashKey(), isTTValid);
    
    SortStruct sortArray[256];
    for(int i = 0; i< list.counter; i++){
        SortStruct entry;
        entry.move = list.moves[i];
        if(isTTValid && equal(list.moves[i], tte->bestMove)){
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
    int mgScore = 0;
    int egScore = 0;
    int gamePhase = 0;
    int materialScore = 0;
    BitBoard allPieces = board.getBitboard(All);
    int square = 0;
    while (allPieces) {
        square = board.popLsb(allPieces);
        BitBoardEnum piece = board.getPieceOnSquare(square);
        mgScore += Material::getPieceSquareScoreMG(piece, square);
        egScore += Material::getPieceSquareScoreEG(piece, square);
        gamePhase += Material::gamePhaseArray[piece];
        materialScore += Material::materialScoreArray[piece];
    }

    //Pesto gamephase handling
    int mgPhase = gamePhase;
    if (mgPhase > 24) mgPhase = 24; /* in case of early promotion */
    int egPhase = 24 - mgPhase;
    int psqt = (mgScore * mgPhase + egScore * egPhase) / 24;
    int score = materialScore+psqt;
    score += evaluatePawns(board);
    score += Evaluation::evaluatePiecePairs(board);


    if (board.getSideToMove() == BitBoardEnum::Black) {
        return score *= -1;
    }
    return score;
}

int Search::evaluatePawns(Board& board) {
    uint64_t hash = board.getPawnHashKey();
    bool isValid = false;
    int score = 0;
    TTEntry* entry = pawnTable.probe(hash, isValid);

    if (!isValid) {
        score = 0;
        score = Evaluation::evaluatePassedPawn(board, White);
        score += Evaluation::evaluatePassedPawn(board, Black);
        score += Evaluation::evaluatePawnShield(board);
        pawnTable.put(hash, score);
        return score;
    }
    else
    {
        pawnTTHits++;
    }
    
    return entry->score;
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
        bool isTTValid = false;
        TTEntry* tte = tt.probe(board.getHashKey(), isTTValid);

        if (isTTValid && tte->type == EXACT) {
            board.makeMove(tte->bestMove);
            list.moves[list.counter++] = tte->bestMove;
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
