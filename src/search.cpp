#include "search.h"
#include "perft/perft.h"
#include "material.h"
#include <algorithm>
#include <chrono>
#include <cassert>
#include "tools/fentools.h"
#include "params.h"
#include "tools.h"
#include "see.h"
#include <vector>

using namespace zaphod::params;

Search::Search() {
}

Score Search::search(Board &board, SearchLimits lim)
{   
    startTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();

    for (int i = 0; i < MAXPLY; i++) {
        ss[i].checkExt = 0;
        ss[i].isNullMove = false;
        ss[i].killerMove[0] = 0;
        ss[i].killerMove[1] = 0;
        ss[i].staticEval = 0;
	ss[i].movedPiece = All;
	ss[i].move = 0;
    }

    history.age();
    
    this->limits = lim;

    if (limits.timeLimit > 0) {
        maxSearchTime = limits.timeLimit;
    }
    else {
        maxSearchTime = std::numeric_limits<int>::max();
    }

    int maxDepth = MAXPLY;
    if (limits.depthLimit > 0) {
        maxDepth = limits.depthLimit;
    }
    
    
    
    stopSearch = false;
    evaluatedNodes = 0;
    pawnTTHits = 0;
    lmrHit = 0;
    lmrResearchHit = 0;
    exactHit = 0;
    upperBoundHit = 0;
    lowerBoundHit = 0;
    qsearchTTHit = 0;
    aspirationHighResearchHit = 0;
    aspirationLowResearchHit = 0;
    reverseFutilityPruningHit = 0;
    futilityPruningHit = 0;
    nullMoveHit = 0;
    razoringEntryHit = 0;
    razoringReturnHit = 0;
    qsearchFutilityPruningHit = 0;
    qsearchMoveCounterPruningHit = 0;

    bestMoveIteration.depth = 0;
    bestMoveIteration.score = 0;
    bestMoveIteration.bestMove = 0;



    bool inIteration = true;
    Score bestScore{};
    constexpr int lowerBound = -std::numeric_limits<int>::max();
    constexpr int upperBound = std::numeric_limits<int>::max();
    int low = lowerBound;
    int upper = upperBound;

    auto start = std::chrono::high_resolution_clock::now();
    

    for (int i = 1; i <= maxDepth; i++) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);

        // Early exit if we have used half the time.  Not doing a another iteration
        if (maxSearchTime / 2 < duration.count()) {
            break;
        }

        int previousScore = i > 1 ? bestScore.score : 0;
        if (i > 4) {
            int aspiration = 20 + i * 5;
            low = previousScore - aspiration;
            upper = previousScore + aspiration;
        }

        currentTargetDepth = i;
        maxQuinesenceDepthThisSearch = 0;
        maxPlyThisIteration = 0;

        //Reset search stack check extension
        ss[0].checkExt = 0;
        int score = negamax(board, i, low, upper,0,true);
        if (score <= low) {
            aspirationLowResearchHit++;
            score = negamax(board, i, lowerBound, upper, 0, true);
        }
        else if (score >= upper) {
            aspirationHighResearchHit++;
            score = negamax(board, i, low, upperBound, 0, true);
        }

        if (stopSearch) {
            break;
        }

        auto stop = std::chrono::high_resolution_clock::now();
        auto npsDuration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
        int nps = (double)evaluatedNodes / ((double)npsDuration.count() / (double)1000000);


        std::string scoreString = " score cp " + std::to_string(score);
        if (score > MATESCORE - MAXPLY) {
            scoreString = " score mate " + std::to_string((MATESCORE - score));
        }
        else if (score < -(MATESCORE - MAXPLY)) {
            scoreString = " score mate " + std::to_string((-(MATESCORE + score)));
        }

        if (printInfo) {
            std::cout << "info depth " << i << " seldepth " << maxPlyThisIteration << scoreString << " nodes " << evaluatedNodes << " nps " << nps << " pv " << Perft::getNotation(bestMoveIteration.bestMove) << std::endl;
        }
        currentFinishedDepth = i;
        bestScore = bestMoveIteration;

        if(isSearchStoppedSoft()){
            break;
        }
    } 

    tt.age();

    ////////////////////////////
    // We might have canceled early and do not have a valid move.
    // We pick one.....  Lets see how that goes
    ////////////////////////////
    if (bestMoveIteration.bestMove) {
        bestScore = bestMoveIteration;
    } else {
        MoveList list;
        MoveGenerator::generateMoves(board, list);
        // Lets try sorting to perhaps hit something in TT
        //sortMoveList(board, list,0,0);

        std::cout << "info string search ended with choosing random move, nodes: " << evaluatedNodes << " Max ply:" << maxPlyThisIteration << " Current target depth: " << currentTargetDepth << " Qsearch depth: " << maxQuinesenceDepthThisSearch << std::endl;
        bestScore = { 0,0, list.moves[0] }; 
    }
    
    return bestScore;
}



int Search::negamax(Board& board, int depth, int alpha, int beta, int ply, bool pvNode)
{
    assert(ply <= MAXPLY);
    
    
    BitBoard key = board.getHashKey();    
    bool isRoot = ply == 0;
    int alphaOrginal = alpha;
    bool improving = false;
    int bestScore = -MATESCORE;


    // Check if max search time has been exhausted
    // Returns beta to prevent things going to shit
    if (isSearchStopped()) {
        return beta;
    }
    //////////////////////////
    // Has repeated 3-fold
    //////////////////////////
    if (board.hasPositionRepeated() || board.getHalfMoveClock() >= 100) {
        return drawScore();
    }

    if (ply >= MAXPLY) {
        return evaluate(board);
    }

    // Mate distance pruning
    alpha = std::max(alpha, -MATESCORE+ply);
    beta = std::min((int)beta, MATESCORE-ply);
    if (alpha >= beta) {
        return alpha;
    }


    if (depth <= 0) return quinesence(board, alpha, beta, 1, ply, pvNode);
    
    auto tte = tt.probe(key);    
    if (tte.type != TType::NO_TYPE) tte.score = scoreFromTT(tte.score, ply);

    if (!pvNode && tte.depth >= depth) {
        
        if (tte.type == EXACT) {
            exactHit++;
            return tte.score;
        }
        else if (tte.type == LOWER && tte.score >= beta) {
            lowerBoundHit++;
            return tte.score;
        }
        else if (tte.type == UPPER && tte.score <= alpha) {
            upperBoundHit++;
            return tte.score;
        }
    }

    bool ttHit = tte.type != TType::NO_TYPE;

    

    int score = 0;

    
    Move alphaMove{};
    

    
    bool inCheck = board.getCheckers() > 0;

    if (inCheck) {
        ss[ply].staticEval = -MATESCORE - 1;
    }
    else if (ttHit && tte.staticEval != (-MATESCORE - 1)) {
        ss[ply].staticEval = tte.staticEval;
    }
    else {
        ss[ply].staticEval = evaluate(board);
    }


    if (ply >= 4 && !inCheck) {
        improving = (ss[ply].staticEval > ss[ply - 2].staticEval && ss[ply - 2].staticEval > ss[ply - 4].staticEval);
    }

    ////////////
    // Razoring
    ////////////
    if (!isRoot && !ttHit && depth <= 4 && ss[ply].staticEval < (alpha - razoringMargin() *depth) ) {
        razoringEntryHit++;

        int value = quinesence(board, alpha-1, alpha, 0, ply, false);
        if (value < alpha && std::abs(value) < 20000) {
            razoringReturnHit++;
            return value;
        }
    }
    
    
    ////////////
    // Reverse futility pruning
    ////////////
    int futilityMargin =(1+ depth) * rfpLinear();
    //futilityMargin += rfpQuadratic()*depth*depth;
    futilityMargin -= rfpImproving()*improving;
    if (!pvNode && !inCheck && depth <= 6 && (ss[ply].staticEval - futilityMargin >= beta) && ss[ply].staticEval < MATESCORE-MAXPLY) {
        reverseFutilityPruningHit++;
        return (ss[ply].staticEval+beta)/2;
    }

    /*
    if (!pvNode && !inCheck && depth <= 3 && (ss[ply].staticEval - futilityMargin[depth]) >= beta && ss[ply].staticEval >= beta) {
        return (2 * beta + ss[ply].staticEval) / 3;
    }
    */
    
    ////////////
    // Null move pruning
    ////////////
    if (!pvNode && !inCheck  && ss[ply].staticEval >= beta && depth >= 3 && !isRoot && !ss[ply - 1].isNullMove) {
        if(board.getNonPawnMaterial(board.getSideToMove()) > 0 ){
            int R = 3 + (depth >= 6) + improving;
            //R = std::clamp(R, 2, 4);
            board.makeNullMove();
            ss[ply].isNullMove = true;
	    ss[ply].movedPiece = All;
            int nullScore = -negamax(board, depth - 1 - R, -beta, -beta + 1, ply + 1,false);
            board.revertNullMove();
	    ss[ply].isNullMove = false;
            if (nullScore >= beta) {
                nullMoveHit++;
                return nullScore;
            }
        }
    }

    MoveGenerator& moveGen = moveGenStack[ply];
    std::vector<Move> failLowQuiet;
    std::vector<Move> failLowNoisy;

    History::ContSlice* conts[History::CONT_PLIES] = {};
    for(int i = 0; i < History::CONT_PLIES; i++){
	int prev = ply - History::contOffset[i];
	if(prev >= 0 && ss[prev].movedPiece != All){
		conts[i] = history.contSlice(BitBoardEnum(ss[prev].movedPiece), ss[prev].move.to());
	} else if(prev >= 0 && ss[prev].isNullMove){
		conts[i] = history.contSlice(BitBoardEnum(P),0);
	}
    }

    moveGen.init(board, ttHit ? tte.move : Move{}, false, ss[ply].killerMove, &history, conts);
    int moveCounter = 0;
    Move move;
    while ((move = moveGen.next())) {
        
        bool isPromo = move.getMoveType() == PROMOTION;
        bool isCapture = board.getPieceOnSquare(move.to()) != All;
        bool isNoisy = isCapture || isPromo;
        int plyCheckExtension = ss[ply].checkExt;
        int extension = 0;
        bool firstMove = moveCounter == 0;
        bool givesCheck = false;
		BitBoard threats = board.getThreats();
		BitBoardEnum stm = board.getSideToMove();
        /*
        if (!isRoot && !pvNode && !inCheck && board.getNonPawnMaterial(board.getSideToMove()) > 0 && bestMoveIteration.score > -10000) {
            if (isCapture) {
                int capturedValue = Material::pieceMaterialScoreArray[board.getPieceOnSquare(move.to())];
                if (eval + capturedValue + 300 < alpha && depth < 8) {
                    continue;
                }
            }
            else {

            }
        }
        */

        ////////////
        // Move loop pruning
        ////////////
        if (!isRoot && board.getNonPawnMaterial(stm) > 0 && bestScore > -MATE_IN_MAX) {
                        
            ////////////
            // Futility pruning
            ////////////
            /*
            constexpr int FpDepth = 5;
            constexpr int FpMult = 148;
            if (!pvNode && !inCheck && depth <= FpDepth && ss[ply].staticEval +18 + FpMult * depth <= alpha) {
                futilityPruningHit++;
                continue;
            }
            */
            
            ////////////
            // Late move pruning
            ////////////
            int lmp = ((4 + depth * depth) / (2 - improving));

            if (moveCounter > lmp) {
                moveGen.skipQuiet();
            }


            ////////////
            // History pruning
            ////////////

            ////////////
            // Continuation pruning
            ////////////


        }
	    BitBoardEnum movedPiece = board.getPieceOnSquare(move.from());
        BitBoardEnum capturedPiece = board.getPieceOnSquare(move.to());
        board.makeMove(move);
	    ss[ply].movedPiece = movedPiece;
	    ss[ply].move = move;
        evaluatedNodes++;

        int newDepth = depth - 1;

        ////////////
        // Check extension
        ////////////
        givesCheck = board.isCheck();

        if (givesCheck && plyCheckExtension < 3 && depth > 1) {
            extension++;
        }

        ss[ply + 1].checkExt = plyCheckExtension + extension;
        

        
        newDepth += extension; 

        ////////////
        // LMR
        ////////////

        if (depth >= 2 && moveCounter > 1 + isRoot) {
            int lnDepth = std::log(depth) * 100;
            int lnMoves = std::log(moveCounter) * 100;
            int base = (isCapture ? lmrBaseNoisy() : lmrBaseQuiet());
            int divider = (isCapture ? lmrDividerNoisy() : lmrDividerQuiet());
            int r = (int)std::max(0, base + lnDepth*lnMoves  / divider);
            int historyScore = 0;            
            if (!isNoisy) {
	    	    historyScore += history.butterflyScore(stm, move, threats);
                historyScore += history.pieceToScore(ss[ply].movedPiece, move, threats);

                historyScore += history.contScore(conts, ss[ply].movedPiece, ss[ply].move.to(),0) * contWeight1Ply()/100;
                historyScore += history.contScore(conts, ss[ply].movedPiece, ss[ply].move.to(),1) * contWeight2Ply()/100;
                historyScore += history.contScore(conts, ss[ply].movedPiece, ss[ply].move.to(),2) * contWeight4Ply()/100;
                historyScore += history.contScore(conts, ss[ply].movedPiece, ss[ply].move.to(),3) * contWeight6Ply()/100;
            }
            else {
                historyScore += history.capturedPieceScore(movedPiece, move.to(), capturedPiece);
            }
            
            


            r += !pvNode*lmrPVReduction();
            r -= improving*lmrImprovingReduction();
            r -= givesCheck*lmrCheckReduction();
            r -= std::clamp(historyScore/lmrHistoryReduction(),-200,200);

            r /= 100;

            int reduction = newDepth - r;
            reduction = std::clamp(reduction, 1, newDepth); // Should the minimum be 1 or 0?
            
            score = -negamax(board, reduction, -(alpha + 1), -alpha, ply + 1,false);
            lmrHit++;

            if (score > alpha && reduction < newDepth) {
                lmrResearchHit++;
                score = -negamax(board, newDepth, -(alpha + 1), -alpha, ply + 1,false);
                
            } 
        }
        else if (!pvNode || !firstMove) {
            int r = 1;

            

            score = -negamax(board, newDepth, -(alpha + 1), -alpha, ply + 1, false);
        }

        if(pvNode && (firstMove || score > alpha)){
            score = -negamax(board, newDepth, -beta, -alpha, ply + 1, true);
        }



        
        board.revertLastMove();
        moveCounter++;

        if (isSearchStopped()) {
            return 0;
        }

        if (score > bestScore) {
            bestScore = score;

            if (score > alpha) {
                alpha = score;
                alphaMove = move;
                if (isRoot) {
                    bestMoveIteration.bestMove = move;
                    bestMoveIteration.score = alpha;
                    bestMoveIteration.depth = depth;
                }
            

                if (bestScore >= beta) {
                    break;
                }
            }
        }

        if (move != alphaMove) {
            if (!isNoisy) {
                failLowQuiet.push_back(move);
            }
            else {
                failLowNoisy.push_back(move);
            }
        }

    }

    if (moveCounter == 0) {
        return inCheck ? -MATESCORE + ply : 0;
    }

    if (alphaMove) {
        bool isNoisy = board.getPieceOnSquare(alphaMove.to()) != All || alphaMove.getMoveType() == PROMOTION;

        if (!isNoisy) {
            int bonus = std::clamp(depth * quietHistBonusDepthScale() - quietHistBonusOffset(), 0, quietHistMaxBonus());
            history.updateButterflyScore(board.getSideToMove(), alphaMove, board.getThreats(), bonus);
	        history.updateContScore(conts, board.getPieceOnSquare(alphaMove.from()), alphaMove.to(), bonus);
            history.updatePieceToScore(board.getPieceOnSquare(alphaMove.from()), alphaMove, board.getThreats(), bonus);

            int penalty = -std::clamp(depth * quietHistPenaltyDepthScale() - quietHistPenaltyOffset(), 0, quietHistMaxPenalty());
            for (Move move : failLowQuiet) {
                history.updateButterflyScore(board.getSideToMove(), move, board.getThreats(), penalty);
		        history.updateContScore(conts, board.getPieceOnSquare(move.from()), move.to(), penalty);
                history.updatePieceToScore(board.getPieceOnSquare(move.from()), move, board.getThreats(), penalty);
            }
        }
        else {
            int bonus = std::clamp(depth * noisyHistBonusDepthScale() - noisyHistBonusOffset(), 0, noisyHistMaxBonus());
            history.updateCapturedPiece(board.getPieceOnSquare(alphaMove.from()), alphaMove.to(), board.getPieceOnSquare(alphaMove.to()),bonus);

            int penalty = -std::clamp(depth * noisyHistPenaltyDepthScale() - noisyHistPenaltyOffset(), 0, noisyHistMaxPenalty());
            for (Move move : failLowNoisy) {
                history.updateCapturedPiece(board.getPieceOnSquare(move.from()), move.to(), board.getPieceOnSquare(move.to()), penalty);
            }
            
        }
    }

    TType bound = bestScore >= beta ? LOWER : bestScore <= alphaOrginal ? UPPER : EXACT;   
    tt.put(key, scoreToTT(bestScore,ply), ss[ply].staticEval, depth, alphaMove, bound, pvNode);
    

    return bestScore;
}



int Search::quinesence(Board &board, int alpha, int beta,int depth, int ply, bool pvNode)
{
    assert(ply <= MAXPLY);
    //////////////////////////
    // Has repeated 3-fold
    //////////////////////////
    if (board.hasPositionRepeated() || board.getHalfMoveClock() >= 100) {
        return drawScore();
    }
    
    if (maxQuinesenceDepthThisSearch < depth) {
        maxQuinesenceDepthThisSearch = depth;
    }

    if (maxPlyThisIteration < ply) {
        maxPlyThisIteration = ply;
    }

    //////////////////////////
    // Check if max search time has been exhausted
    // Returns beta to prevent things going to shit
    //////////////////////////
    if (isSearchStopped()) {
        return beta;
    }

    if (ply >= MAXPLY) {
        return evaluate(board);
    }
    
    // Mate distance pruning
    alpha = std::max(alpha, -MATESCORE + ply);
    beta = std::min((int)beta, MATESCORE - ply);
    if (alpha >= beta) {
        return alpha;
    }


    auto tte = tt.probe(board.getHashKey());
    bool ttHit = tte.type != TType::NO_TYPE;
    if (tte.type != TType::NO_TYPE) tte.score = scoreFromTT(tte.score, ply);


    //////////////////////////
    // Transposition Table
    //////////////////////////
    if (!pvNode &&  (tte.type == EXACT || (tte.type == LOWER && tte.score >= beta) || (tte.type == UPPER && tte.score <= alpha)))  {
        qsearchTTHit++;            
        return tte.score;
    }


    

    bool inCheck = board.getCheckers() > 0;

    if (inCheck) {
        ss[ply].staticEval = -MATESCORE - 1;
    }
    else {
        if (ttHit && tte.staticEval != (-MATESCORE - 1)) {
            ss[ply].staticEval = tte.staticEval;
        }
        else {
            ss[ply].staticEval = evaluate(board);
        }

        if (ss[ply].staticEval >= beta) {
            return beta;
        }

        if (ss[ply].staticEval > alpha) {
            alpha = ss[ply].staticEval;
        }
    }

    int score = 0;
    int futilityValue = ss[ply].staticEval + futilityBaseQsearch();
    
    MoveGenerator& moveGen = moveGenStack[ply];
    moveGen.init(board, tte.type != TType::NO_TYPE ? tte.move : Move{}, true, ss[ply].killerMove, &history);
    int moveCounter = 0;
    Move move;
    while((move = moveGen.next())){
        /*
        if (board.getPieceOnSquare(move.to()) != All) {
            int seeScore = see(board, move.from(), move.to(), board.getSideToMove());
            if (seeScore < -800) {
                continue;
            }
        }
        */

        bool isCapture = board.getPieceOnSquare(move.to()) != All;

        //////////////////////////
        // Futility pruning
        //////////////////////////
        if (isCapture && !inCheck && futilityValue <= alpha && 0 > See::see(board, move.from(), move.to(), board.getSideToMove())) {
            qsearchFutilityPruningHit++;
            continue;
        }
        
        evaluatedNodes++;
        board.makeMove(move);
        score = -quinesence(board,-beta,-alpha,depth-1, ply+1,pvNode);

        if(score > alpha){
            alpha = score;
        }
        if(alpha >= beta){      
            board.revertLastMove();
            return beta;
        }
        

        board.revertLastMove();   
        moveCounter++;

        if (isSearchStopped()) {
            return 0;
        }
    }

    //We check for checkmate, but not stalemate
    if (inCheck && moveCounter == 0) {
        return -MATESCORE + ply;
    }

    return alpha;
}

int Search::drawScore() {
    return 2 - (evaluatedNodes % 4);
}


int Search::evaluate(Board &board)
{
    return board.evaluate();
}


bool Search::equal(Move &a, Move &b)
{
    return (a.from() == b.from() &&
            a.to() == b.to());
}

void Search::setNewGame() {
    tt.clear();
    history.clear();
}

bool Search::isSearchStoppedSoft()
{
    if (stopSearch) {
        return true;
    }

    auto end = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    auto diff = end - startTime;
    if (diff > maxSearchTime) {
        stopSearch = true;
        return true;
    }
    if (limits.nodeLimit > 0 && evaluatedNodes > limits.nodeLimit) {
        stopSearch = true;
        return true;
    }

    return false;
}



bool Search::isSearchStopped()
{
    if (stopSearch) {
        return true;
    }

    // We only check every 1000 nodes, if it is not already stopped, this will return false
    if ((evaluatedNodes % 1000) != 0) {
        return false;
    }


    auto end = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    auto diff = end - startTime;
    if (diff > maxSearchTime) {
        stopSearch = true;
        return true;
    }
    if (limits.nodeLimit > 0 && evaluatedNodes > limits.nodeLimit) {
        stopSearch = true;
        return true;
    }

    return false;
}
