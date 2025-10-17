#include "search.h"
#include "perft.h"
#include "material.h"
#include <algorithm>
#include <chrono>


Search::Search() {
}

Score Search::search(Board &board, int maxDepth, int maxTime)
{   
    startTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    if (clearTTOnSearch) {
        tt.clear();
    }
    

    for (int i = 0; i < 100; i++) {
        ss[i].checkExt = 0;
        ss[i].isNullMove = false;
    }
    
    maxSearchTime = maxTime;
    
    stopSearch = false;
    evaluatedNodes = 0;
    pawnTTHits = 0;
    lmrHit = 0;
    lmrResearchHit = 0;
    exactHit = 0;
    upperBoundHit = 0;
    lowerBoundHit = 0;
    aspirationHighResearchHit = 0;
    aspirationLowResearchHit = 0;
    bestMoveIteration.depth = 0;
    bestMoveIteration.score = 0;
    bestMoveIteration.bestMove = 0;

    

    bool inIteration = true;
    Score bestScore;
    constexpr int lowerBound = -std::numeric_limits<int>::max();
    constexpr int upperBound = std::numeric_limits<int>::max();

    auto start = std::chrono::high_resolution_clock::now();
    

    for (int i = 1; i <= maxDepth; i++) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
        if (maxSearchTime / 2 < duration.count()) {
            break;
        }

        for (int s = 0; s < 2; ++s)
            for (int f = 0; f < 64; ++f)
                for (int t = 0; t < 64; ++t)
                    hist.quiet[s][f][t] /= 2;

        int previousScore = i > 1 ? bestScore.score : 0;
        int aspiration = 20 + i * 5;
        int low = previousScore - aspiration;
        int high = previousScore + aspiration;

        currentTargetDepth = i;
        maxQuinesenceDepthThisSearch = 0;
        maxPlyThisIteration = 0;

        //Reset search stack check extension
        ss[0].checkExt = 0;
        int score = negamax(board, i, low, high,0,false);
        if (score <= low) {
            aspirationLowResearchHit++;
            score = negamax(board, i, lowerBound, high, 0, false);
        }
        else if (score >= high) {
            aspirationHighResearchHit++;
            score = negamax(board, i, low, upperBound, 0, false);
        }

        if (stopSearch) {
            break;
        }

        auto stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
        int nps = (double)evaluatedNodes / ((double)duration.count() / (double)1000);


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
    } 


    ////////////////////////////
    // We might have canceled early and do not have a valid move.
    // We pick one.....  Lets see how that goes
    ////////////////////////////
    if (bestScore.depth == 0) {
        MoveList list;
        MoveGenerator::generateMoves(board, list);
        // Lets try sorting to perhaps hit something in TT
        sortMoveList(board, list,0,0);
        bestScore = { 0,0, list.moves[0] }; 
    }
    
    return bestScore;
}



int Search::negamax(Board& board, int depth, int alpha, int beta, int ply, bool pvNode)
{

    if (depth <= 0) return quinesence(board, alpha, beta, 1,ply,pvNode);
    
    BitBoard key = board.getHashKey();    
    bool isRoot = ply == 0;
    int alphaOrginal = alpha;


    // Check if max search time has been exhausted
    // Returns beta to prevent things going to shit
    if ((evaluatedNodes % 100) == 0 && isSearchStopped()) {
        return beta;
    }
    //////////////////////////
    // Has repeated 3-fold
    //////////////////////////
    if (board.hasPositionRepeated() || board.getHalfMoveClock() >= 100) {
        return 0;
    }
    
    auto tte = tt.probe(key);    

    if (!pvNode && tte && tte->depth >= depth) {
        if (tte->type == EXACT) {
            exactHit++;
            if (isRoot) {
                bestMoveIteration.bestMove = tte->bestMove;
                bestMoveIteration.score = tte->score;
                bestMoveIteration.depth = tte->depth;
            }
            return tte->score;
        }
        else if (tte->type == LOWER && tte->score > alpha) {
            lowerBoundHit++;
            alpha = tte->score;
        }
        else if (tte->type == UPPER && tte->score < beta) {
            upperBoundHit++;
            beta = tte->score;
        }
        if (alpha >= beta) {
            return tte->score;
        }
    }

    

    MoveList moveList;
    MoveGenerator::generateMoves(board, moveList);
    int score = 0;

    
    Move alphaMove{};
    
    
    sortMoveList(board, moveList, ply, tte ? tte->bestMove : 0);

    int validMoves = moveList.counter;
    bool inCheck = moveList.checkers > 0;

    int eval = evaluate(board);


    ////////////
    // Razoring
    ////////////
    if (eval < alpha - 500 * depth*depth) {

        int value = quinesence(board, alpha - 1, alpha, 0, ply, false);
        if (value < alpha && std::abs(value) < 20000) {
            return value;
        }
    }

    ////////////
    // Null move pruning
    ////////////
    if (!pvNode && !inCheck  && eval >= beta && depth >= 3 && !isRoot && !ss[ply - 1].isNullMove) {
        if(board.getNonPawnMaterial(board.getSideToMove()) > 0 || depth >= 5){
            int R = 3 + (depth >= 6) + (eval - beta) / 200; // adaptive
            R = std::clamp(R, 2, 4);
            board.makeNullMove();
            ss[ply].isNullMove = true;
            int nullScore = -negamax(board, depth - 1 - R, -beta, -beta + 1, ply + 1,false);
            board.revertNullMove();
            if (nullScore >= beta) {
                return nullScore;
            }
        }
    }


    
    for (int i = 0; i < moveList.counter; i++) {
        Move move = moveList.moves[i];        
        
        bool isPromo = move.getMoveType() == PROMOTION;
        bool isCapture = board.getPieceOnSquare(move.to()) != All;
        int plyCheckExtension = ss[ply].checkExt;
        int extension = 0;
        bool firstMove = i == 0;
        bool givesCheck = false;
        
        board.makeMove(move);        

        ////////////
        // Check extension
        ////////////
        BitBoard kingBB = board.getBitboard(board.getSideToMove() + BitBoardEnum::K);
        if (board.isSquareAttacked(kingBB, board.getOtherSide())) {           
           givesCheck = true;
        }

        if (givesCheck && plyCheckExtension < 3 && depth > 1) {
            extension++;
        }

        ss[ply + 1].checkExt = plyCheckExtension + extension;

        int newDepth = depth - 1 + extension;

        /*
        if (!pvNode && !inCheck && depth >= 3 && !isCapture && !isPromo) {
            if (i > 8 + depth * 2) {
                if (board.evaluate() + 80 <= alpha) {
                    board.revertLastMove();
                    continue;
                }
            }
        }

        */

        
        if (firstMove || pvNode || inCheck || isCapture || isPromo || isRoot) {
            score = -negamax(board, newDepth, -beta, -alpha, ply + 1,firstMove && pvNode);
        }
        else {
            //int reduction = 0.5 + (std::log(depth) * std::log(i) / 3);
            int reduction = (uint8_t)std::max(0.0,0.75 + std::log(depth) * std::log(i) / 2.5);

            int eval = board.evaluate();

            // We are not improving, reduce more
            if (eval <= alphaOrginal - 50) {
                r + 1;
            }

            if (givesCheck) {
                reduction = std::max(0, reduction - 1);
            }


            

            ////////////
            // LMR
            ////////////
            score = -negamax(board, newDepth-reduction, -(alpha + 1), -alpha, ply + 1,false);
            lmrHit++;

            if (score > alpha) {
                lmrResearchHit++;
                score = -negamax(board, newDepth, -(alpha + 1), -alpha, ply + 1,false);
                if (score > alpha && score < beta) {
                    // Full re-search
                    score = -negamax(board, newDepth, -beta, -alpha, ply + 1,true);
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
            
            if (!isCapture) {
                if (move.value != ss[ply].killerMove[0].value) {
                    ss[ply].killerMove[0] = move;
                }
                else {
                    ss[ply].killerMove[1] = move;
                }

                int side = 0;
                if (board.getSideToMove() == Black) {
                    side = 1;
                }
                hist.quiet[side][move.from()][move.to()] += depth * depth;

            }

            /*if (it == transpositionMap.end() || it->second.depth < depth) {
                transpositionMap[key] = { move, TEType::lower, depth, beta };
            }*/
                
            break;
        }

        

    }

    if (validMoves == 0) {

        if (inCheck) {
            // We are check mate
            alpha = -MATESCORE+ply;
        }
        else {
            // Stalemate
            alpha = 0;
        }
    }

    
    //Replace if depth is higher
    if (!tte || (tte && tte->depth < depth)) {
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



int Search::quinesence(Board &board, int alpha, int beta,int depth, int ply, bool pvNode)
{

    //////////////////////////
    // Has repeated 3-fold
    //////////////////////////
    if (board.hasPositionRepeated() || board.getHalfMoveClock() >= 100) {
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

    MoveList moveList;
    MoveList moveListReduced;
    MoveGenerator::generateMoves(board,moveList);
    for(int i = 0; i < moveList.counter; i++){
        if(moveList.checkers != 0 || board.getPieceOnSquare(moveList.moves[i].to()) != All || moveList.moves[i].getMoveType() == PROMOTION) {
            moveListReduced.moves[moveListReduced.counter++] = moveList.moves[i];
        }
    }

    auto tte = tt.probe(board.getHashKey());

    sortMoveList(board, moveListReduced,ply,tte? tte->bestMove:0);


    int score = 0;
    bool inCheck = moveList.checkers > 0;
    for(int i = 0; i < moveListReduced.counter; i++){
        Move move = moveListReduced.moves[i];
        /*
        if (board.getPieceOnSquare(move.to()) != All) {
            int seeScore = see(board, move.from(), move.to(), board.getSideToMove());
            if (seeScore < -800) {
                continue;
            }
        }
        */

        
        
        bool valid = board.makeMove(move);
        score = -quinesence(board,-beta,-alpha,depth+1, ply+1,pvNode);

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
        
        if (inCheck) {
            // We are check mate
            alpha = -MATESCORE + ply;
        }
        else {
            // Stalemate
            alpha = 0;
        }
    }

    return alpha;
}

int Search::see(Board &board,int fromSq, int toSq, BitBoardEnum sideToMove) {

    BitBoardEnum us = sideToMove;
    BitBoardEnum otherSide = White;
    if (sideToMove == White) {
        otherSide = Black;
    }

    // We only want positive values, so subtracting side to move so we only get white values.    
    int result = Material::materialScoreArray[board.getPieceOnSquare(toSq)-otherSide];
    result = Material::materialScoreArray[board.getPieceOnSquare(fromSq) - sideToMove]- result;
    if (result < 0) {
        return result;
    }

    BitBoard occupied = board.getBitboard(All);
    BitBoard toSqBB = board.sqBB[toSq];


    //Remove pieces
    occupied &= ~board.sqBB[fromSq];
    occupied &= ~toSqBB;


    BitBoard attackersTo = 0;

    
    uint64_t magic = ((board.getBitboard(All) & board.rookMask[toSq]) * board.magicNumberRook[toSq]) >> board.magicNumberShiftsRook[toSq];
    attackersTo |= (*board.magicMovesRook)[toSq][magic] & (board.getBitboard(Q) | board.getBitboard(R) | board.getBitboard(q) | board.getBitboard(r));
        
    magic = ((board.getBitboard(All) & board.bishopMask[toSq]) * board.magicNumberBishop[toSq]) >> board.magicNumberShiftsBishop[toSq];
    attackersTo |= (*board.magicMovesBishop)[toSq][magic] & (board.getBitboard(Q) | board.getBitboard(B) | board.getBitboard(q) | board.getBitboard(b));

    attackersTo |= board.getKnightMask(toSq) & (board.getBitboard(N) | board.getBitboard(n));

    attackersTo |= ((toSqBB & ~board.FileHMask) << 7) & board.getBitboard(P);
    attackersTo |= ((toSqBB & ~board.FileAMask) << 9) & board.getBitboard(P);

    attackersTo |= ((toSqBB & ~board.FileAMask) >> 7) & board.getBitboard(p);
    attackersTo |= ((toSqBB & ~board.FileHMask) >> 9) & board.getBitboard(p);

    //Remove the already capture pieces
    attackersTo &= occupied;
        

    BitBoard attacker = 0;
    BitBoard sideToMoveAttackers = 0;
    while (true) {

        if (sideToMove == White) {
            sideToMove = Black;
            otherSide = White;
        }
        else {
            sideToMove = White;
            otherSide = Black;
        }

        if ((sideToMoveAttackers = attackersTo & board.getBitboard(sideToMove)) == 0) {
            break;
        }

        if ((attacker = sideToMoveAttackers & board.getBitboard(P + sideToMove))) {
            attackersTo &= board.sqBB[board.popLsb(attacker)];
            occupied &= board.sqBB[board.popLsb(attacker)];
            result = Material::materialScoreArray[P] - result;
            if (result < 0) {
                break;
            }
            //TODO: we have removed a piece, we need to check if there are changes to pins!
            magic = ((occupied & board.bishopMask[toSq]) * board.magicNumberBishop[toSq]) >> board.magicNumberShiftsBishop[toSq];
            attackersTo |= (*board.magicMovesBishop)[toSq][magic] & ((board.getBitboard(Q) | board.getBitboard(B) | board.getBitboard(q) | board.getBitboard(b)) & occupied);


        }
        else if ((attacker = sideToMoveAttackers & board.getBitboard(N + sideToMove))) {
            BitBoard bb = board.sqBB[board.popLsb(attacker)];
            attackersTo &= bb;
            occupied &= bb;
            result = Material::materialScoreArray[N] - result;
            if (result < 0) {
                break;
            }
        }
        else if ((attacker = sideToMoveAttackers & board.getBitboard(B + sideToMove))) {
            BitBoard bb = board.sqBB[board.popLsb(attacker)];
            attackersTo &= bb;
            occupied &= bb;
            result = Material::materialScoreArray[B] - result;

            magic = ((occupied & board.bishopMask[toSq]) * board.magicNumberBishop[toSq]) >> board.magicNumberShiftsBishop[toSq];
            attackersTo |= (*board.magicMovesBishop)[toSq][magic] & ((board.getBitboard(Q) | board.getBitboard(B) | board.getBitboard(q) | board.getBitboard(b)) & occupied);
            
            if (result < 0) {
                break;
            }

        }
        else if ((attacker = sideToMoveAttackers & board.getBitboard(R + sideToMove))) {
            BitBoard bb = board.sqBB[board.popLsb(attacker)];
            attackersTo &= bb;
            occupied &= bb;
            result = Material::materialScoreArray[R] - result;
            magic = ((occupied & board.rookMask[toSq]) * board.magicNumberRook[toSq]) >> board.magicNumberShiftsRook[toSq];
            attackersTo |= (*board.magicMovesRook)[toSq][magic] & ((board.getBitboard(Q) | board.getBitboard(R) | board.getBitboard(q) | board.getBitboard(r)) & occupied);

            if (result < 0) {
                break;
            }

        }
        else if ((attacker = sideToMoveAttackers & board.getBitboard(Q + sideToMove))) {
            BitBoard bb = board.sqBB[board.popLsb(attacker)];
            attackersTo &= bb;
            occupied &= bb;
            result = Material::materialScoreArray[Q] - result;

            magic = ((occupied & board.bishopMask[toSq]) * board.magicNumberBishop[toSq]) >> board.magicNumberShiftsBishop[toSq];
            attackersTo |= (*board.magicMovesBishop)[toSq][magic] & ((board.getBitboard(Q) | board.getBitboard(B) | board.getBitboard(q) | board.getBitboard(b)) & occupied);

            magic = ((occupied & board.rookMask[toSq]) * board.magicNumberRook[toSq]) >> board.magicNumberShiftsRook[toSq];
            attackersTo |= (*board.magicMovesRook)[toSq][magic] & ((board.getBitboard(Q) | board.getBitboard(R) | board.getBitboard(q) | board.getBitboard(r)) & occupied);

            if (result < 0) {
                break;
            }
        }
        else if ((attacker = sideToMoveAttackers & board.getBitboard(K + sideToMove))) {
            BitBoard bb = board.sqBB[board.popLsb(attacker)];
            attackersTo &= bb;
            occupied &= bb;
            if (attackersTo) {
                
                return 1000;
            }

        }
        if (attackersTo == 0) {
            break;
        }
    }

    return result;

}

BitBoard Search::getPinned(Board& board, BitBoardEnum sideToMove) {
    BitBoard king = board.getBitboard(K + sideToMove);
    int kingSquare = board.popLsb(king);
    king = board.getBitboard(K + sideToMove);
    BitBoardEnum otherSide = White;
    if (sideToMove == White) {
        otherSide = Black;
    }

    BitBoard snipers = board.getSnipers(kingSquare, otherSide);
    BitBoard sniperCopy = snipers;

    BitBoard pinned = 0;
    BitBoard potentialPinned = 0;


    while (sniperCopy) {
        int sniperSquare = board.popLsb(sniperCopy);
        potentialPinned = board.sqBetween[kingSquare][sniperSquare] & board.getBitboard(All);
        if (board.countSetBits(potentialPinned) == 1) {
            pinned |= potentialPinned & board.getBitboard(board.getSideToMove());
        }
    }
    return pinned;
}



bool compare(SortStruct a, SortStruct b)
{
    return a.score > b.score;
}


void Search::sortMoveList(Board &board, MoveList &list, int ply, Move bestMove)
{    
    int side = 0;
    if (board.getSideToMove() == Black) {
        side = 1;
    }

    SortStruct sortArray[256];
    for (int i = 0; i < list.counter; i++) {
        SortStruct entry;
        entry.move = list.moves[i];
        if (equal(list.moves[i], bestMove)) {
            entry.score = 100000;
        }
        else if (entry.move.getMoveType() == PROMOTION) {
            entry.score = 80000;

        }
        else if (board.getPieceOnSquare(entry.move.to()) != All) {

            BitBoardEnum capturedPiece = board.getPieceOnSquare(entry.move.to());
            BitBoardEnum attacker = board.getPieceOnSquare(entry.move.from());

            if (entry.move.getMoveType() == EN_PASSANT) {
                capturedPiece = P;
            }
            int Mvv = Material::getMaterialScore(capturedPiece);
            int lva = Material::getMaterialScore(attacker);
            int mvvlva = (Mvv - lva) / 100;
            entry.score = 70000 + mvvlva;

            //int score = see(board, entry.move.from(), entry.move.to(), board.getSideToMove());
            //entry.score = 100 - (score / 100);               


        }
        else if (ss[ply].killerMove[0].value == entry.move.value ||
            ss[ply].killerMove[1].value == entry.move.value) {
            entry.score = 60000;
        } else if(hist.quiet[side][entry.move.from()][entry.move.to()] != 0){                
            entry.score = 30000 + hist.quiet[side][entry.move.from()][entry.move.to()];
        }
        else {
            //quiet move
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
    return board.evaluate();
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
        
        auto tte = tt.probe(board.getHashKey());

        if (tte && tte->type == EXACT) {
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
