#ifndef SEARCH_H
#define SEARCH_H

#include "board.h"
#include <vector>
#include "movegenerator.h"
#include <iostream>
#include <chrono>
#include "ttable.h"

struct Score {
    int depth = 0;
    int score = -100000;
    Move bestMove;
};

struct SortStruct {
    int score;
    Move move;
};

struct SearchStack {
    int checkExt = 0;
    bool isNullMove = false;
    Move killerMove[2] = {0};
};

class Search {
    public:
        unsigned long long evaluatedNodes = 0;
        unsigned long long pawnTTHits = 0;
        int64_t lowerBoundHit = 0;
        int64_t upperBoundHit = 0;
        int64_t exactHit = 0;
        int64_t lmrHit = 0;
        int64_t lmrResearchHit = 0;
        Score search(Board &board, int maxDepth, int maxTime);
        int negamax(Board &board, int depth, int alpha, int beta, int ply);
        int quinesence(Board &board, int alpha, int beta, int depth, int ply);
        void sortMoveList(Board &board,MoveList &list, int ply);
        int evaluate(Board &board);
        int evaluatePawns(Board& board);
        bool equal(Move &a, Move &b);
        MoveList reconstructPV(Board& board, int depth);
        bool isSearchStopped();
        int see(Board& board, int fromSq, int toSq, BitBoardEnum sideToMove);
        BitBoard getPinned(Board& board, BitBoardEnum otherSide);
        
        int currentFinishedDepth = 0;
        int maxQuinesenceDepthThisSearch = 0;
        int maxPlyThisIteration = 0;

    private:


        Score bestMoveIteration;        
        TTable pawnTable = TTable(64); 
        TTable tt = TTable(256);
        int currentTargetDepth;
        int64_t startTime = 0;
        int64_t maxSearchTime = 0;
        bool stopSearch = false;
        SearchStack ss[100];
          
};
#endif