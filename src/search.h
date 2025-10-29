#ifndef SEARCH_H
#define SEARCH_H

#include "board.h"
#include <vector>
#include "movegenerator.h"
#include <iostream>
#include <chrono>
#include "ttable.h"
#include "nnueq.h"

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
    int staticEval = 0;
};

struct HistoryTables {
    int16_t quiet[2][64][64] = {};
};

class Search {
    public:
        Search();
        unsigned long long evaluatedNodes = 0;
        unsigned long long pawnTTHits = 0;
        int64_t lowerBoundHit = 0;
        int64_t upperBoundHit = 0;
        int64_t qsearchTTHit = 0;
        int64_t exactHit = 0;
        int64_t lmrHit = 0;
        int64_t lmrResearchHit = 0;
        int64_t aspirationHighResearchHit = 0;
        int64_t aspirationLowResearchHit = 0;
        int64_t reverseFutilityPruningHit = 0;
        int64_t futilityPruningHit = 0;
        int64_t nullMoveHit = 0;
        Score search(Board &board, int maxDepth, int maxTime);
        int negamax(Board &board, int depth, int alpha, int beta, int ply, bool pvNode);
        int quinesence(Board &board, int alpha, int beta, int depth, int ply, bool pvNode);
        void sortMoveList(Board &board,MoveList &list, int ply, Move bestMove);
        int evaluate(Board &board);
        bool equal(Move &a, Move &b);
        MoveList reconstructPV(Board& board, int depth);
        bool isSearchStopped();
        int see(Board& board, int fromSq, int toSq, BitBoardEnum sideToMove);
        BitBoard getPinned(Board& board, BitBoardEnum otherSide);
        
        int currentFinishedDepth = 0;
        int maxQuinesenceDepthThisSearch = 0;
        int maxPlyThisIteration = 0;

        const int MATESCORE = 30000;
        const int MAXPLY = 1024;

        void setPrintInfo(bool on) {
            printInfo = on;
        };

        void setTTclearEnabled(bool enabled) {
            clearTTOnSearch = enabled;
        }

    private:


        Score bestMoveIteration;        
        TTable tt = TTable(256);
        HistoryTables hist{};
        int currentTargetDepth = 0;
        int64_t startTime = 0;
        int64_t maxSearchTime = 0;
        bool stopSearch = false;
        SearchStack ss[100];
        NNUEQ nnueQ;
        bool printInfo = true;
        bool clearTTOnSearch = true;
          
};
#endif