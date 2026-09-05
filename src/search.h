#ifndef SEARCH_H
#define SEARCH_H

#include "board.h"
#include <vector>
#include "movegenerator.h"
#include <iostream>
#include <chrono>
#include "ttable.h"
#include "nnueq.h"
#include "history.h"

struct Score {
    int depth = 0;
    int score = -100000;
    Move bestMove;
};

struct SearchStack {
    int checkExt = 0;
    bool isNullMove = false;
    int staticEval = 0;
    BitBoardEnum movedPiece = All;
    Move move = 0;
};

struct SearchLimits {
    int depthLimit = -1;
    int nodeLimit = -1;
    int timeLimit = -1;
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
        int64_t razoringEntryHit, razoringReturnHit = 0;
        int64_t qsearchFutilityPruningHit;
        int64_t qsearchMoveCounterPruningHit;
        Score search(Board &board, SearchLimits limits);
        int negamax(Board &board, int depth, int alpha, int beta, int ply, bool pvNode);
        int qsearch(Board &board, int alpha, int beta, int depth, int ply, bool pvNode);
        int evaluate(Board &board);
        bool equal(Move &a, Move &b);
        bool isSearchStopped();
        bool isSearchStoppedSoft();

        int drawScore();

        void setNewGame();
        
        int currentFinishedDepth = 0;
        int maxQuinesenceDepthThisSearch = 0;
        int maxPlyThisIteration = 0;

        const int MATESCORE = 30000;
        constexpr static int MAXPLY = 255;

        const int MATE_IN_MAX = MATESCORE - MAXPLY;   // 29745

        int scoreToTT(int score, int ply) {
            if (score >= MATE_IN_MAX) return score + ply;
            if (score <= -MATE_IN_MAX) return score - ply;
            return score;
        }
        int scoreFromTT(int score, int ply) {
            if (score >= MATE_IN_MAX) return score - ply;
            if (score <= -MATE_IN_MAX) return score + ply;
            return score;
        }

        void setPrintInfo(bool on) {
            printInfo = on;
        };

        void setTTSize(int size) {
            tt.setSize(size);
        }

    private:

	std::unique_ptr<MoveGenerator[]> moveGenStack { new MoveGenerator[MAXPLY +1]};

        Score bestMoveIteration;        
        TTable tt = TTable(256);
        History history;
        int currentTargetDepth = 0;
        int64_t startTime = 0;
        int64_t maxSearchTime = 0;
        bool stopSearch = false;
        SearchLimits limits;
        SearchStack ss[MAXPLY + 1];
        bool printInfo = true;
        bool clearTTOnSearch = true;
          
};
#endif
