#ifndef SEARCH_H
#define SEARCH_H

#include "board.h"
#include "movegenerator.h"
#include "ttable.h"
#include "history.h"
#include <cstdint>
#include <memory>

struct Score {
    int depth = 0;
    int score = -100000;
    Move bestMove;
};

struct SearchStack {
    int checkExt = 0;
    bool isNullMove = false;
    Move killerMove[2] = {0};
    int staticEval = 0;
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
        int64_t razoringEntryHit = 0;
        int64_t razoringReturnHit = 0;
        int64_t qsearchFutilityPruningHit = 0;
        int64_t qsearchMoveCounterPruningHit = 0;
        Score search(Board &board, SearchLimits limits);
        int negamax(Board &board, int depth, int alpha, int beta, int ply, bool pvNode);
        int quinesence(Board &board, int alpha, int beta, int depth, int ply, bool pvNode);
        int evaluate(Board &board);
        bool isSearchStopped();
        bool isSearchStoppedSoft();

        int drawScore();

        void setNewGame();
        
        int currentFinishedDepth = 0;
        int maxQuinesenceDepthThisSearch = 0;
        int maxPlyThisIteration = 0;

        const int MATESCORE = 30000;
        constexpr static int MAXPLY = 255;

        void setPrintInfo(bool on) {
            printInfo = on;
        };

        void setTTSize(int size) {
            tt.setSize(size);
        }

    private:

        // Mate scores are counted from the root ("mate in N plies from the root"),
        // but a transposition entry can be probed at a different ply than it was
        // stored at.  Store them counted from the node instead, and convert back
        // on probe, so the mate distance and the cutoff decision stay correct.
        [[nodiscard]] int scoreToTT(int score, int ply) const {
            if (score > MATESCORE - MAXPLY) return score + ply;
            if (score < -(MATESCORE - MAXPLY)) return score - ply;
            return score;
        }

        [[nodiscard]] int scoreFromTT(int score, int ply) const {
            if (score > MATESCORE - MAXPLY) return score - ply;
            if (score < -(MATESCORE - MAXPLY)) return score + ply;
            return score;
        }

	std::unique_ptr<MoveGenerator[]> moveGenStack { new MoveGenerator[MAXPLY +1]};

        Score bestMoveIteration;        
        TTable tt = TTable(256);
        HistoryTables hist{};
        int currentTargetDepth = 0;
        int64_t startTime = 0;
        int64_t maxSearchTime = 0;
        bool stopSearch = false;
        SearchLimits limits;
        SearchStack ss[MAXPLY + 1];
        bool printInfo = true;

};
#endif
