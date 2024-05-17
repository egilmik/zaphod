#ifndef SEARCH_H
#define SEARCH_H

#include "board.h"
#include <vector>
#include "movegenerator.h"
#include <iostream>
#include <chrono>

struct Score {
    int depth = 0;
    int score = -100000;
    Move bestMove;
};

struct SortStruct {
    int score;
    Move move;
};

class Search {
    public:
        unsigned long long evaluatedNodes = 0;
        unsigned long long evaluatedQuinesenceNodes = 0;
        int64_t lowerBoundHit = 0;
        int64_t upperBoundHit = 0;
        int64_t exactHit = 0;
        unsigned long long ttHits = 0;
        Score search(Board &board, int maxDepth, int maxTime);
        int negamax(Board &board, int depth, int alpha, int beta);
        int quinesence(Board &board, int alpha, int beta, int depth);
        void sortMoveList(Board &board,MoveList &list);
        int evaluate(Board &board);
        bool equal(Move &a, Move &b);
        MoveList reconstructPV(Board& board, int depth);
        bool isSearchStopped();
        
        int currentFinishedDepth = 0;
        int maxQuinesenceDepthThisSearch = 0;

    private:


        Score bestMoveIteration;        
        std::unordered_map<BitBoard,TranspositionEntry> transpositionMap;
        int currentTargetDepth;
        Move pvMoves[50];
        int maxQuiesenceDepth = 0;
        int64_t startTime = 0;
        int64_t maxSearchTime = 0;
        bool stopSearch = false;
          
};
#endif