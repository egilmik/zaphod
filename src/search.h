#ifndef SEARCH_H
#define SEARCH_H

#include "board.h"
#include <vector>
#include "movegenerator.h"
#include <iostream>

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
        unsigned long long ttHits = 0;
        Score search(Board board, int maxDepth);
        int searchAlphaBeta(Board board,int depth, int alpha, int beta, bool maximizingPlayer);
        int negaMax(Board board, int alpha, int beta, int depthLeft);
        int quinesence(Board board, int alpha, int beta, int depth);
        void sortMoveList(Board board,MoveList &list);
        int evaluate(Board &board);
        bool equal(Move a, Move b);
        

    private:
        Score bestMove;        
        std::unordered_map<BitBoard,TranspositionEntry> transpositionMap;
        int currentTargetDepth;
        Move pvMoves[50];
        bool isBlackMaxPlayer = false;
          
};
#endif