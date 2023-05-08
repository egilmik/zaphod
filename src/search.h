#ifndef SEARCH_H
#define SEARCH_H

#include "board.h"
#include <vector>
#include "movegenerator.h"
#include <iostream>
#include "transpositiontable.h"

struct Score {
    int depth = 0;
    int score = -100000;
    Move bestMove;
};

class Search {
    public:
        unsigned long long evaluatedNodes = 0;
        unsigned long long evaluatedQuinesenceNodes = 0;
        unsigned long long ttHits = 0;
        Score searchAlphaBeta(Board board,int depth);
        int negaMax(Board board, int alpha, int beta, int depthLeft);
        int quinesence(Board board, int alpha, int beta, int depth);
        void sortMoveList(Board board,MoveList &list);
        int evaluate(Board &board);
        

    private:
        Score bestMove;
        TranspositionTable ttable;
        
          
};
#endif