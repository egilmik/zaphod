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
        unsigned long long pseudoLegalNodeCounter = 0;
        unsigned long long evaluatedNodes = 0;
        unsigned long long ttHits = 0;
        Score searchAlphaBeta(Board board,int depth);
        int negaMax(Board board, int alpha, int, int depthLeft);
        int evaluate(Board &board);
        int getPieceSquareScore(Board &board);
        int getScoreForSpecificPiece(Board &board,BitBoardEnum piece);
        int getMaterialScore(Board &board);

    private:
        Score bestMove;
        int targetDepth;
        TranspositionTable ttable;
        
          
};
#endif