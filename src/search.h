#ifndef SEARCH_H
#define SEARCH_H

#include "board.h"
#include <vector>
#include "movegenerator.h"
#include <iostream>

class Search {
    public:
    
    Move searchAlphaBeta(Board board,int depth);

    int negaMax(Board board, int alpha, int, int depthLeft);

    int evaluate(Board board);
};

#endif