#ifndef PAWNS_H
#define PAWNS_H

#include "board.h"

class Pawns
{
    public:
        Pawns();

        static int evaluate(Board &board, BitBoardEnum color);

    private:
    
};


#endif