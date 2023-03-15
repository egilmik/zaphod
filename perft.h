#ifndef PERFT_H
#define PERFT_H

#include "board.h"
#include <vector>
#include "movegenerator.h"
#include <iostream>

class Perft {
    public:
        static void perft(Board &board, int depth){
            MoveGenerator generator;
            int nrOfNodes = 0;
            if(depth == 0){
                return;
            }
            std::vector<Move> moveVector = generator.generateMoves(board);
            nrOfNodes += moveVector.size();
            for(Move move: moveVector){
                //Board Make move
                perft(board, depth-1);
                //Undo move
            }
            
            std::cout << "Number of nodes " << moveVector.size() << std::endl;
        }
};

#endif