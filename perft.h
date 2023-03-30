#ifndef PERFT_H
#define PERFT_H

#include "board.h"
#include <vector>
#include "movegenerator.h"
#include <iostream>

class Perft {
    public:
        static int perft(Board board, int depth){
            MoveGenerator generator;
            int nrOfNodes = 0;
            if(depth == 0){
                return 0;
            }
            std::vector<Move> moveVector = generator.generateMoves(board);
            nrOfNodes += moveVector.size();
            for(Move move: moveVector){
                board.makeMove(move.fromSq,move.toSq,move.piece,false);

                
                nrOfNodes += perft(board, depth-1);
                board.revertLastMove();
            }
            

            return nrOfNodes;
        }
};

#endif