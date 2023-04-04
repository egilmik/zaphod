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
            std::vector<Move> moveVector;
            generator.generateMoves(board,moveVector);
            nrOfNodes += moveVector.size();
            for(Move move: moveVector){
                bool valid = board.makeMove(move.fromSq,move.toSq,move.piece,false,move.promotion);
                if(valid){
                    nrOfNodes += perft(board, depth-1);
                    //std::cout << board.sqToNotation[move.fromSq] << "" << board.sqToNotation[move.toSq] << std::endl;
                } else {
                    nrOfNodes--;
                }

                board.revertLastMove();               
                
            }
            return nrOfNodes;
        }

        static int dperft(Board board, int depth){
            MoveGenerator generator;
            int divideNodes = 0;
            int nrOfNodes = 0;
            if(depth == 0){
                return 0;
            }
            std::vector<Move> moveVector;
            generator.generateMoves(board,moveVector);
            //nrOfNodes += moveVector.size();
            for(Move move: moveVector){
                bool valid = board.makeMove(move.fromSq,move.toSq,move.piece,false, move.promotion);
                if(valid){
                    divideNodes = perft(board, depth-1);
                    nrOfNodes += divideNodes;
                    std::cout << board.sqToNotation[move.fromSq] << "" << board.sqToNotation[move.toSq] <<": " << divideNodes << std::endl;
                } else {
                    nrOfNodes--;
                }

                board.revertLastMove();               
                
            }
            return nrOfNodes;      
        }
};

#endif