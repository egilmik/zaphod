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

            MoveList moveList;
            generator.generateMoves(board,moveList);
            nrOfNodes += moveList.counter;
            for(int i = 0; i < moveList.counter; i++){
                Move move = moveList.moves[i];
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
            MoveList moveList;
            generator.generateMoves(board,moveList);
            
            for(int i = 0; i < moveList.counter; i++){
                Move move = moveList.moves[i];
                bool valid = board.makeMove(move.fromSq,move.toSq,move.piece,false, move.promotion);
                if(valid){
                    divideNodes = perft(board, depth-1);
                    nrOfNodes += divideNodes;
                    std::string notation = getNotation(move);
                    std::cout << notation << ": " << divideNodes << std::endl;
                } else {
                    nrOfNodes--;
                }

                board.revertLastMove();               
                
            }
            return nrOfNodes;      
        }

        static std::string getNotation(Move move){
            std::string promotion = "";
            
                if(move.promotion == Board::Q){
                    promotion = "Q";
                } else if (move.promotion == Board::q){
                    promotion = "q";
                } else if (move.promotion == Board::B){
                    promotion = "B";
                } else if (move.promotion == Board::b){
                    promotion = "b";
                } else if (move.promotion == Board::r){
                    promotion = "r";
                } else if (move.promotion == Board::R){
                    promotion = "R";
                } else if (move.promotion == Board::n){
                    promotion = "n";
                } else if (move.promotion == Board::N){
                    promotion = "N";
                }
            return Board::sqToNotation[move.fromSq] + Board::sqToNotation[move.toSq] + promotion;
        }
};

#endif