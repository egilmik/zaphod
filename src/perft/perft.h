#ifndef PERFT_H
#define PERFT_H

#include "../board.h"
#include <vector>
#include "../movegenerator.h"
#include <iostream>

class Perft {
    public:

        static unsigned long long perft(Board& board, int depth) {
            History table;
            return perft(board, depth, table);

        }


        static unsigned long long perft(Board &board, int depth, History& table){
            
            if(depth == 0){
                return 0;
            }
                        
            MoveGenerator generator;
            generator.init(board, 0, false, &table);
            
            unsigned long long nrOfNodes = 0;
            if (depth == 1) {
                while (Move move = generator.next()) {
                    nrOfNodes++;
                }
                return nrOfNodes;
            }

            while (Move move = generator.next()) {
                board.makeMove(move);
                nrOfNodes++;
                nrOfNodes += perft(board, depth - 1,table);
                board.revertLastMove();
            }

            return nrOfNodes;
        }

        static std::string getNotation(Move move) {
            return getNotation(move, White);
        }

        static std::string getNotation(Move move, BitBoardEnum color){
            std::string promotion = "";
            BitBoardEnum promotionPiece = move.getPromotionType(color);

            
            if(promotionPiece == BitBoardEnum::Q || promotionPiece == BitBoardEnum::q){
                promotion = "q";
            } else if (promotionPiece == BitBoardEnum::B || promotionPiece == BitBoardEnum::b){
                promotion = "b";
            } else if (promotionPiece == BitBoardEnum::R || promotionPiece == BitBoardEnum::r){
                promotion = "r";
            } else if (promotionPiece == BitBoardEnum::N || promotionPiece == BitBoardEnum::n){
                promotion = "n";
            }

            if (move.getMoveType() != PROMOTION) {
                promotion = "";
            }

            return Board::sqToNotation[move.from()] + Board::sqToNotation[move.to()] + promotion;
        }
};

#endif
