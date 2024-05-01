#ifndef PERFT_H
#define PERFT_H

#include "board.h"
#include <vector>
#include "movegenerator.h"
#include <iostream>

struct PerftResults {
    unsigned long long nodes = 0;
    unsigned long long captures = 0;
    unsigned long long enPassant = 0;
    unsigned long long castle = 0;
    unsigned long long promotions = 0;
    unsigned long long checks = 0;
    unsigned long long checkmate = 0;
};



class Perft {
    public:

        static int64_t invalidPMove;
        static int64_t invalidBMove;
        static int64_t invalidQMove;
        static int64_t invalidKMove;
        static int64_t invalidNMove;
        static int64_t invalidRMove;


        static unsigned long long perft(Board &board, int depth){
            
            if(depth == 0){
                return 0;
            }

            MoveList moveList;
            MoveGenerator::generateMoves(board,moveList);
            unsigned long long nrOfNodes = moveList.counter;
            for(int i = 0; i < moveList.counter; i++){
                if(board.makeMove(moveList.moves[i])){
                    nrOfNodes += perft(board, depth-1);
                } else {
                    nrOfNodes--;
                    BitBoardEnum pcs = moveList.moves[i].piece;
                    if (pcs == P || pcs == p) {
                        board.printBoard();
                        invalidPMove++;
                    }
                    if (pcs == B || pcs == b) {
                        //board.printBoard();
                        invalidBMove++;
                    }
                    if (pcs == R || pcs == r) {
                        board.printBoard();
                        invalidRMove++;
                    }
                    if (pcs == Q || pcs == q) invalidQMove++;
                    if (pcs == K || pcs == k) invalidKMove++;
                    if (pcs == N || pcs == n) {
                        board.printBoard();
                        invalidNMove++;
                    }
                }

                board.revertLastMove();               
                
            }
            return nrOfNodes;
        }

        static void dperft(Board board, int depth){
            unsigned long long divideNodes = 0;
            
            MoveList moveList;
            MoveGenerator::generateMoves(board,moveList);
            
            for(int i = 0; i < moveList.counter; i++){
                Move move = moveList.moves[i];
                bool valid = board.makeMove(move);
                if(valid){
                    unsigned long long nodes = dperftLeafNodeCounter(board, depth-1);
                    divideNodes += nodes;
                    std::string notation = getNotation(move);
                    std::cout << notation << ": " << nodes << std::endl;
                }

                board.revertLastMove();               
                
            }
            std::cout << "Depth: " << depth << " Count: " << divideNodes << std::endl;
        }

        static unsigned long long dperftLeafNodeCounter(Board board, int depth){
            unsigned long long divideNodes = 0;
            
            if(depth == 0){
                return 0;
            }
            MoveList moveList;
            MoveGenerator::generateMoves(board,moveList);
            unsigned long long nrOfNodes = moveList.counter;
            for(int i = 0; i < moveList.counter; i++){
                Move move = moveList.moves[i];
                bool valid = board.makeMove(move);
                if(valid){
                    divideNodes += dperftLeafNodeCounter(board, depth-1);
                } else {                    
                    nrOfNodes--;
                }

                board.revertLastMove();               
                
            }
            if(depth == 1){
                divideNodes+= nrOfNodes;
            }

            return divideNodes;
        }


        static void perftWithStats(Board board, int depth, PerftResults &results){
            
            if(depth == 0){
                return;
            }

            MoveList moveList;
            MoveGenerator::generateMoves(board,moveList);
            if(moveList.counter == 0){
                BitBoard kingSquare = board.sqBB[board.getSideToMove()+BitBoardEnum::K];
                if(board.isSquareAttacked(kingSquare,board.getSideToMove())){
                        
                }
                
            }

            int64_t actualPerformedMoves = moveList.counter;
            results.nodes += moveList.counter;
            for(int i = 0; i < moveList.counter; i++){
                Move move = moveList.moves[i];
                bool valid = board.makeMove(move);
                if(valid){
                    perftWithStats(board, depth-1,results);
                    if(move.capture){
                        results.captures +=1;
                    } 
                    if(move.castling){
                        results.castle += 1;
                    } 
                    if(move.enpassant){
                        results.enPassant += 1;
                    } 
                    if(move.promotion != BitBoardEnum::All){
                        results.promotions += 1;
                    }
                    //board.printBoard();
                    //std::cout << board.sqToNotation[move.fromSq] << "" << board.sqToNotation[move.toSq] << std::endl;
                } else {
                    results.nodes--;
                    actualPerformedMoves--;
                }

                board.revertLastMove();               
                
            }
            if(actualPerformedMoves == 0){
                results.checkmate+=1;
            }
        }

        static Move moveFromNotation(std::string move, Board &board){
            MoveList list;
            MoveGenerator::generateMoves(board,list);
            for(int i = 0; i < list.counter; i++){
                if(move == Perft::getNotation(list.moves[i])){
                    return list.moves[i];
                }
            }
            return {};
        }

        static std::string getNotation(Move move){
            std::string promotion = "";
            
                if(move.promotion == BitBoardEnum::Q){
                    promotion = "Q";
                } else if (move.promotion == BitBoardEnum::q){
                    promotion = "q";
                } else if (move.promotion == BitBoardEnum::B){
                    promotion = "B";
                } else if (move.promotion == BitBoardEnum::b){
                    promotion = "b";
                } else if (move.promotion == BitBoardEnum::r){
                    promotion = "r";
                } else if (move.promotion == BitBoardEnum::R){
                    promotion = "R";
                } else if (move.promotion == BitBoardEnum::n){
                    promotion = "n";
                } else if (move.promotion == BitBoardEnum::N){
                    promotion = "N";
                }
            return Board::sqToNotation[move.fromSq] + Board::sqToNotation[move.toSq] + promotion;
        }
};

#endif