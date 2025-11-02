#ifndef PERFT_H
#define PERFT_H

#include "..\board.h"
#include <vector>
#include "..\movegenerator.h"
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
            if (depth == 1) {
                return nrOfNodes;
            }
            for(int i = 0; i < moveList.counter; i++){
                board.makeMove(moveList.moves[i]);
                nrOfNodes += perft(board, depth-1);
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
                    std::string notation = getNotation(move);
                    unsigned long long nodes = dperftLeafNodeCounter(board, depth-1);
                    divideNodes += nodes;
                    
                    std::cout << notation << ": " << nodes << std::endl;
                }

                board.revertLastMove();               
                
            }
            std::cout << "Depth: " << depth << " Count: " << divideNodes << std::endl;
        }

        static unsigned long long dperftLeafNodeCounter(Board &board, int depth){
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


        static void perftWithStats(Board &board, int depth, PerftResults &results){
            
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
                bool capture = board.getPieceOnSquare(move.to()) != All;
                bool valid = board.makeMove(move);
                if(valid){
                    perftWithStats(board, depth-1,results);
                    
                    if(capture){
                        results.captures +=1;
                    } 
                    if(move.getMoveType() == CASTLING){
                        results.castle += 1;
                    } 
                    if(move.getMoveType() == EN_PASSANT){
                        results.enPassant += 1;
                    } 
                    if(move.getMoveType() == PROMOTION){
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