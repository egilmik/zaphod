#include <iostream>
#include "board.h"
#include "movegenerator.h"
#include "perft.h"
#include "test.h"
#include "search.h"
#include <chrono>

int main(int, char**) {
    Board board;    
    board.parseFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 ");
    /*Search search;
    Move move = search.searchAlphaBeta(board,7);
    std::cout << Perft::getNotation(move) << std::endl;
    board.makeMove(move);

    move = search.searchAlphaBeta(board,7);
    std::cout << Perft::getNotation(move) << std::endl;*/


    Search search;
    for(int i = 0; i < 10; i++){
        Move move = search.searchAlphaBeta(board,3);
        if(move.fromSq == 0 && move.toSq == 0){
            std::cout << "Check mate " << board.getSideToMove() << " lost" << std::endl;
            break;
        }
        std::cout << Perft::getNotation(move) << " " << move.capture << std::endl;
        board.makeMove(move);
        //board.printBoard();
    }
}
