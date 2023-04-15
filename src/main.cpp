#include <iostream>
#include "board.h"
#include "movegenerator.h"
#include "perft.h"
#include "test.h"
#include "search.h"
#include <chrono>

int main(int, char**) {

    Board board;    

    board.parseFen("2k5/8/8/8/8/8/7P/2K5 w - - 0 1");

    /*Search search;
    auto start = std::chrono::high_resolution_clock::now();    

    for(int i = 0; i < 100; i++){
        Move move = search.searchAlphaBeta(board,7);
        if(move.fromSq == 0 && move.toSq == 0){
            std::cout << "Check mate " << board.getSideToMove() << " lost" << std::endl;
            break;
        }
        std::cout << Perft::getNotation(move) << " " << move.capture << std::endl;
        board.makeMove(move.fromSq,move.toSq,move.piece,move.capture,move.enpassant,move.doublePawnPush,move.castling,move.promotion);
        //board.printBoard();
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "Playtime " << (duration.count()/1000) << " s" << std::endl;
    
    */
    Perft::dperft(board,6);
    //for(int i = 0; i< 64; i++){
    //    board.printBoard(Board::kingMask[i],i);
    //}
    

    /*int actual = Perft::perft(board,8);
    //Tested with qperft
    int expected = 7+44+356+2482+21066+156403+1319736+10148801;

    std::cout << "Actual: " << expected << " Expected: " << expected << std::endl;
    */
}
