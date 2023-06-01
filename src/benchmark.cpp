#include <iostream>
#include "board.h"
#include "movegenerator.h"
#include "perft.h"
#include "test.h"
#include "search.h"
#include <chrono>

int main(int, char**) {

    Board board;    
    board.parseFen("2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - -");
    Search search;

    auto start = std::chrono::high_resolution_clock::now();    

    Score move = search.search(board,8);
    
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Playtime " << (duration.count()) << " ms" << std::endl;
    std::cout << std::endl;

    /*
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
    */
    /*
    Search search;
    Board board;
    board.parseFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 ");
    
    auto start = std::chrono::high_resolution_clock::now();    
    Move move = search.searchAlphaBeta(board,9);
    std::cout << Perft::getNotation(move) << " " << move.capture << std::endl;
    */
    /*
    for(int i = 0; i < 6; i++){
        Move move = search.searchAlphaBeta(board,4);
        if(move.fromSq == 0 && move.toSq == 0){
            std::cout << "Check mate " << board.getSideToMove() << " lost" << std::endl;
            break;
        }
        std::cout << Perft::getNotation(move) << " " << move.capture << std::endl;
        board.makeMove(move.fromSq,move.toSq,move.piece,move.capture,move.enpassant,move.doublePawnPush,move.castling,move.promotion);
        //board.printBoard();
    }
    


    std::cout << "Playtime " << (duration.count()) << " ms" << std::endl;
    */
    /*
    Search search;
    Move move = search.searchAlphaBeta(board,2);
    std::cout << Perft::getNotation(move) << " " << move.capture << std::endl;
    */

    //UCI uci;
    //uci.loop();

    //unsigned long long nodes = Perft::perft(board,6);

    //std::cout << "Perf nodes: " << nodes << " Seached nodes: " << search.evaluatedNodes;
    //for(int i = 0; i< 64; i++){
    //    board.printBoard(Board::kingMask[i],i);
    //}
    

    //Perft::dperft(board,6);
    /*
    //Tested with qperft
    int expected = 7+44+356+2482+21066+156403+1319736+10148801;

    std::cout << "Actual: " << expected << " Expected: " << expected << std::endl;
    */
}
