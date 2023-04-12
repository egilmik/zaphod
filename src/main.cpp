#include <iostream>
#include "board.h"
#include "movegenerator.h"
#include "perft.h"
#include "test.h"
#include "search.h"
#include <chrono>

int main(int, char**) {

    Board board;    

    board.parseFen("rnb1k1nr/ppp2pbR/8/3pp3/8/5P2/PPPPPP2/RNBQKBN1 b Qkq - 0 6");

    Search search;
    Move move = search.searchAlphaBeta(board,3);
    std::cout << Perft::getNotation(move) << std::endl;

    /*int actual = Perft::perft(board,8);
    //Tested with qperft
    int expected = 7+44+356+2482+21066+156403+1319736+10148801;

    std::cout << "Actual: " << expected << " Expected: " << expected << std::endl;
    */
}
