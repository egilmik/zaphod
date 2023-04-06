#include <iostream>
#include "board.h"
#include "movegenerator.h"
#include "perft.h"
#include "test.h"
#include <chrono>

int main(int, char**) {

    Board board;    

    board.parseFen("8/8/8/K7/5p1k/8/4P3/8 w - - 0 1");
    int actual = Perft::perft(board,8);
    //Tested with qperft
    int expected = 7+44+356+2482+21066+156403+1319736+10148801;

    std::cout << "Actual: " << expected << " Expected: " << expected << std::endl;
}
