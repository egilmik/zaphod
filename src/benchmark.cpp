#include <iostream>
#include "board.h"
#include "movegenerator.h"
#include "perft.h"
#include "test.h"
#include "search.h"
#include <chrono>

int main(int, char**) {

    Board board;
    BitBoard array[15];

    for(int i = 0; i < 100000; i++){
        BitBoard tmp = board.getBitboard(Board::All);
    }

    for(int i = 0; i < 100000; i++){
        BitBoard tmp = array[0];
    }
}
