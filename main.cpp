#include <iostream>
#include "board.h"

int main(int, char**) {
    Board board;
    std::string startingFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    board.parseFen(startingFen);
    board.printBoard();
}
