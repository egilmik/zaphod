#include <iostream>
#include "board.h"
#include "movegenerator.h"
#include "perft.h"

int main(int, char**) {

    //Perft results and FEN, https://www.chessprogramming.org/Perft_Results
    Board startingBoard;
    std::string startingFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    startingBoard.parseFen(startingFen);
    //startingBoard.printBoard();

    Board test1;
    //Position 2 - Kiwipete
    //Expected at depth 1 - 48 nodes and 8 captures
    std::string test1Fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - ";
    test1.parseFen(test1Fen);
    //test1.printBoard();

    Perft::perft(startingBoard,3);
    
    

    
    
}
