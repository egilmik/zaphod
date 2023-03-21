#include <iostream>
#include "board.h"
#include "movegenerator.h"
#include "perft.h"
#include <chrono>

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


    auto start = std::chrono::high_resolution_clock::now();
    int depth = 4;
    int nrOfNodes = Perft::perft(startingBoard,depth);
    std::cout << "Depth: " << depth << " - nr of nodes: " << nrOfNodes << std::endl;
    
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    int nps = nrOfNodes/(duration.count()/1000);
    std::cout << duration.count() << " ms" << " NPS " << nps << std::endl;
    
    /*
    for(int i = 0; i< 64; i++){
        BitBoard mask = startingBoard.knightmask[i];
        startingBoard.printBoard(mask,i);
    }
    */    
}
