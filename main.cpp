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
    

    Board test1;
    //Position 2 - Kiwipete
    //Expected at depth 1 - 48 nodes and 8 captures
    std::string test1Fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - ";
    test1.parseFen(test1Fen);
    
    std::string singleWhiteRookFen = "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1";
    Board singleWhiteRook;
    singleWhiteRook.parseFen(singleWhiteRookFen);
    singleWhiteRook.printBoard();

    auto start = std::chrono::high_resolution_clock::now();
    int depth = 2;
    int nrOfNodes = Perft::perft(singleWhiteRook,depth);
    std::cout << "Depth: " << depth << " - nr of nodes: " << nrOfNodes << std::endl;
    
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    int nps = (double)nrOfNodes/((double)duration.count()/(double)1000);
    std::cout << duration.count() << " ms" << " NPS " << nps << std::endl;
    

/*    
    //Array test

    BitBoard array[12];
    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 1000000; i++){
        for(int x = 0; x < 12; x++){
            BitBoard board = array[x];
        }
    }
    
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << duration.count() << " ms array test"  << std::endl;

    // map test
    std::map<int,BitBoard> bitboardMap;
    for(int i=0; i < 12; i++){
        bitboardMap.insert({i,0});
    }
    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 1000000; i++){
        for(int x = 0; x < 12; x++){
            BitBoard board = bitboardMap.at(x);
        }
    }
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << duration.count() << " ms map test"  << std::endl;
*/

    /*
    for(int i = 0; i< 64; i++){
        BitBoard mask = startingBoard.rayAttackEast[i];
        startingBoard.printBoard(mask,i);
    }
    */
    
        
}
