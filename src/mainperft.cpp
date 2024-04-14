#include <iostream>
#include "board.h"
#include "movegenerator.h"
#include "perft.h"
#include "test.h"
#include <chrono>

int main(int, char**) {

    std::cout << "Starting Perft" << std::endl;

    std::vector<TestDefinition> testVector;
    /*testVector.push_back({4,197281 + 8902 + 400 + 20,"Starting position","rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",true});
    testVector.push_back({3,97862+2039+48, "Position 2 - chessprogramming.org", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - ",true});
    testVector.push_back({5,4865609+197281+8902+400+20," Position 1  - chess programming Starting position","rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",true});
    testVector.push_back({2,264+6,"Position 4 - chessprogramming.org","r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",true});
    testVector.push_back({4,43238+2812+191+14, "Position 3 - chessprogramming.org", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - ",true});
    */
    testVector.push_back({5,4865609+197281+8902+400+20," Position 1  - chess programming Starting position","rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",true});
    
    testVector.push_back({6,119060324+4865609+197281+8902+400+20,"Starting position","rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",true});
    
    testVector.push_back({7,3195901860+119060324+4865609+197281+8902+400+20,"Starting position","rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",true});



    Test test;
    test.runAllTest(testVector);    

    return 0;
        
}
