#include <iostream>
#include "board.h"
#include "movegenerator.h"
#include "perft.h"
#include "test.h"
#include <chrono>

int main(int, char**) {

    std::vector<TestDefinition> testVector;
    testVector.push_back({3,497+40+9, "Pawn promotion","4k3/1P6/8/8/8/8/8/2K5 w - - 0 1"});
    testVector.push_back({1,16,"Basic bishop moves", "4k3/8/8/2B5/8/8/8/2K5 w - - 0 1"});
    testVector.push_back({1,26,"Basic queen moves","4k3/8/8/8/8/8/3Q4/2K5 w - - 0 1"});
    testVector.push_back({2,255,"Basic rook moves", "4k3/8/8/2R2r2/8/8/8/2K5 w - - 0 1"});
    testVector.push_back({2,20,"Simple check test with kings and pawn","4k3/4P3/8/8/8/8/8/2K5 w - - 0 1"});
    testVector.push_back({2,51+13,"Simple check test with kings and a knight","4k3/8/4N3/8/8/8/8/2K5 w - - 0 1"});
    testVector.push_back({4,197281+8902+400+20,"Starting position","rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"});
    //testVector.push_back({3,97862+2039+48, "Position 2 - chessprogramming.org", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - "});
    //testVector.push_back({5,4865609+197281+8902+400+20," Position 1  - chess programming Starting position","rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"});
    testVector.push_back({2,264+6,"Position 4 - chessprogramming.org","r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"});
    testVector.push_back({4,43238+2812+191+14, "Position 3 - chessprogramming.org", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - "});
    
    testVector.push_back({6,4865609+197281+8902+400+20,"Starting position","rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"});



    Test test;
    test.runAllTest(testVector);    
        
}
