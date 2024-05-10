#include <iostream>"
#include "test.h"
#include "perft.h"


int main(int, char**) {

    std::cout << "Starting Perft" << std::endl;

    std::vector<TestDefinition> testVector;
    
    
    testVector.push_back({ 7,3195901860 + 119060324 + 4865609 + 197281 + 8902 + 400 + 20,"Position 1 - Starting position","rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",true });
    testVector.push_back({ 6,8031647685 + 193690690 + 4085603 + 97862 + 2039 + 48, "Position 2 - chessprogramming.org", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - ",true });
    testVector.push_back({ 8,3009794393 + 178633661 + 11030083 + 674624 + 43238 + 2812 + 191 + 14, "Position 3 - chessprogramming.org", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - ",true });
    testVector.push_back({ 6,706045033 + 15833292 + 422333 + 9467 + 264 + 6,"Position 4 - chessprogramming.org","r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",true });
    testVector.push_back({ 5,89941194 + 2103487+ 62379 + 1486+ 44,"Position 5 - chessprogramming.org","rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",true });
    testVector.push_back({ 6,6923051137 + 164075551 + 3894594 + 89890 + 2079 + 46,"Position 6 - chessprogramming.org","r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",true });

    Test test;
    test.runAllTest(testVector);    

    return 0;
        
}
