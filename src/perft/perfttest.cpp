#include "perfttest.h"
#include "perft.h"
#include <iostream>
#include <chrono>
#include <vector>

void PerftTest::runAllTest()
{
    std::vector<PerftTestDefinition> testVector;


    testVector.push_back({ 7,3195901860 + 119060324 + 4865609 + 197281 + 8902 + 400 + 20,"Position 1 - Starting position","rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",true });
    testVector.push_back({ 6,8031647685 + 193690690 + 4085603 + 97862 + 2039 + 48, "Position 2 - chessprogramming.org", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - ",true });
    testVector.push_back({ 8,3009794393 + 178633661 + 11030083 + 674624 + 43238 + 2812 + 191 + 14, "Position 3 - chessprogramming.org", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - ",true });
    testVector.push_back({ 6,706045033 + 15833292 + 422333 + 9467 + 264 + 6,"Position 4 - chessprogramming.org","r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",true });
    testVector.push_back({ 5,89941194 + 2103487 + 62379 + 1486 + 44,"Position 5 - chessprogramming.org","rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",true });
    testVector.push_back({ 6,6923051137 + 164075551 + 3894594 + 89890 + 2079 + 46,"Position 6 - chessprogramming.org","r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",true });

    for(PerftTestDefinition def: testVector){
        PertTestResult result = standardTest(def);
        if(!result.result || def.printResult){
            printResult(result,def);
        }
    }
}

PertTestResult PerftTest::standardTest(PerftTestDefinition definition)
{
    Board board;
    board.parseFen(definition.fen);
    auto start = std::chrono::high_resolution_clock::now();    
    unsigned long long nrOfNodes = Perft::perft(board,definition.depth);
    
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    int nps = (double)nrOfNodes/((double)duration.count()/(double)1000);

    bool expected = (nrOfNodes == definition.expectedNodes);

    return {definition.text, expected, duration.count(),nps, nrOfNodes};
}

void PerftTest::printResult(PertTestResult result, PerftTestDefinition def)
{
    if(result.result){
        std::cout << "[Passed]";
    } else {
        std::cout << "[Failed]";
    }
    std::string increment = "         ";
    std::cout << " " << result.text << std::endl;
    std::cout << increment << "Depth: " << def.depth << " nps: " << result.nps << " runtime: "  << result.runTimeMilliseconds << " ms" << std::endl;
    std::cout << increment << "Expected nodes: " << def.expectedNodes << " Actual nodes: " << result.actualNodes << std::endl;
    std::cout << std::endl;
}
