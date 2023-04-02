#include "test.h"
#include "perft.h"
#include <iostream>
#include <chrono>
#include <vector>

void Test::runAllTest()
{
    std::vector<TestDefinition> testVector;
    testVector.push_back({3,2473+223+13,"Basic pawn moves","4k3/8/8/8/8/5r2/PPPPP3/2K5 w - - 0 1"});
    testVector.push_back({1,16,"Basic bishop moves", "4k3/8/8/2B5/8/8/8/2K5 w - - 0 1"});
    testVector.push_back({1,26,"Basic queen moves","4k3/8/8/8/8/8/3Q4/2K5 w - - 0 1"});
    testVector.push_back({2,255,"Basic rook moves", "4k3/8/8/2R2r2/8/8/8/2K5 w - - 0 1"});
    testVector.push_back({2,20,"Simple check test with kings and pawn","4k3/4P3/8/8/8/8/8/2K5 w - - 0 1"});
    testVector.push_back({2,51+13,"Simple check test with kings and a knight","4k3/8/4N3/8/8/8/8/2K5 w - - 0 1"});
    testVector.push_back({3,8902+400+20,"Starting position","rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"});
    testVector.push_back({4,197281+8902+400+20,"Starting position","rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"});
    
    //testVector.push_back({5,4865609+197281+8902+400+20,"Starting position","rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"});


    for(TestDefinition def: testVector){
        TestResult result = standardTest(def);
        printResult(result,def);
    }
}

TestResult Test::standardTest(TestDefinition definition)
{
    Board board;
    board.parseFen(definition.fen);
    auto start = std::chrono::high_resolution_clock::now();    
    int nrOfNodes = Perft::perft(board,definition.depth);
    
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    int nps = (double)nrOfNodes/((double)duration.count()/(double)1000);

    bool expected = (nrOfNodes == definition.expectedNodes);

    return {definition.text, expected, duration.count(),nps, nrOfNodes};
}

void Test::printResult(TestResult result, TestDefinition def)
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
