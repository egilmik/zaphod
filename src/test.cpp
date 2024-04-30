#include "test.h"
#include "perft.h"
#include <iostream>
#include <chrono>
#include <vector>

void Test::runAllTest(std::vector<TestDefinition> &testVector)
{
    for(TestDefinition def: testVector){
        TestResult result = standardTest(def);
        if(!result.result || def.printResult){
            printResult(result,def);
        }
    }
}

TestResult Test::standardTest(TestDefinition definition)
{
    Board board;
    board.parseFen(definition.fen);
    Perft::invalidPMove = 0;
    Perft::invalidBMove = 0;
    Perft::invalidRMove = 0;
    Perft::invalidQMove = 0;
    Perft::invalidNMove = 0;
    Perft::invalidKMove = 0;
    auto start = std::chrono::high_resolution_clock::now();    
    unsigned long long nrOfNodes = Perft::perft(board,definition.depth);
    
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    int nps = (double)nrOfNodes/((double)duration.count()/(double)1000);

    bool expected = (nrOfNodes == definition.expectedNodes);
    std::cout << "Invalid moves Pawn: " << Perft::invalidPMove << std::endl;
    std::cout << "Invalid moves Queen: " << Perft::invalidQMove << std::endl;
    std::cout << "Invalid moves Bishop: " << Perft::invalidBMove << std::endl;
    std::cout << "Invalid moves Rook: " << Perft::invalidRMove << std::endl;
    std::cout << "Invalid moves Knight: " << Perft::invalidNMove << std::endl;
    std::cout << "Invalid moves King: " << Perft::invalidKMove << std::endl;

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
