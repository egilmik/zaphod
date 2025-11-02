#ifndef PERFTTEST_H
#define PERFTTEST_H

#include "../board.h"
#include <vector>

struct PertTestResult {
    std::string text;
    bool result;
    long long runTimeMilliseconds;
    int nps;
    unsigned long long actualNodes;
};

struct PerftTestDefinition {
    int depth;
    unsigned long long expectedNodes;
    std::string text;
    std::string fen;  
    bool printResult = false;
};

class PerftTest {
    public:
        void runAllTest();

    private:
        void printResult(PertTestResult result, PerftTestDefinition def);

        PertTestResult standardTest(PerftTestDefinition definition);
};

#endif