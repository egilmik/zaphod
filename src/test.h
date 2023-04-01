#ifndef TEST_H
#define TEST_H

#include "board.h"
#include <vector>

struct TestResult {
    std::string text;
    bool result;
    long runTimeMilliseconds;
    int nps;
    long actualNodes;
};

struct TestDefinition {
    int depth;
    long expectedNodes;
    std::string text;
    std::string fen;  
};

class Test {
    public:
        void runAllTest();

    private:
        void printResult(TestResult result, TestDefinition def);

        TestResult standardTest(TestDefinition definition);
};

#endif