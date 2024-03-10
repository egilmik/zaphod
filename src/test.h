#ifndef TEST_H
#define TEST_H

#include "board.h"
#include <vector>

struct TestResult {
    std::string text;
    bool result;
    long long runTimeMilliseconds;
    int nps;
    unsigned long long actualNodes;
};

struct TestDefinition {
    int depth;
    unsigned long long expectedNodes;
    std::string text;
    std::string fen;  
    bool printResult = false;
};

class Test {
    public:
        void runAllTest(std::vector<TestDefinition> &testVector);

    private:
        void printResult(TestResult result, TestDefinition def);

        TestResult standardTest(TestDefinition definition);
};

#endif