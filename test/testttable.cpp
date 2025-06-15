#include <gtest/gtest.h>
#include "../src/ttable.h"

class TTableTest : public ::testing::Test {
protected:
    void SetUp() override {

    }
};
/*
TEST_F(TTableTest, itIsWorking) {
    TTable table(1);

    uint64_t hash = 15124567851251;
    int score = 100;
    table.put(hash, score);
    int newScore = 0;
    bool valid = false;
    table.probe(hash, valid, newScore);
   
    EXPECT_EQ(score, newScore);
}

*/