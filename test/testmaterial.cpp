#include <gtest/gtest.h>
#include "../src/material.h"

class MaterialTest : public ::testing::Test {
protected:
    void SetUp() override {

    }
};

TEST_F(MaterialTest, KnightSquareScoreTest) {
    

    EXPECT_EQ(Material::getPieceSquareScore(N, 0), -Material::getPieceSquareScore(n, 56));
}