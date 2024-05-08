#include <gtest/gtest.h>
#include "../src/move.h"

class MoveTest : public ::testing::Test {
protected:
    void SetUp() override {
        //GTEST_SKIP() << "Skipping all tests for this fixture";
    }
};

TEST_F(MoveTest, normalMoveReturnsRightType) {
    Move move = Move::make<NORMAL>(1, 2);

    EXPECT_EQ(move.getMoveType(), MoveType::NORMAL);

}

TEST_F(MoveTest, enpassantMoveReturnsRightType) {
    Move move = Move::make<EN_PASSANT>(1, 2);

    EXPECT_EQ(move.getMoveType(), MoveType::EN_PASSANT);

}

TEST_F(MoveTest, prmotionlMoveReturnsRightType) {
    Move move = Move::make<PROMOTION>(1, 2);

    EXPECT_EQ(move.getMoveType(), MoveType::PROMOTION);

}

TEST_F(MoveTest, castlingMoveReturnsRightType) {
    Move move = Move::make<CASTLING>(1, 2);

    EXPECT_EQ(move.getMoveType(), MoveType::CASTLING);

}