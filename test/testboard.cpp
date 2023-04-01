#include <gtest/gtest.h>
#include "../src/board.h"

// Demonstrate some basic assertions.
TEST(BoardTest, a1NotationTest) {
  Board board;
  std::string actualNotation = board.sqToNotation[0];

  EXPECT_EQ(actualNotation, "a1");
}

TEST(BoardTest, a8NotationTest) {
  Board board;
  std::string actualNotation = board.sqToNotation[56];

  EXPECT_EQ(actualNotation, "a8");
}