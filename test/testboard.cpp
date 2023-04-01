#include <gtest/gtest.h>
#include "../src/board.h"

// Demonstrate some basic assertions.
TEST(BoardTest, BasicNotationTest) {
  Board board;
  std::string actualNotation = board.sqToNotation[0];

  EXPECT_EQ(actualNotation, "a1");
}
