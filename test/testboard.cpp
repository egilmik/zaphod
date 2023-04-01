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

TEST(BoardTest, rookTargetsSquaresAroundKing){
    Board board;
    board.parseFen("4k3/8/8/5R2/8/8/8/2K5 b - - 0 1");
    BitBoard attackedSquares = 0;
    board.setBit(attackedSquares,61);
    board.setBit(attackedSquares,53);
    bool isAttacked = board.isSquareAttacked(attackedSquares, Board::BitBoardEnum::Black);
    EXPECT_TRUE(isAttacked);

}