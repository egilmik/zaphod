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
    bool isAttacked = board.isSquareAttacked(attackedSquares, BitBoardEnum::White);
    EXPECT_TRUE(isAttacked);

}

TEST(BoardTest, fenparsingEnpassantNoSquare){
    Board board;
    board.parseFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    EXPECT_EQ(board.getEnPassantSq(), -1);
}


TEST(BoardTest, fenparsingEnpassantE3){
    Board board;
    board.parseFen("rn1qkbnr/ppp1pppp/8/8/3pP3/5N1b/PPPP1PPP/RNBQKB1R b KQkq e3 0 4");

    EXPECT_EQ(board.getEnPassantSq(), 20);
}