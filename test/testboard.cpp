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

TEST(BoardTest, rookOccupancyAtSquareA1) {

    Board board;
    board.initRookMask();
    
    //Edges are removed, so this should remove an empty bitboard
    EXPECT_EQ(board.rookMask[0], 0);
}

TEST(BoardTest, rookOccupancyMask) {
    Board board;
    board.initRookMask();

    BitBoard tempMask = 0;
    /*
        BitBoard structure
        56 57 58 59 60 61 62 63
        48 49 50 51 52 53 54 55
        40 41 42 43 44 45 46 47
        32 33 34 35 36 37 38 39
        24 25 26 27 28 29 30 31
        16 17 18 19 20 21 22 23
         8  9 10 11 12 13 14 15
         0  1  2  3  4  5  6  7
        */


    Board::setBit(tempMask, 19);
    Board::setBit(tempMask, 27);
    Board::setBit(tempMask, 35);
    Board::setBit(tempMask, 43);
    Board::setBit(tempMask, 51);
    Board::setBit(tempMask, 9);
    Board::setBit(tempMask, 10);
    Board::setBit(tempMask, 12);
    Board::setBit(tempMask, 13);
    Board::setBit(tempMask, 14);


    
    EXPECT_EQ(board.rookMask[11], tempMask);
}