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


TEST(BoardTest, rookOccupancyMask) {
    Board board;
    board.initMagicMasks();

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

TEST(BoardTest, OneBlackBishopSniperPinnedPawn) {
    Board board;
    board.parseFen("rnbqk1nr/pppp1ppp/8/4p3/1b2P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1");
    BitBoard hasSnipers = board.getSnipers(4, Black); // King square
    ;
    EXPECT_TRUE(board.countSetBits(hasSnipers) == 1);
}

TEST(BoardTest, OneBlackBishopAndQueenSniper) {
    Board board;
    board.parseFen("rnb1k1nr/ppppqp1p/6p1/8/1bN1P3/8/PPPP1PPP/RNBQKB1R w KQkq - 0 1");
    BitBoard hasSnipers = board.getSnipers(4, Black); // King square
    ;
    EXPECT_TRUE(board.countSetBits(hasSnipers) == 2);
}


TEST(BoardTest, NoSniper) {
    Board board;
    board.parseFen("rnbqk1nr/pppp1ppp/8/2b1p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1");
    BitBoard hasSnipers = board.getSnipers(4, Black); // King square
    EXPECT_TRUE(hasSnipers == 0);
}

TEST(BoardTest, SquareBetweenSameRank) {
    Board board;
    BitBoard between = board.sqBetween[8][24];
    BitBoard testBoard = 0;
    board.setBit(testBoard, 16);
    EXPECT_EQ(between, testBoard);
}

TEST(BoardTest, SquareBetweenDiagonal) {
    Board board;
    BitBoard between = board.sqBetween[8][53];
    BitBoard testBoard = 0;
    board.setBit(testBoard, 17);
    board.setBit(testBoard, 26);
    board.setBit(testBoard, 35);
    board.setBit(testBoard, 44);
    EXPECT_EQ(between, testBoard);
}

TEST(BoardTest, HalfMoveClockIsCorrectFromFenString) {
    Board board;
    board.parseFen("rn1qkbnr/ppp1pppp/8/3p1b2/3P4/4B3/PPP1PPPP/RN1QKBNR w KQkq - 2 3");

    EXPECT_EQ(2, board.getHalfMoveClock());
}


TEST(BoardTest, HalfMoveClockIsCorrectFromFenStringTwoDigits) {
    Board board;
    board.parseFen("rn2kb1r/ppp1pppp/Q4n1q/3p2B1/3P4/2N2N1b/PPP1PPPP/R3KB1R b KQkq - 11 7");

    EXPECT_EQ(11, board.getHalfMoveClock());
}

TEST(BoardTest, HalfMoveClockIsResetWithPawnMove) {
    Board board;
    board.parseFen("r3kb1r/ppp1pppp/Q1n2n1q/3p2B1/3P4/2N2N1b/PPP1PPPP/R3KB1R w KQkq - 12 8");
    board.makeMove(Move::make<NORMAL>(8, 16));

    EXPECT_EQ(0, board.getHalfMoveClock());
}

TEST(BoardTest, HalfMoveClockIsResetWithCapture) {
    Board board;
    board.parseFen("r3kb1r/ppp1pppp/Q1n2n1q/3p2B1/3P4/2N2N1b/PPP1PPPP/R3KB1R w KQkq - 12 8");
    board.makeMove(Move::make<NORMAL>(14, 23));

    EXPECT_EQ(0, board.getHalfMoveClock());
}
