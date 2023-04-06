#include <gtest/gtest.h>
#include "../src/perft.h"

TEST(PerftTest, enPassantBasicTest){
    Board board;
    board.parseFen("8/8/8/K7/5p1k/8/4P3/8 w - - 0 1");
    int actual = Perft::perft(board,2);

    //Tested with stockfish
    int expected = 7+44;
    
    EXPECT_EQ(actual,expected);

}