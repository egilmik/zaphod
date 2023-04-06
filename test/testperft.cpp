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

TEST(PerftTest, perftStartingPosition){
    // https://www.chessprogramming.org/Perft_Results#Initial_Position

    Board board;
    board.parseFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 ");
    PerftResults results;
    Perft::perftWithStats(board,6,results);

    EXPECT_EQ(20+400+8902+197281+4865609+119060324 ,results.nodes);
    EXPECT_EQ(34+1576+82719+2812008, results.captures);
    EXPECT_EQ(258+5248, results.enPassant);
    EXPECT_EQ(0,results.castle);
    EXPECT_EQ(0, results.promotions);
}