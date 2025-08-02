#include <gtest/gtest.h>
#include "../src/search.h"

class SearchTest : public ::testing::Test {
 protected:
  void SetUp() override {
     
  }
};

TEST_F(SearchTest, symmetricPositionReturnsZeroScore){
    GTEST_SKIP();
    Board board;    
    board.parseFen("rnbqkbnr/ppp2ppp/8/3pp3/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3");

    Search search;
    int score = search.evaluate(board);
    
    EXPECT_EQ(0,score);

}

TEST_F(SearchTest, startingPosShouldReturnZeroScore){
    Board board;    
    board.parseFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    Search search;
    int score = search.evaluate(board);
    
    EXPECT_TRUE(score < 100);
}

TEST_F(SearchTest, seePositiveExchangeForBlack1) {
    Board board;
    board.parseFen("5rk1/1pp2q1p/p1pb4/8/3P1NP1/2P5/1P1BQ1P1/5RK1 b - -");
    Move move = Move::make<NORMAL>(43, 29);

    Search search;
    int score = search.see(board,43,29,Black);
    std::cout << score << std::endl;
    EXPECT_TRUE(score > 0);
}

TEST_F(SearchTest, seePositiveExchangeForBlack2) {
    Board board;
    board.parseFen("3qk3/3r4/8/8/8/3R4/8/3QK3 b HAha - 0 1");
    Move move = Move::make<NORMAL>(51, 19);

    Search search;
    int score = search.see(board, 51, 19, Black);
    std::cout << score << std::endl;
    EXPECT_TRUE(score > 0);
}


