#include <gtest/gtest.h>
#include "../src/search.h"

class SearchTest : public ::testing::Test {
 protected:
  void SetUp() override {
     
  }
};

TEST_F(SearchTest, symmetricPositionReturnsZeroScore){
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
    
    EXPECT_EQ(0,score);
}


