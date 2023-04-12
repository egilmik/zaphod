#include <gtest/gtest.h>
#include "../src/search.h"

class SearchTest : public ::testing::Test {
 protected:
  void SetUp() override {
     
  }
};

TEST_F(SearchTest, returnBestMove){
    Board board;    
    board.parseFen("4k2r/pp5R/8/8/8/8/2PP4/4K3 w k - 0 7");

    Search search;
    Move move = search.searchAlphaBeta(board,1);
    
    EXPECT_EQ(move.fromSq,55);
    EXPECT_EQ(move.toSq,63);   
}

