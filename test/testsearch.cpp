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
    Score move = search.searchAlphaBeta(board,1);
    
    EXPECT_EQ(move.bestMove.fromSq,55);
    EXPECT_EQ(move.bestMove.toSq,63);   
}

TEST_F(SearchTest, returnPositiveScoreWhite){
    Board board;    
    board.parseFen("rnbqkbnr/1pppppp1/7p/8/p2PP3/P1N5/1PP2PPP/R1BQKBNR w KQkq - 0 5");

    Search search;
    Score move = search.searchAlphaBeta(board,5);
    
    EXPECT_TRUE(move.score > 100);
}


TEST_F(SearchTest, returnPositiveScoreBlack){
    Board board;    
    board.parseFen("rnbqkbnr/ppp2ppp/3p4/4p3/4P1Q1/5P2/PPPP2PP/RNB1KBNR b KQkq - 0 3");

    Search search;
    Score move = search.searchAlphaBeta(board,2);
    
    EXPECT_TRUE(move.score > 100);
}

