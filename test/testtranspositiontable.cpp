#include <gtest/gtest.h>
#include "../src/search.h"

class TranspositionTableTest : public ::testing::Test {
 protected:
  void SetUp() override {
     
  }
};

TEST_F(TranspositionTableTest, samePositionGivesSameResult){
    Board board1;    
    board1.parseFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    Move move1 = {11,19,false,BitBoardEnum::All,false,false,false,BitBoardEnum::P};
    Move move2 = {52,44,false,BitBoardEnum::All,false,false,false,BitBoardEnum::p};
    Move move3 = {12,20,false,BitBoardEnum::All,false,false,false,BitBoardEnum::P};

    board1.makeMove(move1);
    board1.makeMove(move2);
    board1.makeMove(move3);

    Board board2;    
    board2.parseFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    board2.makeMove(move3);
    board2.makeMove(move2);
    board2.makeMove(move1);

    TranspositionTable table;
    BitBoard key1 = table.generateKey(board1);
    BitBoard key2 = table.generateKey(board2);

    
    EXPECT_EQ(key1,key2);
}