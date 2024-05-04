#include <gtest/gtest.h>
#include "../src/board.h"
#include "../src/tools.h"

class ToolsTest : public ::testing::Test {
protected:
	void SetUp() override {

	}
};

// Demonstrate some basic assertions.
TEST(ToolsTest, consistentBoardReturnsTrue) {
	Board board;
	board.parseFen("4k3/8/8/8/8/5r2/PPPPP3/2K5 w - - 0 1");


	EXPECT_TRUE(Tools::isBoardConsistent(board));
}

TEST(ToolsTest, inconsistentBoardReturnsFalse) {
	Board board;
	board.parseFen("4k3/8/8/8/8/5r2/PPPPP3/2K5 w - - 0 1");
	board.setBit(B, 15);

	EXPECT_FALSE(Tools::isBoardConsistent(board));

}