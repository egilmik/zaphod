#include <gtest/gtest.h>
#include "../src/material.h"
#include "../src/board.h"
#include "../src/evaluation.h"
#include "../src/search.h"

class EvaluationTest : public ::testing::Test {
protected:
    void SetUp() override {

    }
};


TEST_F(EvaluationTest, PassedPawnTest1) {
    Board board;
    board.parseFen("rn2kb1r/8/4p3/8/3P4/2N5/8/R1BQKBNR w KQkq - 0 1");
    int score = Evaluation::evaluatePassedPawn(board, White);
    score += Evaluation::evaluatePassedPawn(board, Black);

    // Should not give passed pawn score
    EXPECT_TRUE(score == 0);
}

TEST_F(EvaluationTest, PassedPawnTest2) {
    Board board;
    board.parseFen("rn2kb1r/3p4/8/8/3P4/2N5/8/R1BQKBNR w KQkq - 0 1");
    int score = Evaluation::evaluatePassedPawn(board, White);
    score += Evaluation::evaluatePassedPawn(board, Black);

    // Should not give passed pawn score
    EXPECT_TRUE(score == 0);
}

TEST_F(EvaluationTest, PassedPawnTest3) {
    Board board;
    board.parseFen("rn2kb1r/8/8/8/3P4/2N5/8/R1BQKBNR w KQkq - 0 1");
    int score = Evaluation::evaluatePassedPawn(board, White);
    score += Evaluation::evaluatePassedPawn(board, Black);

    // Passed pawn to white
    EXPECT_TRUE(score > 0);
}

TEST_F(EvaluationTest, PassedPawnTest4) {
    Board board;
    board.parseFen("rn2kb1r/8/7p/8/8/P1N5/7P/R1BQKBNR w KQkq - 0 1");
    int score = Evaluation::evaluatePassedPawn(board, White);
    score += Evaluation::evaluatePassedPawn(board, Black);

    // Passed pawn to white, could be A/H rank wrap around
    EXPECT_TRUE(score > 0);
}

TEST_F(EvaluationTest, PassedPawnTest5) {
    Board board;
    board.parseFen("rn2kb1r/8/p7/8/7P/P1N5/8/R1BQKBNR w KQkq - 0 1");
    int score = Evaluation::evaluatePassedPawn(board, White);
    score += Evaluation::evaluatePassedPawn(board, Black);

    // Passed pawn to white, could be A/H rank wrap around
    EXPECT_TRUE(score > 0);
}

TEST_F(EvaluationTest, PassedPawnTest6) {
    Board board;
    board.parseFen("rn2kb1r/7p/p7/8/7P/2N5/3Q4/R1B1KBNR b KQkq - 1 1");
    int score = Evaluation::evaluatePassedPawn(board, White);
    score += Evaluation::evaluatePassedPawn(board, Black);

    // Passed pawn to black, could be A/H rank wrap around
    EXPECT_TRUE(score < 0);
}

TEST_F(EvaluationTest, PassedPawnTest7) {
    Board board;
    board.parseFen("rn2kb1r/8/3p4/8/8/2N5/R7/2BQKBNR b Kkq - 1 1");
    int score = Evaluation::evaluatePassedPawn(board, White);
    score += Evaluation::evaluatePassedPawn(board, Black);

    // Passed pawn to black
    EXPECT_TRUE(score < 0);
}

TEST_F(EvaluationTest, PassedPawnSymmetricEvaluation) {
    Board board;
    board.parseFen("8/8/k1p5/8/8/5P1K/8/8 w - - 0 1");
    int score = Evaluation::evaluatePassedPawn(board, White);
    score += Evaluation::evaluatePassedPawn(board, Black);

    // Passed pawn to black
    EXPECT_EQ(score ,0);
}

TEST_F(EvaluationTest, SymmetricPositionWithPassedPawnIsEqual) {
    Board blackBoard;
    blackBoard.parseFen("4k3/8/8/3pp3/8/8/8/4K3 w - - 0 1");
    Search search;
    int blackScore = search.evaluate(blackBoard);
    Board whiteBoard;
    whiteBoard.parseFen("4k3/8/8/8/3PP3/8/8/4K3 w - - 0 1");
    int whiteScore = search.evaluate(whiteBoard);
    
    // Invert black score
    EXPECT_EQ(whiteScore, blackScore*-1);
}

TEST_F(EvaluationTest, DoubledPawn1) {
    Board board;
    board.parseFen("8/8/k1p5/8/8/5P1K/5P2/8 w - - 0 1");
    int whiteScore = Evaluation::evaluatePassedPawn(board, White);
    int blackScore = Evaluation::evaluatePassedPawn(board, Black);

    // Passed pawn to black
    EXPECT_TRUE(whiteScore < blackScore*-1);
}

TEST_F(EvaluationTest, DoubledPawn2) {
    Board board;
    board.parseFen("8/2p5/k1p5/8/8/5P1K/8/8 w - - 0 1");
    int whiteScore = Evaluation::evaluatePassedPawn(board, White);
    int blackScore = Evaluation::evaluatePassedPawn(board, Black);

    // Passed pawn to black
    EXPECT_TRUE(whiteScore > blackScore * -1);
}