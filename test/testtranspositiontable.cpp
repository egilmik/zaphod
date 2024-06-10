#include <gtest/gtest.h>
#include "../src/search.h"
#include "../src/movegenerator.h"
#include "../src/perft.h"

class TranspositionTableTest : public ::testing::Test {
 protected:
  void SetUp() override {
     
  }
};

TEST_F(TranspositionTableTest, samePositionGivesSameResult){
    Board board1;    
    board1.parseFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    Board board2;    
    board2.parseFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    
    BitBoard key1 = board1.generateHashKey();
    BitBoard key2 = board2.generateHashKey();

    
    EXPECT_EQ(key1,key2);
}

TEST_F(TranspositionTableTest, transisitionTableKeysAreDeterministic){
    Board board1;    
    board1.parseFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    Board board2;    
    board2.parseFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    
    BitBoard key1 = board1.ttable.pieceKeys[0][0];
    BitBoard key2 = board2.ttable.pieceKeys[0][0];

    
    EXPECT_EQ(key1,key2);
}

TEST_F(TranspositionTableTest, samePositionGivesSameResultAfterDifferentMoves){
    Board board1;    
    board1.parseFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    Move move1 = Move::make<NORMAL>(11, 19);
    Move move2 = Move::make<NORMAL>(52, 44);
    Move move3 = Move::make<NORMAL>(12, 20);
    
    board1.makeMove(move1);
    board1.makeMove(move2);
    board1.makeMove(move3);

    Board board2;    
    board2.parseFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    board2.makeMove(move3);
    board2.makeMove(move2);
    board2.makeMove(move1);

    TranspositionTable table;
    BitBoard key1 = board1.getHashKey();
    BitBoard key2 = board2.generateHashKey();
    BitBoard pawnKey1 = board1.getPawnHashKey();
    BitBoard pawnKey2 = board2.generatePawnHashKey();

    
    EXPECT_EQ(key1,key2);
    EXPECT_EQ(pawnKey1, pawnKey2);
}

TEST_F(TranspositionTableTest, incrementalHashKeyHandlingEnpassantSquare){
    GTEST_SKIP();
    Board board;    
    board.parseFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");


    Move move = Perft::moveFromNotation("d2d3", board);
    board.makeMove(move);

    move = Perft::moveFromNotation("a7a5",board);
    board.makeMove(move);    

    board.revertLastMove();    
    
    board.makeMove(move);
    
    BitBoard key1 = board.getHashKey();
    BitBoard key2 = board.generateHashKey();
    
    EXPECT_EQ(key1,key2);
}

TEST_F(TranspositionTableTest, movingRookDisallowingCastlingWhiteQueenSideCorrectIncrementalKey){
    GTEST_SKIP();
    Board board;
    board.parseFen("rnbqkbnr/pp1ppppp/8/2p5/8/2N5/PPPPPPPP/R1BQKBNR w KQkq - 0 2");

    Move move = Perft::moveFromNotation("a1b1",board);
    board.makeMove(move);
    
    EXPECT_EQ(board.getHashKey(),board.generateHashKey());
}

TEST_F(TranspositionTableTest, captureGivesCorrectIncrementalKey){
    GTEST_SKIP();
    Board board;
    board.parseFen("rnbqkbnr/pppp1ppp/8/4p3/5P2/8/PPPPP1PP/RNBQKBNR w KQkq - 0 2");

    Move move = Perft::moveFromNotation("f4e5",board);
    board.makeMove(move);
    
    EXPECT_EQ(board.getHashKey(),board.generateHashKey());
}

TEST_F(TranspositionTableTest, capturingRookRemovesCastlingRightInIncrementalKey){
    GTEST_SKIP();
    Board board;
    board.parseFen("rn1qkbnr/pbpp1ppp/1p6/4p3/5P2/1P2P1P1/P1PP3P/RNBQKBNR b KQkq - 0 4");

    Move move = Perft::moveFromNotation("b7h1", board);
    board.makeMove(move);

    EXPECT_EQ(board.getHashKey(),board.generateHashKey());
}

TEST_F(TranspositionTableTest, enpassantGivesCorrectIncrementalHashKey){
    GTEST_SKIP();
    Board board;
    board.parseFen("rn1qkbnr/ppp1pppp/8/8/3pP3/5N1b/PPPP1PPP/RNBQKB1R b KQkq e3 0 4");

    Move move = Perft::moveFromNotation("d4e3", board);
    board.makeMove(move);

    EXPECT_EQ(board.getHashKey(),board.generateHashKey());
}

TEST_F(TranspositionTableTest, castlingGivesCorrectIncrementalHashKeyWhiteKingSide){
    Board board1;    
    board1.parseFen("rnb1k1nr/ppp1bppp/3p4/4q3/2B1P3/5N2/PPP2PPP/RNBQK2R w KQkq - 2 6");

    Move move1 = Move::make<CASTLING>(4, 6);
    board1.makeMove(move1);
    
    Board board2;    
    board2.parseFen("rnb1k1nr/ppp1bppp/3p4/4q3/2B1P3/5N2/PPP2PPP/RNBQK2R w KQkq - 2 6");
    board2.makeMove(move1);

    TranspositionTable table;
    BitBoard key1 = board1.getHashKey();
    BitBoard key2 = board2.generateHashKey();

    
    EXPECT_EQ(key1,key2);
}