#include <gtest/gtest.h>
#include "../src/perft.h"

TEST(PerftTest, enPassantBasicTest){
    GTEST_SKIP();
    Board board;
    board.parseFen("8/8/8/K7/5p1k/8/4P3/8 w - - 0 1");
    int actual = Perft::perft(board,5);

    //Tested with stockfish
    int expected = 7+44+356+2482+21066;//+156403;
    
    EXPECT_EQ(actual,expected);
}

TEST(PerftTest, BasicPawnMoves){
    GTEST_SKIP();
    Board board;
    board.parseFen("4k3/8/8/8/8/5r2/PPPPP3/2K5 w - - 0 1");
    int actual = Perft::perft(board,3);
    int expected = 2473+223+13;

    EXPECT_EQ(actual,expected);
}

TEST(PerftTest, perftStartingPosition){
    GTEST_SKIP();
    // https://www.chessprogramming.org/Perft_Results#Initial_Position
    Board board;
    board.parseFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 ");
    PerftResults results;
    Perft::perftWithStats(board,6,results);

    int expectedNodes =20+400+8902+197281+4865609+119060324;
    int expectedCaptures = 34+1576+82719+2812008;
    int expectedEnPassant = 258+5248;

    EXPECT_EQ(expectedNodes ,results.nodes);
    EXPECT_EQ(expectedCaptures, results.captures);
    EXPECT_EQ(expectedEnPassant, results.enPassant);
    EXPECT_EQ(0,results.castle);
    EXPECT_EQ(0, results.promotions);
}

TEST(PerftTest,Position2ChessProgramming){
    GTEST_SKIP();
    //https://www.chessprogramming.org/Perft_Results#Position_2
    Board board;
    board.parseFen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - ");
    PerftResults results;
    Perft::perftWithStats(board,4,results);

    int expectedNodes =48+2039+97862+4085603;
    int expectedCaptures = 8+351+17102+757163;
    int expectedEnPassant = 1+45+1929;
    int expectedCastle = 2+91+3162+128013;
    int expectedPromotions = 15172;

    EXPECT_EQ(expectedNodes ,results.nodes);
    EXPECT_EQ(expectedCaptures, results.captures);
    EXPECT_EQ(expectedEnPassant, results.enPassant);
    EXPECT_EQ(expectedCastle,results.castle);
    EXPECT_EQ(expectedPromotions, results.promotions);
}

TEST(PerftTest,Position3ChessProgramming){   
    //GTEST_SKIP();
    //https://www.chessprogramming.org/Perft_Results#Position_3
    Board board;
    board.parseFen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - ");
    PerftResults results;
    Perft::perftWithStats(board,6,results);

    int expectedNodes = 14+191+2812+43238+674624+11030083;
    int expectedCaptures = 1+14+209+3348+52051+940350;
    int expectedEnPassant = 2+123+1165+33325;
    int expectedPromotions = 7552;

    EXPECT_EQ(expectedNodes ,results.nodes);
    EXPECT_EQ(expectedCaptures, results.captures);
    EXPECT_EQ(expectedEnPassant, results.enPassant);
    EXPECT_EQ(0,results.castle);
    EXPECT_EQ(expectedPromotions, results.promotions);
}

TEST(PerftTest,Position4ChessProgramming){
    //GTEST_SKIP();
    //https://www.chessprogramming.org/Perft_Results#Position_4
    Board board;
    board.parseFen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");
    PerftResults results;
    Perft::perftWithStats(board,5,results);

    int expectedNodes =6+264+9467+422333+15833292;
    int expectedCaptures = 87+1021+131393+2046173;
    int expectedEnPassant = 4+6512;
    int expectedCastle = 7795;
    int expectedPromotions = 48+120+60032+329464;

    EXPECT_EQ(expectedNodes ,results.nodes);
    EXPECT_EQ(expectedCaptures, results.captures);
    EXPECT_EQ(expectedEnPassant, results.enPassant);
    EXPECT_EQ(expectedCastle,results.castle);
    EXPECT_EQ(expectedPromotions, results.promotions);
}

TEST(PerftTest,Position5ChessProgramming){
    
    //https://www.chessprogramming.org/Perft_Results#Position_5
    Board board;
    board.parseFen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
    PerftResults results;
    Perft::perftWithStats(board,5,results);

    int expectedNodes =44+1486+62379+2103487+89941194;    

    EXPECT_EQ(expectedNodes ,results.nodes);
}