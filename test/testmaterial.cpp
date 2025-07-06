#include <gtest/gtest.h>
#include "../src/material.h"

class MaterialTest : public ::testing::Test {
protected:
    void SetUp() override {

    }
};

TEST_F(MaterialTest, KnightSquareScoreMGTest) {
    for (int square = 0; square < 64; square++) {
        EXPECT_EQ(Material::pieceSquareScoreArrayMG[N][square], -Material::pieceSquareScoreArrayMG[n][Material::flip[square]]);
    }
}

TEST_F(MaterialTest, RookSquareScoreMGTest) {
    for (int square = 0; square < 64; square++) {
        EXPECT_EQ(Material::pieceSquareScoreArrayMG[R][square], -Material::pieceSquareScoreArrayMG[r][Material::flip[square]]);
    }
}

TEST_F(MaterialTest, BishopSquareScoreMGTest) {
    for (int square = 0; square < 64; square++) {
        EXPECT_EQ(Material::pieceSquareScoreArrayMG[B][square], -Material::pieceSquareScoreArrayMG[b][Material::flip[square]]);
    }
}

TEST_F(MaterialTest, QueenSquareScoreMGTest) {
    for (int square = 0; square < 64; square++) {
        EXPECT_EQ(Material::pieceSquareScoreArrayMG[Q][square], -Material::pieceSquareScoreArrayMG[q][Material::flip[square]]);
    }
}

TEST_F(MaterialTest, KingSquareScoreMGTest) {
    for (int square = 0; square < 64; square++) {
        EXPECT_EQ(Material::pieceSquareScoreArrayMG[K][square], -Material::pieceSquareScoreArrayMG[k][Material::flip[square]]);
    }
}

TEST_F(MaterialTest, PawnSquareScoreMGTest) {
    for (int square = 0; square < 64; square++) {
        EXPECT_EQ(Material::pieceSquareScoreArrayMG[P][square], -Material::pieceSquareScoreArrayMG[p][Material::flip[square]]);
    }
}

TEST_F(MaterialTest, KnightSquareScoreEGTest) {
    for (int square = 0; square < 64; square++) {
        EXPECT_EQ(Material::pieceSquareScoreArrayEG[N][square], -Material::pieceSquareScoreArrayEG[n][Material::flip[square]]);
    }
}

TEST_F(MaterialTest, RookSquareScoreEGTest) {
    for (int square = 0; square < 64; square++) {
        EXPECT_EQ(Material::pieceSquareScoreArrayEG[R][square], -Material::pieceSquareScoreArrayEG[r][Material::flip[square]]);
    }
}

TEST_F(MaterialTest, BishopSquareScoreEGTest) {
    for (int square = 0; square < 64; square++) {
        EXPECT_EQ(Material::pieceSquareScoreArrayEG[B][square], -Material::pieceSquareScoreArrayEG[b][Material::flip[square]]);
    }
}

TEST_F(MaterialTest, QueenSquareScoreEGTest) {
    for (int square = 0; square < 64; square++) {
        EXPECT_EQ(Material::pieceSquareScoreArrayEG[Q][square], -Material::pieceSquareScoreArrayEG[q][Material::flip[square]]);
    }
}


TEST_F(MaterialTest, KingSquareScoreEGTest) {
    for (int square = 0; square < 64; square++) {
        EXPECT_EQ(Material::pieceSquareScoreArrayEG[K][square], -Material::pieceSquareScoreArrayEG[k][Material::flip[square]]);
    }
}

TEST_F(MaterialTest, PawnSquareScoreEGTest) {
    for (int square = 0; square < 64; square++) {
        EXPECT_EQ(Material::pieceSquareScoreArrayEG[P][square], -Material::pieceSquareScoreArrayEG[p][Material::flip[square]]);
    }
}