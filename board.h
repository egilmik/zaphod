#ifndef BITBOARD_H
#define BITBOARD_H

#include <string>
#include <map>

#define BitBoard __UINT64_TYPE__


class Board {

    public:
        Board();

        enum BitBoardEnum {All,White,Black,R,r,N,n,B,b,Q,q,K,k,P,p};

        static constexpr BitBoard FileHMask = 0b0000000100000001000000010000000100000001000000010000000100000001;
        static constexpr BitBoard FileGHMask = 0b0000001100000011000000110000001100000011000000110000001100000011;
        static constexpr BitBoard FileABMask = 0b1100000011000000110000001100000011000000110000001100000011000000;
        static constexpr BitBoard FileAMask = 0b1000000010000000100000001000000010000000100000001000000010000000;

        static constexpr BitBoard Rank1Mask = 0xFF;
        static constexpr BitBoard Rank2Mask = Rank1Mask << (8 * 1);
        static constexpr BitBoard Rank3Mask = Rank1Mask << (8 * 2);
        static constexpr BitBoard Rank4Mask = Rank1Mask << (8 * 3);
        static constexpr BitBoard Rank5Mask = Rank1Mask << (8 * 4);
        static constexpr BitBoard Rank6Mask = Rank1Mask << (8 * 5);
        static constexpr BitBoard Rank7Mask = Rank1Mask << (8 * 6);
        static constexpr BitBoard Rank8Mask = Rank1Mask << (8 * 7);

        BitBoard knightmask[64];
        BitBoard kingMask[64];
        BitBoard rayAttackNorth[64];
        BitBoard rayAttackSouth[64];
        BitBoard rayAttackNW[64];
        BitBoard rayAttackNE[64];
        BitBoard rayAttackWest[64];
        BitBoard rayAttackEast[64];
        BitBoard rayAttackSE[64];
        BitBoard rayAttackSW[64];

        std::string sqToNotation[64] = {    "a1","b1","c1","d1","e1","f1","g1","h1",
                                            "a2","b2","c2","d2","e2","f2","g2","h2",
                                            "a3","b3","c3","d3","e3","f3","g3","h3",
                                            "a4","b4","c4","d4","e4","f4","g4","h4",
                                            "a5","b5","c5","d5","e5","f5","g5","h5",
                                            "a6","b6","c6","d6","e6","f6","g6","h6",
                                            "a7","b7","c7","d7","e7","f7","g7","h7",
                                            "a8","b8","c8","d8","e8","f8","g8","h8"
                                        };

        void initRayAttacks();
        void initKnightMask();
        void initKingMask();
        BitBoard getKnightMask(int square);
        BitBoard getKingMask(int square);
        BitBoard getRankMask(int square);
        BitBoard getLineMask(int square);

        void parseFen(std::string fen);
        void printBoard();
        void printBoard(BitBoard board);
        void printBoard(BitBoard board, int origin);
        void popBit(BitBoard &board, int bitNr);
        void setBit(BitBoard &board, int bitNr);
        void setBit(BitBoard &board, bool highLow, int bitNr);
        void setBit(BitBoardEnum piece, bool highLow, int bitNr);
        bool checkBit(BitBoard board, int bitNr);
        bool checkBit(BitBoardEnum piece, int bitNr);
        int popLsb(BitBoard& board);
        void makeMove(int fromSq, int toSq,BitBoardEnum piece, bool capture);
        void revertLastMove();
        bool isSquareAttacked(BitBoard targetSquares, BitBoardEnum attackingSide);


        
        BitBoard southOccludedMoves(BitBoard pieces, BitBoard empty);
        BitBoard northOccludedMoves(BitBoard pieces, BitBoard empty);
        BitBoard eastOccludedMoves(BitBoard pieces, BitBoard empty);
        BitBoard westOccludedMoves(BitBoard pieces, BitBoard empty);
        BitBoard northEastOccludedMoves(BitBoard pieces, BitBoard empty);
        BitBoard northWestccludedMoves(BitBoard pieces, BitBoard empty);
        BitBoard southEastOccludedMoves(BitBoard pieces, BitBoard empty);
        BitBoard southWestOccludedMoves(BitBoard pieces, BitBoard empty);
        BitBoard northEastOne(BitBoard pieces);
        BitBoard northWestOne(BitBoard pieces);
        BitBoard southEastOne(BitBoard pieces);
        BitBoard southWestOne(BitBoard pieces);

        BitBoard getBitboard(BitBoardEnum piece);
        BitBoard getEnemyBoard();
        BitBoard getOwnBoard();
        void changeSideToMove();
        BitBoardEnum getSideToMove();

    private:
        void parseFenPosition(char value, int &bitCout);

        std::map<BitBoardEnum,BitBoard> PreviousbitBoardMap = {};
        BitBoardEnum sideToMoveCopy;

        BitBoardEnum sideToMove = White;
        std::map<BitBoardEnum,BitBoard> bitBoardMap = {
            {All, 0},
            {White, 0},
            {Black,0},
            {R,0},
            {r,0},
            {N,0},
            {n,0},
            {B,0},
            {b,0},
            {Q,0},
            {q,0},
            {K,0},
            {k,0},
            {P,0},
            {p,0},
        };

        int fenToBitMapping[64] = {56,57,58,59,60,61,62,63,
                                   48,49,50,51,52,53,54,55,
                                   40,41,42,43,44,45,46,47,
                                   32,33,34,35,36,37,38,39,
                                   24,25,26,27,28,29,30,31,
                                   16,17,18,19,20,21,22,23,
                                    8, 9,10,11,12,13,14,15,
                                    0, 1, 2, 3, 4, 5, 6, 7 };
        

        std::map<char,BitBoardEnum> fenToEnumBoardMap = {{'r', BitBoardEnum::r},
                                                    {'R', BitBoardEnum::R},
                                                    {'n', BitBoardEnum::n},
                                                    {'N', BitBoardEnum::N},
                                                    {'b', BitBoardEnum::b},
                                                    {'B', BitBoardEnum::B},
                                                    {'p', BitBoardEnum::p},
                                                    {'P', BitBoardEnum::P},
                                                    {'q', BitBoardEnum::q},
                                                    {'Q', BitBoardEnum::Q},
                                                    {'k', BitBoardEnum::k},
                                                    {'K', BitBoardEnum::K}};

        


        
        /*
        BitBoard structure
        56 57 58 59 60 61 62 63
        48 49 50 51 52 53 54 55
        40 41 42 43 44 45 46 47
        32 33 34 35 36 37 38 39
        24 25 26 27 28 29 30 31
        16 17 18 19 20 21 22 23
         8  9 10 11 12 13 14 15
         0  1  2  3  4  5  6  7
        */
    
};


#endif