#ifndef BOARD_H
#define BOARD_H

#include <string>
#include <map>
#include "bitboard.h"
#include "move.h"
#include "transpositiontable.h"
#include <array>

struct MoveStruct {
    BitBoard bitBoardArrayCopy[15];
    BitBoardEnum sideToMoveCopy = BitBoardEnum::White;
    int enPassantSqCopy = -1;
    bool castleWKCopy = true;
    bool castleWQCopy = true;
    bool castleBKCopy = true;
    bool castleBQCopy = true;
    int materialScoreCopy = 0;
    int pieceSquareScoreCopy = 0;
    BitBoard hashKeyCopy = 0;
};


class Board {

    public:
        Board();

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

        // Converst a square to unit64 with the appropriate bits set.
        static constexpr BitBoard sqBB[64] = {1,
                                        2,  
                                        4,
                                        8,
                                        16,
                                        32,
                                        64,
                                        128,
                                        256,
                                        512,
                                        1024,
                                        2048,
                                        4096,
                                        8192,
                                        16384,
                                        32768,
                                        65536,
                                        131072,
                                        262144,
                                        524288,
                                        1048576,
                                        2097152,
                                        4194304,
                                        8388608,
                                        16777216,
                                        33554432,
                                        67108864,
                                        134217728,
                                        268435456,
                                        536870912,
                                        1073741824,
                                        2147483648,
                                        4294967296,
                                        8589934592,
                                        17179869184,
                                        34359738368,
                                        68719476736,
                                        137438953472,
                                        274877906944,
                                        549755813888,
                                        1099511627776,
                                        2199023255552,
                                        4398046511104,
                                        8796093022208,
                                        17592186044416,
                                        35184372088832,
                                        70368744177664,
                                        140737488355328,
                                        281474976710656,
                                        562949953421312,
                                        1125899906842624,
                                        2251799813685248,
                                        4503599627370496,
                                        9007199254740992,
                                        18014398509481984,
                                        36028797018963968,
                                        72057594037927936,
                                        144115188075855872,
                                        288230376151711744,
                                        576460752303423488,
                                        1152921504606846976,
                                        2305843009213693952,
                                        4611686018427387904,
                                        9223372036854775808ull
                                    };
        static const std::array<BitBoard,64> knightmask;
        static const std::array<BitBoard,64> kingMask;

        static const int noSq = -1;
        
        inline static const std::string sqToNotation[64] = {"a1","b1","c1","d1","e1","f1","g1","h1",
                                            "a2","b2","c2","d2","e2","f2","g2","h2",
                                            "a3","b3","c3","d3","e3","f3","g3","h3",
                                            "a4","b4","c4","d4","e4","f4","g4","h4",
                                            "a5","b5","c5","d5","e5","f5","g5","h5",
                                            "a6","b6","c6","d6","e6","f6","g6","h6",
                                            "a7","b7","c7","d7","e7","f7","g7","h7",
                                            "a8","b8","c8","d8","e8","f8","g8","h8"
                                        };


        std::array<std::array<BitBoard, 4096>, 64> magicMovesRook{{}};
        std::array<BitBoard, 64> magicNumberRook{};
        std::array<BitBoard, 64> magicNumberShiftsRook{};
        std::array<BitBoard, 64> rookMask{};

        void initMagicRook();

        BitBoard getRookAttacks(BitBoard occlusion, int square, BitBoard friendlyPieces) {
            BitBoard bbBlockers = occlusion & rookMask[square];

            int databaseIndex = (int)((bbBlockers * magicNumberRook[square]) >> magicNumberShiftsRook[square]);

            BitBoard bbMoveSquares = magicMovesRook[square][databaseIndex] & ~friendlyPieces;

            return bbMoveSquares;
        };

        /*
        struct Magic {
            BitBoard mask;
            BitBoard magic;
        };

        static const std::array<std::array<BitBoard,4096>,64> mRookAttacks;

        Magic mRookTable[64];

        static const std::array<BitBoard, 64> rookMask;

        BitBoard getRookAttacks(BitBoard occlusion, int square) {
            //Reduce the occlusion map with the mask
            occlusion &= mRookTable[square].mask;
            //Multiply with magic number
            occlusion *= mRookTable[square].magic;
            //bitshift to get the top 9 bits
            occlusion >>= 64 - 9;
            return mRookAttacks[square][occlusion];
        };

        */

        BitBoard getKnightMask(int square);
        BitBoard getKingMask(int square);
        static BitBoard getRankMask(int square);
        static BitBoard getLineMask(int square);

        void parseFen(std::string fen);
        void printBoard();
        void printBoard(BitBoard board);
        void printBoard(BitBoard board, int origin);
        void popBit(BitBoard &board, int bitNr);
        static void setBit(BitBoard &board, int bitNr);
        static void setBit(BitBoard &board, bool highLow, int bitNr);
        void setBit(BitBoardEnum piece, int bitNr);
        void popBit(BitBoardEnum piece, int bitNr);
        bool checkBit(BitBoard &board, int bitNr);
        bool checkBit(BitBoardEnum piece, int bitNr);
        int popLsb(BitBoard& board);
        int countSetBits(BitBoardEnum piece);
        static int countSetBits(unsigned long long board);
        bool makeMove(Move move);
        bool makeMove(int fromSq, int toSq,BitBoardEnum piece, bool capture,bool enPassant, bool doublePush, bool castling, BitBoardEnum promotion);
        void revertLastMove();
        bool isSquareAttacked(BitBoard targetSquares, const BitBoardEnum attacker);
        BitBoardEnum getPieceOnSquare(int sq);


        
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
        BitBoard getBitboard(int piece);
        BitBoard getEnemyBoard();
        BitBoard getOwnBoard();
        void changeSideToMove();
        BitBoardEnum getSideToMove();
        BitBoardEnum getOtherSide();

        void setEnPassantSq(int sq){enPassantSq = sq;};
        int getEnPassantSq(){return enPassantSq;};
        bool getCastleRightsWK(){return castleWK;};
        bool getCastleRightsWQ(){return castleWQ;};
        bool getCastleRightsBK(){return castleBK;};
        bool getCastleRightsBQ(){return castleBQ;};

        int getPieceSquareScore(){ return pieceSquareScore;};
        int getMaterialScore(){ return materialScore;};

        BitBoard generateHashKey();
        BitBoard getHashKey(){ return hashKey;};

        TranspositionTable ttable;

    private:
        void parseFenPosition(char value, int &bitCout);
        void clearBoard();

        MoveStruct moveHistory[128];
        int historyPly = 0;
        
        BitBoard bitBoardArray[15];
        BitBoardEnum sideToMove = White;
        int enPassantSq = noSq;
        bool castleWK = false;
        bool castleWQ = false;
        bool castleBK = false;
        bool castleBQ = false;
        int pieceSquareScore = 0;
        int materialScore = 0;
        BitBoard hashKey = 0;
        

        inline static const int fenToBitMapping[64] = { 56,57,58,59,60,61,62,63,
                                                        48,49,50,51,52,53,54,55,
                                                        40,41,42,43,44,45,46,47,
                                                        32,33,34,35,36,37,38,39,
                                                        24,25,26,27,28,29,30,31,
                                                        16,17,18,19,20,21,22,23,
                                                        8, 9,10,11,12,13,14,15,
                                                        0, 1, 2, 3, 4, 5, 6, 7 };
        

        inline static const std::map<char,BitBoardEnum> fenToEnumBoardMap = {{'r', BitBoardEnum::r},
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