#ifndef BITBOARD_H
#define BITBOARD_H

#include <string>
#include <map>

#define BitBoard __UINT64_TYPE__


class Board {

    public:
        Board();

        enum BitBoardEnum {All,White,Black,R,r,N,n,B,b,Q,q,K,k,P,p};
        void parseFen(std::string fen);
        void printBoard();
        void setBit(BitBoard &board, bool highLow, int bitNr);
        void setBit(BitBoardEnum piece, bool highLow, int bitNr);
        bool checkBit(BitBoard board, int bitNr);
        bool checkBit(BitBoardEnum piece, int bitNr);
        int popLsb(BitBoard& board);
        void makeMove(int fromSq, int toSq,BitBoardEnum piece, bool capture);
        void revertLastMove();

        

        

        //enum side {White,Black};
        
        
        BitBoard getBitboard(BitBoardEnum piece);
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
       /*
        a-file             0x0101010101010101
        h-file             0x8080808080808080
        1st rank           0x00000000000000FF
        8th rank           0xFF00000000000000
        a1-h8 diagonal     0x8040201008040201
        h1-a8 antidiagonal 0x0102040810204080
        light squares      0x55AA55AA55AA55AA
        dark squares       0xAA55AA55AA55AA55
       */


};


#endif