#ifndef BITBOARD_H
#define BITBOARD_H

#include <string>
#include <map>

#define BitBoard __UINT64_TYPE__

class Board {

    public:
        Board();
        void parseFen(std::string fen);
        void printBoard();
        void setBit(BitBoard &board, bool highLow, int bitNr);
        bool checkBit(BitBoard board, int bitNr);
        int popLsb(BitBoard& board);
        void makeMove(int fromSq, int toSq, bool capture);

        //enum side {White,Black};
        enum BitBoardEnum {All,White,Black,R,r,N,n,B,b,Q,q,K,k,P,p};
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

        
        BitBoard pieceses = 0;
        BitBoard white = 0;
        BitBoard whiteKing = 0;
        BitBoard whiteQueens = 0;
        BitBoard whiteRooks = 0;
        BitBoard whiteBishops = 0;
        BitBoard whiteKnights = 0;
        BitBoard whitePawns = 0;
        BitBoard black = 0;
        BitBoard blackKing = 0;
        BitBoard blackQueens= 0;
        BitBoard blackRooks = 0;
        BitBoard blackBishops = 0;
        BitBoard blackKnights= 0;
        BitBoard blackPawns= 0;
        

    private:
        void parseFenPosition(char value, int &bitCout);

        

        int fenToBitMapping[64] = {56,57,58,59,60,61,62,63,
                                   48,49,50,51,52,53,54,55,
                                   40,41,42,43,44,45,46,47,
                                   32,33,34,35,36,37,38,39,
                                   24,25,26,27,28,29,30,31,
                                   16,17,18,19,20,21,22,23,
                                    8, 9,10,11,12,13,14,15,
                                    0, 1, 2, 3, 4, 5, 6, 7 };

        std::map<char,BitBoard*> fenToBitboardMap = {{'r', &blackRooks},
                                                    {'R', &whiteRooks},
                                                    {'n', &blackKnights},
                                                    {'N', &whiteKnights},
                                                    {'b', &blackBishops},
                                                    {'B', &whiteBishops},
                                                    {'p', &blackPawns},
                                                    {'P', &whitePawns},
                                                    {'q', &blackQueens},
                                                    {'Q', &whiteQueens},
                                                    {'k', &blackKing},
                                                    {'K', &whiteKing}};
        
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