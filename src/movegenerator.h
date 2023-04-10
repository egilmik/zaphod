#ifndef MOVEGENERATOR_H
#define MOVEGENERATOR_H

#include "board.h"
#include <vector>

struct Move {
    int fromSq;
    int toSq;
    bool capture;
    Board::BitBoardEnum promotion;
    bool doublePawnPush;
    bool enpassant;
    bool castling;
    Board::BitBoardEnum piece;
};

struct MoveList{
    //218 seems to be the largest nr of moves for a position https://www.chessprogramming.org/Chess_Position
    //No additional instructions to allocate 250, just to sure :)
    Move moves[250];
    int counter = 0;
};

class MoveGenerator {

    public:
        void static generateMoves(Board &board,MoveList &moveList);      

    private:
        void static generatePawnMoves(Board &board,MoveList &moveList);
        void static generateKnightMoves(Board &board, MoveList &moveList);
        void static generateRookMoves(Board &board, MoveList &moveList);
        void static generateBishopMoves(Board &board, MoveList &moveList);
        void static generateQueenMoves(Board &board, MoveList &moveList);
        void static generateKingMoves(Board &board, MoveList &moveList);


};

#endif