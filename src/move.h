#ifndef MOVE_H
#define MOVE_H

#include "bitboard.h"

struct Move {
    int fromSq;
    int toSq;
    bool capture;
    BitBoardEnum promotion;
    bool doublePawnPush;
    bool enpassant;
    bool castling;
    BitBoardEnum piece;
};

struct MoveList{
    //218 seems to be the largest nr of moves for a position https://www.chessprogramming.org/Chess_Position
    //No additional instructions to allocate 250, just to sure :)
    Move moves[250];
    int64_t counter = 0;
};

#endif