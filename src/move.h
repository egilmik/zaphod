#ifndef MOVE_H
#define MOVE_H

#include "bitboard.h"
#include <cstdint>

/*
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
*/

enum MoveType {
    NORMAL,
    PROMOTION = 1 << 14,
    EN_PASSANT = 2 << 14,
    CASTLING = 3 << 14
};

class Move {
    // 1-6 bit to
    // 7-12 bit from
    // 13-14 bit promotion type
    // 15-16 bit move type
    public:
        Move() = default;

        Move(uint16_t move) {
            value = move;
        }

        // Promotion piece, is white pieces regardless of color.  Subtract 1 to get it to start from 0 to fit into 4-bits
        template<MoveType T>
        static Move make(uint32_t from, uint32_t to, BitBoardEnum promotionPiece = R) {
            return Move(T + ((promotionPiece-1) << 12) + (from << 6) + to);
        }

        constexpr uint32_t from() const {
            return (value >> 6) & 0x3F;
        }

        constexpr uint32_t to() const {
            return value & 0x3F;
        }

        constexpr MoveType getMoveType() const { return MoveType(value & (3 << 14)); }

        // Add in color plus the 1 subtracted when saving the promotionType
        constexpr BitBoardEnum getPromotionType(BitBoardEnum color) const { 
            return BitBoardEnum(((value >> 12) & 3) + color +1); 
        }

        uint32_t value;
    protected:
};

struct MoveList{
    //218 seems to be the largest nr of moves for a position https://www.chessprogramming.org/Chess_Position
    //No additional instructions to allocate 250, just to sure :)
    Move moves[250];
    BitBoard checkers = 0;
    int64_t counter = 0;
};

#endif
