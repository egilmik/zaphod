#ifndef MOVEGENERATOR_H
#define MOVEGENERATOR_H

#include "board.h"
#include "move.h"
#include <vector>

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