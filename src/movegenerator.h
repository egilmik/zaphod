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
        void static generateKnightMoves(Board &board, MoveList &moveList, BitBoard pinned);
        void static generateRookMoves(Board &board, MoveList &moveList, BitBoard checkers, int kingSquare);
        void static generateBishopMoves(Board &board, MoveList &moveList, BitBoard checkers, int kingSquare, BitBoard pinned, BitBoard snipers);
        void static generateQueenMoves(Board &board, MoveList &moveList, BitBoard checkers, int kingSquare, BitBoard pinned, BitBoard snipers);
        void static generateKingMoves(Board &board, MoveList &moveList);


};

#endif