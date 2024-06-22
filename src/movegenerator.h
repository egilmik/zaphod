#ifndef MOVEGENERATOR_H
#define MOVEGENERATOR_H

#include "board.h"
#include "move.h"
#include <vector>

class MoveGenerator {

    public:
        void static generateMoves(Board &board,MoveList &moveList, bool isQuinesence = false);

    private:
        void static generatePawnMoves(Board &board,MoveList &moveList,BitBoard checkers, int kingSquare, BitBoard pinned, BitBoard snipers, bool doQuinesenceReduction);
        void static generateKnightMoves(Board &board, MoveList &moveList, BitBoard checkers, int kingSquare, BitBoard pinned, BitBoard snipers, bool doQuinesenceReduction);
        void static generateRookMoves(Board &board, MoveList &moveList, BitBoard checkers, int kingSquare, BitBoard pinned, BitBoard snipers, bool doQuinesenceReduction);
        void static generateBishopMoves(Board &board, MoveList &moveList, BitBoard checkers, int kingSquare, BitBoard pinned, BitBoard snipers, bool doQuinesenceReduction);
        void static generateQueenMoves(Board &board, MoveList &moveList, BitBoard checkers, int kingSquare, BitBoard pinned, BitBoard snipers, bool doQuinesenceReduction);
        void static generateKingMoves(Board &board, MoveList &moveList, BitBoard checkers, int kingSquare, BitBoard pinned, BitBoard snipers, bool doQuinesenceReduction);

        BitBoard static makeLegalMoves(Board& board, BitBoard moves, BitBoard pinned, BitBoard checkers, BitBoard snipers, int fromSq, int kingSquare);
        BitBoard static pawnAttacks(Board& board, BitBoardEnum color);

};

#endif