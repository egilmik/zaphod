#ifndef MOVEGENERATOR_H
#define MOVEGENERATOR_H

#include "board.h"
#include <vector>

struct Move {
    int fromSq;
    int toSq;
    Board::BitBoardEnum piece;
};

class MoveGenerator {

    public:
        std::vector<Move> generateMoves(Board board);      

    private:
        void generatePawnPush(Board board,std::vector<Move> &moveVector);
        void generatePawnCaptures(Board board);
        void generateRookMove(Board board);
        void addMove(int fromSq, int toSq, Board::BitBoardEnum piece);
};

#endif