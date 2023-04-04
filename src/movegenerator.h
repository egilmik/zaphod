#ifndef MOVEGENERATOR_H
#define MOVEGENERATOR_H

#include "board.h"
#include <vector>

struct Move {
    int fromSq;
    int toSq;
    bool capture;
    Board::BitBoardEnum promotion;
    Board::BitBoardEnum piece;
};

class MoveGenerator {

    public:
        void generateMoves(Board &board,std::vector<Move> &moveVector);      

    private:
        void generatePawnMoves(Board board,std::vector<Move> &moveVector);
        void generateKnightMoves(Board board, std::vector<Move> &moveVector);
        void generateRookMoves(Board board, std::vector<Move> &moveVector);
        void generateBishopMoves(Board board, std::vector<Move> &moveVector);
        void generateQueenMoves(Board board, std::vector<Move> &moveVector);
        void generateKingMoves(Board board, std::vector<Move> &moveVector);


};

#endif