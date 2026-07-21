#ifndef MOVEGENERATOR_H
#define MOVEGENERATOR_H

#include "board.h"
#include "move.h"
#include <vector>
#include "history.h"

enum Stage {
    TT_MOVE,
    GEN_NOISY,
    GOOD_NOISY,
    KILLER,
    GEN_QUIET,
    QUIET,
    BAD_NOISY,
    END
};

struct ScoredMove {
    int score = 0;
    Move move;
};

class MoveGenerator {

    public:
        void static generateMoves(Board &board,MoveList &moveList);
        void init(Board &board, Move ttMove, bool onlyCaptures, Move *killer, HistoryTables *hist);
        Move next();

        int getLegalMoves() {
            return noisyCount + quietCount;
        }

        BitBoard getCheckers() {
            return checkers;
        }

        ScoredMove noisyMoves[256];
        ScoredMove quietMoves[256];
	int noisyCount = 0;
	int quietCount = 0;

    private:
        void static generatePawnMoves(Board &board,MoveList &moveList,BitBoard checkers, int kingSquare, BitBoard pinned, BitBoard snipers);
        void static generateKnightMoves(Board &board, MoveList &moveList, BitBoard checkers, int kingSquare, BitBoard pinned, BitBoard snipers);
        void static generateRookMoves(Board &board, MoveList &moveList, BitBoard checkers, int kingSquare, BitBoard pinned, BitBoard snipers);
        void static generateBishopMoves(Board &board, MoveList &moveList, BitBoard checkers, int kingSquare, BitBoard pinned, BitBoard snipers);
        void static generateQueenMoves(Board &board, MoveList &moveList, BitBoard checkers, int kingSquare, BitBoard pinned, BitBoard snipers);
        void static generateKingMoves(Board &board, MoveList &moveList, BitBoard checkers, int kingSquare, BitBoard pinned, BitBoard snipers);

        BitBoard static makeLegalMoves(Board& board, BitBoard moves, BitBoard pinned, BitBoard checkers, BitBoard snipers, int fromSq, int kingSquare);
        BitBoard static pawnAttacks(Board& board, BitBoardEnum color);
        BitBoard static pawnAttacks(BitBoard pawns, BitBoardEnum color);

        bool isMoveLegal(Move move);
        bool isMoveLegalSliders(Move move, bool isCapture, BitBoard moves, BitBoard pieceBoard, BitBoard enemyBoard, BitBoard emptySquares);

	inline void addNoisy(Move m){ noisyMoves[noisyCount++] = { 0, m}; }
	inline void addQuiet(Move m){ quietMoves[quietCount++] = { 0, m}; }

        void generatePawnNoisy();
        void generatePawnQuiet();
        void generateKnightNoisy();
        void generateKnightQuiet();
        void generateBishopNoisy();
        void generateBishopQuiet();
        void generateRookNoisy();
        void generateRookQuiet();
        void generateQueenNoisy();
        void generateQueenQuiet();
        void generateKingNoisy();
        void generateKingQuiet();
        
        void sortNoisyMoves();
        void scoreQuietMoves();

        Board* board;
        Move *killerMove;
        HistoryTables* histTable;
        
            
        int noisyIdx = 0;        
        int quietIdx = 0;
        //int moveListIdx = 0;
        Stage currentStage = TT_MOVE;
        Move ttMove{};
        BitBoard checkers = 0;
        BitBoard pinned = 0;
        BitBoard snipers = 0;
        int kingSquare = 0;
        bool onlyNoisy = false;
        bool ttMoveFound = false;
};

#endif
