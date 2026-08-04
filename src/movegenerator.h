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

        ScoredMove noisyMoves[256];
        ScoredMove quietMoves[256];
	    int noisyCount = 0;
	    int quietCount = 0;

    private:
        void generatePawnMoves(Board &board,MoveList &moveList, int kingSquare, BitBoard pinned, BitBoard snipers);
        void generateKingMoves(Board &board, MoveList &moveList, BitBoard checkers, int kingSquare, BitBoard pinned, BitBoard snipers);

        BitBoard static makeLegalMoves(Board& board, BitBoard moves, BitBoard pinned, BitBoard checkMask, BitBoard snipers, int fromSq, int kingSquare);
        BitBoard static pawnAttacks(Board& board, BitBoardEnum color);
        BitBoard static pawnAttacks(BitBoard pawns, BitBoardEnum color);
        void computeKingDanger();
        int generateCastlingMoves(int kingSq, Move out[2]);

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
        int killerIdx = 0;
        Stage currentStage = TT_MOVE;
        Move ttMove{};
        int kingSquare = 0;
        BitBoard checkMask = 0;
        BitBoard kingDangerMask = 0;
        bool onlyNoisy = false;
        bool ttMoveFound = false;
};

#endif
