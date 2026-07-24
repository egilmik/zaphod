#ifndef TRANSPOSITIONTABLE_H
#define TRANSPOSITIONTABLE_H

#include <random>
#include "bitboard.h"
#include "move.h"

enum TEType {exact, upper,lower};

struct TranspositionEntry {
    Move bestMove;
    TEType type;
    int depth;
    int score;
};

class TranspositionTable {
    public:

        BitBoard pieceKeys[15][64];
        BitBoard sideToMoveKey[2];
        BitBoard castlingRightsKeys[4];
        BitBoard enPassantKeys[64];

        void initKeys(){
            std::mt19937 gen;
            gen.seed(static_cast<std::mt19937::result_type>(220818100915ULL));
            std::uniform_int_distribution<unsigned long long> dis;

            for(int i = 0; i < 15; i++){
                for(int x = 0; x < 64; x++){
                    pieceKeys[i][x] = dis(gen);
                }
            }
            sideToMoveKey[0] = dis(gen);
            sideToMoveKey[1] = dis(gen);
            for(int x = 0; x < 64; x++){
                enPassantKeys[x] = dis(gen);
            }

            for(int x = 0; x < 4; x++){
                castlingRightsKeys[x] = dis(gen);
            }
        };


        
};

#endif