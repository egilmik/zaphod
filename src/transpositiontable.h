#ifndef TRANSPOSITIONTABLE_H
#define TRANSPOSITIONTABLE_H

#include "board.h"
#include <random>
#include <unordered_map>

struct TranspositionEntry {
    Move bestMove;
    int depth;
    int score;
};

class TranspositionTable {
    public:

        BitBoard pieceKeys[12][64];
        BitBoard sideToMoveKey[2];
        BitBoard castlingRightsKeys[4];
        BitBoard enPassantKeys[64];

        void initKeys(){
            std::mt19937 gen;
            gen.seed(220818100915);
            std::uniform_int_distribution<unsigned long long> dis;

            for(int i = 0; i < 12; i++){
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

        BitBoard generateKey(Board &board){
            BitBoard key = 0;

            for (int pieceValue = BitBoardEnum::R; pieceValue != BitBoardEnum::All; pieceValue++ ){
                if(pieceValue != BitBoardEnum::All && pieceValue != BitBoardEnum::White && pieceValue != BitBoardEnum::Black){
                    BitBoardEnum pieceEnum = static_cast<BitBoardEnum>(pieceValue);
                    BitBoard pieceBoard = board.getBitboard(pieceEnum);

                    while(pieceBoard != 0){
                        key ^= pieceKeys[pieceEnum][board.popLsb(pieceBoard)];
                    }
                }
            }
            if(board.getCastleRightsWK()){
                key ^= castlingRightsKeys[0];
            }
            if(board.getCastleRightsWQ()){
                key ^= castlingRightsKeys[1];
            }
            if(board.getCastleRightsBK()){
                key ^= castlingRightsKeys[1];
            }
            if(board.getCastleRightsBQ()){
                key ^= castlingRightsKeys[1];
            }

            if(board.getEnPassantSq() != board.noSq){
                key ^= enPassantKeys[board.getEnPassantSq()];
            }
            return key;
        }

        std::unordered_map<BitBoard,TranspositionEntry> transpositionMap;

        
};

#endif