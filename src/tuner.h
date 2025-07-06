#pragma once
#ifndef TUNER_H
#define TUNER_H

#include "board.h"
#include "search.h"
#include <array>
#include "material.h"

struct FenEvalStruct {
	std::string fen;
    MoveStruct boardState;
	float score;
};

class Tuner {

	public:
		float calculateMSE(std::vector<FenEvalStruct> *positions, Board &board) {
            
            Search search;

            float error = 0;

            for (int i = 0; i < positions->size(); i++) {
                FenEvalStruct fenEval = positions->at(i);
                board.setBoardState(fenEval.boardState);
                int eval = search.evaluate(board);
                float score = sigmoid(eval);

                error += pow(fenEval.score - score, 2);

            }

            return error / positions->size();

		};

        float sigmoid(int score) {
            return 1.0 / (1.0 + pow(10.0, ( - 0.4 * score) / 400));
        };

        float tuneMaterial(std::vector<FenEvalStruct>* positions, Board& board, float bestError ) {
            for (int i = 1; i < 14; i++) {

                Material::materialScoreArray[i] += 1;

                float newError = calculateMSE(positions, board);

                if (newError < bestError) {
                    bestError = newError;
                }
                else {
                    Material::materialScoreArray[i] -= 2;
                    newError = calculateMSE(positions, board);

                    if (newError < bestError) {
                        bestError = newError;
                    }
                    else {
                        // No improvement, back to normal
                        Material::materialScoreArray[i] += 1;
                    }
                }
            }
            return bestError;
        }

        float tunePSQT(std::vector<FenEvalStruct>* positions, Board& board, float bestError) {
            for (int i = 1; i < 7; i++) {
                for (int x = 0; x < 64; x++) {
                    Material::pieceSquareScoreArrayMG[i][x] += 1;

                    float newError = calculateMSE(positions, board);

                    if (newError < bestError) {
                        bestError = newError;
                    }
                    else {
                        Material::pieceSquareScoreArrayMG[i][x] -= 2;
                        newError = calculateMSE(positions, board);

                        if (newError < bestError) {
                            bestError = newError;
                        }
                        else {
                            // No improvement, back to normal
                            Material::pieceSquareScoreArrayMG[i][x] += 1;
                        }
                    }

                }
            }
            return bestError;
        };

        float tunePSQTEG(std::vector<FenEvalStruct>* positions, Board& board, float bestError) {
            for (int i = 1; i < 7; i++) {
                for (int x = 0; x < 64; x++) {
                    Material::pieceSquareScoreArrayEG[i][x] += 1;

                    float newError = calculateMSE(positions, board);

                    if (newError < bestError) {
                        bestError = newError;
                    }
                    else {
                        Material::pieceSquareScoreArrayEG[i][x] -= 2;
                        newError = calculateMSE(positions, board);

                        if (newError < bestError) {
                            bestError = newError;
                        }
                        else {
                            // No improvement, back to normal
                            Material::pieceSquareScoreArrayEG[i][x] += 1;
                        }
                    }

                }
            }

            return bestError;
        };


};
#endif
