#pragma once
#ifndef TUNER_H
#define TUNER_H

#include "board.h"
#include "search.h"
#include <array>

struct FenEvalStruct {
	std::string fen;
	float score;
};

class Tuner {

	public:
		float calculateMSE(std::vector<FenEvalStruct> *positions, Board &board) {
            
            Search search;

            float error = 0;

            for (int i = 0; i < positions->size(); i++) {
                FenEvalStruct fenEval = positions->at(i);
                board.parseFen(fenEval.fen);
                int eval = search.evaluate(board);
                float score = sigmoid(eval);

                error += pow((fenEval.score - score), 2);

            }

            return error / positions->size();

		};

        float sigmoid(int score) {
            return 1.0 / (1.0 + pow(10.0, -0.1 * score / 400));
        };


};
#endif
