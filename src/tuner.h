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
		static int calculateMSE(std::vector<FenEvalStruct> &positions) {
            Board board;
            Search search;

            int error = 0;

            for (int i = 0; i < positions.size(); i++) {
                FenEvalStruct fenEval = positions.at(i);
                board.parseFen(fenEval.fen);
                int score = search.evaluate(board);

                error += pow((fenEval.score - score), 2);
            }

            return error = error / positions.size();

		};


};
#endif
