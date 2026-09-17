#ifndef PARAMS_H
#define PARAMS_H

#include <string>
#include <deque>


namespace zaphod::params {

	

	struct Parameter {
		std::string name;
		int32_t value;
		int32_t min;
		int32_t max;
		int32_t step;
	};


	inline std::deque<Parameter>& registry()
	{
		static std::deque<Parameter> tunableParameters;
		return tunableParameters;
	}


	inline Parameter& addParameter(std::string name, int32_t value, int32_t min, int32_t max, int32_t step) {
		registry().push_back(Parameter{ name, value, min, max, step });
		return registry().back();
	};

#define ZAP_TUNABLE_INT(Name, Value, Min, Max, Step) \
        inline Parameter& param_##Name = addParameter(#Name, Value, Min, Max, Step); \
        [[nodiscard]] inline int32_t Name() { \
            return param_##Name.value; \
        }
	///////////
	// LMR
	///////////
	ZAP_TUNABLE_INT(lmrDividerQuiet, 206, 100, 350, 10)
	ZAP_TUNABLE_INT(lmrDividerNoisy, 210, 100, 350, 10)
	ZAP_TUNABLE_INT(lmrBaseQuiet, 77, 0, 150, 10)
	ZAP_TUNABLE_INT(lmrBaseNoisy, -60, -100, 100, 10)
	ZAP_TUNABLE_INT(lmrPVReduction, 98, 0, 200, 10)
	ZAP_TUNABLE_INT(lmrImprovingReduction, 84, 0, 200, 10)
	ZAP_TUNABLE_INT(lmrCheckReduction, 101, 0, 200, 10)
	ZAP_TUNABLE_INT(lmrHistoryReduction, 201,20,600,20)
	
	ZAP_TUNABLE_INT(lmrButterflyWeight, 95, 0, 200, 10)
	ZAP_TUNABLE_INT(lmrPieceToWeight, 89, 0, 200, 10)

	ZAP_TUNABLE_INT(lmrContWeight1Ply, 129, 0, 200, 10)
	ZAP_TUNABLE_INT(lmrContWeight2Ply, 111, 0, 200, 10)
	ZAP_TUNABLE_INT(lmrContWeight4Ply, 105, 0, 200, 10)
	ZAP_TUNABLE_INT(lmrContWeight6Ply, 24, 0, 200, 10)
	
	ZAP_TUNABLE_INT(lmrCorrWeight, 37, 0, 200, 10)
	ZAP_TUNABLE_INT(lmrCorrMax, 92, 0, 300, 20)

	//Razoring
	ZAP_TUNABLE_INT(razoringMargin, 242, 100, 400, 20)

	//Qsearch
	ZAP_TUNABLE_INT(futilityBaseQsearch, 72, 0, 300, 20)

	//Reverse Futility Pruning
	ZAP_TUNABLE_INT(rfpLinear, 63, 0, 200, 20)

	ZAP_TUNABLE_INT(rfpImproving, 82, 0, 200, 20)

	///////////
	// Move generator
	///////////
	ZAP_TUNABLE_INT(movegenContWeight1Ply, 108, 0, 200, 10)
	ZAP_TUNABLE_INT(movegenContWeight2Ply, 99, 0, 200, 10)
	ZAP_TUNABLE_INT(movegenContWeight4Ply, 109, 0, 200, 10)
	ZAP_TUNABLE_INT(movegenContWeight6Ply, 20, 0, 200, 10)

	///////////
	//History
	///////////
	ZAP_TUNABLE_INT(quietHistBonusDepthScale, 371, 0, 500, 20)
	ZAP_TUNABLE_INT(quietHistBonusOffset, 329, 0, 500, 20)
	ZAP_TUNABLE_INT(quietHistMaxBonus, 16280, 0, 32000, 256)
	ZAP_TUNABLE_INT(quietHistPenaltyDepthScale, 277, 0, 500, 20)
	ZAP_TUNABLE_INT(quietHistPenaltyOffset, 219, 0, 500, 20)
	ZAP_TUNABLE_INT(quietHistMaxPenalty, 15951, 0, 32000, 256)
	ZAP_TUNABLE_INT(maxButterflyHistory, 16349, 256, 32000, 256)
	ZAP_TUNABLE_INT(butterflyAging, 809, 0, 2000,50) 

	ZAP_TUNABLE_INT(maxCapturePieceHistoryBonus,16000,256,32000,256)
	ZAP_TUNABLE_INT(noisyHistBonusDepthScale, 288, 0, 500, 20)
	ZAP_TUNABLE_INT(noisyHistBonusOffset, 323, 0, 500, 20)
	ZAP_TUNABLE_INT(noisyHistMaxBonus, 16150, 0, 32000, 256)
	ZAP_TUNABLE_INT(noisyHistPenaltyDepthScale, 323, 0, 500, 20)
	ZAP_TUNABLE_INT(noisyHistPenaltyOffset, 248, 0, 500, 20)
	ZAP_TUNABLE_INT(noisyHistMaxPenalty, 15942, 0, 32000, 256)

	ZAP_TUNABLE_INT(maxContHistory, 16288, 256, 32000, 256)
	ZAP_TUNABLE_INT(maxPieceToHistory, 16288, 256, 32000, 256)

	ZAP_TUNABLE_INT(pawnCorrectionWeight, 97, 0, 400, 10)
	ZAP_TUNABLE_INT(nonPawnCorrectionWeight, 75, 0, 400, 10)
	ZAP_TUNABLE_INT(minorCorrectionWeight, 44, 0, 400, 10)
	ZAP_TUNABLE_INT(majorCorrectionWeight, 58, 0, 400, 10)
	ZAP_TUNABLE_INT(contCorrectionWeight, 56, 0, 400, 10)

	
};
#endif

