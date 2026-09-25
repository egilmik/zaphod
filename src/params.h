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
	ZAP_TUNABLE_INT(lmrDividerQuiet, 175, 100, 350, 10)
	ZAP_TUNABLE_INT(lmrDividerNoisy, 209, 100, 350, 10)
	ZAP_TUNABLE_INT(lmrBaseQuiet, 75, 0, 150, 10)
	ZAP_TUNABLE_INT(lmrBaseNoisy, -58, -100, 100, 10)
	ZAP_TUNABLE_INT(lmrPVReduction, 92, 0, 200, 10)
	ZAP_TUNABLE_INT(lmrImprovingReduction, 92, 0, 200, 10)
	ZAP_TUNABLE_INT(lmrCheckReduction, 100, 0, 200, 10)
	ZAP_TUNABLE_INT(lmrHistoryReduction, 185,20,600,20)
	
	ZAP_TUNABLE_INT(lmrButterflyWeight, 101, 0, 200, 10)
	ZAP_TUNABLE_INT(lmrPieceToWeight, 93, 0, 200, 10)

	ZAP_TUNABLE_INT(lmrContWeight1Ply, 110, 0, 200, 10)
	ZAP_TUNABLE_INT(lmrContWeight2Ply, 75, 0, 200, 10)
	ZAP_TUNABLE_INT(lmrContWeight4Ply, 88, 0, 200, 10)
	ZAP_TUNABLE_INT(lmrContWeight6Ply, 19, 0, 200, 10)
	
	ZAP_TUNABLE_INT(lmrCorrWeight, 47, 0, 200, 10)
	ZAP_TUNABLE_INT(lmrCorrMax, 105, 0, 300, 20)

	//Razoring
	ZAP_TUNABLE_INT(razoringMargin, 261, 100, 400, 20)

	//Qsearch
	ZAP_TUNABLE_INT(futilityBaseQsearch, 50, 0, 300, 20)

	//Reverse Futility Pruning
	ZAP_TUNABLE_INT(rfpLinear, 43, 0, 200, 20)
	ZAP_TUNABLE_INT(rfpCorrection, 20, 0, 200, 10)
	ZAP_TUNABLE_INT(rfpImproving, 35, 0, 200, 20)

	///////////
	// Move generator
	///////////
	ZAP_TUNABLE_INT(movegenContWeight1Ply, 101, 0, 200, 10)
	ZAP_TUNABLE_INT(movegenContWeight2Ply, 86, 0, 200, 10)
	ZAP_TUNABLE_INT(movegenContWeight4Ply, 85, 0, 200, 10)
	ZAP_TUNABLE_INT(movegenContWeight6Ply, 12, 0, 200, 10)

	///////////
	//History
	///////////
	ZAP_TUNABLE_INT(quietHistBonusDepthScale, 401, 0, 500, 20)
	ZAP_TUNABLE_INT(quietHistBonusOffset, 314, 0, 500, 20)
	ZAP_TUNABLE_INT(quietHistMaxBonus, 16280, 0, 32000, 256)
	ZAP_TUNABLE_INT(quietHistPenaltyDepthScale, 312, 0, 500, 20)
	ZAP_TUNABLE_INT(quietHistPenaltyOffset, 214, 0, 500, 20)
	ZAP_TUNABLE_INT(quietHistMaxPenalty, 15951, 0, 32000, 256)
	ZAP_TUNABLE_INT(maxButterflyHistory, 16349, 256, 32000, 256)
	ZAP_TUNABLE_INT(butterflyAging, 859, 0, 2000,50) 

	ZAP_TUNABLE_INT(maxCapturePieceHistoryBonus,16000,256,32000,256)
	ZAP_TUNABLE_INT(noisyHistBonusDepthScale, 267, 0, 500, 20)
	ZAP_TUNABLE_INT(noisyHistBonusOffset, 381, 0, 500, 20)
	ZAP_TUNABLE_INT(noisyHistMaxBonus, 16150, 0, 32000, 256)
	ZAP_TUNABLE_INT(noisyHistPenaltyDepthScale, 381, 0, 500, 20)
	ZAP_TUNABLE_INT(noisyHistPenaltyOffset, 232, 0, 500, 20)
	ZAP_TUNABLE_INT(noisyHistMaxPenalty, 15942, 0, 32000, 256)

	ZAP_TUNABLE_INT(maxContHistory, 16288, 256, 32000, 256)
	ZAP_TUNABLE_INT(maxPieceToHistory, 16288, 256, 32000, 256)

	ZAP_TUNABLE_INT(pawnCorrectionWeight, 90, 0, 400, 10)
	ZAP_TUNABLE_INT(nonPawnCorrectionWeight, 63, 0, 400, 10)
	ZAP_TUNABLE_INT(minorCorrectionWeight, 64, 0, 400, 10)
	ZAP_TUNABLE_INT(majorCorrectionWeight, 39, 0, 400, 10)
	ZAP_TUNABLE_INT(contCorrectionWeight1Ply, 57, 0, 400, 10)
	ZAP_TUNABLE_INT(contCorrectionWeight2Ply, 42, 0, 400, 10)
	ZAP_TUNABLE_INT(contCorrectionWeight4Ply, 48, 0, 400, 10)
	ZAP_TUNABLE_INT(contCorrectionWeight6Ply, 61, 0, 400, 10)

	
};
#endif

