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
	ZAP_TUNABLE_INT(lmrDividerQuiet, 199, 100, 350, 10)
	ZAP_TUNABLE_INT(lmrDividerNoisy, 206, 100, 350, 10)
	ZAP_TUNABLE_INT(lmrBaseQuiet, 84, 0, 150, 10)
	ZAP_TUNABLE_INT(lmrBaseNoisy, -49, -100, 100, 10)
	ZAP_TUNABLE_INT(lmrPVReduction, 95, 0, 200, 10)
	ZAP_TUNABLE_INT(lmrImprovingReduction, 83, 0, 200, 10)
	ZAP_TUNABLE_INT(lmrCheckReduction, 93, 0, 200, 10)
	ZAP_TUNABLE_INT(lmrHistoryReduction, 213,0,600,20)

	//Razoring
	ZAP_TUNABLE_INT(razoringMargin, 252, 100, 400, 20)

	//Qsearch
	ZAP_TUNABLE_INT(futilityBaseQsearch, 111, 0, 300, 20)

	//Reverse Futility Pruning
	ZAP_TUNABLE_INT(rfpLinear, 44, 0, 200, 20)

	ZAP_TUNABLE_INT(rfpImproving, 75, 0, 200, 20)

	///////////
	//History
	///////////
	ZAP_TUNABLE_INT(quietHistBonusDepthScale, 353, 0, 500, 20)
	ZAP_TUNABLE_INT(quietHistBonusOffset, 317, 0, 500, 20)
	ZAP_TUNABLE_INT(quietHistMaxBonus, 16280, 0, 32000, 256)
	ZAP_TUNABLE_INT(quietHistPenaltyDepthScale, 302, 0, 500, 20)
	ZAP_TUNABLE_INT(quietHistPenaltyOffset, 259, 0, 500, 20)
	ZAP_TUNABLE_INT(quietHistMaxPenalty, 15951, 0, 32000, 256)
	ZAP_TUNABLE_INT(maxButterflyHistory, 16349, 0, 32000, 256)
	ZAP_TUNABLE_INT(butterflyAging, 797, 0, 2000,50) 
	ZAP_TUNABLE_INT(contAging, 727, 0, 2000,50) 

	ZAP_TUNABLE_INT(maxCapturePieceHistoryBonus,16000,256,32000,256)
	ZAP_TUNABLE_INT(noisyHistBonusDepthScale, 318, 0, 500, 20)
	ZAP_TUNABLE_INT(noisyHistBonusOffset, 312, 0, 500, 20)
	ZAP_TUNABLE_INT(noisyHistMaxBonus, 16150, 0, 32000, 256)
	ZAP_TUNABLE_INT(noisyHistPenaltyDepthScale, 282, 0, 500, 20)
	ZAP_TUNABLE_INT(noisyHistPenaltyOffset, 260, 0, 500, 20)
	ZAP_TUNABLE_INT(noisyHistMaxPenalty, 15942, 0, 32000, 256)

	ZAP_TUNABLE_INT(maxContHistory, 16288, 256, 32000, 256)

	ZAP_TUNABLE_INT(contWeight1Ply, 107, 0, 200, 10)
	ZAP_TUNABLE_INT(contWeight2Ply, 104, 0, 200, 10)
	ZAP_TUNABLE_INT(contWeight4Ply, 88, 0, 200, 10)
	ZAP_TUNABLE_INT(contWeight6Ply, 32, 0, 200, 10)

	ZAP_TUNABLE_INT(movegenContWeight1Ply, 107, 0, 200, 10)
	ZAP_TUNABLE_INT(movegenContWeight2Ply, 104, 0, 200, 10)
	ZAP_TUNABLE_INT(movegenContWeight4Ply, 88, 0, 200, 10)
	ZAP_TUNABLE_INT(movegenContWeight6Ply, 32, 0, 200, 10)
};
#endif
