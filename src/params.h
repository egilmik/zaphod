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

	// LMR
	ZAP_TUNABLE_INT(lmrDividerQuiet, 184, 100, 350, 10)
	ZAP_TUNABLE_INT(lmrDividerNoisy, 221, 100, 350, 10)
	ZAP_TUNABLE_INT(lmrBaseQuiet, 96, 0, 150, 10)
	ZAP_TUNABLE_INT(lmrBaseNoisy, -41, -100, 100, 10)
	ZAP_TUNABLE_INT(lmrPVReduction, 79, 0, 200, 10)
	ZAP_TUNABLE_INT(lmrImprovingReduction, 87, 0, 200, 10)
	ZAP_TUNABLE_INT(lmrCheckReduction, 104, 0, 200, 10)
	ZAP_TUNABLE_INT(lmrHistoryReduction, 250,0,600,20)

	//Razoring
	ZAP_TUNABLE_INT(razoringMargin, 271, 100, 400, 20)

	//Qsearch
	ZAP_TUNABLE_INT(futilityBaseQsearch, 164, 0, 300, 20)
//	ZAP_TUNABLE_INT(seeMarginQsearch, -166, -200, 100, 20)

	//Reverse Futility Pruning
	ZAP_TUNABLE_INT(rfpLinear, 97, 0, 200, 20)
//	ZAP_TUNABLE_INT(rfpQuadratic, 10, 0, 200, 20)
	ZAP_TUNABLE_INT(rfpImproving, 104, 0, 200, 20)

	//History
	ZAP_TUNABLE_INT(quietHistBonusDepthScale, 325, 0, 500, 20)
	ZAP_TUNABLE_INT(quietHistBonusOffset, 333, 0, 500, 20)
	ZAP_TUNABLE_INT(quietHistMaxBonus, 16129, 0, 32000, 256)
	ZAP_TUNABLE_INT(quietHistPenaltyDepthScale, 300, 0, 500, 20)
	ZAP_TUNABLE_INT(quietHistPenaltyOffset, 292, 0, 500, 20)
	ZAP_TUNABLE_INT(quietHistMaxPenalty, 15835, 0, 32000, 256)
	ZAP_TUNABLE_INT(maxButterflyHistory, 16240, 0, 32000, 256)
	ZAP_TUNABLE_INT(butterflyAging, 750, 0, 2000,50) 
	ZAP_TUNABLE_INT(contAging, 750, 0, 2000,50) 

	ZAP_TUNABLE_INT(maxContHistory, 16000, 256, 32000, 256)
	ZAP_TUNABLE_INT(contWeight1Ply, 120, 0, 200, 10)
	ZAP_TUNABLE_INT(contWeight2Ply, 120, 0, 200, 10)
	ZAP_TUNABLE_INT(contWeight4Ply, 60, 0, 200, 10)
	ZAP_TUNABLE_INT(contWeight6Ply, 30, 0, 200, 10)
};
#endif
