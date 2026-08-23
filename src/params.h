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
	ZAP_TUNABLE_INT(lmrDividerQuiet, 207, 100, 350, 10)
	ZAP_TUNABLE_INT(lmrDividerNoisy, 218, 100, 350, 10)
	ZAP_TUNABLE_INT(lmrBaseQuiet, 100, 0, 150, 10)
	ZAP_TUNABLE_INT(lmrBaseNoisy, -23, -100, 100, 10)
	ZAP_TUNABLE_INT(lmrPVReduction, 101, 0, 200, 10)
	ZAP_TUNABLE_INT(lmrImprovingReduction, 94, 0, 200, 10)
	ZAP_TUNABLE_INT(lmrCheckReduction, 101, 0, 200, 10)

	//Razoring
	ZAP_TUNABLE_INT(razoringMargin, 293, 100, 400, 20)

	//Qsearch
	ZAP_TUNABLE_INT(futilityBaseQsearch, 180, 0, 300, 20)
	ZAP_TUNABLE_INT(seeMarginQsearch, -166, -200, 100, 20)

	//Reverse Futility Pruning
	ZAP_TUNABLE_INT(rfpLinear, 76, 0, 200, 20)
	ZAP_TUNABLE_INT(rfpQuadratic, 10, 0, 200, 20)
	ZAP_TUNABLE_INT(rfpImproving, 80, 0, 200, 20)

	//History
	ZAP_TUNABLE_INT(quietHistBonusDepthScale, 300, 0, 500, 20)
	ZAP_TUNABLE_INT(quietHistBonusOffset, 300, 0, 500, 20)
	ZAP_TUNABLE_INT(quietHistMaxBonus, 16000, 0, 32000, 256)
	ZAP_TUNABLE_INT(quietHistPenaltyDepthScale, 300, 0, 500, 20)
	ZAP_TUNABLE_INT(quietHistPenaltyOffset, 300, 0, 500, 20)
	ZAP_TUNABLE_INT(quietHistMaxPenalty, 16000, 0, 32000, 256)
	ZAP_TUNABLE_INT(maxButterflyHistory, 16000, 0, 32000, 256)
};
#endif
