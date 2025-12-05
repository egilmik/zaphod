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
	ZAP_TUNABLE_INT(lmrDividerQuiet, 221, 100, 350, 10)
	ZAP_TUNABLE_INT(lmrDividerNoisy, 226, 100, 350, 10)
	ZAP_TUNABLE_INT(lmrBaseQuiet, 87, 0, 150, 10)
	ZAP_TUNABLE_INT(lmrBaseNoisy, -7, -100, 100, 10)
	ZAP_TUNABLE_INT(lmrPVReduction, 101, 0, 200, 10)
	ZAP_TUNABLE_INT(lmrImprovingReduction, 99, 0, 200, 10)
	ZAP_TUNABLE_INT(lmrCheckReduction, 101, 0, 200, 10)

	//Razoring
	ZAP_TUNABLE_INT(razoringMargin, 210, 100, 400, 20)

	//Qsearch
	ZAP_TUNABLE_INT(futilityBaseQsearch, 195, 0, 300, 20)
	ZAP_TUNABLE_INT(seeMarginQsearch, -153, -200, 100, 20)

};
#endif