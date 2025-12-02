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

	ZAP_TUNABLE_INT(lmrDivider, 225, 150, 300, 10)
	ZAP_TUNABLE_INT(lmrBaseQuiet, 75, 0, 150, 10)
	ZAP_TUNABLE_INT(lmrBaseNoisy, -10, -100, 100, 10)
	ZAP_TUNABLE_INT(lmrPVReduction, 100, 0, 200, 10)
	ZAP_TUNABLE_INT(lmrImprovingReduction, 100, 0, 200, 10)
	ZAP_TUNABLE_INT(lmrCheckReduction, 100, 0, 200, 10)

};
#endif