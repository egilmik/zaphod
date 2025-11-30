#ifndef PARAMS_H
#define PARAMS_H

#include <string>
#include <algorithm>

namespace zaphod::params {

	struct Parameter {
		std::string name;
		int32_t default;
		int32_t min;
		int32_t max;
		int32_t step;
	};

	Parameter& addParameter(std::string name, int32_t default, int32_t min, int32_t max, int32_t step);

#define ZAP_TUNABLE_INT(Name, Default, Min, Max, Step) \
	inline Parameter& param_##Name = addParameter(#Name, Default, Min, Max, Step); \
	inline int32_t Name() { \
		return param_##Name.D; \
	}


	ZAP_TUNABLE_INT(lmrDivider, 2.25, 1.5, 3, 0.1);
};
#endif