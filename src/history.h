#ifndef HISTORY_H
#define HISTORY_H

#include <cstdint>

struct HistoryTables {
    // int32 rather than int16: updates are += depth*depth, so at high depth a
    // handful of cutoffs on the same move overflow a 16-bit counter and the
    // best quiet move wraps negative, sorting it last.  32 KiB total.
    int32_t quiet[2][64][64] = {};
};

#endif