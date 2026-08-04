#ifndef MATERIAL_H
#define MATERIAL_H

#include "bitboard.h"
#include <array>

class Material {

public:
    inline static const std::array<int, 14> pieceMaterialScoreArray = { 0,500,320,330,900,2000,100,0,500,320,330,900,2000,100 };
};

#endif
