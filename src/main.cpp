#include <iostream>
#include "board.h"
#include "movegenerator.h"
#include "perft.h"
#include "test.h"
#include "search.h"
#include "uci.h"
#include <chrono>

int main(int, char**) {

    UCI uci;
    uci.loop();
}
