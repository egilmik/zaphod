#include <iostream>
#include "board.h"
#include "movegenerator.h"
#include "search.h"
#include "uci.h"
#include <chrono>

int main(int argc, char* argv[]) {

    std::string networkPath;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-network" && i + 1 < argc) {
            networkPath = argv[i + 1];
        }
    }

    UCI uci;
    if (!networkPath.empty()) {
        uci.setNetworkPath(networkPath);
    }

    
    uci.loop();
}
