#ifndef UCI_H
#define UCI_H

#ifndef ZAPHOD_VERSION
#define ZAPHOD_VERSION "2.0-dev"
#endif

#include <vector>
#include <iostream>
#include "board.h"
#include "movegenerator.h"
#include "search.h"


class UCI {
    public:
        UCI() {};
        void loop();
        void setNetworkPath(std::string networkPath);

    private:
        void setUCINewGame();
        void setPosition(std::istringstream &is);
        bool parseMove(std::string token);
        void startSearch(std::istringstream &is);
        void staticEvaluation();
        void sendID();
        void setOption(std::istringstream& is);

        void perft();
        void printFen();
        void bench();

        Board motherBoard;
        Search search;
        std::string networkPath;
};

#endif