#ifndef UCI_H
#define UCI_H

#include <vector>
#include <iostream>
#include "board.h"
#include "movegenerator.h"
#include "search.h"


class UCI {
    public:
        void loop();
        

    private:
        void setPosition(std::istringstream &is);
        bool parseMove(std::string token);
        void startSearch(std::istringstream &is);
        void staticEvaluation();
        void sendID();

        Board motherBoard;
        Search search;
};

#endif