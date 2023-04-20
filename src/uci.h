#ifndef UCI_H
#define UCI_H

#include "board.h"
#include <vector>
#include "movegenerator.h"
#include <iostream>

class UCI {
    public:
        void loop();
        

    private:
        void setPosition(std::istringstream &is);
        bool parseMove(std::string token);
        void startSearch(std::istringstream &is);
        void sendID();

        

        Board motherBoard;
};

#endif