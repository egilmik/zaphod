#include "movegenerator.h"
#include <iostream>

void MoveGenerator::generateMoves(Board board)
{
    BitBoard pawnMoves = NULL;

    BitBoard pawns = board.whitePawns;

    
    int fromSq = board.popLsb(pawns);;
    while (fromSq != 0)
    {
        int toSq = fromSq+8;

        if(!board.checkBit(board.pieceses,toSq)){
            std::cout << "pseudo legal move from " << fromSq << " to " << toSq << std::endl;
        }
        
        if(fromSq > 7 && fromSq < 16){
            //Two squares forward from starting position
            toSq+=8;

            if(!board.checkBit(board.pieceses,toSq)){
                std::cout << "pseudo legal move from " << fromSq << " to " << toSq << std::endl;
            }
        }
        fromSq = board.popLsb(pawns);

    }
    
    
    


}
