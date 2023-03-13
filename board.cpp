#include "board.h"
#include <iostream>

Board::Board(){

}

void Board::parseFen(std::string fen){
    int count = 0;
    int state = 0;
    for(std::string::size_type i = 0; i < fen.size(); ++i) {

        switch (state)
        {
        case 0:
            parseFenPosition(fen[i], count);
            break;
        
        default:
            break;
        }
    }


}

void Board::parseFenPosition(char value, int &count){

    int bitNr = fenToBitMapping[count];
    
    if(fenToBitboardMap.count(value) > 0){
        BitBoard board = *fenToBitboardMap.find(value)->second;
        setBit(pieceses, true, bitNr);
        setBit(board,true, bitNr);
        count++;
    }
    
}

void Board::setBit(BitBoard &board, bool highLow, int bitNr)
{
    board |= 1ULL << bitNr;
}

bool Board::checkBit(BitBoard board, int bitNr)
{
    return (board >> bitNr) & 1U;
}

void Board::printBoard(){
    uint64_t count = 0;
    char printBoard[64];

    for(int i = 0; i < 64; i++){
        printBoard[i] = '*';
        if(checkBit(whiteRooks,i)){
            printBoard[i] = 'R';
        } else if(checkBit(blackRooks,i)){
            printBoard[i] = 'r';
        } else if(checkBit(whiteKnights,i)){
            printBoard[i] = 'N';
        } else if(checkBit(blackKnights,i)){
            printBoard[i] = 'n';
        }
    }

    for(int i = 64;i > 0; i--){
        if(i%8== 0){
            std::cout << std::endl;
        }
        std::cout << printBoard[i-1] << " ";
    }

    std::cout << std::endl;


    
}

