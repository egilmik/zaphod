#include "board.h"
#include <iostream>
#include <string>

Board::Board(){

}

void Board::parseFen(std::string fen){
    int count = 0;
    int state = 0;
    for(std::string::size_type i = 0; i < fen.size(); ++i) {

        if(fen[i] == ' '){
            state++;
        }

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

    if(isdigit(value)){
        std::string s = &value;
        int increment = std::stoi(s);
        count+= increment;
    } else { 

        switch (value)
        {
        case 'r':
            setBit(pieceses, true, bitNr);
            setBit(blackRooks,true, bitNr);
            count++;
            break;
        case 'R':
            setBit(pieceses, true, bitNr);
            setBit(whiteRooks,true, bitNr);
            count++;
            break;
        case '/':
            //Do nothing
            break;
        case 'n':
            setBit(pieceses,true, bitNr);
            setBit(blackKnights,true,bitNr);
            count++;
            break;
        case 'N':
            setBit(pieceses,true, bitNr);
            setBit(whiteKnights,true,bitNr);
            count++;
            break;
        case 'b':
            setBit(pieceses,true, bitNr);
            setBit(blackBishops,true,bitNr);
            count++;
            break;
        case 'B':
            setBit(pieceses,true, bitNr);
            setBit(whiteBishops,true,bitNr);
            count++;
            break;
        case 'q':
            setBit(pieceses,true, bitNr);
            setBit(blackQueens,true,bitNr);
            count++;
            break;
        case 'Q':
            setBit(pieceses,true, bitNr);
            setBit(whiteQueens,true,bitNr);
            count++;
            break;
        case 'k':
            setBit(pieceses,true, bitNr);
            setBit(blackKing,true,bitNr);
            count++;
            break;
        case 'K':
            setBit(pieceses,true, bitNr);
            setBit(whiteKing,true,bitNr);
            count++;
            break;
        case 'p':
            setBit(pieceses,true, bitNr);
            setBit(blackPawns,true,bitNr);
            count++;
            break;
        case 'P':
            setBit(pieceses,true, bitNr);
            setBit(whitePawns,true,bitNr);
            count++;
            break;

    
        default:
            count++;
            break;
        }
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
        } else if(checkBit(whiteBishops,i)){
            printBoard[i] = 'B';
        } else if(checkBit(blackBishops,i)){
            printBoard[i] = 'b';
        } else if(checkBit(whiteQueens,i)){
            printBoard[i] = 'Q';
        } else if(checkBit(blackQueens,i)){
            printBoard[i] = 'q';
        } else if(checkBit(whiteKing,i)){
            printBoard[i] = 'K';
        } else if(checkBit(blackKing,i)){
            printBoard[i] = 'k';
        } else if(checkBit(whitePawns,i)){
            printBoard[i] = 'P';
        } else if(checkBit(blackPawns,i)){
            printBoard[i] = 'p';
        }
    }

    for(int i = 0;i < 64; i++){
        if(i%8== 0){
            std::cout << std::endl;
        }
        std::cout << printBoard[i] << " ";
    }

    std::cout << std::endl;

    
}

