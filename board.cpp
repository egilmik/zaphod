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
        case 1:
            if(fen[i] == 'w'){
                sideToMove = White;
            }
        
        default:
            break;
        }
    }


}

BitBoard Board::getBitboard(BitBoardEnum piece)
{
    return bitBoardMap.at(piece);
}

void Board::parseFenPosition(char value, int &count)
{

    int bitNr = fenToBitMapping[count];

    if(isdigit(value)){
        std::string s = &value;
        int increment = std::stoi(s);
        count+= increment;
    } else { 

        if (fenToEnumBoardMap.find(value) != fenToEnumBoardMap.end()){
            setBit(fenToEnumBoardMap.at(value),true,bitNr);
            setBit(BitBoardEnum::All, true, bitNr);
            count++;
        } else if(value == '/'){

        }
    }
}

void Board::setBit(BitBoard &board, bool highLow, int bitNr)
{
    board |= 1ULL << bitNr;
}

void Board::setBit(BitBoardEnum piece, bool highLow, int bitNr)
{
    std::map<BitBoardEnum,BitBoard>::iterator itr;
    itr = bitBoardMap.find(piece);
    BitBoard board = itr->second;
    board |= 1ULL << bitNr;
    itr->second = board;

}

bool Board::checkBit(BitBoard board, int bitNr)
{
    return (board >> bitNr) & 1U;
}

bool Board::checkBit(BitBoardEnum piece, int bitNr)
{
    BitBoard board = bitBoardMap.at(piece);
    return (board >> bitNr) & 1U;    
}

int Board::popLsb(BitBoard& board)
{   
    int lsb = __builtin_ctzll(board);
    board &= board - 1;
    return lsb;
}

void Board::makeMove(int fromSq, int toSq, bool capture)
{
    
}

void Board::printBoard(){
    uint64_t count = 0;
    char printBoard[64];

    for(int i = 0; i < 64; i++){
        printBoard[i] = '*';
        if(checkBit(Board::R,i)){
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

