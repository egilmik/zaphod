#include "board.h"
#include <iostream>
#include <string>

Board::Board(){
    initKnightMask();
    initKingMask();
    initRayAttacks();
}

void Board::initRayAttacks()
{
    BitBoard north = FileHMask;
    BitBoard south = FileHMask;
    

    for(int i = 0; i < 64; i++){
        rayAttackNorth[i] = north <<=1;
    }

    for(int i = 63; i >= 0; i--){
        rayAttackSouth[i] = south >>=1;
    }

    for(int i = 0; i< 64; i++){
        BitBoard one = 1;
        rayAttackEast[i] = ((one << (i|7)));//- (one << i))); 
    }
}

void Board::initKingMask(){
    for(int i =0; i< 64;i++){
        BitBoard piece = 0;
        BitBoard moves = 0;
        
        setBit(piece,true,i);

        if(!(piece & FileAMask)){
            moves |= piece >> 7 ;
            moves |= piece << 1 ;
            moves |= piece << 9 ;
        }
        

        if(!(piece & FileHMask)){
            moves |= piece >> 9;
            moves |= piece >> 1;
            moves |= piece << 7 ;
        }
        moves |= piece >> 8;
        moves |= piece << 8;

        kingMask[i] = moves;
        
    }
}

void Board::initKnightMask()
{  
    for(int i =0; i< 64;i++){
        BitBoard piece = 0;
        BitBoard moves = 0;

        
        setBit(piece,true,i);

        if(!(piece & FileAMask)){
            moves |= piece >> 15;
            moves |= piece << 17;
        }
        if(!(piece & FileABMask)){
            moves |= piece >> 6;
            moves |= piece << 10;
        }

        if(!(piece & FileHMask)){
            moves |= piece >> 17;
            moves |= piece << 15;
        }

        if(!(piece & FileGHMask)){
            moves |= piece >> 10;
            moves |= piece << 6;
        }

        knightmask[i] = moves;
        
    }

}

BitBoard Board::getKnightMask(int square)
{
    return knightmask[square];
}

BitBoard Board::getKingMask(int square)
{
    return kingMask[square];
}

BitBoard Board::getRankMask(int square)
{
    BitBoard mask = 0xff;
    return mask << (square & 56);
}

BitBoard Board::getLineMask(int square)
{
    BitBoard mask = 0x0101010101010101;
    return mask << (square & 7);
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

BitBoard Board::getEnemyBoard()
{
    if(sideToMove == White){
        return bitBoardMap.at(Black);
    }
    return bitBoardMap.at(White);
}

BitBoard Board::getOwnBoard()
{
    return bitBoardMap.at(sideToMove);
}

void Board::changeSideToMove()
{
    if(sideToMove == BitBoardEnum::White){
       sideToMove = BitBoardEnum::Black;     
    } else {
        sideToMove = BitBoardEnum::White;
    }
}

Board::BitBoardEnum Board::getSideToMove()
{
    return sideToMove;
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
            if(islower(value)){
                setBit(BitBoardEnum::Black, true, bitNr);
            } else {
                setBit(BitBoardEnum::White,true, bitNr);
            }
            count++;
        } else if(value == '/'){

        }
    }
}

void Board::popBit(BitBoard &board, int bitNr)
{
    board &= ~(1ULL << bitNr);
}

void Board::setBit(BitBoard &board, int bitNr)
{
    board |= 1ULL << bitNr;
}

void Board::setBit(BitBoard &board, bool highLow, int bitNr)
{
    board |= 1ULL << bitNr;
}

// TODO Possible performance hog!
void Board::setBit(BitBoardEnum piece, bool highLow, int bitNr)
{
    BitBoard board = bitBoardMap[piece];
    if(highLow){
        board |= 1ULL << bitNr;
    } else {
        board &= ~(1ULL <<bitNr);
    }    
    bitBoardMap[piece] = board;

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

void Board::makeMove(int fromSq, int toSq,BitBoardEnum piece, bool capture)
{
    std::map<BitBoardEnum,BitBoard> copy(bitBoardMap);
    PreviousbitBoardMap = copy;
    sideToMoveCopy = sideToMove;

    bool inAllBoard = checkBit(BitBoardEnum::All,fromSq);
    bool inPieceSpecificBoard = checkBit(piece,fromSq);

    if(!inAllBoard || !inPieceSpecificBoard){
        std::cout << "Error in makeMove() " << piece << " from " << fromSq << " to " << toSq << " in all board:" << inAllBoard << " in piece specific:" << inPieceSpecificBoard  <<  std::endl;
        return;
    }

    setBit(BitBoardEnum::All,false,fromSq);
    setBit(BitBoardEnum::All,true,toSq);
    setBit(piece, false, fromSq);
    setBit(piece,true, toSq);
    if(sideToMove == BitBoardEnum::White){
        // TODO Possible bug in using setbit for popbit
        setBit(BitBoardEnum::White,false,fromSq);
        setBit(BitBoardEnum::White,true,toSq);
    } else {
        setBit(BitBoardEnum::Black,false,fromSq);
        setBit(BitBoardEnum::Black,true,toSq);
    }

    std::map<BitBoardEnum,BitBoard>::iterator itr;
    for(itr = bitBoardMap.begin(); itr != bitBoardMap.end(); ++itr){
        if(itr->first != BitBoardEnum::Black &&
            itr->first != BitBoardEnum::White &&
            itr->first != BitBoardEnum::All &&
            itr->first != piece){
                setBit(itr->first,false,toSq);
            }

    }

    changeSideToMove();
}

void Board::revertLastMove()
{
    bitBoardMap = PreviousbitBoardMap;
    sideToMove = sideToMoveCopy;
}

void Board::printBoard(){
    uint64_t count = 0;
    char printBoard[64];

    for(int i = 0; i < 64; i++){
        printBoard[i] = '*';

        if(checkBit(Board::R,i)){
            printBoard[i] = 'R';
        } else if(checkBit(Board::r,i)){
            printBoard[i] = 'r';
        } else if(checkBit(Board::N,i)){
            printBoard[i] = 'N';
        } else if(checkBit(Board::n,i)){
            printBoard[i] = 'n';
        } else if(checkBit(Board::B,i)){
            printBoard[i] = 'B';
        } else if(checkBit(Board::b,i)){
            printBoard[i] = 'b';
        } else if(checkBit(Board::Q,i)){
            printBoard[i] = 'Q';
        } else if(checkBit(Board::q,i)){
            printBoard[i] = 'q';
        } else if(checkBit(Board::K,i)){
            printBoard[i] = 'K';
        } else if(checkBit(Board::k,i)){
            printBoard[i] = 'k';
        } else if(checkBit(Board::P,i)){
            printBoard[i] = 'P';
        } else if(checkBit(Board::p,i)){
            printBoard[i] = 'p';
        }
    }

    for(int i = 0;i < 64; i++){
        if((i)%8== 0){
            std::cout << std::endl;
        }
        std::cout << printBoard[i] << " ";
    }

    std::cout << std::endl;
    std::cout << std::endl;

    
}

void Board::printBoard(BitBoard board)
{

    char printBoard[64];

    for(int i = 0; i< 64; i++){
        if(checkBit(board,i)){
            printBoard[i] = 'X';
        } else {
            printBoard[i] = '*';
        }
        
    }
    
    for(int i = 63;i >= 0; i--){
        if((i+1)%8== 0){
            std::cout << std::endl;
        }
        std::cout << printBoard[i] << " ";
    }

    std::cout << std::endl;
    std::cout << std::endl;

}

void Board::printBoard(BitBoard board, int origin)
{
    char printBoard[64];

    for(int i = 0; i < 64; i++){
        printBoard[i] = '*';
        if(checkBit(board,i)){
            printBoard[i] = 'X';
        }
        if(origin == i){
            printBoard[i] = 'O';
        }
    }

    for(int i = 63;i >= 0; i--){
        if((i+1)%8== 0){
            std::cout << std::endl;
        }
        std::cout << printBoard[i] << " ";
    }

    std::cout << std::endl;
    std::cout << std::endl;
}
