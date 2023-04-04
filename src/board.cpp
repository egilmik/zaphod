#include "board.h"
#include <iostream>
#include <string>

Board::Board(){
    for(int i = 0; i< 15; i++){
        bitBoardArray[i] = 0;
    }
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
            } else {
                sideToMove = Black;
            }
        
        default:
            break;
        }
    }

}


BitBoard Board::southOccludedMoves(BitBoard pieces, BitBoard empty)
{
   BitBoard flood = 0;
   while (pieces) {
      flood |= pieces;
      pieces = (pieces >> 8) & empty;
   }
   return (flood >> 8);
}

BitBoard Board::northOccludedMoves(BitBoard pieces, BitBoard empty)
{
   BitBoard flood = 0;
   while (pieces) {
      flood |= pieces;
      pieces = (pieces << 8) & empty;
   }
   return (flood << 8);
}

BitBoard Board::eastOccludedMoves(BitBoard pieces, BitBoard empty)
{
    BitBoard flood = 0;
    empty &= ~Board::FileAMask;
    while(pieces){
        flood |= pieces;
        pieces = (pieces >> 1) & empty;
    }
    return (flood >> 1) & ~Board::FileAMask;
}

BitBoard Board::westOccludedMoves(BitBoard pieces, BitBoard empty)
{
   BitBoard flood = 0;
    empty &= ~Board::FileHMask;
    while(pieces){
        flood |= pieces;
        pieces = (pieces << 1) & empty;
    }
    return (flood << 1) & ~Board::FileHMask;
}

BitBoard Board::northEastOccludedMoves(BitBoard pieces, BitBoard empty)
{
    BitBoard flood = 0;
    empty &= ~Board::FileAMask;
    while(pieces){
        flood |= pieces;
        pieces = (pieces >> 9) & empty;
    }
    return (flood >> 9) & ~Board::FileAMask;
}

BitBoard Board::northWestccludedMoves(BitBoard pieces, BitBoard empty)
{
    BitBoard flood = 0;
    empty &= ~Board::FileHMask;
    while(pieces){
        flood |= pieces;
        pieces = (pieces >> 7) & empty;
    }
    return (flood >> 7) & ~Board::FileHMask;
}

BitBoard Board::southEastOccludedMoves(BitBoard pieces, BitBoard empty)
{
    BitBoard flood = 0;
    empty &= ~Board::FileAMask;
    while(pieces){
        flood |= pieces;
        pieces = (pieces << 7) & empty;
    }
    return (flood << 7) & ~Board::FileAMask;
}

BitBoard Board::southWestOccludedMoves(BitBoard pieces, BitBoard empty)
{
    BitBoard flood = 0;
    empty &= ~Board::FileHMask;
    while(pieces){
        flood |= pieces;
        pieces = (pieces << 9) & empty;
    }
    return (flood << 9) & ~Board::FileHMask;
}

BitBoard Board::northEastOne(BitBoard pieces)
{
    return (pieces >> 9) & ~Board::FileAMask;
}

BitBoard Board::northWestOne(BitBoard pieces)
{
    return (pieces >> 7) & ~Board::FileHMask;
}

BitBoard Board::southEastOne(BitBoard pieces)
{
    return (pieces << 7) & ~Board::FileAMask;
}

BitBoard Board::southWestOne(BitBoard pieces)
{
    return (pieces << 9) & ~Board::FileHMask;
}

BitBoard Board::getBitboard(BitBoardEnum piece)
{
    return bitBoardArray[piece];
}

BitBoard Board::getEnemyBoard()
{
    if(sideToMove == White){
        return bitBoardArray[Black];
    }
    return bitBoardArray[White];
}

BitBoard Board::getOwnBoard()
{
    return bitBoardArray[sideToMove];
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
            setBit(fenToEnumBoardMap.at(value),bitNr);
            setBit(BitBoardEnum::All, bitNr);
            if(islower(value)){
                setBit(BitBoardEnum::Black, bitNr);
            } else {
                setBit(BitBoardEnum::White, bitNr);
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

void Board::setBit(BitBoardEnum piece, int bitNr)
{
    BitBoard board = bitBoardArray[piece];    
    board |= 1ULL << bitNr;    
    bitBoardArray[piece] = board;

}

void Board::popBit(BitBoardEnum piece, int bitNr)
{
    BitBoard board = bitBoardArray[piece];    
    board &= ~(1ULL <<bitNr);    
    bitBoardArray[piece] = board;
}

bool Board::checkBit(BitBoard &board, int bitNr)
{
    return (board >> bitNr) & 1U;
}

bool Board::checkBit(BitBoardEnum piece, int bitNr)
{
    BitBoard board = bitBoardArray[piece];
    return (board >> bitNr) & 1U;    
}

int Board::popLsb(BitBoard& board)
{   
    int lsb = __builtin_ctzll(board);
    board &= board - 1;
    return lsb;
}

bool Board::makeMove(int fromSq, int toSq,BitBoardEnum piece, bool capture, BitBoardEnum promotion)
{
    for(int i = 0; i< 15; i++){
        bitBoardArrayCopy[i] = bitBoardArray[i];
    }
    sideToMoveCopy = sideToMove;

    bool inAllBoard = checkBit(BitBoardEnum::All,fromSq);
    bool inPieceSpecificBoard = checkBit(piece,fromSq);

    if(!inAllBoard || !inPieceSpecificBoard){
        std::cout << "Error in makeMove() " << piece << " from " << fromSq << " to " << toSq << " in all board:" << inAllBoard << " in piece specific:" << inPieceSpecificBoard  <<  std::endl;
        return false;
    }

    popBit(BitBoardEnum::All,fromSq);
    setBit(BitBoardEnum::All,toSq);
    popBit(piece,  fromSq);
    setBit(piece, toSq);
    if(sideToMove == BitBoardEnum::White){
        popBit(BitBoardEnum::White,fromSq);
        setBit(BitBoardEnum::White,toSq);
    } else {
        popBit(BitBoardEnum::Black,fromSq);
        setBit(BitBoardEnum::Black,toSq);
    }


    int black = BitBoardEnum::Black;
    int white = BitBoardEnum::White;
    int all = BitBoardEnum::All;
    int currentPiece = piece;

    for(int i = 0; i < 15; i++){
        if((i != black) &&
            (i != white) &&
            (i != all) &&
            (i != currentPiece)){
                BitBoardEnum val = static_cast<BitBoardEnum>(i);
                popBit(val,toSq);
            }
    }

    if(promotion != Board::All){        
        popBit(piece,toSq);
        setBit(promotion,toSq);
    }

    if(sideToMove == White){
        if(isSquareAttacked(bitBoardArray[K], White)){
            return false;
        }
        
    } else {
        if(isSquareAttacked(bitBoardArray[k], Black)){
            return false;
        }
    }

    changeSideToMove();
    return true;
}

void Board::revertLastMove()
{
    for(int i = 0; i< 15; i++){
        bitBoardArray[i] = bitBoardArrayCopy[i];
    }
    sideToMove = sideToMoveCopy;
}

bool Board::isSquareAttacked(BitBoard targetSquares, BitBoardEnum sideAttacked)
{
    BitBoard allPieces = bitBoardArray[All];
    BitBoard queenRooks = 0;
    BitBoard queenBishops = 0;
    BitBoard knights = 0;
    BitBoard king = 0;

    if(sideAttacked == BitBoardEnum::White){
        BitBoard blackPawns = bitBoardArray[p];
        
        if((northWestOne(blackPawns) & targetSquares) != 0) return true; 
        if((northEastOne(blackPawns) & targetSquares) != 0) return true;

        king = bitBoardArray[k];
        knights = bitBoardArray[n];
        queenRooks = bitBoardArray[q] | bitBoardArray[r];
        queenBishops = bitBoardArray[q] | bitBoardArray[b];
        

    } else {
        BitBoard whitePawns = bitBoardArray[P];
        if((southEastOne(whitePawns) & targetSquares) != 0) return true; 
        if((southWestOne(whitePawns) & targetSquares) != 0) return true;
        
        king = bitBoardArray[K];
        knights = bitBoardArray[N];
        queenRooks = bitBoardArray[Q] | bitBoardArray[R];
        queenBishops = bitBoardArray[Q] | bitBoardArray[B];
    }

        
        int knightSquare = 0;
        while(knights != 0){
            knightSquare = popLsb(knights);
            if((knightmask[knightSquare] & targetSquares) != 0) return true;
        }

        if((southOccludedMoves(queenRooks, ~allPieces) & targetSquares) != 0) return true;
        if((westOccludedMoves(queenRooks, ~allPieces) & targetSquares) != 0) return true;
        if((eastOccludedMoves(queenRooks, ~allPieces) & targetSquares) != 0) return true;
        if((northOccludedMoves(queenRooks, ~allPieces) & targetSquares) != 0) return true;
        if((northEastOccludedMoves(queenBishops, ~allPieces) & targetSquares) != 0) return true;
        if((northWestccludedMoves(queenBishops, ~allPieces) & targetSquares) != 0) return true;
        if((southEastOccludedMoves(queenBishops, ~allPieces) & targetSquares) != 0) return true;
        if((southWestOccludedMoves(queenBishops, ~allPieces) & targetSquares) != 0) return true;

    return false;
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
