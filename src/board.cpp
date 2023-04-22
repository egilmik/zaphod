#include "board.h"
#include <iostream>
#include <string>
#include <cstring>

static std::array<BitBoard,64> initSqToBitMapping(){
    std::array<BitBoard,64> mapping;


    for(int i = 0; i < 64; i++){
        BitBoard bb = 0;
        Board::setBit(bb,i);
        mapping[i] = bb;
    }

    return mapping;
}

static std::array<BitBoard,64> initInvertedSqToBitMapping(){
    std::array<BitBoard,64> mapping;


    for(int i = 0; i < 64; i++){
        BitBoard bb = 0;
        Board::setBit(bb,i);
        mapping[i] = ~bb;
    }

    return mapping;
}

static std::array<BitBoard,64> initKingMask(){
    std::array<BitBoard,64> kingMask;
    for(int i =0; i< 64;i++){
        BitBoard piece = 0;
        BitBoard moves = 0;
        
        Board::setBit(piece,true,i);

        if(!(piece & Board::FileAMask)){
            moves |= piece >> 7 ;
            moves |= piece << 1 ;
            moves |= piece << 9 ;
        }
        

        if(!(piece & Board::FileHMask)){
            moves |= piece >> 9;
            moves |= piece >> 1;
            moves |= piece << 7 ;
        }
        moves |= piece >> 8;
        moves |= piece << 8;

        kingMask[i] = moves;
    }

    return kingMask;
}

static std::array<BitBoard,64> initKnightMask()
{  
    std::array<BitBoard,64> knightmask;
    for(int i =0; i< 64;i++){
        BitBoard piece = 0;
        BitBoard moves = 0;

        
        Board::setBit(piece,true,i);

        if(!(piece & Board::FileAMask)){
            moves |= piece >> 15;
            moves |= piece << 17;
        }
        if(!(piece & Board::FileABMask)){
            moves |= piece >> 6;
            moves |= piece << 10;
        }

        if(!(piece & Board::FileHMask)){
            moves |= piece >> 17;
            moves |= piece << 15;
        }

        if(!(piece & Board::FileGHMask)){
            moves |= piece >> 10;
            moves |= piece << 6;
        }

        knightmask[i] = moves;
        
    }
    return knightmask;

}


const std::array<BitBoard,64> Board::kingMask = initKingMask();
const std::array<BitBoard,64> Board::knightmask = initKnightMask();

Board::Board(){
    for(int i = 0; i< 15; i++){
        bitBoardArray[i] = 0;
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

void Board::clearBoard()
{
    for(int i = 0; i < 15; i++){
        bitBoardArray[i] = 0;
    }
    sideToMove = White;
    enPassantSq = noSq;
    castleWK = false;
    castleWQ = false;
    castleBK = false;
    castleBQ = false;
}

void Board::parseFen(std::string fen){
    clearBoard();
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
            break;
        case 2:
            if(fen[i] == 'K'){ castleWK = true;}
            if(fen[i] == 'Q'){ castleWQ = true;}
            if(fen[i] == 'k'){ castleBK = true;}
            if(fen[i] == 'q'){ castleBQ = true;}
        
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

BitBoardEnum Board::getSideToMove()
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
    bitBoardArray[piece] = bitBoardArray[piece] &= ~(1ULL <<bitNr);
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

int Board::countSetBits(BitBoardEnum piece)
{
    BitBoard board = bitBoardArray[piece];
    int count = 0;
    while(board != 0){
        popLsb(board);
        count++;
    }
    return count;
}

int Board::popLsb(BitBoard& board)
{   
    int lsb = __builtin_ctzll(board);
    board &= board - 1;
    return lsb;
}

bool Board::makeMove(Move move)
{
    return makeMove(move.fromSq,move.toSq,move.piece,move.capture,move.enpassant,move.doublePawnPush,move.castling,move.promotion);
}

bool Board::makeMove(int fromSq, int toSq,BitBoardEnum piece, bool capture,bool enPassant, bool doublePush,bool castling, BitBoardEnum promotion)
{
    int size = 15*sizeof(bitBoardArray[0]);
    std::memcpy(&bitBoardArrayCopy,&bitBoardArray,size);
    sideToMoveCopy = sideToMove;
    enPassantSqCopy = enPassantSqCopy;
    castleWKCopy = castleWK;
    castleWQCopy= castleWQ;
    castleBKCopy = castleBK;
    castleBQCopy = castleBQ;

    
    /*
    bool inAllBoard = checkBit(BitBoardEnum::All,fromSq);
    bool inPieceSpecificBoard = checkBit(piece,fromSq);

    if(!inAllBoard || !inPieceSpecificBoard){
        printBoard();
        std::cout << "Error in makeMove() " << piece << " from " << fromSq << " to " << toSq << " in all board:" << inAllBoard << " in piece specific:" << inPieceSpecificBoard  <<  std::endl;
        return false;
    }
    */

    BitBoardEnum enemy = BitBoardEnum::White;
    if(sideToMove == White){
        enemy = Black;
    }

    BitBoard fromToBoard = 0;
    fromToBoard = sqBB[fromSq] ^ sqBB[toSq];

    // Pop and set bits in piece and all board
    bitBoardArray[All] &= ~sqBB[fromSq];
    bitBoardArray[All] |= sqBB[toSq];
    bitBoardArray[piece] &= ~sqBB[fromSq];
    bitBoardArray[piece] |= sqBB[toSq];
    bitBoardArray[sideToMove] &= ~sqBB[fromSq];
    bitBoardArray[sideToMove] |= sqBB[toSq];


    if(capture){
        bitBoardArray[enemy] &= ~sqBB[toSq];
        bitBoardArray[P+enemy] &= ~sqBB[toSq];
        bitBoardArray[N+enemy] &= ~sqBB[toSq];
        bitBoardArray[Q+enemy] &= ~sqBB[toSq];
        bitBoardArray[B+enemy] &= ~sqBB[toSq];
        bitBoardArray[R+enemy] &= ~sqBB[toSq];
    }

    

    if(sideToMove == BitBoardEnum::White){

        if(doublePush){
            setEnPassantSq(toSq-8);
        } else {
            setEnPassantSq(noSq);
        }

        if(castling){
            if(toSq == 2){
                popBit(BitBoardEnum::All,0);
                popBit(BitBoardEnum::White,0);
                popBit(BitBoardEnum::R, 0);
                setBit(BitBoardEnum::All,3);
                setBit(BitBoardEnum::White,3);
                setBit(BitBoardEnum::R,3);
            } else {
                popBit(BitBoardEnum::All,7);
                popBit(BitBoardEnum::White,7);
                popBit(BitBoardEnum::R, 7);
                setBit(BitBoardEnum::All,5);
                setBit(BitBoardEnum::White,5);
                setBit(BitBoardEnum::R,5);
            }
        }

    } else {

        if(doublePush){
            setEnPassantSq(toSq+8);
        } else {
            setEnPassantSq(noSq);
        }

        if(castling){
            if(toSq == 58){
                popBit(BitBoardEnum::All,56);
                popBit(BitBoardEnum::Black,56);
                popBit(BitBoardEnum::r, 56);
                setBit(BitBoardEnum::All,59);
                setBit(BitBoardEnum::Black,59);
                setBit(BitBoardEnum::r,59);
            } else {
                popBit(BitBoardEnum::All,63);
                popBit(BitBoardEnum::Black,63);
                popBit(BitBoardEnum::r, 63);
                setBit(BitBoardEnum::All,61);
                setBit(BitBoardEnum::Black,61);
                setBit(BitBoardEnum::r,61);
            }
        }
    }

    if(enPassant){
        if(sideToMove == BitBoardEnum::White){
            popBit(p,toSq-8);
            popBit(All, toSq-8);
            popBit(Black, toSq-8);
        } else {
            popBit(P,toSq+8);
            popBit(All, toSq+8);
            popBit(White, toSq+8);
            
        }  
        //std::cout << "En passant " << sqToNotation[fromSq] << " " << sqToNotation[toSq] << std::endl;
    }    
    if(promotion != BitBoardEnum::All){        
        popBit(piece,toSq);
        setBit(promotion,toSq);
    }

    //TODO Castline status overly complex
    //Update castling rights
    if(piece == K){
        castleWK = false;
        castleWQ = false;
    }

    if(piece == k){
        castleBK = false;
        castleBQ = false;
    }


    if(piece == R){
        if(fromSq == 0){
            castleWQ = false;
        } else if(fromSq == 7) {
            castleWK = false;
        }
    } else if( piece == r){
        if(fromSq == 56){
            castleBQ = false;
        } else if( fromSq == 63){
            castleBK = false;
        }
    }



    if(toSq == 0 && capture){
        castleWQ = false;
    }

    if(toSq == 7 && capture){
        castleWK = false;
    }

    if(toSq == 56 && capture){
        castleBQ = false;
    }

    if(toSq == 63 && capture){
        castleBK = false;
    }

    
    
    if(isSquareAttacked(bitBoardArray[K+sideToMove], sideToMove)){
        return false;
    }
        

    changeSideToMove();
    return true;
}

void Board::revertLastMove()
{
    int size = 15*sizeof(bitBoardArray[0]);
    std::memcpy(&bitBoardArray,&bitBoardArrayCopy,size);
    sideToMove = sideToMoveCopy;
    enPassantSq = enPassantSqCopy;
    castleWK = castleWKCopy;
    castleWQ = castleWQCopy;
    castleBK = castleBKCopy;
    castleBQ = castleBQCopy;
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

    if((kingMask[popLsb(king)] & targetSquares) != 0) return true;

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

        if(checkBit(BitBoardEnum::R,i)){
            printBoard[i] = 'R';
        } else if(checkBit(BitBoardEnum::r,i)){
            printBoard[i] = 'r';
        } else if(checkBit(BitBoardEnum::N,i)){
            printBoard[i] = 'N';
        } else if(checkBit(BitBoardEnum::n,i)){
            printBoard[i] = 'n';
        } else if(checkBit(BitBoardEnum::B,i)){
            printBoard[i] = 'B';
        } else if(checkBit(BitBoardEnum::b,i)){
            printBoard[i] = 'b';
        } else if(checkBit(BitBoardEnum::Q,i)){
            printBoard[i] = 'Q';
        } else if(checkBit(BitBoardEnum::q,i)){
            printBoard[i] = 'q';
        } else if(checkBit(BitBoardEnum::K,i)){
            printBoard[i] = 'K';
        } else if(checkBit(BitBoardEnum::k,i)){
            printBoard[i] = 'k';
        } else if(checkBit(BitBoardEnum::P,i)){
            printBoard[i] = 'P';
        } else if(checkBit(BitBoardEnum::p,i)){
            printBoard[i] = 'p';
        }
    }

    for(int i = 7; i >= 0; i--){
        int startSquare = 8 * i;
        for(int x = 0; x < 8; x++){
            std::cout << printBoard[startSquare+x] << " ";    
        }

        std::cout << std::endl;

    }

    /*
    for(int i = 63;i >= 0; i--){
        if((i+1)%8== 0){
            std::cout << std::endl;
        }
        std::cout << printBoard[i] << " ";
    }
    */

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
