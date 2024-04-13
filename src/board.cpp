#include "board.h"
#include <iostream>
#include <string>
#include <cstring>
#include "material.h"
#include <array>
#include <random>

static std::array<BitBoard,64> initSqToBitMapping(){
    std::array<BitBoard, 64> mapping;


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

void Board::initMagicMasks() {

    for (BitBoard index = 0; index < 64; index++)
    {
        BitBoard mask = 0;

        for (int i = index + 8; i < 56; i += 8) {
            setBit(mask, i);
        }
        for (int i = index - 8; i > 7; i -= 8) {
            setBit(mask, i);
        }

        for (int i = index + 1; i % 8 != 7 && i % 8 != 0; i++) {
            setBit(mask, i);
        }

        for (int i = index - 1; i % 8 != 7 && i % 8 != 0 && i >=0 ; i--) {
            setBit(mask, i);
        }

		rookMask[index] = mask;

        mask = 0;
        for (int i = index + 9; i % 8 != 7 && i % 8 != 0 && i < 56; i += 9) {
            setBit(mask, i);
        }
        for (int i = index - 9; i % 8 != 7 && i % 8 != 0 && i >= 8; i -= 9) {
            setBit(mask, i);
        }
        for (int i = index + 7; i % 8 != 7 && i % 8 != 0 && i <= 55; i += 7) {
            setBit(mask, i);
        }
        for (int i = index - 7; i % 8 != 7 && i % 8 != 0 && i >= 8; i -= 7) {
            setBit(mask, i);
        }
        bishopMask[index] = mask;
    }
}


// Inspired by http://web.archive.org/web/20160314001240/http://www.afewmorelines.com/understanding-magic-bitboards-in-chess-programming/
void Board::initMagics(bool isRook, std::array<std::array<BitBoard, 4096>, 64>* magicMoves, std::array<BitBoard,64> &moveMask, std::array<BitBoard, 64>& magicNumberArray, std::array<BitBoard, 64>& magicShiftArray) {
    //magicMovesRook = new std::array<std::array<BitBoard, 4096>, 64>();

    std::array<BitBoard, 4096> occupancy{};
    std::array<BitBoard, 4096> attackSet{};


    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<uint64_t> distribution(0, UINT64_MAX);

    for (int square = 0; square < 64; square++) {


        std::array<BitBoard, 4096> epoch{};

        //Carry-Ripler to enumerate all subsets of mask
        BitBoard mask = moveMask[square];

        BitBoard origin = 0;
        setBit(origin, square);

        int size = 0;
        BitBoard n = 0;
        do {
            occupancy[size] = n;
            
            BitBoard moves = 0;
            if (isRook) {
                moves = southOccludedMoves(origin, ~n);
                moves |= northOccludedMoves(origin, ~n);
                moves |= westOccludedMoves(origin, ~n);
                moves |= eastOccludedMoves(origin, ~n);
            }
            else {

                moves = northEastOccludedMoves(origin, ~n);
                moves |= northWestccludedMoves(origin, ~n);
                moves |= southEastOccludedMoves(origin, ~n);
                moves |= southWestOccludedMoves(origin, ~n);
            }

            attackSet[size] = moves;

            //TODO generate attackset
            n = (n - mask) & mask;
            size++;
        } while (n);

        BitBoard magicNumber = 0;
        int attempts = 0;

        bool fail = false;

        uint32_t magicShift = 52;

        do
        {
            // Make sure 
            //for (magicNumber = 0; countSetBits((rookMask[square] * magicNumber) >> magicShift) < 6; )
            magicNumber = 0;
            while (magicNumber == 0) {
                magicNumber = distribution(rng) & distribution(rng) & distribution(rng) & distribution(rng); // generate a random number with not many bits set
            }
            
            //for (int j = 0; j < size; j++) (*magicMovesRook)[square][j] = 0;
            attempts++;

            uint64_t index = 0;
            fail = false;



            for (int i = 0; i < size && !fail ; i++)
            {

                BitBoard mask = (occupancy[i] & moveMask[square]);

                index = (mask * magicNumber) >> magicShift;

                if (epoch[index] < attempts) {
                    epoch[index] = attempts;
                    (*magicMoves)[square][index] = attackSet[i];
                }
                else if ((*magicMoves)[square][index] != attackSet[i]) {
                    fail = true;
                }


                
            }
        } while (fail);
        magicNumberArray[square] = magicNumber;
        magicShiftArray[square] = magicShift;

    }



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

    ttable.initKeys();
    initMagicMasks();
    magicMovesRook = new std::array<std::array<BitBoard, 4096>, 64>();
    magicMovesBishop = new std::array<std::array<BitBoard, 4096>, 64>();
    initMagics(true, magicMovesRook, rookMask, magicNumberRook, magicNumberShiftsRook);
    initMagics(false, magicMovesBishop, bishopMask, magicNumberBishop, magicNumberShiftsBishop);

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
            break;
        case 3:
            for(int x = 0; x < 64; x++){
                if(fen.substr(i,2) == sqToNotation[x]){
                    enPassantSq = x;
                }
            }
            break;
            
        
        default:
            break;
        }
    }

    materialScore = Material::getMaterialScore(*this);
    pieceSquareScore = Material::getPieceSquareScore(*this);
    hashKey = generateHashKey();
    historyPly = 0;
}

BitBoard Board::generateHashKey(){
    BitBoard key = 0;

    for (int pieceValue = BitBoardEnum::R; pieceValue != BitBoardEnum::All; pieceValue++ ){
        if(pieceValue != BitBoardEnum::All && pieceValue != BitBoardEnum::White && pieceValue != BitBoardEnum::Black){
            BitBoardEnum pieceEnum = static_cast<BitBoardEnum>(pieceValue);
            BitBoard pieceBoard = getBitboard(pieceEnum);

            while(pieceBoard != 0){
                key ^= ttable.pieceKeys[pieceEnum][popLsb(pieceBoard)];
            }
        }
    }
    if(getCastleRightsWK()){
        key ^= ttable.castlingRightsKeys[0];
    }
    if(getCastleRightsWQ()){
        key ^= ttable.castlingRightsKeys[1];
    }
    if(getCastleRightsBK()){
        key ^= ttable.castlingRightsKeys[2];
    }
    if(getCastleRightsBQ()){
        key ^= ttable.castlingRightsKeys[3];
    }

    if(getEnPassantSq() != noSq){
        key ^= ttable.enPassantKeys[getEnPassantSq()];
    }
    return key;
}



bool Board::checkSnipers(int sq, BitBoardEnum color)
{
    BitBoard snipers = 0;

    //
    uint64_t magic = ((getBitboard(All) & rookMask[sq]) * magicNumberRook[sq]) >> magicNumberShiftsRook[sq];
    BitBoard magicBoard = (*magicMovesRook)[sq][magic] & bitBoardArray[Q+color] & bitBoardArray[R+color];


    // For bishop
    //uint64_t magic = ((getBitboard(All) & rookMask[sq]) * magicNumberRook[sq]) >> magicNumberShiftsRook[sq];
    //BitBoard magicBoard = (*magicMovesRook)[sq][magic];

    BitBoard bishopsBoard = 0;
    setBit(bishopsBoard, sq);

    BitBoard moves = northEastOccludedMoves(bishopsBoard, ~bitBoardArray[All]);
    moves |= northWestccludedMoves(bishopsBoard, ~bitBoardArray[All]);
    moves |= southEastOccludedMoves(bishopsBoard, ~bitBoardArray[All]);
    moves |= southWestOccludedMoves(bishopsBoard, ~bitBoardArray[All]);
    moves &= bitBoardArray[Q + color] & bitBoardArray[B + color];

    if ((moves == 0) && (magicBoard == 0)) {
        return false;
    }

    return true;

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


BitBoard Board::getBitboard(int piece)
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

BitBoardEnum Board::getOtherSide()
{
    if(sideToMove == BitBoardEnum::White){
        return BitBoardEnum::Black;
    }
    return BitBoardEnum::White;
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
    #ifdef LINUX
        return __builtin_popcountll(bitBoardArray[piece]);
    #elif WIN32
        return __popcnt64(bitBoardArray[piece]);
    #endif
    
}

int Board::countSetBits(unsigned long long board)
{
    #ifdef LINUX
        return __builtin_popcountll(board);
    #elif WIN32
        return __popcnt64(board);
    #endif
}

int Board::popLsb(BitBoard& board)
{   
    #ifdef LINUX
        int lsb = __builtin_ctzll(board);
        board &= board - 1;
        return lsb;
    #elif WIN32
        int lsb = _tzcnt_u64(board);
        // This is here because all other logic is depdent on getting 0 when there are no more set bits, while _tzcnt_u64 counts trailing zeros
        if (lsb == 64) lsb = 0;
        board &= board - 1;
        return lsb;
    #endif


}

bool Board::makeMove(Move move) {

    MoveStruct *histMove = &moveHistory[historyPly];
    
    const int size = 15*sizeof(bitBoardArray[0]);
    std::memcpy(&histMove->bitBoardArrayCopy,&bitBoardArray,size);
    histMove->sideToMoveCopy = sideToMove;
    histMove->enPassantSqCopy = enPassantSq;
    histMove->castleWKCopy = castleWK;
    histMove->castleWQCopy= castleWQ;
    histMove->castleBKCopy = castleBK;
    histMove->castleBQCopy = castleBQ;
    histMove->pieceSquareScoreCopy = pieceSquareScore;
    histMove->materialScoreCopy = materialScore;
    histMove->hashKeyCopy = hashKey;

    historyPly++;


    BitBoardEnum attacker = BitBoardEnum::White;
    if(sideToMove == White){
        attacker = Black;
    }
    

    // Pop and set bits in piece and all board
    bitBoardArray[All] &= ~sqBB[move.fromSq];
    bitBoardArray[All] |= sqBB[move.toSq];
    bitBoardArray[move.piece] &= ~sqBB[move.fromSq];
    bitBoardArray[move.piece] |= sqBB[move.toSq];
    bitBoardArray[sideToMove] &= ~sqBB[move.fromSq];
    bitBoardArray[sideToMove] |= sqBB[move.toSq];

    hashKey ^= ttable.pieceKeys[move.piece][move.fromSq];
    hashKey ^= ttable.pieceKeys[move.piece][move.toSq];


    // Upadet score for moved piece
    pieceSquareScore -= Material::pieceSquareScoreArray[move.piece][move.fromSq];
    pieceSquareScore += Material::pieceSquareScoreArray[move.piece][move.toSq];


    if(move.capture){
        if(move.enpassant){
            if(sideToMove == BitBoardEnum::White){
                popBit(p,move.toSq-8);
                popBit(All, move.toSq-8);
                popBit(Black, move.toSq-8);
                pieceSquareScore -= Material::pieceSquareScoreArray[p][move.toSq-8];

                hashKey ^= ttable.pieceKeys[p][move.toSq-8];
            } else {
                popBit(P, move.toSq+8);
                popBit(All, move.toSq+8);
                popBit(White, move.toSq+8);
                pieceSquareScore -= Material::pieceSquareScoreArray[P][move.toSq+8];
                hashKey ^= ttable.pieceKeys[P][move.toSq+8];
            }
        } else {       
            bitBoardArray[attacker] &= ~sqBB[move.toSq];
            BitBoardEnum capturedPiece = All;
            if((bitBoardArray[P+attacker] & sqBB[move.toSq]) != 0){
                bitBoardArray[P+attacker] &= ~sqBB[move.toSq];
                capturedPiece = static_cast<BitBoardEnum>(P+attacker);
            } else if((bitBoardArray[N+attacker] & sqBB[move.toSq]) != 0){
                bitBoardArray[N+attacker] &= ~sqBB[move.toSq];
                capturedPiece = static_cast<BitBoardEnum>(N+attacker);
            } else if((bitBoardArray[B+attacker] & sqBB[move.toSq]) != 0){
                bitBoardArray[B+attacker] &= ~sqBB[move.toSq];
                capturedPiece = static_cast<BitBoardEnum>(B+attacker);
            } else if((bitBoardArray[R+attacker] & sqBB[move.toSq]) != 0){
                bitBoardArray[R+attacker] &= ~sqBB[move.toSq];
                capturedPiece = static_cast<BitBoardEnum>(R+attacker);
            } else if((bitBoardArray[Q+attacker] & sqBB[move.toSq]) != 0){
                bitBoardArray[Q+attacker] &= ~sqBB[move.toSq];
                capturedPiece = static_cast<BitBoardEnum>(Q+attacker);
            }
            
            // Update score for captured piece
            pieceSquareScore -= Material::pieceSquareScoreArray[capturedPiece][move.fromSq];        
            hashKey ^= ttable.pieceKeys[capturedPiece][move.toSq];
        }   
        materialScore = Material::getMaterialScore(*this);     
    }

    if(sideToMove == BitBoardEnum::White){

        if(move.doublePawnPush){
            // Remove the previous enpassantSquare
            if(enPassantSq != noSq){
                hashKey ^= ttable.enPassantKeys[enPassantSq];
            }
            enPassantSq = move.toSq-8;            
            hashKey ^= ttable.enPassantKeys[enPassantSq];
        } else {
            if(enPassantSq != noSq){
                hashKey ^= ttable.enPassantKeys[enPassantSq];
            }
            enPassantSq = noSq;
        }

        if(move.castling){
            if(move.toSq == 2){
                popBit(BitBoardEnum::All,0);
                popBit(BitBoardEnum::White,0);
                popBit(BitBoardEnum::R, 0);
                setBit(BitBoardEnum::All,3);
                setBit(BitBoardEnum::White,3);
                setBit(BitBoardEnum::R,3);

                hashKey ^= ttable.pieceKeys[R][0];
                hashKey ^= ttable.pieceKeys[R][3];

                // Upadet score for rook
                pieceSquareScore -= Material::pieceSquareScoreArray[R][0];
                pieceSquareScore += Material::pieceSquareScoreArray[R][3];

            } else {
                popBit(BitBoardEnum::All,7);
                popBit(BitBoardEnum::White,7);
                popBit(BitBoardEnum::R, 7);
                setBit(BitBoardEnum::All,5);
                setBit(BitBoardEnum::White,5);
                setBit(BitBoardEnum::R,5);
                // Upadet score for rook
                pieceSquareScore -= Material::pieceSquareScoreArray[R][7];
                pieceSquareScore += Material::pieceSquareScoreArray[R][5];

                hashKey ^= ttable.pieceKeys[R][7];
                hashKey ^= ttable.pieceKeys[R][5];
            }
        }

    } else {

        if(move.doublePawnPush){
            // Remove the previous enpassantSquare
            if(enPassantSq != noSq){
                hashKey ^= ttable.enPassantKeys[enPassantSq];
            }
            enPassantSq = move.toSq + 8;
            hashKey ^= ttable.enPassantKeys[enPassantSq];
        } else {
            if(enPassantSq != noSq){
                hashKey ^= ttable.enPassantKeys[enPassantSq];
            }
            enPassantSq = noSq;
        }

        if(move.castling){
            if(move.toSq == 58) {
                popBit(BitBoardEnum::All,56);
                popBit(BitBoardEnum::Black,56);
                popBit(BitBoardEnum::r, 56);
                setBit(BitBoardEnum::All,59);
                setBit(BitBoardEnum::Black,59);
                setBit(BitBoardEnum::r,59);

                hashKey ^= ttable.pieceKeys[r][56];
                hashKey ^= ttable.pieceKeys[r][59];

                // Upadet score for rook
                pieceSquareScore -= Material::pieceSquareScoreArray[R][56];
                pieceSquareScore += Material::pieceSquareScoreArray[R][59];
            } else {
                popBit(BitBoardEnum::All,63);
                popBit(BitBoardEnum::Black,63);
                popBit(BitBoardEnum::r, 63);
                setBit(BitBoardEnum::All,61);
                setBit(BitBoardEnum::Black,61);
                setBit(BitBoardEnum::r,61);

                hashKey ^= ttable.pieceKeys[r][63];
                hashKey ^= ttable.pieceKeys[r][61];

                // Upadet score for rook
                pieceSquareScore -= Material::pieceSquareScoreArray[R][63];
                pieceSquareScore += Material::pieceSquareScoreArray[R][61];
            }
        }
    }

    if(move.promotion != BitBoardEnum::All){        
        hashKey ^= ttable.pieceKeys[move.piece][move.toSq];
        pieceSquareScore -= Material::pieceSquareScoreArray[move.piece][move.toSq];
        hashKey ^= ttable.pieceKeys[move.promotion][move.toSq];
        pieceSquareScore += Material::pieceSquareScoreArray[move.promotion][move.toSq];
        materialScore = Material::getMaterialScore(*this);
    }

    //TODO Castline status overly complex
    //Update castling rights
    if(move.piece == K){
        if(castleWK){
            hashKey ^= ttable.castlingRightsKeys[0];
            castleWK = false;
        }
        if(castleWQ){
            hashKey ^= ttable.castlingRightsKeys[1];
            castleWQ = false;
        }
        
        
    }

    if(move.piece == k){
        if(castleBK){
            hashKey ^= ttable.castlingRightsKeys[2];
            castleBK = false;
        }
        if(castleBQ){
            hashKey ^= ttable.castlingRightsKeys[3];
            castleBQ = false;
        }
    }


    if(move.piece == R){
        if(move.fromSq == 0){
            if(castleWQ){
                hashKey ^= ttable.castlingRightsKeys[1];
                castleWQ = false;
            }   
            castleWQ = false;
        } else if(move.fromSq == 7) {
            if(castleWK){
                hashKey ^= ttable.castlingRightsKeys[0];
                castleWK = false;
            }
        }
    } else if( move.piece == r){
        if(move.fromSq == 56){            
            if(castleBQ){
                hashKey ^= ttable.castlingRightsKeys[3];
                castleBQ = false;
            }            
        } else if( move.fromSq== 63){
            if(castleBK){
                hashKey ^= ttable.castlingRightsKeys[2];
                castleBK = false;
            }
        }
    }



    if(move.toSq == 0 && move.capture){
        if(castleWQ){
            hashKey ^= ttable.castlingRightsKeys[1];
            castleWQ = false;
        }
    }

    if(move.toSq == 7 && move.capture){
        if(castleWK){
            hashKey ^= ttable.castlingRightsKeys[0];
            castleWK = false;
        }
    }

    if(move.toSq == 56 && move.capture){
        if(castleBQ){
            hashKey ^= ttable.castlingRightsKeys[3];
            castleBQ = false;
        }
    }

    if(move.toSq == 63 && move.capture){
        if(castleBK){
            hashKey ^= ttable.castlingRightsKeys[2];
            castleBK = false;
        }
    }



    if(isSquareAttacked(bitBoardArray[K+sideToMove], attacker)){
        return false;
    }
    

    changeSideToMove();
    return true;
}

void Board::revertLastMove()
{
    historyPly--;
    MoveStruct *move = &moveHistory[historyPly];    

    int size = 15*sizeof(bitBoardArray[0]);
    std::memcpy(&bitBoardArray,&move->bitBoardArrayCopy,size);
    sideToMove = move->sideToMoveCopy;
    enPassantSq = move->enPassantSqCopy;
    castleWK = move->castleWKCopy;
    castleWQ = move->castleWQCopy;
    castleBK = move->castleBKCopy;
    castleBQ = move->castleBQCopy;
    pieceSquareScore = move->pieceSquareScoreCopy;
    materialScore = move->materialScoreCopy;
    hashKey = move->hashKeyCopy;
}

bool Board::isSquareAttacked(BitBoard targetSquares, const BitBoardEnum attacker)
{
    BitBoard empty = ~bitBoardArray[All];
    BitBoard queenRooks = bitBoardArray[Q+attacker] | bitBoardArray[R+attacker];
    BitBoard queenBishops = bitBoardArray[Q+attacker] | bitBoardArray[B+attacker];
    BitBoard knights = bitBoardArray[N+attacker];
    BitBoard king = bitBoardArray[K+attacker];;

    if(attacker == BitBoardEnum::Black){
        if(((northWestOne(bitBoardArray[p]) | northEastOne(bitBoardArray[p])) & targetSquares) != 0) return true;         

    } else {
        if(((southEastOne(bitBoardArray[P]) | southWestOne(bitBoardArray[P])) & targetSquares) != 0) return true; 
    }


    int knightSquare = 0;
    while(knights != 0){
        knightSquare = popLsb(knights);
        if((knightmask[knightSquare] & targetSquares) != 0) return true;
    }

    if((kingMask[popLsb(king)] & targetSquares) != 0) return true;

    int queenRookSquare = 0;
    while (queenRooks != 0) {
        queenRookSquare = popLsb(queenRooks);


        uint64_t magic = ((getBitboard(All) & rookMask[queenRookSquare]) * magicNumberRook[queenRookSquare]) >> magicNumberShiftsRook[queenRookSquare];
        BitBoard magicBoard = (*magicMovesRook)[queenRookSquare][magic];
        if ((magicBoard & targetSquares) != 0) {
            return true;
        }
    }

    int queenBishopSquare = 0;
    while (queenBishops != 0) {
        queenBishopSquare = popLsb(queenBishops);
        BitBoard magicBoard = getBishopMagics(queenBishopSquare);

        if ((magicBoard & targetSquares) != 0) {
            return true;
        }
    }

    return false;
}

BitBoardEnum Board::getPieceOnSquare(int sq)
{
    for (int i = 0; i < 14; i++) {
        if (i != BitBoardEnum::Black && i != BitBoardEnum::White && (bitBoardArray[i] & sqBB[sq]) != 0) {
            return static_cast<BitBoardEnum>(i);
        }
    }
    return BitBoardEnum::All;

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
    
    for (int i = 7; i >= 0; i--) {
        int startSquare = 8 * i;
        for (int x = 0; x < 8; x++) {
            std::cout << printBoard[startSquare + x] << " ";
        }

        std::cout << std::endl;
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

    for (int i = 7; i >= 0; i--) {
        int startSquare = 8 * i;
        for (int x = 0; x < 8; x++) {
            std::cout << printBoard[startSquare + x] << " ";
        }

        std::cout << std::endl;

    }

    std::cout << std::endl;
    std::cout << std::endl;
}
