#include "board.h"
#include <iostream>
#include <string>
#include <cstring>
#include "material.h"
#include <array>
#include <random>
#include "tools.h"
#include <cstdlib>
#include <cassert>

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

void Board::initSqBetween(){

    for (int sq1 = 0; sq1 < 64; sq1++) {
        for (int sq2 = 0; sq2 < 64; sq2++) {
            BitBoard squares = sqBB[sq1] | sqBB[sq2];


            //Following code is from https://www.chessprogramming.org/Square_Attacked_By#cite_note-5
            const BitBoard m1 = BitBoard(-1);
            const BitBoard a2a7 = BitBoard(0x0001010101010100);
            const BitBoard b2g7 = BitBoard(0x0040201008040200);
            const BitBoard h1b7 = BitBoard(0x0002040810204080); /* Thanks Dustin, g2b7 did not work for c1-a3 */
            BitBoard btwn, line, rank, file;

            btwn = (m1 << sq1) ^ (m1 << sq2);
            file = (sq2 & 7) - (sq1 & 7);
            rank = ((sq2 | 7) - sq1) >> 3;
            line = ((file & 7) - 1) & a2a7; /* a2a7 if same file */
            line += 2 * (((rank & 7) - 1) >> 58); /* b1g1 if same rank */
            line += (((rank - file) & 15) - 1) & b2g7; /* b2g7 if same diagonal */
            line += (((rank + file) & 15) - 1) & h1b7; /* h1b7 if same antidiag */
            line *= btwn & -btwn; /* mul acts like shift by smaller square */
            sqBetween[sq1][sq2] =  line & btwn;   /* return the bits on that line in-between */                              

        }
    }
}

void Board::initMagicMasks() {

    for (int index = 0; index < 64; index++)
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
    std::array<BitBoard, 64> rookMagicStart{};
    rookMagicStart = {36028901172019232,18023263326183424,9007886516619266,9011666154947074,20266206980212740,720716884428980258,36591749136842762,
                        72058438544916736,4512396831887616,87995558404098,77691493766660352,43980738003072,141905786570768,5070964808253464,
                        158331834466817,36046389746115600,18014948267425810,1161964163179221504,36187135351458048,905223662575027232,564049532749824,1152922604185714824,
                        4611690429363523712,1136895090327875,162199957477007408,2326566877011968,283674136285312,17596481282050,4611967802776485890,
                        9228157116357810178,72620546142109953,2269396831600704,16176948004534616096,18014948401094658,578730145377027072,1130323724206096,
                        35253225816384,1100049563664,282024736719113,70369318797440,36169681609981952,576478387977061376,1125934270777856,8933599101186,
                        3458768911900622980,562952369471489,1892640084159697024,9223377268704411649,18019896352849992,35529043353620,9223654062125482144,281543700516880,
                        1153488857203737088,70918533808256,549761319456,71470405386816,72568052678690,162129623093625601,106108274348042,577595586010711297,
                        4908928026778026497,71571402166274,2305852114812862980,5915373099352450 };


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

        //Generate first magic or use already known from array
        BitBoard magicNumber = distribution(rng) & distribution(rng) & distribution(rng) & distribution(rng);;
        if (isRook) {
            magicNumber = rookMagicStart[square];
        }
        int attempts = 0;

        bool fail = false;

        uint32_t magicShift = 52;

        do
        {
            // First time we might already know the magic from the array
            if (attempts > 0) {
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
        //std::cout << magicNumber << std::endl;
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

    for (int i = 0; i < 64; i++) {
        mailBoxBoard[i] = All;
    }

    ttable.initKeys();
    initMagicMasks();
    magicMovesRook = new std::array<std::array<BitBoard, 4096>, 64>();
    magicMovesBishop = new std::array<std::array<BitBoard, 4096>, 64>();
    initMagics(true, magicMovesRook, rookMask, magicNumberRook, magicNumberShiftsRook);
    initMagics(false, magicMovesBishop, bishopMask, magicNumberBishop, magicNumberShiftsBishop);
    initSqBetween();
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
    for (int i = 0; i < 64; i++) {
        mailBoxBoard[i] = All;
    }
    historyPly = 0;
    halfMoveClock = 0;

    sideToMove = White;
    enPassantSq = noSq;
    castleWK = false;
    castleWQ = false;
    castleBK = false;
    castleBQ = false;
}

void Board::addPiece(int sq, BitBoardEnum piece, BitBoardEnum color)
{
    if (piece == All) {
        int x = 0;
    }

    mailBoxBoard[sq] = piece;
    bitBoardArray[All] |= sqBB[sq];
    bitBoardArray[piece] |= sqBB[sq];
    bitBoardArray[color] |= sqBB[sq];
}

void Board::removePiece(int sq, BitBoardEnum color)
{
    bitBoardArray[All] &= ~sqBB[sq];
    bitBoardArray[color] &= ~sqBB[sq];
    bitBoardArray[mailBoxBoard[sq]] &= ~sqBB[sq];
    mailBoxBoard[sq] = All;

}

bool Board::hasPositionRepeated() {
    int moves = std::min(halfMoveClock, historyPly);
    int moveCounter = 0;

    for (int i = historyPly-1; i > historyPly - moves; i--) {

        if (moveHistory[i].hashKeyCopy == hashKey) {
            moveCounter++;
        }
    }

    return moveCounter > 2;
}

void Board::parseFen(std::string fen){
    clearBoard();
    int count = 0;
    int state = 0;
    for(std::string::size_type i = 0; i < fen.size(); ++i) {

        if (count > 64) {
            std::cout << "How did this happen" << std::endl;
        }


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
        {
            std::string enpassant = fen.substr(i, 2);
            for (int x = 0; x < 64; x++) {
                if (enpassant == sqToNotation[x]) {
                    enPassantSq = x;
                }
            }
            break;
        }
        case 4:
        {
            if (fen.size() > i + 3) {
                //Half move clock
                std::string halfString = "";
                halfString += fen[i];

            
                if (fen[i + 1] != ' ') {
                    halfString += fen[i + 1];
                    i++;
                }
                if (fen[i + 1] != ' ') {
                    halfString += fen[i + 1];
                    i++;
                }

                i++;

                std::string fullmoveString = "";
                fullmoveString += fen[i];

                //Double for double digit positions
                if (fen[i + 1] != ' ') {
                    fullmoveString += fen[i + 1];
                    i++;
                }
                if (fen[i + 1] != ' ') {
                    fullmoveString += fen[i + 1];
                    i++;
                }

                halfMoveClock = std::stoi(halfString);
                int fullMove = std::stoi(fullmoveString);
            }





            break;
        }
        case 5:
            //This is the move counter
        default:
            break;
        }
    }

    hashKey = generateHashKey();
    pawnHash = generatePawnHashKey();
    historyPly = 0;
}

BitBoard Board::generatePawnHashKey() {
    BitBoard whitePawns = getBitboard(P);
    BitBoard blackPawns = getBitboard(p);
    BitBoard key = 0;
    int sq = 0;
    while (whitePawns != 0) {
        sq = popLsb(whitePawns);
        key ^= ttable.pieceKeys[P][sq];
    }
    while (blackPawns != 0) {
        sq = popLsb(blackPawns);
        key ^= ttable.pieceKeys[p][sq];
    }
    return key;
}

BitBoard Board::generateHashKey(){
    BitBoard key = 0;

    for (int pieceValue = BitBoardEnum::R; pieceValue != BitBoardEnum::All; pieceValue++ ){
        if(pieceValue != BitBoardEnum::All && pieceValue != BitBoardEnum::White && pieceValue != BitBoardEnum::Black){
            BitBoardEnum pieceEnum = static_cast<BitBoardEnum>(pieceValue);
            BitBoard pieceBoard = getBitboard(pieceEnum);

            int sq = 0;
            while(pieceBoard != 0){
                sq = popLsb(pieceBoard);
                key ^= ttable.pieceKeys[pieceEnum][sq];
            }
        }
    }

    key ^= sideToMove;

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



BitBoard Board::getSnipers(int kingSquare, BitBoardEnum attackerColor)
{
    BitBoard snipers = 0;

    // Rook and Queen
    uint64_t magic = ((rookMask[kingSquare] & bitBoardArray[attackerColor]) * magicNumberRook[kingSquare]) >> magicNumberShiftsRook[kingSquare];
    BitBoard boardQR = (*magicMovesRook)[kingSquare][magic] & (bitBoardArray[Q+attackerColor] | bitBoardArray[R+attackerColor]);


    // For bishop and queen
    magic = ((bishopMask[kingSquare] & bitBoardArray[attackerColor]) * magicNumberBishop[kingSquare]) >> magicNumberShiftsBishop[kingSquare];
    BitBoard boardQB = (*magicMovesBishop)[kingSquare][magic] & (bitBoardArray[Q + attackerColor] | bitBoardArray[B + attackerColor]);

    return (boardQB | boardQR);

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
    hashKey ^= sideToMove;
    if(sideToMove == BitBoardEnum::White){
       sideToMove = BitBoardEnum::Black;     
    } else {
        sideToMove = BitBoardEnum::White;
    }
    hashKey ^= sideToMove;
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
    assert(count < 64);

    int bitNr = fenToBitMapping[count];

    if(isdigit(value)){
        std::string s = &value;
        int increment = std::stoi(s);
        count+= increment;
    } else { 

        if (fenToEnumBoardMap.find(value) != fenToEnumBoardMap.end()){
            BitBoardEnum color;
            if (islower(value)) {
                color = Black;
            }
            else {
                color = White;
            }
            
            addPiece(bitNr, fenToEnumBoardMap.at(value), color);
            
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
        unsigned long idx = 0;
        _BitScanForward64(&idx, board);
        board &= board - 1;
        return idx;
    #endif


}

bool Board::makeMove(Move move) {

    
    MoveUndoInfo *histMove = &moveHistory[historyPly];
    
    histMove->halfMoveClock = halfMoveClock;
    histMove->sideToMove = static_cast<uint8_t>(sideToMove);
    histMove->enPassantSqCopy = enPassantSq;
    histMove->castleMask = (castleWK ? 1 : 0) | (castleWQ ? 2 : 0) | (castleBK ? 4 : 0) | (castleBQ ? 8 : 0);
    histMove->hashKeyCopy = hashKey;
    histMove->pawnHashCopy = pawnHash;
    histMove->move = move;

    historyPly++;

    int toSq = move.to();
    int fromSq = move.from();
    BitBoardEnum piece = mailBoxBoard[fromSq];
    BitBoardEnum capturedPiece = mailBoxBoard[toSq];
    histMove->movedPiece = static_cast<uint8_t>(piece);
    histMove->capturedPiece = static_cast<uint8_t>(capturedPiece);

    MoveType moveType = move.getMoveType();
    bool enpassant = (moveType == EN_PASSANT);
    bool capture = (capturedPiece != All) || enpassant;
    bool isPawn = (piece == p || piece == P);
    bool doublePush = isPawn && std::abs(fromSq - toSq) == 16;
    halfMoveClock++;


    BitBoardEnum otherSide = BitBoardEnum::White;
    int enpassantModifier = -8;
    if(sideToMove == White){
        otherSide = Black;
        enpassantModifier = 8;
    }

    if (capture) {
        if (enpassant) {
            capturedPiece = mailBoxBoard[toSq - enpassantModifier];
            removePiece(toSq - enpassantModifier, otherSide);
            hashKey ^= ttable.pieceKeys[otherSide + P][toSq - enpassantModifier];
        }
        else {
            capturedPiece = mailBoxBoard[toSq];
            removePiece(toSq, otherSide);
            hashKey ^= ttable.pieceKeys[capturedPiece][toSq];
        }
        // Capture resets halfmoveclock
        halfMoveClock = 0;
    }

    // update Pawn hash
    if (capturedPiece == P + otherSide) {
        pawnHash ^= ttable.pieceKeys[P+otherSide][toSq];
    }
    if (piece == P + sideToMove) {
        if(moveType != PROMOTION){
            pawnHash ^= ttable.pieceKeys[P + sideToMove][toSq];
        }
        pawnHash ^= ttable.pieceKeys[P+sideToMove][fromSq];
    }

    // Pop and set bits in piece and all board
    addPiece(toSq, piece, sideToMove);
    removePiece(fromSq, sideToMove);

    hashKey ^= ttable.pieceKeys[piece][fromSq];
    hashKey ^= ttable.pieceKeys[piece][toSq];

    // Reset halfmoveclock if there is a pawn move
    if (piece == P + sideToMove) {
        halfMoveClock = 0;
    }

    if (doublePush) {
        // Remove the previous enpassantSquare
        if (enPassantSq != noSq) {
            hashKey ^= ttable.enPassantKeys[enPassantSq];
        }
        enPassantSq = toSq - enpassantModifier;
        hashKey ^= ttable.enPassantKeys[enPassantSq];

    }
    else {
        if (enPassantSq != noSq) {
            hashKey ^= ttable.enPassantKeys[enPassantSq];
        }
        enPassantSq = noSq;
    }

    if(sideToMove == BitBoardEnum::White){

        if(moveType == MoveType::CASTLING){
            if(toSq == 2){
                removePiece(0,White);
                addPiece(3, R, White);

                hashKey ^= ttable.pieceKeys[R][0];
                hashKey ^= ttable.pieceKeys[R][3];
            } else {
                removePiece(7, White);
                addPiece(5, R, White);
                
                hashKey ^= ttable.pieceKeys[R][7];
                hashKey ^= ttable.pieceKeys[R][5];
            }
        }

    } else {

        if(moveType == MoveType::CASTLING){
            if(toSq == 58) {
                removePiece(56, Black);
                addPiece(59, r, Black);

                hashKey ^= ttable.pieceKeys[r][56];
                hashKey ^= ttable.pieceKeys[r][59];

            } else {
                removePiece(63, Black);
                addPiece(61, r, Black);
                
                hashKey ^= ttable.pieceKeys[r][63];
                hashKey ^= ttable.pieceKeys[r][61];

            }
        }
    }   

    if(moveType == MoveType::PROMOTION){  
        BitBoardEnum promotionPiece = move.getPromotionType(sideToMove);
        hashKey ^= ttable.pieceKeys[piece][toSq];
        hashKey ^= ttable.pieceKeys[promotionPiece][toSq];
        removePiece(toSq,sideToMove);
        addPiece(toSq, promotionPiece, sideToMove);

    }

    //TODO Castline status overly complex
    //Update castling rights
    if(piece == K){
        if(castleWK){
            hashKey ^= ttable.castlingRightsKeys[0];
            castleWK = false;
        }
        if(castleWQ){
            hashKey ^= ttable.castlingRightsKeys[1];
            castleWQ = false;
        }
        
        
    }

    if(piece == k){
        if(castleBK){
            hashKey ^= ttable.castlingRightsKeys[2];
            castleBK = false;
        }
        if(castleBQ){
            hashKey ^= ttable.castlingRightsKeys[3];
            castleBQ = false;
        }
    }


    if(piece == R){
        if(fromSq == 0){
            if(castleWQ){
                hashKey ^= ttable.castlingRightsKeys[1];
                castleWQ = false;
            }   
            castleWQ = false;
        } else if(fromSq == 7) {
            if(castleWK){
                hashKey ^= ttable.castlingRightsKeys[0];
                castleWK = false;
            }
        }
    } else if( piece == r){
        if(fromSq == 56){            
            if(castleBQ){
                hashKey ^= ttable.castlingRightsKeys[3];
                castleBQ = false;
            }            
        } else if( fromSq== 63){
            if(castleBK){
                hashKey ^= ttable.castlingRightsKeys[2];
                castleBK = false;
            }
        }
    }



    if(toSq == 0 && capture){
        if(castleWQ){
            hashKey ^= ttable.castlingRightsKeys[1];
            castleWQ = false;
        }
    }

    if(toSq == 7 && capture){
        if(castleWK){
            hashKey ^= ttable.castlingRightsKeys[0];
            castleWK = false;
        }
    }

    if(toSq == 56 && capture){
        if(castleBQ){
            hashKey ^= ttable.castlingRightsKeys[3];
            castleBQ = false;
        }
    }

    if(toSq == 63 && capture){
        if(castleBK){
            hashKey ^= ttable.castlingRightsKeys[2];
            castleBK = false;
        }
    } 
    
    changeSideToMove();
    return true;
}

void Board::revertLastMove()
{
    historyPly--;
    MoveUndoInfo *info = &moveHistory[historyPly];

    sideToMove = static_cast<BitBoardEnum>(info->sideToMove);
    halfMoveClock = info->halfMoveClock;
    enPassantSq = info->enPassantSqCopy;

    castleWK = (info->castleMask & 1) != 0;
    castleWQ = (info->castleMask & 2) != 0;
    castleBK = (info->castleMask & 4) != 0;
    castleBQ = (info->castleMask & 8) != 0;

    MoveType moveType = info->move.getMoveType();

    BitBoardEnum movedPiece = static_cast<BitBoardEnum>(info->movedPiece);
    BitBoardEnum capturedPiece = static_cast<BitBoardEnum>(info->capturedPiece);
    

    if (info->move.getMoveType() == CASTLING) {
        if (sideToMove == White) {
            if (info->move.to() == 2) {
                removePiece(3, White);
                addPiece(0, R, White);
            }
            else {
                removePiece(5, White);
                addPiece(7, R, White);
            }
        }
        else {
            if (info->move.to() == 58) {
                removePiece(59, Black);
                addPiece(56, r, Black);
            }
            else {
                removePiece(61, Black);
                addPiece(63, r, Black);
            }
        }
    }

    if (info->move.getMoveType() == PROMOTION) {
        removePiece(info->move.to(), sideToMove);
        addPiece(info->move.from(), static_cast<BitBoardEnum>(P + sideToMove), sideToMove);
    }
    else {
        removePiece(info->move.to(), sideToMove);
        addPiece(info->move.from(), movedPiece, sideToMove);
    }

    if (info->move.getMoveType() == EN_PASSANT) {
        int enpassantModifier = -8;
        if (sideToMove == White) {
            enpassantModifier = 8;
        }
        addPiece(info->move.to() - enpassantModifier, static_cast<BitBoardEnum>(P + getOtherSide()), getOtherSide());

    }

    if (capturedPiece != All) {     
        addPiece(info->move.to(), capturedPiece, getOtherSide());
    }

    hashKey = info->hashKeyCopy;
    pawnHash = info->pawnHashCopy;
    
}

void Board::makeNullMove() {
    /*
    MoveStruct* histMove = &moveHistory[historyPly];

    int sizeBB = 15 * sizeof(bitBoardArray[0]);
    int sizeMB = 64 * sizeof(mailBoxBoard[0]);
    std::memcpy(&histMove->bitBoardArrayCopy, &bitBoardArray, sizeBB);
    std::memcpy(&histMove->mailBox, &mailBoxBoard, sizeMB);
    histMove->halfMoveClock = halfMoveClock;
    histMove->sideToMoveCopy = sideToMove;
    histMove->enPassantSqCopy = enPassantSq;
    histMove->castleWKCopy = castleWK;
    histMove->castleWQCopy = castleWQ;
    histMove->castleBKCopy = castleBK;
    histMove->castleBQCopy = castleBQ;
    histMove->hashKeyCopy = hashKey;
    histMove->pawnHashCopy = pawnHash;

    if (enPassantSq != noSq) {
        hashKey ^= ttable.enPassantKeys[enPassantSq];
    }
    enPassantSq = noSq;

    historyPly++;

    changeSideToMove();
    */
}

void Board::revertNullMove() {
    /*
    historyPly--;
    MoveStruct* move = &moveHistory[historyPly];

    setBoardState(*move);
    */
}

bool Board::isSquareAttacked(BitBoard targetSquares, const BitBoardEnum attacker)
{
    BitBoard empty = ~bitBoardArray[All];
    BitBoard queenRooks = bitBoardArray[Q+attacker] | bitBoardArray[R+attacker];
    BitBoard queenBishops = bitBoardArray[Q+attacker] | bitBoardArray[B+attacker];
    BitBoard knights = bitBoardArray[N+attacker];
    BitBoard king = bitBoardArray[K+attacker];

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
    return mailBoxBoard[sq];
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
