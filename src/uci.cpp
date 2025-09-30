#include "uci.h"
#include <sstream>
#include "perft.h"
#include "material.h"

void UCI::setNetworkPath(std::string path) {
    motherBoard.loadNetwork(path);
}

void UCI::loop(/*int argc, char* argv[]*/) {

    std::string token, cmd;
    do {
        if(!std::getline(std::cin,cmd)){
            std::cout << "Is this a problem in the UCI loop?";
        }

        std::istringstream is(cmd);

        token.clear(); // Avoid a stale if getline() returns nothing or a blank line
        is >> std::skipws >> token;
        
        if (token == "quit") break;
        else if (token == "uci") sendID();
        else if (token == "position") setPosition(is);
        else if (token == "go") startSearch(is);
        else if (token == "eval") staticEvaluation();
        else if (token == "isready") std::cout << "readyok" << std::endl;
        else if (token == "d") motherBoard.printBoard();
    } while (token != "quit" /*&& argc == 1*/); // The command-line arguments are one-shot
}

void UCI::setPosition(std::istringstream &is)
{
    std::string nextToken;
    std::string fenString;
    is >> std::skipws >> nextToken;

    if(nextToken == "fen"){
        is >> std::skipws >> nextToken;
        fenString += nextToken + " ";
        while (is >> nextToken && nextToken != "moves"){
            fenString += nextToken + " ";
        }
    } else if(nextToken == "startpos"){
        fenString = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        is >> nextToken; // Move string is just consumed....
    }
    motherBoard.parseFen(fenString);
    is >> nextToken;
    while(!nextToken.empty() && parseMove(nextToken)){
        is >> nextToken;
    }

    
}

void UCI::startSearch(std::istringstream &is)
{

    std::string nextToken;
    int wTime = 30000*30;
    int bTime = 30000*30;
    int wIncrement = 0;
    int bIncrement = 0;

    bool fixedSearchTime = false;
    int searchTime = 30000;
    int depth = 20;

    //https://gist.github.com/DOBRO/2592c6dad754ba67e6dcaec8c90165bf
    while (is >> nextToken) {
        if (nextToken == "searchmoves") {
            /*
            while (is >> nextToken) {

            }
            */

        }
        else if (nextToken == "wtime") {
            is >> nextToken;
            wTime = stoi(nextToken);
        }
        else if (nextToken == "btime") {
            is >> nextToken;
            bTime = stoi(nextToken);
        }
        else if (nextToken == "winc") {
            is >> nextToken;
            wIncrement = stoi(nextToken);
        }
        else if (nextToken == "binc") {
            is >> nextToken;
            bIncrement = stoi(nextToken);
        }
        else if (nextToken == "movetime") {
            is >> nextToken;
            searchTime = stoi(nextToken);
            fixedSearchTime = true;
        }
        else if (nextToken == "depth") {
            is >> nextToken;
            depth = stoi(nextToken);
        }
    }
    

    /////////////////////
    // Sets searchtime, based on amount left and/or increment time per move
    /////////////////////
    if(!fixedSearchTime){
        if (motherBoard.getSideToMove() == White) {
            searchTime = wTime / 20 + wIncrement / 2;
        }
        else {
            searchTime = bTime / 20 + bIncrement / 2;
        }
    }
    

    Score move;
    move = search.search(motherBoard,depth,searchTime);
    std::string bestMove = Perft::getNotation(move.bestMove);
    //std::cout << "info depth " << newSearch.currentFinishedDepth << std::endl;
    std::cout << "bestmove " << bestMove << std::endl;
}

void UCI::sendID()
{
    std::cout << "id name Zaphod 1.9" << std::endl;
    std::cout << "id author Egil Tennfjord Mikalsen" << std::endl;
    std::cout << "uciok" << std::endl;
}

bool UCI::parseMove(std::string token)
{
    MoveList list;
    MoveGenerator::generateMoves(motherBoard,list);
    for(int i = 0; i < list.counter; i++){
        if(token == Perft::getNotation(list.moves[i], motherBoard.getSideToMove())){
            motherBoard.makeMove(list.moves[i]);
            return true;
        }
    }
    return false;
}

void UCI::staticEvaluation() {
    
    int mgScore = 0;
    int egScore = 0;
    int gamePhase = 0;
    int materialScore = 0;
    BitBoard allPieces = motherBoard.getBitboard(All);
    int square = 0;
    while (allPieces) {
        square = motherBoard.popLsb(allPieces);
        BitBoardEnum piece = motherBoard.getPieceOnSquare(square);
        mgScore += Material::pieceSquareScoreArrayMG[piece][square];
        egScore += Material::pieceSquareScoreArrayEG[piece][square];
        gamePhase += Material::gamePhaseArray[piece];
        materialScore += Material::materialScoreArray[piece];
    }

    //Pesto gamephase handling
    int mgPhase = gamePhase;
    if (mgPhase > 24) mgPhase = 24; /* in case of early promotion */
    int egPhase = 24 - mgPhase;
    int psqt = (mgScore * mgPhase + egScore * egPhase) / 24;
    int score = materialScore + psqt;
    


    if (motherBoard.getSideToMove() == BitBoardEnum::Black) {
        score *= -1;
    }
    std::cout << "eval " << score << std::endl;
}
