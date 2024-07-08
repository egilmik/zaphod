#include "uci.h"
#include <sstream>
#include "perft.h"

void UCI::loop(/*int argc, char* argv[]*/) {

  /*Position pos;
  string token, cmd;
  StateListPtr states(new std::deque<StateInfo>(1));

  pos.set(StartFEN, false, &states->back(), Threads.main());

  for (int i = 1; i < argc; ++i)
      cmd += std::string(argv[i]) + " ";
*/
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
        else if (token == "isready") std::cout << "readyok" << std::endl;
        else if (token == "d") motherBoard.printBoard();


        /*
        if (argc == 1 && !getline(cin, cmd)) // Wait for an input or an end-of-file (EOF) indication
            cmd = "quit";

        istringstream is(cmd);

        token.clear(); // Avoid a stale if getline() returns nothing or a blank line
        is >> skipws >> token;

        if (    token == "quit"
            ||  token == "stop")
            Threads.stop = true;

        // The GUI sends 'ponderhit' to tell that the user has played the expected move.
        // So, 'ponderhit' is sent if pondering was done on the same move that the user
        // has played. The search should continue, but should also switch from pondering
        // to the normal search.
        else if (token == "ponderhit")
            Threads.main()->ponder = false; // Switch to the normal search

        else if (token == "uci")
            sync_cout << "id name " << engine_info(true)
                        << "\n"       << Options
                        << "\nuciok"  << sync_endl;

        else if (token == "setoption")  setoption(is);
        else if (token == "go")         go(pos, is, states);
        else if (token == "position")   position(pos, is, states);
        else if (token == "ucinewgame") Search::clear();
        else if (token == "isready")    sync_cout << "readyok" << sync_endl;

        // Add custom non-UCI commands, mainly for debugging purposes.
        // These commands must not be used during a search!
        else if (token == "flip")     pos.flip();
        else if (token == "bench")    bench(pos, is, states);
        else if (token == "d")        sync_cout << pos << sync_endl;
        else if (token == "eval")     trace_eval(pos);
        else if (token == "compiler") sync_cout << compiler_info() << sync_endl;
        else if (token == "export_net")
        {
            std::optional<std::string> filename;
            std::string f;
            if (is >> skipws >> f)
                filename = f;
            Eval::NNUE::save_eval(filename);
        }
        else if (token == "--help" || token == "help" || token == "--license" || token == "license")
            sync_cout << "\nStockfish is a powerful chess engine for playing and analyzing."
                        "\nIt is released as free software licensed under the GNU GPLv3 License."
                        "\nStockfish is normally used with a graphical user interface (GUI) and implements"
                        "\nthe Universal Chess Interface (UCI) protocol to communicate with a GUI, an API, etc."
                        "\nFor any further information, visit https://github.com/official-stockfish/Stockfish#readme"
                        "\nor read the corresponding README.md and Copying.txt files distributed along with this program.\n" << sync_endl;
        else if (!token.empty() && token[0] != '#')
            sync_cout << "Unknown command: '" << cmd << "'. Type help for more information." << sync_endl;
    */
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
            searchTime = std::min(searchTime, wTime / 30);

            if (wIncrement > 0) {
                searchTime = std::max(searchTime, (int)(wIncrement*0.95));
            }
        }
        else {
            searchTime = std::min(searchTime, bTime / 30);
            if (bIncrement > 0) {
                searchTime = std::max(searchTime, (int)(bIncrement*0.95));
            }
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
    std::cout << "id name Zaphod 1.3" << std::endl;
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
