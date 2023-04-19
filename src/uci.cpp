#include "uci.h"
#include <sstream>
#include "search.h"
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
        
        if(token == "quit") break;
        else if(token == "uci") sendID();
        else if(token == "position") setPosition(is);
        else if(token == "go") startSearch(is);
        else if(token == "isready") std::cout << "readyok" << std::endl;


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
        if(nextToken == "startpos"){
            fenString = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        }
    }
    motherBoard.parseFen(fenString);

    is >> std::skipws >> nextToken;
    if(nextToken == "moves"){
        while(!nextToken.empty()){
            parseMove(nextToken);
            is >> std::skipws >> nextToken;

        }
    }

    
}

void UCI::startSearch(std::istringstream &is)
{
    Search search;
    Move move = search.searchAlphaBeta(motherBoard,2);
    std::string bestMove = Perft::getNotation(move);
    std::cout << "bestmove " << bestMove << std::endl;
}

void UCI::sendID()
{
    std::cout << "id name Zaphod 0.1" << std::endl;
    std::cout << "id author Egil Tennfjord Mikalsen" << std::endl;
    std::cout << "uciok" << std::endl;
}

void UCI::parseMove(std::string token)
{
    MoveList list;
    MoveGenerator::generateMoves(motherBoard,list);
    for(int i = 0; i < list.counter; i++){
        if(token == Perft::getNotation(list.moves[i])){
            motherBoard.makeMove(list.moves[i]);
            break;
        }
    }
}
