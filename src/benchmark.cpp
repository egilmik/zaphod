#include <iostream>
#include "board.h"
#include "movegenerator.h"
#include "perft/perft.h"

#include "search.h"
#include <chrono>
#include <fstream>

struct BenchmarkDefinition {
    std::string text;
    std::string fen;
    std::string bestMove;
};

 int main(int argc, char* argv[]) {

     std::string networkPath;
     
     for (int i = 1; i < argc; ++i) {
         std::string arg = argv[i];
         if(arg == "-network" && i+1 < argc){
             networkPath = argv[i + 1];
         }
     }

     if (networkPath.empty()) {
         std::cout << "No network provided" << std::endl;
         return 0;
     }

    std::vector<BenchmarkDefinition> benchVector;
    
    //https://www.chessprogramming.org/CCR_One_Hour_Test
    benchVector.push_back({ "CCR01","rn1qkb1r/pp2pppp/5n2/3p1b2/3P4/2N1P3/PP3PPP/R1BQKBNR w KQkq - 0 1", "d1b3" });
    benchVector.push_back({ "CCR02","rn1qkb1r/pp2pppp/5n2/3p1b2/3P4/1QN1P3/PP3PPP/R1B1KBNR b KQkq - 1 1", "f5c8" });
    benchVector.push_back({ "CCR03","r1bqk2r/ppp2ppp/2n5/4P3/2Bp2n1/5N1P/PP1N1PP1/R2Q1RK1 b kq - 1 10", "g4h6" });
    benchVector.push_back({ "CCR04","r1bqrnk1/pp2bp1p/2p2np1/3p2B1/3P4/2NBPN2/PPQ2PPP/1R3RK1 w - - 1 12","b2b4" });
    benchVector.push_back({ "CCR05","rnbqkb1r/ppp1pppp/5n2/8/3PP3/2N5/PP3PPP/R1BQKBNR b KQkq - 3 5","e7e5" });
    benchVector.push_back({ "CCR06","rnbq1rk1/pppp1ppp/4pn2/8/1bPP4/P1N5/1PQ1PPPP/R1B1KBNR b KQ - 1 5","b4c3" });
    benchVector.push_back({ "CCR07","r4rk1/3nppbp/bq1p1np1/2pP4/8/2N2NPP/PP2PPB1/R1BQR1K1 b - - 1 12","f8b8" });
    benchVector.push_back({ "CCR08","rn1qkb1r/pb1p1ppp/1p2pn2/2p5/2PP4/5NP1/PP2PPBP/RNBQK2R w KQkq c6 1 6","d4d5" });
    benchVector.push_back({ "CCR09","r1bq1rk1/1pp2pbp/p1np1np1/3Pp3/2P1P3/2N1BP2/PP4PP/R1NQKB1R b KQ - 1 9","c6d4" });
    /*
    benchVector.push_back({ "CCR10","rnbqr1k1/1p3pbp/p2p1np1/2pP4/4P3/2N5/PP1NBPPP/R1BQ1RK1 w - - 1 11","a2a4" });
    benchVector.push_back({ "","","" });
    benchVector.push_back({ "","","" });
    benchVector.push_back({ "","","" });
    benchVector.push_back({ "","","" });
    benchVector.push_back({ "","","" });
    benchVector.push_back({ "","","" });
    benchVector.push_back({ "","","" });
    benchVector.push_back({ "","","" });
    */




    benchVector.push_back({ "Random1","1R6/1brk2p1/4p2p/p1P1Pp2/P7/6P1/1P4P1/2R3K1 w - - 0 1","b8b7" });
    benchVector.push_back({ "Random2","4r1k1/p1qr1p2/2pb1Bp1/1p5p/3P1n1R/1B3P2/PP3PK1/2Q4R w - - 0 1","c1f4" });

    benchVector.push_back({ "Mate In 3 - 01","8/8/8/8/1p1N4/1Bk1K3/3N4/b7 w - -","d4e6" });
    benchVector.push_back({ "Mate In 3 - 01","1N4k1/2p5/3p3Q/p3p3/4P3/2P1B1Pp/PP6/3R1RK1 w - - 0 35 ","f1f8" });
    

    benchVector.push_back({ "Why do I not see the mate?", "4rk1r/p2b1ppp/8/1p1pR1N1/3P4/2P5/P4PPP/4R1K1 w - - 6 23 ","g5f3" });

    std::ofstream csvFile("benchmark.csv");
    csvFile << "id,status,expMove,selMove,depth,qdepth,score,elapsedtime,nps,nodes\n";

    int64_t npsTotal = 0;
    int64_t nodesTotal = 0;
    
    Search search;
    for (BenchmarkDefinition def : benchVector) {
        auto start = std::chrono::high_resolution_clock::now();

        std::cout << def.text << " " << def.fen << std::endl;
        Board board;
        board.loadNetwork(networkPath);
        board.parseFen(def.fen);
        int depth = 12;
                
        Score move = search.search(board, depth,1000000);

        //MoveList pvList = search.reconstructPV(board,depth);

        

        std::string status = "[Passed]";

        if (Perft::getNotation(move.bestMove) != def.bestMove) {
            status = "[Failed]";
        }
        

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        int nps = (double)search.evaluatedNodes / ((double)duration.count() / (double)1000);
        std::cout << status << " Score: " << (double)move.score / 100.0 << " Depth: ";
        std::cout << move.depth <<"/"<< search.maxQuinesenceDepthThisSearch << " NPS: " << nps << " Nodes: " << search.evaluatedNodes << " Playtime " << (duration.count()) << " ms" << std::endl;
        std::cout << "Best move " << Perft::getNotation(move.bestMove)  << " Expected best move " << def.bestMove << std::endl;
        std::cout << "TT stats " << " Upper bound hit: " << search.upperBoundHit << " Lower bound hit " << search.lowerBoundHit << " Exact hit " << search.exactHit << " qsearch hits: " << search.qsearchTTHit << std::endl;
        std::cout << "LMR stats " << " LMR hit: " << search.lmrHit << " LMR re-search: " << search.lmrResearchHit << std::endl;
        std::cout << "Aspiration window stats " << " Low research : " << search.aspirationLowResearchHit << " High research: " << search.aspirationHighResearchHit << std::endl;
        std::cout << "Reverse futility: " << search.reverseFutilityPruningHit << " Futility pruning: " << search.futilityPruningHit << std::endl;
        std::cout << "Null move: " << search.nullMoveHit << std::endl;
        /*std::cout << "PV ";
        for (int i = 0; i < pvList.counter; i++) {
            std::cout << Perft::getNotation(pvList.moves[i]) << " ";
        }*/
        //csvFile << "id,expMove,selMove,depth,score,elapsedtime,nps,nodes\n";
        csvFile << def.text << "," << status << "," << def.bestMove << "," << Perft::getNotation(move.bestMove);
        csvFile << "," << move.depth << "," << search.maxQuinesenceDepthThisSearch << "," << move.score << "," << duration.count() << "," << nps << "," << search.evaluatedNodes << "\n";

        std::cout << std::endl;
        std::cout << "Playtime " << (duration.count()) << " ms" << std::endl;
        std::cout << std::endl;

        npsTotal += nps;
        nodesTotal += search.evaluatedNodes;

    }

    std::cout << "Average NPS: " << (npsTotal / benchVector.size()) << " Average nodes: " << (nodesTotal/benchVector.size()) << std::endl;

    csvFile.close();

}
