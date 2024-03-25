#include <iostream>
#include "board.h"
#include "movegenerator.h"
#include "perft.h"
#include "test.h"
#include "search.h"
#include <chrono>

struct BenchmarkDefinition {
    std::string text;
    std::string fen;
    std::string bestMove;
};

int main(int, char**) {

    std::vector<BenchmarkDefinition> benchVector;

    //https://www.chessprogramming.org/CCR_One_Hour_Test
    benchVector.push_back({ "CCR01","rn1qkb1r/pp2pppp/5n2/3p1b2/3P4/2N1P3/PP3PPP/R1BQKBNR w KQkq - 0 1", "d1b3" });
    benchVector.push_back({ "CCR02","rn1qkb1r/pp2pppp/5n2/3p1b2/3P4/1QN1P3/PP3PPP/R1B1KBNR b KQkq - 1 1", "f5c8" });
    benchVector.push_back({ "CCR03","r1bqk2r/ppp2ppp/2n5/4P3/2Bp2n1/5N1P/PP1N1PP1/R2Q1RK1 b kq - 1 10", "g4h6" });
    benchVector.push_back({ "CCR04","r1bqrnk1/pp2bp1p/2p2np1/3p2B1/3P4/2NBPN2/PPQ2PPP/1R3RK1 w - - 1 12","b2b4" });
    benchVector.push_back({ "CCR05","rnbqkb1r/ppp1pppp/5n2/8/3PP3/2N5/PP3PPP/R1BQKBNR b KQkq - 3 5","e7e5" });
    benchVector.push_back({ "CCR06","rnbq1rk1/pppp1ppp/4pn2/8/1bPP4/P1N5/1PQ1PPPP/R1B1KBNR b KQ - 1 5","b4c3" });
    benchVector.push_back({ "CCR07","r4rk1/3nppbp/bq1p1np1/2pP4/8/2N2NPP/PP2PPB1/R1BQR1K1 b - - 1 12","f8b8" });

    //benchVector.push_back({ "","1R6/1brk2p1/4p2p/p1P1Pp2/P7/6P1/1P4P1/2R3K1 w - - 0 1","b8b7" });
    //benchVector.push_back({ "","4r1k1/p1qr1p2/2pb1Bp1/1p5p/3P1n1R/1B3P2/PP3PK1/2Q4R w - - 0 1","c1f4" });
    
    for (BenchmarkDefinition def : benchVector) {
        auto start = std::chrono::high_resolution_clock::now();

        std::cout << def.text << " " << def.fen << std::endl;
        Board board;
        board.parseFen(def.fen);
        int depth = 15;
        Search search;        
        Score move = search.search(board, depth,60000);

        MoveList pvList = search.reconstructPV(board,depth);

        

        std::string status = "[Passed]";

        if (Perft::getNotation(move.bestMove) != def.bestMove) {
            status = "[Failed]";
        }
        

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        int nps = (double)search.evaluatedNodes / ((double)duration.count() / (double)1000);
        std::cout << status << " Score: " << (double)move.score / 100.0 << " Depth: ";
        std::cout << move.depth << " NPS: " << nps << " Nodes: " << search.evaluatedNodes << " Playtime " << (duration.count()) << " ms" << std::endl;
        std::cout << "Best move " << Perft::getNotation(move.bestMove)  << " Expected best move " << def.bestMove << std::endl;
        std::cout << "TT stats " << " Upper bound hit: " << search.upperBoundHit << " Lower bound hit " << search.lowerBoundHit << " Exact hit " << search.exactHit << std::endl;
        std::cout << "PV ";
        for (int i = 0; i < pvList.counter; i++) {
            std::cout << Perft::getNotation(pvList.moves[i]) << " ";
        }

        std::cout << std::endl;
        std::cout << "Playtime " << (duration.count()) << " ms" << std::endl;
        std::cout << std::endl;

    }

}
