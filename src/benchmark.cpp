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
    benchVector.push_back({ "CCR01","rn1qkb1r/pp2pppp/5n2/3p1b2/3P4/2N1P3/PP3PPP/R1BQKBNR w KQkq - 0 1", "Qb3" });
    benchVector.push_back({ "CCR02","rn1qkb1r/pp2pppp/5n2/3p1b2/3P4/1QN1P3/PP3PPP/R1B1KBNR b KQkq - 1 1", "Bc8" });
    benchVector.push_back({ "CCR03","r1bqk2r/ppp2ppp/2n5/4P3/2Bp2n1/5N1P/PP1N1PP1/R2Q1RK1 b kq - 1 10", "Nh6" });


    benchVector.push_back({ "","1R6/1brk2p1/4p2p/p1P1Pp2/P7/6P1/1P4P1/2R3K1 w - - 0 1","b8b7" });
    benchVector.push_back({ "","4r1k1/p1qr1p2/2pb1Bp1/1p5p/3P1n1R/1B3P2/PP3PK1/2Q4R w - - 0 1","c1f4" });
    
    // "Fails" to search depth 9
    //benchVector.push_back({ "","r1b2rk1/ppq1bppp/2p1pn2/8/2NP4/2N1P3/PP2BPPP/2RQK2R w K - 0 1","b8b7" });

    for (BenchmarkDefinition def : benchVector) {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::cout << def.text << " " << def.fen << std::endl;
        Board board;
        board.parseFen(def.fen);
        Search search;        
        Score move = search.search(board, 7);

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        int nps = (double)search.evaluatedNodes / ((double)duration.count() / (double)1000);
        std::cout << Perft::getNotation(move.bestMove) << " Score: " << (double)move.score / 100.0 << " Depth: ";
        std::cout << move.depth << " NPS: " << nps << " Nodes: " << search.evaluatedNodes << std::endl;
        std::cout << "Expected best move " << def.bestMove << std::endl;
        std::cout << "Playtime " << (duration.count()) << " ms" << std::endl;
        std::cout << std::endl;

    }

}
