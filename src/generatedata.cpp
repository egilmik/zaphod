#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <format>
#include <random>
#include <iomanip>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>
#include "board.h"
#include "search.h"
#include "nnueq.h"
#include "tools/openingbook.h"
#include "tools/fentools.h"

static std::chrono::steady_clock::time_point gStart;

static inline uint64_t mix64(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33; return x;
}

struct WorkerArgs {
    int id;
    uint64_t quota;                  // global target
    std::atomic<uint64_t>* produced; // global produced counter
    std::string outPath;
    std::string networkPath;
    int nodes = 10000;
    OpeningBook* book;
    int depth = 4;
};


struct PositionData {
    std::string fen;
    int score = 0; //white relative
};

void worker_fn(WorkerArgs a) {
    std::ofstream out(a.outPath);
    if (!out) {
        std::cerr << "[t" << a.id << "] cannot open " << a.outPath << "\n";
        return;
    }

    // Per-thread RNG
    std::random_device rd;
    std::mt19937 gen(static_cast<uint32_t>(mix64(rd() ^ (0x9e3779b97f4a7c15ULL * (a.id + 1)))));

    Board board;
    board.loadNetwork(a.networkPath);
    Search search;
    search.setTTclearEnabled(false);
    search.setPrintInfo(false);

    SearchLimits limits{};
    limits.nodeLimit = a.nodes;    

    int evalLimit = 3000;

    const std::string startFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    std::vector<PositionData> posData{};
    float wdl = 1;
    while (a.produced->load(std::memory_order_relaxed) < a.quota) {

        if (a.book) {
            std::string fen = a.book->nextFen();
	    if(fen.empty()){
	    	std::cout << "Empty fen" << std::endl;
		continue;
	    }
            board.parseFen(fen);
        }
        else {

            board.parseFen(startFen);

            // randomize opening a bit
            for (int i = 0; i < 4; ++i) {
                MoveList l;
                MoveGenerator::generateMoves(board, l);
                if (l.counter == 0) break;
                std::uniform_int_distribution<> d(0, l.counter - 1);
                board.makeMove(l.moves[d(gen)]);
            }
        }
        

        // Here we randomize the first moves, since the opening book is looping to not generate the same position over and over
        for (int i = 0; i < 4; ++i) {
            MoveList l;
            MoveGenerator::generateMoves(board, l);
            if (l.counter == 0) break;
            std::uniform_int_distribution<> d(0, l.counter - 1);
            board.makeMove(l.moves[d(gen)]);
        }
        

        while(true){
            MoveList list;
            MoveGenerator::generateMoves(board, list);
            if (list.counter == 0 || board.hasPositionRepeated() || board.hasInsufficientMaterial()) {
                wdl = 0.5;
                break;
            }


            Score sc = search.search(board, limits);
            Move  best = sc.bestMove;

            // Skip noisy: in-check or capture-to-play
            bool isCapture = board.getPieceOnSquare(best.to()) != All;

            // Eval is noisy
            bool isNoisyEval = std::abs(search.evaluate(board) - sc.score) > 60;

            if (sc.score > evalLimit) {
                wdl = 1;
                break;
            }
            else if(sc.score < -evalLimit) {
                wdl = 0;
                break;
            }

            if (list.checkers == 0 && !isCapture && !isNoisyEval) {
                // Collect active indices
                PositionData data;

                int scoreModifier = 1;
                if (board.getSideToMove() == Black) {
                    scoreModifier = -1;
                }

                //White POV score;
                data.score = sc.score * scoreModifier;
                data.fen = FenTools::boardToFen(board);                
                posData.push_back(data);
        
        
            }

            if (board.getFullMoveClock() > 200) {
                wdl = 0.5;
                break;
            }

            
        
            board.makeMove(best);
        }
        a.produced->fetch_add(posData.size(), std::memory_order_relaxed);
        
        // 1.0 White win, 0.5 draw, 0 black win
        for (int i = 0; i < posData.size(); i++) {
            out << posData[i].fen << " | " << posData[i].score << " | " << std::format("{:.1f}", wdl) << "\n";
        }
        posData.clear();
    }

    out.flush();
    out.close();
}

void monitor(std::atomic<uint64_t> &produced, const uint64_t target) {

    while (true) {

        std::this_thread::sleep_for(std::chrono::seconds(10));

        int prod = produced.load(std::memory_order_relaxed);

        std::cout << prod << "/" << target << std::endl;

        if (prod > target) {
            return;
        }
    }

    
}

int main(int argc, char* argv[]) {

    std::string networkPath;
    std::string bookPath;
    uint64_t targetPositions = 1000000;
    int threads = 6;
    int depth = 4;
    int nodes = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-network" && i + 1 < argc) {
            networkPath = argv[i + 1];
        }

        if (arg == "-target_positions" && i + 1 < argc) {
            targetPositions = std::stoull(argv[i + 1]);
        }

        if (arg == "-threads" && i + 1 < argc) {
            threads = std::stoi(argv[i + 1]);
        }

        if (arg == "-depth" && i + 1 < argc) {
            depth = std::stoi(argv[i + 1]);
        }

        if (arg == "-book" && i + 1 < argc) {
            bookPath = argv[i + 1];
        }

        if (arg == "-nodes" && i + 1 < argc) {
            nodes = std::stoi(argv[i + 1]);
        }
    }

    OpeningBook* openingBook = nullptr;
    if (!bookPath.empty()) {
        openingBook = new OpeningBook();
        bool success = openingBook->loadBook(bookPath);
        if (success) {
            std::cout << "Opening book loaded: " << bookPath << std::endl;
        }
        else {
            std::cout << "Opening book not loaded" << std::endl;
        }


    }

    std::cout.setf(std::ios::unitbuf); // line-buffered

    std::atomic<uint64_t> produced{ 0 };
    std::vector<std::thread> pool;
    std::thread monitorThread(monitor,std::ref(produced), std::ref(targetPositions));
    pool.reserve(threads);

    auto t0 = std::chrono::steady_clock::now();

    for (int i = 0; i < threads; ++i) {
        WorkerArgs a;
        a.networkPath = networkPath;
        a.depth = depth;
        a.id = i;
        a.nodes = nodes;
        a.quota = targetPositions;
        a.produced = &produced;
        a.book = openingBook;
        a.outPath = "part_" + std::to_string(i) + ".txt";
        pool.emplace_back(worker_fn, a);
    }

    for (auto& th : pool) th.join();

    monitorThread.join();

    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "TOTAL produced=" << produced.load()
        << " in " << secs << " s using " << threads << " threads\n";
    std::cout << "Merge files (example):\n"
        << "  cat positions_indices.part_*.txt > positions_indices.txt\n";

    return 0;
}
