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
#include <cmath>
#include <cstdint>
#include <limits>
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
    search.setPrintInfo(false);

    SearchLimits limits{};
    limits.nodeLimit = a.nodes;    

    int evalLimit = 3000;

    const std::string startFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    std::vector<PositionData> posData{};
    float wdl = 1;
    while (a.produced->load(std::memory_order_relaxed) < a.quota) {
        search.setNewGame();
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
        
        int moveCounter = 0;

        while(true){
            MoveList list;
            MoveGenerator::generateMoves(board, list);
            if (list.counter == 0 || board.hasPositionRepeated() || board.hasInsufficientMaterial() || moveCounter > 200) {
                wdl = 0.5;
                break;
            }


            Score sc = search.search(board, limits);
            Move best = sc.bestMove;

            
            if (board.getSideToMove() == Black) {
                sc.score = sc.score * -1;
            }

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

                data.score = sc.score;
                data.fen = FenTools::boardToFen(board);                
                posData.push_back(data);
        
        
            }

            if (board.getFullMoveClock() > 200) {
                wdl = 0.5;
                break;
            }

            
        
            board.makeMove(best);
            moveCounter++;
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

    int lastProduced = 0;
    const int sleepSeconds = 10;

    int hoursLeft = 0;
    int minLeft = 0;

    while (true) {

        std::this_thread::sleep_for(std::chrono::seconds(sleepSeconds));

        int prod = produced.load(std::memory_order_relaxed);

        if (prod > target) {
            return;
        }

        int delta = prod - lastProduced;
        if (delta > sleepSeconds) {
            int perSecond = delta / sleepSeconds;
            int secondsLeft = (target - prod) / perSecond;
            hoursLeft = secondsLeft / 60 / 60;
            minLeft = secondsLeft / 60 % 60;
        }

        std::cout << prod << "/" << target << " ETA: " << hoursLeft << " h " << minLeft << " m" << std::endl;

        lastProduced = prod;
    }

    
}


// ---------------------------------------------------------------------------
// Depth benchmark
//
// Instrumentation used to compare how deep the engine gets on a fixed node
// budget. Every position is searched from a clean slate (setNewGame clears the
// transposition table and the history tables), so the result for a position
// does not depend on the positions searched before it, on the thread that
// happened to pick it up, or on the wall clock. That makes the run
// reproducible and lets two builds be compared position by position.
// ---------------------------------------------------------------------------

struct DepthBenchResult {
    int depth = 0;       // last completed iterative deepening iteration
    int seldepth = 0;    // deepest ply reached in that last started iteration
    int score = 0;
    uint64_t nodes = 0;
};

struct DepthBenchArgs {
    const std::vector<std::string>* fens = nullptr;
    std::vector<DepthBenchResult>* results = nullptr;
    std::string networkPath;
    size_t begin = 0;
    size_t end = 0;
    int nodes = 10000;
    int ttSizeMB = 0;    // 0 keeps the engine default
};

static void depth_bench_worker(DepthBenchArgs a) {
    Board board;
    if (!a.networkPath.empty()) {
        board.loadNetwork(a.networkPath);
    }

    Search search;
    search.setPrintInfo(false);
    if (a.ttSizeMB > 0) {
        search.setTTSize(a.ttSizeMB);
    }

    SearchLimits limits{};
    limits.nodeLimit = a.nodes;

    for (size_t i = a.begin; i < a.end; ++i) {
        // Clean slate so the positions stay independent of each other.
        search.setNewGame();
        search.currentFinishedDepth = 0;
        search.maxPlyThisIteration = 0;
        search.evaluatedNodes = 0;

        board.parseFen((*a.fens)[i]);

        Score sc = search.search(board, limits);

        DepthBenchResult r;
        r.depth = search.currentFinishedDepth;
        r.seldepth = search.maxPlyThisIteration;
        r.score = sc.score;
        r.nodes = search.evaluatedNodes;
        (*a.results)[i] = r;
    }
}

static uint64_t fnv1a(const std::string& s, uint64_t h = 1469598103934665603ULL) {
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    return h;
}

static int run_depth_bench(const std::string& bookPath, const std::string& networkPath,
                           size_t positions, int nodes, int threads, int ttSizeMB,
                           const std::string& outPath, const std::string& label) {
    if (bookPath.empty()) {
        std::cerr << "-depth_bench needs -book <path to epd/fen file>" << std::endl;
        return 1;
    }

    OpeningBook book;
    if (!book.loadBook(bookPath)) {
        std::cerr << "Could not load book " << bookPath << std::endl;
        return 1;
    }

    // Pulled on one thread so the set of positions, and their order, is the
    // same for every run regardless of the thread count.
    std::vector<std::string> fens;
    fens.reserve(positions);
    uint64_t bookHash = 1469598103934665603ULL;
    for (size_t i = 0; i < positions; ++i) {
        std::string fen = book.nextFen();
        if (fen.empty()) {
            std::cerr << "Book returned an empty fen at " << i << std::endl;
            return 1;
        }
        bookHash = fnv1a(fen, bookHash);
        fens.push_back(std::move(fen));
    }

    std::cout << "depth_bench label=" << label
              << " positions=" << fens.size()
              << " nodes=" << nodes
              << " threads=" << threads
              << " book=" << bookPath
              << " book_hash=" << bookHash << std::endl;

    std::vector<DepthBenchResult> results(fens.size());

    if (threads < 1) threads = 1;
    if (static_cast<size_t>(threads) > fens.size()) threads = static_cast<int>(fens.size());

    auto t0 = std::chrono::steady_clock::now();

    std::vector<std::thread> pool;
    pool.reserve(threads);
    const size_t chunk = (fens.size() + threads - 1) / threads;
    for (int t = 0; t < threads; ++t) {
        DepthBenchArgs a;
        a.fens = &fens;
        a.results = &results;
        a.networkPath = networkPath;
        a.begin = std::min(fens.size(), chunk * static_cast<size_t>(t));
        a.end = std::min(fens.size(), a.begin + chunk);
        a.nodes = nodes;
        a.ttSizeMB = ttSizeMB;
        pool.emplace_back(depth_bench_worker, a);
    }
    for (auto& th : pool) th.join();

    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // Aggregate
    double depthSum = 0;
    double seldepthSum = 0;
    double nodeSum = 0;
    int minDepth = std::numeric_limits<int>::max();
    int maxDepth = 0;
    std::vector<int> depths;
    depths.reserve(results.size());
    for (const auto& r : results) {
        depthSum += r.depth;
        seldepthSum += r.seldepth;
        nodeSum += static_cast<double>(r.nodes);
        minDepth = std::min(minDepth, r.depth);
        maxDepth = std::max(maxDepth, r.depth);
        depths.push_back(r.depth);
    }
    const double n = static_cast<double>(results.size());
    const double avgDepth = depthSum / n;

    double var = 0;
    for (int d : depths) {
        const double diff = d - avgDepth;
        var += diff * diff;
    }
    var /= n;
    const double sd = std::sqrt(var);
    const double stderrMean = sd / std::sqrt(n);

    std::sort(depths.begin(), depths.end());
    const int median = depths[depths.size() / 2];

    if (!outPath.empty()) {
        std::ofstream out(outPath);
        if (!out) {
            std::cerr << "Cannot open " << outPath << std::endl;
        }
        else {
            out << "index,depth,seldepth,nodes,score,fen\n";
            for (size_t i = 0; i < results.size(); ++i) {
                out << i << "," << results[i].depth << "," << results[i].seldepth << ","
                    << results[i].nodes << "," << results[i].score << ","
                    << fens[i] << "\n";
            }
            out.flush();
            std::cout << "Per position data written to " << outPath << std::endl;
        }
    }

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "RESULT label=" << label
              << " positions=" << results.size()
              << " node_limit=" << nodes
              << " avg_depth=" << avgDepth
              << " sd_depth=" << sd
              << " stderr=" << stderrMean
              << " median_depth=" << median
              << " min_depth=" << minDepth
              << " max_depth=" << maxDepth
              << " avg_seldepth=" << (seldepthSum / n)
              << " avg_nodes=" << (nodeSum / n)
              << " seconds=" << secs << std::endl;

    std::cout << "depth histogram (depth: count)" << std::endl;
    for (int d = minDepth; d <= maxDepth; ++d) {
        const auto count = std::count(depths.begin(), depths.end(), d);
        if (count) {
            std::cout << "  " << d << ": " << count << std::endl;
        }
    }

    return 0;
}

int main(int argc, char* argv[]) {

    std::string networkPath;
    std::string bookPath;
    uint64_t targetPositions = 1000000;
    int threads = 6;
    int depth = 4;
    int nodes = 0;

    // Depth benchmark options
    bool depthBench = false;
    uint64_t benchPositions = 10000;
    int benchTTSizeMB = 0;
    std::string benchOut;
    std::string benchLabel = "engine";

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

        if (arg == "-depth_bench") {
            depthBench = true;
        }

        if (arg == "-positions" && i + 1 < argc) {
            benchPositions = std::stoull(argv[i + 1]);
        }

        if (arg == "-bench_out" && i + 1 < argc) {
            benchOut = argv[i + 1];
        }

        if (arg == "-label" && i + 1 < argc) {
            benchLabel = argv[i + 1];
        }

        if (arg == "-tt" && i + 1 < argc) {
            benchTTSizeMB = std::stoi(argv[i + 1]);
        }
    }

    std::cout.setf(std::ios::unitbuf); // line-buffered

    if (depthBench) {
        if (nodes <= 0) {
            nodes = 10000;
        }
        return run_depth_bench(bookPath, networkPath, static_cast<size_t>(benchPositions),
                               nodes, threads, benchTTSizeMB, benchOut, benchLabel);
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
