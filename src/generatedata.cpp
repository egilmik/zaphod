#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>
#include "board.h"
#include "search.h"
#include "nnue.h"

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
    Search search;

    uint64_t local_written = 0;
    const std::string startFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    while (a.produced->load(std::memory_order_relaxed) < a.quota) {

        board.parseFen(startFen);

        // randomize opening a bit
        for (int i = 0; i < 10; ++i) {
            MoveList l;
            MoveGenerator::generateMoves(board, l);
            if (l.counter == 0) break;
            std::uniform_int_distribution<> d(0, l.counter - 1);
            board.makeMove(l.moves[d(gen)]);
        }

        while (a.produced->load(std::memory_order_relaxed) < a.quota) {
            MoveList list;
            MoveGenerator::generateMoves(board, list);
            if (list.counter == 0 || board.hasPositionRepeated()) break;

            // Search label (adjust depth/time as you wish)
            Score sc = search.search(board, /*depth*/4, /*time_ms*/150);
            Move  best = sc.bestMove;

            // Skip noisy: in-check or capture-to-play
            bool isCapture = board.getPieceOnSquare(best.to()) != All;
            if (list.checkers == 0 && !isCapture) {
                // Collect active indices
                int idxs[64]; // enough (<= pieces on board)
                int n = 0;

                BitBoard all = board.getBitboard(All);
                while (all) {
                    int sq = board.popLsb(all);
                    BitBoardEnum piece = board.getPieceOnSquare(sq);
                    int pl = NNUE::plane_index_from_piece(piece); // 0..11
                    if (pl >= 0) idxs[n++] = pl * 64 + sq;
                }

                if (n > 0) {
                    std::sort(idxs, idxs + n);

                    // white-POV score
                    int score_cp = sc.score;
                    if (board.getSideToMove() == Black) score_cp = -score_cp;

                    // Reserve a slot *now*; stop if quota reached
                    uint64_t slot = a.produced->fetch_add(1, std::memory_order_relaxed);
                    if (slot < a.quota) {
                        // write line
                        out << idxs[0];
                        for (int i = 1; i < n; ++i) out << ' ' << idxs[i];
                        out << " ; " << score_cp << '\n';
                        ++local_written;

                        if ((slot + 1) % 100000 == 0) {
                            std::cout << "[t" << a.id << "] produced " << (slot + 1) << "\n";
                        }
                    }
                    else {
                        // exceeded quota: undo reservation and exit
                        a.produced->fetch_sub(1, std::memory_order_relaxed);
                        break;
                    }
                }
            }

            // advance game
            board.makeMove(best);
        }
    }

    out.flush();
    out.close();
    std::cout << "[t" << a.id << "] wrote " << local_written
        << " lines to " << a.outPath << "\n";
}

int main(int argc, char** argv) {
    const uint64_t target_positions = (argc >= 2) ? std::stoull(argv[1]) : 100'000ULL;
    int threads = (argc >= 3) ? std::stoi(argv[2]) : (int)std::thread::hardware_concurrency();
    if (threads <= 0) threads = 1;

    std::cout.setf(std::ios::unitbuf); // line-buffered

    std::atomic<uint64_t> produced{ 0 };
    std::vector<std::thread> pool;
    pool.reserve(threads);

    auto t0 = std::chrono::steady_clock::now();

    for (int i = 0; i < threads; ++i) {
        WorkerArgs a;
        a.id = i;
        a.quota = target_positions;
        a.produced = &produced;
        a.outPath = "positions_indices.part_" + std::to_string(i) + ".txt";
        pool.emplace_back(worker_fn, a);
    }

    for (auto& th : pool) th.join();

    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "TOTAL produced=" << produced.load()
        << " in " << secs << " s using " << threads << " threads\n";
    std::cout << "Merge files (example):\n"
        << "  cat positions_indices.part_*.txt > positions_indices.txt\n";

    return 0;
}
