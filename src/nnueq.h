#ifndef NNUEQ_H
#define NNUEQ_H

#include <string>
#include <array>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <memory>

#include "bitboard.h"

namespace NNUE{
    constexpr int IN = 768;
    constexpr int H = 512;
    constexpr int OUT = 1;

    struct Network {
        alignas(32) std::array<int16_t, IN*H>  l0w;
        alignas(32) std::array<int16_t, H>  l0b;
        alignas(32) std::array<int16_t, H*2>  l1w;
        int16_t l1b;
    };

    // Both perspectives of the first layer, [0] = white, [1] = black
    struct alignas(32) Accumulator {
        int16_t pre[2][H];
    };

    // One accumulator per move made, so unmaking a move is a pop. Matches
    // Board::MAXMOVEHISTORY, plus the root position.
    constexpr int ACC_STACK_SIZE = 1024 + 1;
}

using namespace NNUE;

class NNUEQ {
public:
    
    NNUEQ() : net(std::make_unique<Network>()) {}
  
    bool load(const std::string& path);
    bool loadEmbedded();
    int forward(BitBoardEnum stm);

    // Queue a feature change. Nothing is computed until push() or refresh().
    void addPiece(BitBoardEnum piece, int sq) { queue(piece, sq, true); }
    void removePiece(BitBoardEnum piece, int sq) { queue(piece, sq, false); }

    // New accumulator on the stack: the current one plus the queued changes,
    // applied in a single pass over the weights. Called once per makeMove.
    void push();
    // Back to the previous accumulator, called by revertLastMove. Drops
    // whatever was queued while the move was taken back.
    void pop() { top--; pendingCount = 0; }
    // Resets the stack to the bias and applies the queued changes (a new position).
    void refresh();

    void clear();

    int64_t VectorizedSCReLU_AVX2(const int16_t* __restrict stmAcc, const int16_t* __restrict nstmAcc, const int16_t* __restrict wStm, const int16_t* __restrict wNstm);

    static int plane_index_from_piece(BitBoardEnum piece);
    static int encodeFeature(int piece, int sq, BitBoardEnum color);


private:
    // Set to true when network is loaded.
    bool isInitialized = false;
    std::unique_ptr<Network> net;

    struct PendingChange {
        uint8_t piece;
        uint8_t sq;
        bool add;
    };

    void queue(BitBoardEnum piece, int sq, bool add) {
        pending[pendingCount++] = { static_cast<uint8_t>(piece), static_cast<uint8_t>(sq), add };
    }
    void applyPending(const Accumulator& from, Accumulator& to);
    bool checkAccumulatorRange() const;

    std::unique_ptr<Accumulator[]> accStack = std::make_unique<Accumulator[]>(ACC_STACK_SIZE);
    int top = 0;
    // Enough for a full position set up from a FEN
    PendingChange pending[64];
    int pendingCount = 0;
    
    int16_t SCALE = 400;
    int16_t QA = 255;
    int16_t QB = 64;
};

#endif
