#ifndef NNUE_H
#define NNUE_H

#include <string>
#include <array>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "bitboard.h"
#include "board.h"

class NNUE {
	public:
		void addPiece(int sq, BitBoardEnum piece, BitBoardEnum color);
		void removePiece(int sq, BitBoardEnum piece, BitBoardEnum color);
		float forward(Board &board);
		bool load(const std::string& path);

        int plane_index_from_piece(BitBoardEnum piece);

	private:
        static constexpr int IN = 768;
        static constexpr int H = 32;
        static constexpr int OUT = 1;

        // parameters
        std::vector<float> W1; // [H][IN] row-major
        std::vector<float> B1; // [H]
        std::vector<float> W2; // [OUT][H] (i.e. 32)
        float B2 = 0.f;
        float scale_cp = 600.f;


};

#endif