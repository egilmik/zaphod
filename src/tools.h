#ifndef TOOLS_H
#define TOOLS_H

#include "board.h"

class Tools {
public:
	static bool isBoardConsistent(Board& board) {
		bool isConsistent = false;

		if (checkPiecesAgainstColorBoard(board, White) &&
			checkPiecesAgainstColorBoard(board, Black)) {
			isConsistent = true;
		}

		if (checkPiecesAgainstAll(board, White) &&
			checkPiecesAgainstAll(board, Black)) {
			isConsistent = true;
		}


		return isConsistent;
	};

	static bool isMailBoxConsistent(Board& board) {
		BitBoard allPieces = board.getBitboard(All);
		int square = 0;
		while (allPieces) {
			square = board.popLsb(allPieces);
			BitBoardEnum piece = board.getPieceOnSquare(square);
			if (piece == All) {
				return false;
			}
		}
		return true;

	}

	static bool checkPiecesAgainstColorBoard(Board& board, BitBoardEnum color) {
		BitBoard colorBoard = board.getBitboard(color);
		BitBoard pawnBoard = board.getBitboard(P + color);
		BitBoard knightBoard = board.getBitboard(N + color);
		BitBoard bishopBoard = board.getBitboard(B + color);
		BitBoard queenBoard = board.getBitboard(Q + color);
		BitBoard kingBoard = board.getBitboard(K + color);
		BitBoard rookBoard = board.getBitboard(R + color);

		bool isConsistent = false;

		if (pawnBoard == (colorBoard & pawnBoard) &&
			knightBoard == (colorBoard & knightBoard) &&
			bishopBoard == (colorBoard & bishopBoard) && 
			queenBoard == (colorBoard & queenBoard) && 
			kingBoard == (colorBoard & kingBoard)  &&
			rookBoard == (colorBoard & rookBoard)) {
			isConsistent = true;
		}
		return isConsistent;
	}

	static bool checkPiecesAgainstAll(Board& board, BitBoardEnum color) {
		BitBoard allBoard = board.getBitboard(All);
		BitBoard pawnBoard = board.getBitboard(P + color);
		BitBoard knightBoard = board.getBitboard(N + color);
		BitBoard bishopBoard = board.getBitboard(B + color);
		BitBoard queenBoard = board.getBitboard(Q + color);
		BitBoard kingBoard = board.getBitboard(K + color);
		BitBoard rookBoard = board.getBitboard(R + color);

		bool isConsistent = false;

		if (pawnBoard == (allBoard & pawnBoard) &&
			knightBoard == (allBoard & knightBoard) &&
			bishopBoard == (allBoard & bishopBoard) &&
			queenBoard == (allBoard & queenBoard) &&
			kingBoard == (allBoard & kingBoard) &&
			rookBoard == (allBoard & rookBoard)) {
			isConsistent = true;
		}
		return isConsistent;
	}


};

#endif