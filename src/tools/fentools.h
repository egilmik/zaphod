#ifndef FENTOOLS_H
#define FENTOOLS_H

#include <string>
#include "../board.h"

class FenTools {
public:
	
	std::string boardToFen(Board& board) {
		std::string fenString;
		fenString.reserve(32);
		for (int r = 7; r >= 0; r--) {
			int increment = 0;
			for (int f = 0;f < 8; f++) {
				BitBoardEnum piece = board.getPieceOnSquare(r * 8 + f);
				if (piece == All) {
					increment++;
				}
				else {
					if (increment > 0) {
						fenString.push_back(char('0' + increment));
						increment = 0;
					}
					fenString.push_back(pieceToChar(piece));
				}
			}
			if (increment > 0) {
				fenString.push_back(char('0' + increment));
			}
			if (r > 0) {
				fenString.push_back('/');
			}
		}

		fenString.push_back(' ');
		bool whiteToMove = board.getSideToMove() == White;

		fenString.push_back(whiteToMove ? 'w' : 'b');
		
		fenString.push_back(' ');
		if (board.getCastleRightsBK() || board.getCastleRightsBQ() || board.getCastleRightsWK() || board.getCastleRightsWQ()) {
			if (board.getCastleRightsWK()) fenString.push_back('K');
			if (board.getCastleRightsWQ()) fenString.push_back('Q');
			if (board.getCastleRightsBK()) fenString.push_back('k');
			if (board.getCastleRightsBQ()) fenString.push_back('q');
		}
		else {
			fenString.push_back('-');
		}

		fenString.push_back(' ');
		if (board.getEnPassantSq() >= 0) {
			fenString.push_back(char('a' + (board.getEnPassantSq() & 7)));
			fenString.push_back(char('1' + (board.getEnPassantSq() >> 3)));
		}
		else {
			fenString.push_back('-');
		}

		fenString.push_back(' ');
		fenString += std::to_string(board.getHalfMoveClock());
		fenString.push_back(' ');
		fenString += std::to_string(std::max(1, board.getFullMoveClock()));

		return fenString;
	}

	char pieceToChar(BitBoardEnum piece) {
		if (piece == p) {
			return 'p';
		}
		else if (piece == P) {
			return 'P';
		}
		else if (piece == n) {
			return 'n';
		}
		else if (piece == N) {
			return 'N';
		}
		else if (piece == b) {
			return 'b';
		}
		else if (piece == B) {
			return 'B';
		}
		else if (piece == r) {
			return 'r';
		}
		else if (piece == R) {
			return 'R';
		}
		else if (piece == q) {
			return 'q';
		}
		else if (piece == Q) {
			return 'Q';
		}
		else if (piece == k) {
			return 'k';
		}
		else if (piece == K) {
			return 'K';
		}
		return 'X';
	}

private:
	
};

#endif
