#include "openingbook.h"
#include <fstream>
#include <random>
#include <algorithm>

bool OpeningBook::loadBook(std::string path) {

	std::ifstream in(path);

	if (!in) {
		return false;
	}

	std::string line;
	while (std::getline(in, line)) {
		if (line.empty()) {
			continue;
		}

		fens.emplace_back(line);
	}

	std::shuffle(std::begin(fens), std::end(fens), std::default_random_engine{});
}

std::string OpeningBook::nextFen() {
	std::unique_lock lock(mutex);

	auto& fen = fens[index++];
	if (index >= fens.size()) {
		index = 0;
	}
	return fen;
}