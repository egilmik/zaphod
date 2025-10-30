#ifndef OPENINGBOOK_H
#define OPENINGBOOK_H

#include <string>
#include <vector>
#include <mutex>

class OpeningBook {
public:
	bool loadBook(std::string path);
	std::string nextFen();

private:
	std::vector<std::string> fens;
	uint64_t index = 0;
	std::mutex mutex{};
};

#endif
