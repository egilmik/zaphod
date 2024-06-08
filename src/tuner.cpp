#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "board.h"
#include "search.h";

struct FenEvalStruct {
    std::string fen;
    float score;
};

int main() {
    // Name of the CSV file
    std::string filename = "D:\\chess\\chessdb\\output.csv";

    // Open the file
    std::ifstream file(filename);

    // Check if the file was opened successfully
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return 1;
    }
    std::vector<FenEvalStruct> vector;

    std::string line;

    //Swallow first line
    std::getline(file, line);
    
    // Read each line from the file
    while (std::getline(file, line) && vector.size() < 100) {
        std::stringstream ss(line);
        std::string column1;
        std::string column2_str;
        float column2;

        // Read the first column
        if (!std::getline(ss, column1, ',')) {
            std::cerr << "Error reading column 1" << std::endl;
            continue;
        }

        // Read the second column as string
        if (!std::getline(ss, column2_str, ',')) {
            std::cerr << "Error reading column 2" << std::endl;
            continue;
        }

        // Convert the second column to float
        try {
            column2 = std::stof(column2_str);
        }
        catch (const std::invalid_argument& e) {
            std::cerr << "Invalid number: " << column2_str << std::endl;
            continue;
        }
        catch (const std::out_of_range& e) {
            std::cerr << "Number out of range: " << column2_str << std::endl;
            continue;
        }
        FenEvalStruct fenEval = { column1,column2 };

        vector.push_back(fenEval);
    }
    std::cout << "Finished parsing csv" << std::endl;

    // Close the file
    file.close();
    Board board;
    Search search;

    for (int i = 0; i < vector.size(); i++) {
        FenEvalStruct fenEval = vector.at(i);
        board.parseFen(fenEval.fen);
        int score = search.evaluate(board);

        int x = 0;
    }

    return 0;
}