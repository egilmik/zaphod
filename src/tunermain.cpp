#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "board.h"
#include "search.h";
#include "material.h"
#include "tuner.h"

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
    std::vector<FenEvalStruct> *vector = new std::vector<FenEvalStruct>();

    std::string line;
    Tuner tuner;

    //Swallow first line
    std::getline(file, line);
    
    // Read each line from the file
    while (std::getline(file, line) && vector->size() < 100000) {
        std::stringstream ss(line);
        std::string column1;
        std::string column2_str;
        int column2;

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
            column2 = std::stof(column2_str)*100;
        }
        catch (const std::invalid_argument& e) {
            std::cerr << "Invalid number: " << column2_str << std::endl;
            continue;
        }
        catch (const std::out_of_range& e) {
            std::cerr << "Number out of range: " << column2_str << std::endl;
            continue;
        }
        float score = tuner.sigmoid(column2);
        FenEvalStruct fenEval = { column1,score };

        vector->push_back(fenEval);
    }
    std::cout << "Finished parsing csv" << std::endl;

    // Close the file
    file.close();
    
    bool improved = true;
    int epochs = 0;
    Board board;
    float bestError = tuner.calculateMSE(vector, board);
    
    while (improved) {
        epochs++;
        improved = false;

        for (int i = 1; i < 14; i++) {            

            Material::materialScoreArray[i] += 1;

            float newError = tuner.calculateMSE(vector, board);

            if (newError < bestError) {
                bestError = newError;
                improved = true;
            }
            else {
                Material::materialScoreArray[i] -= 2;
                newError = tuner.calculateMSE(vector, board);

                if (newError < bestError) {
                    bestError = newError;
                    improved = true;
                }
            }
        }


    }

    std::cout << "Epoch: " << epochs << std::endl;

    for (int i = 0; i < 14; i++) {
        std::cout << Material::materialScoreArray[i] << std::endl;
    }

    return 0;
}