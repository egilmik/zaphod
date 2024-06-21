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
    //std::string filename = "D:\\chess\\chessdb\\output.csv";
    std::string filename = "D:\\chess\\tuner\\wukong_positions.txt";

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
    Board board;

    //Swallow first line
    std::getline(file, line);

    while (std::getline(file, line) && vector->size() < 30000) {
        std::string fen = line.substr(0, line.size() - 5);
        std::string resultString = line.substr(line.size() - 4, 3);
        float result = std::stof(resultString);
        board.parseFen(fen);

        vector->push_back({fen,board.getBoardState(), result});
    }
    /*
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
    */
    std::cout << "Finished parsing csv.  Number of entries: " << vector->size() << std::endl;

    // Close the file
    file.close();
    
    bool improved = true;
    int epochs = 0;
    float bestError = tuner.calculateMSE(vector, board);
    float error = 0;
    
    while (improved) {
        std::cout << "Epoch: " << epochs++ << " Error: " << bestError << std::endl;
        
        improved = false;

        error = tuner.tuneMaterial(vector, board, bestError);
        if (error < bestError) {
            bestError = error;
            improved = true;
        }
        
        error = tuner.tunePSQT(vector, board, bestError);
        if (error < bestError) {
            bestError = error;
            improved = true;
        }

        error = tuner.tunePSQTEG(vector, board, bestError);
        if (error < bestError) {
            bestError = error;
            improved = true;
        }
        


        if (epochs % 10 == 0) {
            std::cout << "Material" << std::endl;
            std::cout << "{";
            for (int i = 0; i < 14; i++) {
                std::cout << Material::materialScoreArray[i] << ",";
            }
            std::cout << "};" << std::endl;

            std::cout << "PSQT EG" << std::endl;
            for (int i = 1; i < 7; i++) {
                std::cout << "{";
                for (int x = 0; x < 64; x++) {
                    std::cout << Material::pieceSquareScoreArrayEG[i][x] << ",";
                }
                std::cout << "}," << std::endl;
            }

            std::cout << "PSQT MG" << std::endl;
            std::cout << "{";
            for (int i = 1; i < 7; i++) {
                std::cout << "{";
                for (int x = 0; x < 64; x++) {
                    std::cout << Material::pieceSquareScoreArray[i][x] << ",";
                }
                std::cout << "}," << std::endl;
            }
        }

    }

    for (int i = 0; i < 14; i++) {
        std::cout << Material::materialScoreArray[i] << std::endl;
    }

    return 0;
}