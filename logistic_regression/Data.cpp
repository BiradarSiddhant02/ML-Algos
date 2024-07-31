#include "Data.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>

Data::Data(const std::string& dataFilePath) : sourceFile(dataFilePath) {
    auto dims = getDims(dataFilePath);
    rows = dims.first;
    cols = dims.second;
    dataFrame = parseCSV(dataFilePath, rows, cols);
}

Data::~Data() {}

std::vector<std::string> Data::tokenize(const std::string& line) {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string cell;

    while (std::getline(ss, cell, ',')) {
        tokens.push_back(cell);
    }

    return tokens;
}

std::pair<int, int> Data::getDims(const std::string& path) {
    std::ifstream csvFile(path);

    if (!csvFile.is_open()) {
        std::cerr << "Error Opening File" << std::endl;
        return {0, 0};
    }

    std::string line;
    std::getline(csvFile, line);
    std::vector<std::string> tokens = tokenize(line);
    int columns = tokens.size();

    int rows = 0;
    while (std::getline(csvFile, line)) {
        rows++;
    }

    csvFile.close();

    return {rows, columns};
}

void Data::head(const int numRows) {
    int numRowsToPrint = std::min(numRows, rows);
    for (int i = 0; i < numRowsToPrint; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << dataFrame[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

std::vector<std::vector<float>> Data::parseCSV(const std::string& path, const int rows, const int cols) {
    std::ifstream csvFile(path);
    std::vector<std::vector<float>> data;

    if (!csvFile.is_open()) {
        std::cerr << "Error Opening File" << std::endl;
        return data;
    }

    std::string line;
    // Consume header line
    std::getline(csvFile, line);
    headers = tokenize(line);

    while (std::getline(csvFile, line)) {
        std::vector<std::string> tokens = tokenize(line);

        // Check if the number of tokens matches the expected number of columns
        if (tokens.size() != cols) {
            std::cerr << "Row has incorrect number of columns: " << line << std::endl;
            continue;
        }

        std::vector<float> tokens_f;
        for (const auto& token : tokens) {
            tokens_f.push_back(stof(token));
        }
        data.push_back(tokens_f);
    }

    csvFile.close();
    return data;
}

std::vector<float>& Data::operator[](int index) {
    if (index < 0 || index >= rows) {
        throw std::out_of_range("Index out of range");
    }
    return dataFrame[index];
}
