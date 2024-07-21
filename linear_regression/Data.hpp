#ifndef DATA_HPP
#define DATA_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <utility>

class Data {
public:
    // Constructor
    Data(const std::string& dataFilePath);

    // Destructor
    ~Data();

    // Function to print the first few rows of the data
    void head(const int numRows = 5);

    // Getter to get individual rows and target
    std::vector<float> getRow(int index) const;
    float getTarget(int index) const;

    // Member variables
    int rows;
    int cols;
    std::vector<std::vector<float>> dataFrame;
    std::vector<std::string> headers;
    std::string sourceFile;

    // Function to parse CSV file and fill dataFrame
    std::vector<std::vector<float>> parseCSV(const std::string& path, const int rows, const int cols);

    // Helper function to tokenize a string based on a delimiter
    std::vector<std::string> tokenize(const std::string& line);

    // Function to get the dimensions (rows and columns) of the CSV file
    std::pair<int, int> getDims(const std::string& path);

    std::vector<float> &operator [] (int index); 
};

#endif // DATA_HPP
