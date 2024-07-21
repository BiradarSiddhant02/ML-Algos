#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

class Data {
private:
    std::vector<std::vector<int>> m_dataFrame;
    int m_rows;
    int m_cols;
    std::string m_sourceFile;

public:
    Data(int rows, int cols, const std::string& dataFilePath) 
        : m_rows(rows), m_cols(cols), m_sourceFile(dataFilePath) {
        m_dataFrame = parseCSV(dataFilePath);
    }

    ~Data() {}

    std::vector<std::vector<int>> parseCSV(const std::string& path);
};

std::vector<std::vector<int>> Data::parseCSV(const std::string& path) {
    std::ifstream csvFile(path);
    
    if (csvFile.fail()) {
        std::cout << "Error Opening File" << std::endl;
        return {};
    }

    std::string sample;
    int rows = 0;
    int cols = 0;

    while (getline(csvFile, sample)) {
        std::cout << sample << std::endl;
        rows++;
    }
    
    csvFile.close();
    return {};
} 

std::pair<int, int> getDims(const std::string& path){
    std::ifstream csvFile(path);

    if (csvFile.fail()) {
        std::cout << "Error Opening File" << std::endl;
        return {};
    }

    std::string line;
    getline(csvFile, line);
    std::vector<std::string> tokens;
    std::stringstream check(line);
    std::string intermediate;
    char delim  = ',';

    while (getline(check, intermediate, delim))
        tokens.push_back(intermediate);

    int columns = tokens.size();

    csvFile.close();

    csvFile.open(path);
    if (csvFile.fail()) {
        std::cout << "Error Opening File" << std::endl;
        return {};
    }

    int rows = 0;
    std::string sample;

    while (getline(csvFile, sample)) {
        std::cout << sample << std::endl;
        rows++;
    }

    csvFile.close();

    return {rows, columns};
}

int main() {

    std::string dataFilePath = "data.csv";
    std::pair<int, int> dim = getDims(dataFilePath);
    Data linearRegressionData(dim.first, dim.second, dataFilePath);
    return 0;

}

