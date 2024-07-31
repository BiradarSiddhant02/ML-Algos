#include <vector>
#include <string>

#ifndef DATATABLE_HPP
#define DATATABLE_HPP

struct Column {
    unsigned int index = 0;
    std::string name = "";
    std::vector<double> dataPoints = {};
};


struct BestSplit{
    unsigned int featureIndex;
    double threshold;
    DataTable datasetLeft;
    DataTable datasetRight;
    double infoGain;
};

class DataTable {
public:
    std::vector<Column> columns;
    std::vector<std::string> columnNames;
    unsigned int columnCount;
    unsigned int rows;
    std::string sourceFile;

    DataTable();
    DataTable(const std::string& path);
    DataTable(const std::string& name, const std::vector<double>& data);
    void addColumn(const std::string& name, const std::vector<double>& data);

    const std::vector<double>& operator[] (const std::string& columnName) const;

    std::vector<double> getRow(const unsigned int index) const;

private:

    void loadData();

};

#endif