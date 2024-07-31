#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>
#include <sstream>

#include "DataTable.hpp"

DataTable::DataTable() : columns({}), columnNames({}), columnCount(0), rows(0), sourceFile("") {}

DataTable::DataTable(const std::string& path) : sourceFile(path), rows(0), columnCount(0) {
    loadData();
}

DataTable::DataTable(const std::string& name, const std::vector<double>& data)
    : sourceFile(""), columnCount(1), rows(data.size()) {
    Column newCol;
    newCol.name = name;
    columnNames.push_back(name);
    newCol.dataPoints = data;
    columns.push_back(newCol);
}

void DataTable::loadData() {
    std::ifstream file(sourceFile);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    std::string line;
    if (std::getline(file, line)) {
        std::istringstream stream(line);
        std::string header;
        int index = 0;
        while (std::getline(stream, header, ',')) {
            Column newColumn;
            newColumn.index = index++;
            newColumn.name = header;
            columnNames.push_back(header);
            columns.push_back(newColumn);
        }
        columnCount = columns.size();
    }

    while (std::getline(file, line)) {
        std::istringstream stream(line);
        std::string cell;
        int colIndex = 0;

        while (std::getline(stream, cell, ',')) {
            if (colIndex < columnCount) {
                columns[colIndex].dataPoints.push_back(std::stod(cell));
            }
            colIndex++;
        }
        rows++;
    }
}

const std::vector<double>& DataTable::operator[] (const std::string& columnName) const {
    for (const auto& column : columns) {
        if (column.name == columnName) {
            return column.dataPoints;
        }
    }
    throw std::out_of_range("Column name not found");
}

void DataTable::addColumn(const std::string& name, const std::vector<double>& data) {
    Column newColumn;
    newColumn.index = columns.size() + 1;
    newColumn.name = name;
    newColumn.dataPoints = data;

    columns.push_back(newColumn);
    columnNames.push_back(name);
    columnCount++;
    sourceFile = "";
}

std::vector<double> DataTable::getRow(const unsigned int index) const{
    std::vector<double> row;
    for(const auto& column : columns) {
        row.push_back(column.dataPoints[index]);
    }
    return row;
}
