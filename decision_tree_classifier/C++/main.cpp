#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <map>

#include "DataTable.hpp"

std::ostream& operator<<(std::ostream& os, const std::vector<double>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i != vec.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

int main() {
    DataTable dataTable("../data/data.csv");
    return 0;
}
