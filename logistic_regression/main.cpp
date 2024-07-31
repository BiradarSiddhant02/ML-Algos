#include "Data.hpp"
#include <iostream>
#include <string>

int main(){
    std::string path = "creditcard.csv";
    Data df(path);
    df.head();
}