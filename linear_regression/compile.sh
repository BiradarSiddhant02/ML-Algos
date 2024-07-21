#!/bin/bash

g++ -c -o build/Data.o Data.cpp
g++ -c -o build/main.o main.cpp
g++ -o build/linear_regression build/*.o