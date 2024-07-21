#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>

#include "Data.hpp"

#define LR 0.000001
#define EPOCHS 1000

std::vector<float> genWeights(const int cols) {
    std::vector<float> weights;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_real(0., 1.);
    
    for (int i = 0; i < cols; i++) {
        weights.push_back(dis_real(gen));
    }
    
    return weights;
}

std::pair<float, float> predict(const std::vector<float>& weights, const std::vector<float>& vector) {
    float product = weights.back(); // Initialize with the bias term
    std::pair<float, float> pred_gt_pair;

    // Check if the input vector size matches the weight vector size (excluding bias term)
    if (vector.size() != weights.size() - 1) {
        throw std::runtime_error("Input vector does not match weight vector size.");
    }

    // Set the ground truth value
    pred_gt_pair.second = vector.back(); 

    // Calculate the dot product plus bias
    for (int i = 0; i < vector.size(); i++) {
        product += weights[i] * vector[i];
    }

    // Apply ReLU activation
    pred_gt_pair.first = std::max(0.0f, product); // ReLU function

    return pred_gt_pair;
}


std::vector<std::pair<float, float>> prediction(const std::vector<float>& weights, const Data& df) {
    std::vector<std::pair<float, float>> predictions;

    for (int i = 0; i < df.rows; i++) {
        auto prediction = predict(weights, df.dataFrame[i]);
        predictions.push_back(prediction);
        // std::cout << i << ' ' << prediction.first << ' ' << prediction.second << std::endl;
    }

    return predictions;
}

float RMSE(const std::vector<std::pair<float, float>>& predictions) {
    int N = predictions.size();
    if (N == 0) {
        throw std::runtime_error("Predictions vector is empty");
    }

    float SE = 0;
    for (int i = 0; i < N; i++) {
        SE += std::pow((predictions[i].first - predictions[i].second), 2);
    }

    return std::sqrt(SE / N); // Corrected the division to be outside the loop
}

float step(float x) {
    return x >= 0 ? 1.0f : 0.0f;
}

std::vector<float> gradient(const std::vector<float>& weights, const Data& df) {
    int N = df.rows;
    std::vector<std::pair<float, float>> predictions = prediction(weights, df);
    std::vector<float> grad(weights.size(), 0.0);

    float rmse = RMSE(predictions);

    for (int i = 0; i < weights.size(); i++) {
        float grad_i = 0;
        for (int j = 0; j < N; j++) {
            float err_i = predictions[j].first - predictions[j].second;
            if (i < weights.size() - 1) {
                grad_i += df.dataFrame[j][i] * err_i;
            } else {
                grad_i += err_i; // Update bias term gradient
            }
        }
        grad[i] = (1 / rmse) * (1.0f / N) * grad_i;
    }
    return grad;
}


void printVec(const std::vector<float>& vec) {
    for (const auto& val : vec) {
        std::cout << val << ",";
    }
    std::cout << std::endl;
}

std::vector<float> gradientDescent(const Data& df, std::vector<float>& weights, const float lr, const int epochs) {
    std::cout << "Initial Weights:" << std::endl;
    printVec(weights);

    for (int epoch = 0; epoch < epochs; epoch++) {
        std::vector<std::pair<float, float>> predictions = prediction(weights, df);
        float error = RMSE(predictions);
        std::vector<float> grad = gradient(weights, df);

        for (int i = 0; i < weights.size(); i++) {
            weights[i] -= lr * grad[i];
        }

        std::cout << "------" << std::endl;
        std::cout << "Epoch: " << epoch << std::endl;
        std::cout << "Gradients:" << std::endl;
        printVec(grad);
        std::cout << "Weights:" << std::endl;
        printVec(weights);
        std::cout << "Error: " << error << std::endl;
    }
    return weights;
}


int main() {
    std::string path = "data.csv";
    Data data(path);

    // Generate weights with bias term included
    std::vector<float> weights = genWeights(data.cols + 1); // +1 for the bias term

    // Run gradient descent
    std::vector<float> finalWeights = gradientDescent(data, weights, LR, EPOCHS);

    return 0;
}

