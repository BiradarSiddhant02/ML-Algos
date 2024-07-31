#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>

// Hyper Parameters
#define LR .001
#define LAMBDA .01
#define ITERS 1000

void readCSV(const std::string& filename, std::vector<std::vector<double>>& X, std::vector<double>& y) {
    auto start = std::chrono::high_resolution_clock::now();  // Start time

    std::ifstream file(filename);
    std::string line;

    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string item;
        std::vector<double> features;
        double label;

        // Read features
        while (std::getline(ss, item, ',')) {
            features.push_back(std::stod(item));
        }
        
        // Separate label
        label = features.back();
        features.pop_back();

        X.push_back(features);
        y.push_back(label);
    }

    auto end = std::chrono::high_resolution_clock::now();  // End time
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "File processed in " << elapsed.count() << " seconds" << std::endl;
}

void trainTestSplit(const std::vector<std::vector<double>>& X, const std::vector<double>& y, 
                    std::vector<std::vector<double>>& X_train, std::vector<double>& y_train, 
                    std::vector<std::vector<double>>& X_test, std::vector<double>& y_test, 
                    double test_size = 0.2) {
    size_t total_size = X.size();
    size_t test_size_count = static_cast<size_t>(total_size * test_size);

    // Calculate split index
    size_t split_index = total_size - test_size_count;

    // Split data
    X_train.assign(X.begin(), X.begin() + split_index);
    y_train.assign(y.begin(), y.begin() + split_index);
    X_test.assign(X.begin() + split_index, X.end());
    y_test.assign(y.begin() + split_index, y.end());
}


std::vector<double> fit(const std::vector<std::vector<double>>& X_train, const std::vector<double>& y_train) {
    auto start = std::chrono::high_resolution_clock::now();  // Start time

    size_t n_samples = X_train.size();
    size_t n_features = X_train[0].size();
    
    std::vector<double> weights(n_features, 0.0);
    double bias = 0;

    std::vector<double> y_train_new(y_train.size());
    std::transform(y_train.begin(), y_train.end(), y_train_new.begin(), [](double label) {
        return label <= 0 ? -1 : 1;
    });

    for (int iter = 0; iter < ITERS; ++iter) {
        std::cout << "Iteration: " << iter << std::endl;
        for (size_t idx = 0; idx < n_samples; ++idx) {
            const std::vector<double>& x_i = X_train[idx];
            double inner_prod = std::inner_product(x_i.begin(), x_i.end(), weights.begin(), -bias);
            bool condition = y_train_new[idx] * inner_prod >= 1.0;

            if (condition) {
                for (size_t k = 0; k < n_features; ++k) {
                    weights[k] -= LR * (2 * LAMBDA * weights[k]);
                }
            } else {
                double dot_product = std::inner_product(x_i.begin(), x_i.end(), weights.begin(), 0.0);
                for (size_t k = 0; k < n_features; ++k) {
                    weights[k] -= LR * (2 * LAMBDA * weights[k] - y_train_new[idx] * x_i[k]);
                }
                bias -= LR * y_train_new[idx];
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();  // End time
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Model trained in " << elapsed.count() << " seconds" << std::endl;

    return weights;
}

std::vector<double> predict(const std::vector<std::vector<double>>& X, const std::vector<double>& weights, double bias) {
    std::vector<double> predictions;
    for (const auto& features : X) {
        double inner_prod = std::inner_product(features.begin(), features.end(), weights.begin(), bias);
        predictions.push_back(inner_prod >= 0 ? 1 : 0);
    }
    return predictions;
}

double calculateAccuracy(const std::vector<double>& true_labels, const std::vector<double>& predictions) {
    if (true_labels.size() != predictions.size()) {
        std::cerr << "True labels and predictions size mismatch!" << std::endl;
        return 0.0;
    }

    size_t correct = 0;
    for (size_t i = 0; i < true_labels.size(); ++i) {
        if (true_labels[i] == predictions[i]) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / true_labels.size();
}


int main() {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    readCSV("nn_data.csv", X, y);

    std::vector<std::vector<double>> X_train, X_test;
    std::vector<double> y_train, y_test;

    trainTestSplit(X, y, X_train, y_train, X_test, y_test);

    std::vector<double> weights = fit(X_train, y_train);

    double bias = 0;
    std::vector<double> predictions = predict(X_test, weights, bias);

    // std::cout << "Test Set Features and Expected Outputs:\n";
    // for (size_t i = 0; i < X_test.size(); ++i) {
    //     for (const auto& feature : X_test[i]) {
    //         std::cout << feature << " ";
    //     }
    //     std::cout << y_test[i] << ' ' << predictions[i] << std::endl;
    // }

    double accuracy = calculateAccuracy(y_test, predictions);
    std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;

    return 0;
}

