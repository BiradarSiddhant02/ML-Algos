#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip> 

class NeuralNetwork {
public:
    NeuralNetwork(int num_features, int hidden_size, int num_classes) {
        this->num_features = num_features;
        this->hidden_size = hidden_size;
        this->num_classes = num_classes;

        weights_input_hidden.resize(hidden_size, std::vector<double>(num_features));
        weights_hidden_output.resize(num_classes, std::vector<double>(hidden_size));
        initializeWeights();
    }

    std::vector<double> forward(const std::vector<double>& input) {
        std::vector<double> hidden(hidden_size);
        std::vector<double> output(num_classes);

        for (int i = 0; i < hidden_size; ++i) {
            hidden[i] = 0.0;
            for (int j = 0; j < num_features; ++j) {
                hidden[i] += weights_input_hidden[i][j] * input[j];
            }
            hidden[i] = relu(hidden[i]);
        }

        for (int i = 0; i < num_classes; ++i) {
            output[i] = 0.0;
            for (int j = 0; j < hidden_size; ++j) {
                output[i] += weights_hidden_output[i][j] * hidden[j];
            }
        }

        return output;
    }

    void train(const std::vector<std::vector<double>>& X, const std::vector<int>& y, int epochs, double learning_rate) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double epoch_loss = 0.0;
            for (size_t i = 0; i < X.size(); ++i) {
                auto output = forward(X[i]);
                double loss = binaryCrossEntropyLoss(output[0], y[i]);
                epoch_loss += loss;
                backpropagation(X[i], y[i], output, learning_rate);
            }
            std::cout << "Epoch [" << epoch + 1 << "/" << epochs << "], Loss: " << epoch_loss / X.size() << std::endl;
        }
    }

private:
    int num_features, hidden_size, num_classes;
    std::vector<std::vector<double>> weights_input_hidden;
    std::vector<std::vector<double>> weights_hidden_output;

    void initializeWeights() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-0.1, 0.1);

        for (auto& row : weights_input_hidden) {
            for (auto& weight : row) {
                weight = dis(gen);
            }
        }

        for (auto& row : weights_hidden_output) {
            for (auto& weight : row) {
                weight = dis(gen);
            }
        }
    }

    double relu(double x) {
        return std::max(0.0, x);
    }

    double sigmoid(double x) {
        return 1 / (1 + std::exp(-x));
    }

    double binaryCrossEntropyLoss(double pred, double target) {
        pred = sigmoid(pred);
        pred = std::max(1e-12, std::min(1 - 1e-12, pred));
        return -(target * std::log(pred) + (1 - target) * std::log(1 - pred));
    }

    void backpropagation(const std::vector<double>& input, int target, const std::vector<double>& output, double learning_rate) {
        double pred = sigmoid(output[0]);
        double output_error = pred - target;

        std::vector<double> hidden_errors(hidden_size);
        for (int j = 0; j < hidden_size; ++j) {
            hidden_errors[j] = output_error * weights_hidden_output[0][j] * (pred * (1 - pred));
        }

        for (int j = 0; j < hidden_size; ++j) {
            weights_hidden_output[0][j] -= learning_rate * output_error * (pred * (1 - pred));
        }

        for (int j = 0; j < hidden_size; ++j) {
            for (int k = 0; k < num_features; ++k) {
                weights_input_hidden[j][k] -= learning_rate * hidden_errors[j] * input[k];
            }
        }
    }
};

bool is_number(const std::string& s) {
    std::istringstream iss(s);
    double d;
    char c;
    return iss >> d && !(iss >> c);
}


std::pair<std::vector<std::vector<double>>, std::vector<int>> readCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }

    std::string line;
    std::vector<std::vector<double>> data;
    std::vector<int> labels;

    std::getline(file, line);
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string item;
        std::vector<double> row;

        while (std::getline(ss, item, ',')) {
            if (is_number(item)) {
                row.push_back(std::stod(item));
            } else {
                std::cerr << "Non-numeric data encountered: " << item << std::endl;
                row.push_back(0.0);
            }
        }

        if (!row.empty()) {
            labels.push_back(static_cast<int>(row.back()));
            row.pop_back();
            data.push_back(row);
        }
    }

    return {data, labels};
}


int main() {
    auto [data, labels] = readCSV("nn_data.csv");

    std::transform(labels.begin(), labels.end(), labels.begin(), [](int label) {
        return label == -1 ? 0 : 1;
    });

    NeuralNetwork nn(data[0].size(), 10, 1);
    nn.train(data, labels, 150, 0.01);

    int correct = 0;
    std::vector<int> predictions;
    for (size_t i = 0; i < data.size(); ++i) {
        auto output = nn.forward(data[i]);
        int prediction = output[0] >= 0.5 ? 1 : 0;
        predictions.push_back(prediction);
        if (prediction == labels[i]) {
            correct++;
        }
    }

    std::cout << "Accuracy: " << static_cast<double>(correct) / data.size() << std::endl;

    return 0;
}
