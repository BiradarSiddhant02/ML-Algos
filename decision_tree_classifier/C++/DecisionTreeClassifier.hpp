#include <string>
#include <vector>

#include "DataTable.hpp"

class Node{
public:
    unsigned int featureIndex;
    double threshold;
    Node* left;
    Node* right;
    double infoGain;
    double value;

    Node(const unsigned int featureValue, const double threshold, Node* left, Node* right, const double infoGain, const double value);
    Node(const unsigned int featureValue, const double threshold, Node* left, Node* right, const double infoGain);
    Node(const double value);

private:
};

class DecisionTreeClassifier{
public:
    Node root;
    unsigned int min_samples_split;
    unsigned int max_depth;

    DecisionTreeClassifier(const Node& root, const unsigned int min_samples_split, const unsigned int max_depth);

private:
    Node buildTree(const DataTable& DataTable, unsigned int currDepth);
    BestSplit getBestSplit(const DataTable& dataset, const unsigned int num_samples, unsigned int num_features);
    std::pair<DataTable, DataTable> split(const DataTable& dataset, const unsigned int featureIndex, const double threshold);
    double informationGain(const DataTable& parent, const DataTable& leftChild, const DataTable& rightChild, const std::string mode = "entropy");
    double calculateLeafValue(const DataTable& Y);
};