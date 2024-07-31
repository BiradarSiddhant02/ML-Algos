#include <iostream>
#include <vector>
#include <string>
#include <limits>
#include "DataTable.hpp"
#include "DecisionTreeClassifier.hpp"

Node::Node(const unsigned int featureValue, const double threshold, Node* left, 
           Node* right, const double infoGain, const double value) :
            featureIndex(featureIndex), threshold(threshold), left(left),
            right(right), infoGain{infoGain}, value(value) {}

Node::Node(const unsigned int featureValue, const double threshold, Node* left, 
           Node* right, const double infoGain) :
            featureIndex(featureIndex), threshold(threshold), left(left),
            right(right), infoGain{infoGain}, value(NULL) {}

Node::Node(const double value) : featureIndex(0), threshold(0.), left(nullptr), right(nullptr),
                                 infoGain(0.), value(value){}

DecisionTreeClassifier::DecisionTreeClassifier(const Node& root, 
                                               const unsigned int min_samples_split, 
                                               const unsigned int max_depth) :
                                               root(root), min_samples_split(min_samples_split), 
                                               max_depth(max_depth) {}

Node DecisionTreeClassifier::buildTree(const DataTable& dataset, unsigned int currDepth) {
    DataTable X, Y;
    
    for(int col = 0; col < dataset.columnCount - 1; col++) {
        Column current = dataset.columns[col];
        X.addColumn(current.name, current.dataPoints);
    }

    Y.addColumn(dataset.columns.back().name, dataset.columns.back().dataPoints);

    unsigned int num_samples = dataset.rows;
    unsigned int num_features = dataset.columnCount;

    if(num_samples >= min_samples_split & currDepth <= max_depth) {
        BestSplit bestSplit = getBestSplit(dataset, num_samples, num_features);
        if(bestSplit.infoGain > 0){
            Node left_subTree = buildTree(bestSplit.datasetLeft, currDepth++);
            Node right_subTree = buildTree(bestSplit.datasetRight, currDepth++);
            Node resultNode(
                bestSplit.featureIndex, 
                bestSplit.threshold,
                &left_subTree, &right_subTree,
                bestSplit.infoGain
                );
            return resultNode;
        }
    }
    double leafValue = calculateLeafValue(Y);
    Node resultNode(leafValue);
    return resultNode;
}

BestSplit DecisionTreeClassifier::getBestSplit(const DataTable& dataset, const unsigned int num_samples, unsigned int num_features){
    BestSplit bestSplit;
    double maxInfoGain = -std::numeric_limits<double>::max();
    for(unsigned int featureIndex = 0; featureIndex < num_features; featureIndex++) {
        std::vector<double> featureValues = dataset.getRow(featureIndex);
        for(const double& threshold : featureValues) {
            std::pair<DataTable, DataTable> splitDataset = split(dataset, featureIndex, threshold);
            DataTable datasetLeft = splitDataset.first;
            DataTable datasetRight = splitDataset.second;
            if(splitDataset.first.rows > 0 & splitDataset.second.rows > 0) {
                DataTable y(dataset.columnNames.back(), dataset.columns.back().dataPoints);
                DataTable lefty(datasetLeft.columnNames.back(), datasetLeft.columns.back().dataPoints);
                DataTable righty(datasetRight.columnNames.back(), datasetRight.columns.back().dataPoints);
                double currInfoGain = informationGain(y, lefty, righty);
                if(currInfoGain > maxInfoGain) {
                    bestSplit.featureIndex = featureIndex;
                    bestSplit.threshold = threshold;
                    bestSplit.datasetLeft = datasetLeft;
                    bestSplit.datasetRight = datasetRight;
                    bestSplit.infoGain = currInfoGain;
                    maxInfoGain = currInfoGain;
                }
            }
        }
    }
    return bestSplit;
}

std::pair<DataTable, DataTable> DecisionTreeClassifier::split(const DataTable& dataset, 
                                                              const unsigned int featureIndex, 
                                                              const double threshold)
{

}

