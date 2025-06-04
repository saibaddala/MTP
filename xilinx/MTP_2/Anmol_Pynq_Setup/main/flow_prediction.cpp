#include "flow_prediction.h"
#include <algorithm>
#include <iomanip>

// Constructor initializes probabilities, count, and class labels
FlowPrediction::FlowPrediction(const std::vector<float> &initial_probabilities, const int num_classes)
    : probabilities(initial_probabilities), count(1), num_classes(num_classes)
{
    updateCurrentPredictionClass();
}

void FlowPrediction::addProbabilities(const std::vector<float> &new_probabilities)
{
    for (std::size_t i = 0; i < probabilities.size(); ++i)
    {
        probabilities[i] = (count * probabilities[i] + new_probabilities[i]) / (count + 1);
    }
    count += 1;
    updateCurrentPredictionClass();
}

std::vector<float> FlowPrediction::getProbabilities() const
{
    return probabilities;
}

int FlowPrediction::getCount() const
{
    return count;
}

int FlowPrediction::getCurrentPredictionClass() const
{
    return current_prediction_class;
}

void FlowPrediction::updateCurrentPredictionClass()
{
    // Find the index of the maximum probability
    auto max_it = std::max_element(probabilities.begin(), probabilities.end());
    size_t max_index = std::distance(probabilities.begin(), max_it);
    // Update the current prediction class
    current_prediction_class = max_index;
}

// Overloading << operator for FlowPrediction
std::ostream& operator<<(std::ostream &os, const FlowPrediction &flowPrediction)
{
    os << "Probabilities: [";
    for (size_t i = 0; i < flowPrediction.probabilities.size(); ++i)
    {
        os << std::fixed << std::setprecision(2) << flowPrediction.probabilities[i];
        if (i != flowPrediction.probabilities.size() - 1)
            os << ", ";
    }
    os << "]";
    return os;
}

// Manager class methods
FlowPredictionManager::FlowPredictionManager(const int num_classes)
    : num_classes(num_classes) {}

void FlowPredictionManager::updatePrediction(const std::string &flow_key, const std::vector<float> &new_probabilities)
{
    if (flow_predictions.find(flow_key) != flow_predictions.end())
    {
        flow_predictions.at(flow_key).addProbabilities(new_probabilities);
    }
    else
    {
        flow_predictions.emplace(flow_key, FlowPrediction(new_probabilities, num_classes));
    }
}

const std::map<std::string, FlowPrediction> &FlowPredictionManager::getPredictions() const
{
    return flow_predictions;
}

void FlowPredictionManager::deleteFlow(const std::string &flow_key)
{
    flow_predictions.erase(flow_key);
}