#ifndef FLOW_PREDICTION_H
#define FLOW_PREDICTION_H

#include <vector>
#include <string>
#include <map>

class FlowPrediction
{
public:
    FlowPrediction(const std::vector<float> &initial_probabilities, const int num_classes);
    void addProbabilities(const std::vector<float> &new_probabilities);
    std::vector<float> getProbabilities() const;
    int getCount() const;
    int getCurrentPredictionClass() const;
    friend std::ostream& operator<<(std::ostream &os, const FlowPrediction &flowPrediction);

private:
    void updateCurrentPredictionClass();

    std::vector<float> probabilities;
    int count;
    std::size_t current_prediction_class;
    int num_classes;
};

class FlowPredictionManager
{
public:
    FlowPredictionManager(const int num_classes);
    void updatePrediction(const std::string &flow_key, const std::vector<float> &new_probabilities);
    const std::map<std::string, FlowPrediction> &getPredictions() const;
    void deleteFlow(const std::string &flow_key);

private:
    std::map<std::string, FlowPrediction> flow_predictions;
    int num_classes;
};

#endif // FLOW_PREDICTION_H
