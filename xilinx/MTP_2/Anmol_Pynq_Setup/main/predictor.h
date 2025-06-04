// predictor.h

#ifndef PREDICTOR_H
#define PREDICTOR_H

#include <string>
#include <vector>
#include <map>
#include <c_api.h>
#include "feature_extraction.h"

class Predictor
{
public:
    Predictor(const std::string &model_path);
    ~Predictor();
    std::map<std::string, std::vector<float>> predict(const std::map<std::string, Features> &flow_features);

private:
    TfLiteModel* model;
    TfLiteInterpreter* interpreter;
    TfLiteInterpreterOptions* options;
    const int num_classes = 4;

    void loadModel(const std::string &model_path);
    std::vector<float> extractFeatureVector(const Features &features);
};

#endif // PREDICTOR_H
