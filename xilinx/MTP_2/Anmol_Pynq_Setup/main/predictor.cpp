// predictor.cpp

#include "predictor.h"
#include <iostream>
#include <tensorflow/lite/c/c_api.h>
#include <stdexcept>
#include <thread>

Predictor::Predictor(const std::string &model_path) : model(nullptr), interpreter(nullptr), options(nullptr)
{
    loadModel(model_path);
}

Predictor::~Predictor()
{
    if (interpreter) TfLiteInterpreterDelete(interpreter);
    if (model) TfLiteModelDelete(model);
    if (options) TfLiteInterpreterOptionsDelete(options);
}

void Predictor::loadModel(const std::string &model_path)
{
    model = TfLiteModelCreateFromFile(model_path.c_str());
    if (!model)
    {
        throw std::runtime_error("Failed to load model.");
    }

    
    // Get the number of hardware threads (CPU cores) available
    unsigned int num_threads = std::thread::hardware_concurrency();

    // If the system cannot determine the number of hardware threads, default to 1
    if (num_threads == 0) {
        num_threads = 1;
    }

    options = TfLiteInterpreterOptionsCreate();

    TfLiteInterpreterOptionsSetNumThreads(options, num_threads);

    interpreter = TfLiteInterpreterCreate(model, options);
    if (!interpreter)
    {
        throw std::runtime_error("Failed to create interpreter.");
    }

    if (TfLiteInterpreterAllocateTensors(interpreter) != kTfLiteOk)
    {
        throw std::runtime_error("Failed to allocate tensors.");
    }
}

std::vector<float> Predictor::extractFeatureVector(const Features &features)
{
    return {features.min_iat, features.max_iat, features.mean_iat, features.std_iat,
            features.flowPktsPerSecond, features.flowBytesPerSecond};
}

std::map<std::string, std::vector<float>> Predictor::predict(const std::map<std::string, Features> &flow_features)
{
    std::map<std::string, std::vector<float>> prediction_probabilities;

    for (const auto &flow : flow_features)
    {
        const auto &flow_key = flow.first;
        const auto &features = flow.second;

        // Prepare the feature vector for the model input
        std::vector<float> feature_vector = extractFeatureVector(features);
        TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);

        if (TfLiteTensorCopyFromBuffer(input_tensor, feature_vector.data(), feature_vector.size() * sizeof(float)) != kTfLiteOk)
        {
            std::cerr << "Failed to copy data to input tensor for flow: " << flow_key << std::endl;
            continue;
        }

        // Run inference
        if (TfLiteInterpreterInvoke(interpreter) != kTfLiteOk)
        {
            std::cerr << "Error during inference for flow: " << flow_key << std::endl;
            continue;
        }

        // Get the model output (probabilities for each class)
        const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
        std::vector<float> output_data(num_classes);
        if (TfLiteTensorCopyToBuffer(output_tensor, output_data.data(), num_classes * sizeof(float)) != kTfLiteOk)
        {
            std::cerr << "Failed to copy output tensor for flow: " << flow_key << std::endl;
            continue;
        }

        prediction_probabilities[flow_key] = output_data;
    }

    return prediction_probabilities;
}
