#include "feature_extraction.h"
#include <numeric>
#include <cmath>
#include <algorithm>
#include <iostream>

Features extract_features(const std::vector<std::size_t> &timestamps, const std::vector<int> &packet_sizes)
{
    Features features;
    float epsilon = 1e-6;
    float total_bytes = std::accumulate(packet_sizes.begin(), packet_sizes.end(), 0);

    // Calculate inter-arrival times (IATs)
    std::vector<float> iats;
    for (std::size_t i = 1; i < timestamps.size(); ++i)
    {
        iats.push_back(timestamps[i] - timestamps[i - 1]);
    }

    if (!iats.empty())
    {
        features.min_iat = *std::min_element(iats.begin(), iats.end());
        features.max_iat = *std::max_element(iats.begin(), iats.end());
        features.mean_iat = std::accumulate(iats.begin(), iats.end(), 0.0) / iats.size();
        features.std_iat = std::sqrt(std::accumulate(iats.begin(), iats.end(), 0.0,
                                                     [mean = features.mean_iat](float sum, float iat)
                                                     {
                                                         return sum + (iat - mean) * (iat - mean);
                                                     }) /
                                     (iats.size() - 1));
    }
    else
    {
        features.min_iat = features.max_iat = features.mean_iat = features.std_iat = 0.0;
    }

    float duration = timestamps.back() - timestamps.front() + epsilon; // Avoid division by zero
    features.flowPktsPerSecond = timestamps.size() / duration;
    features.flowBytesPerSecond = total_bytes / duration;

    return features;
}
