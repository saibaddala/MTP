#ifndef FEATURE_EXTRACTION_H
#define FEATURE_EXTRACTION_H

#include <vector>

struct Features
{
    float min_iat;
    float max_iat;
    float mean_iat;
    float std_iat;
    float flowPktsPerSecond;
    float flowBytesPerSecond;
};

// Feature extraction function
Features extract_features(const std::vector<std::size_t> &timestamps, const std::vector<int> &packet_sizes);

// 🆕 Add this line to enable printing features from any file
void print_features(const Features &f);

#endif // FEATURE_EXTRACTION_H

