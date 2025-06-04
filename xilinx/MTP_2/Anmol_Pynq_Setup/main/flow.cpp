#include "flow.h"
#include <numeric>
#include <cmath>

// Constructor initializes window duration and start time
FlowManager::FlowManager()
{
}

// Generate a unique key based on src/dst IP, ports, and protocol
std::string FlowManager::createFlowKey(const PacketHeaders &packet)
{
    return packet.src_ip + "-" + packet.dst_ip + "-" + std::to_string(packet.src_port) + "-" +
           std::to_string(packet.dst_port) + "-" + packet.protocol;
}

// Add packet to the appropriate flow based on its key
void FlowManager::addPacket(const PacketHeaders &packet)
{
    std::string flowKey = createFlowKey(packet);
    std::size_t timestamp = std::chrono::duration_cast<std::chrono::microseconds>(packet.timestamp.time_since_epoch()).count();
    int packet_size = packet.length;

    flows[flowKey].timestamps.push_back(timestamp);
    flows[flowKey].packet_sizes.push_back(packet_size);
}

std::map<std::string, Features> FlowManager::extractFeatures()
{
    std::vector<float> min_iat_vals, max_iat_vals, mean_iat_vals, std_iat_vals, pkts_per_second_vals, bytes_per_second_vals;

    // First pass: extract features and store in a temporary map
    std::map<std::string, Features> raw_features;
    for (const auto &[flow_key, flow] : getFlows())
    {
        Features extracted_features = extract_features(flow.timestamps, flow.packet_sizes);
        raw_features[flow_key] = extracted_features;

        // Collect feature values for standardization
        min_iat_vals.push_back(extracted_features.min_iat);
        max_iat_vals.push_back(extracted_features.max_iat);
        mean_iat_vals.push_back(extracted_features.mean_iat);
        std_iat_vals.push_back(extracted_features.std_iat);
        pkts_per_second_vals.push_back(extracted_features.flowPktsPerSecond);
        bytes_per_second_vals.push_back(extracted_features.flowBytesPerSecond);
    }

    // Compute mean and standard deviation for each feature
    auto compute_mean_std = [](const std::vector<float> &values) -> std::pair<float, float> {
        float mean = std::accumulate(values.begin(), values.end(), 0.0f) / values.size();
        float variance = std::accumulate(values.begin(), values.end(), 0.0f, 
                         [mean](float acc, float val) { return acc + (val - mean) * (val - mean); }) / values.size();
        return std::make_pair(mean, std::sqrt(variance));
    };

    auto [mean_min_iat, std_min_iat] = compute_mean_std(min_iat_vals);
    auto [mean_max_iat, std_max_iat] = compute_mean_std(max_iat_vals);
    auto [mean_mean_iat, std_mean_iat] = compute_mean_std(mean_iat_vals);
    auto [mean_std_iat, std_std_iat] = compute_mean_std(std_iat_vals);
    auto [mean_pkts_per_sec, std_pkts_per_sec] = compute_mean_std(pkts_per_second_vals);
    auto [mean_bytes_per_sec, std_bytes_per_sec] = compute_mean_std(bytes_per_second_vals);

    // Handle the case of a single feature value by checking std deviation
    for (auto &[flow_key, features] : raw_features)
    {
        features.min_iat = std_min_iat == 0.0f ? 0.0f : (features.min_iat - mean_min_iat) / std_min_iat;
        features.max_iat = std_max_iat == 0.0f ? 0.0f : (features.max_iat - mean_max_iat) / std_max_iat;
        features.mean_iat = std_mean_iat == 0.0f ? 0.0f : (features.mean_iat - mean_mean_iat) / std_mean_iat;
        features.std_iat = std_std_iat == 0.0f ? 0.0f : (features.std_iat - mean_std_iat) / std_std_iat;
        features.flowPktsPerSecond = std_pkts_per_sec == 0.0f ? 0.0f : (features.flowPktsPerSecond - mean_pkts_per_sec) / std_pkts_per_sec;
        features.flowBytesPerSecond = std_bytes_per_sec == 0.0f ? 0.0f : (features.flowBytesPerSecond - mean_bytes_per_sec) / std_bytes_per_sec;
    }

    return raw_features;
}


// Retrieve current flows
std::map<std::string, Flow> FlowManager::getFlows()
{
    return flows;
}

std::map<std::string, Features> FlowManager::getAllFlowFeatures()
{
    return features;
}

// Clear flow data
void FlowManager::clearFlows()
{
    flows.clear();
}
