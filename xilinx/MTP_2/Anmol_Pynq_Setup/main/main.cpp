#include <iostream>
#include <thread>
#include <chrono>
#include <libnetfilter_queue/libnetfilter_queue.h>
#include "packet_parser.h"
#include "flow.h"
#include "predictor.h"
#include "feature_extraction.h"
#include "flow_prediction.h"
#include <arpa/inet.h>
#include <linux/netfilter.h>
#include <atomic>

const std::string MODEL_NAME = "model.tflite"; 
const int PROFILING_INTERVAL = 10;
const int WINDOW_DURATION = 2;

// Global atomic variable to keep track of parsed packets
std::atomic<int> packet_count(0);
std::vector<PacketHeaders> packets;
std::chrono::steady_clock::time_point last_profiling_time = std::chrono::steady_clock::now();
const std::vector<std::string> classes = {"Chat", "File Transfer", "Streaming", "VoIP"};
FlowPredictionManager predictionManager(4);
const bool PRINT = true;
Predictor predictor(MODEL_NAME);

std::chrono::system_clock::time_point get_timestamp(struct nfq_data *nfa)
{
    struct timeval timestamp;
    std::chrono::system_clock::time_point packet_timestamp;

    if (nfq_get_timestamp(nfa, &timestamp) == 0)
    {
        packet_timestamp = std::chrono::system_clock::from_time_t(timestamp.tv_sec) +
                           std::chrono::microseconds(timestamp.tv_usec);
    }
    else
    {
        // Fallback to current system time if timestamp isn't available
        packet_timestamp = std::chrono::system_clock::now();
        // Commenting out the log to prevent flooding
        // std::cerr << "Timestamp not available for this packet" << std::endl;
    }

    return packet_timestamp;
}


FlowManager split_into_flows()
{
    FlowManager flowManager;
    for (const auto &packet : packets)
    {
        flowManager.addPacket(packet);
    }
    packets.clear();
    return flowManager;
}

void predict(std::map<std::string, Features> features)
{
    std::map<std::string, std::vector<float>> prediction_probabilities = predictor.predict(features);

    for (const auto &entry : prediction_probabilities)
    {
        const auto &flow_key = entry.first;
        const auto &new_probabilities = entry.second;
        predictionManager.updatePrediction(flow_key, new_probabilities);
    }

    // Delete all non active flows
    std::vector<std::string> inactive_flows;
    for (const auto &entry : predictionManager.getPredictions())
    {
        const auto &flow_key = entry.first;
        if (prediction_probabilities.find(flow_key) == prediction_probabilities.end())
        {
            inactive_flows.push_back(flow_key);
        }
    }
    for (const auto &flow_key : inactive_flows)
    {
        predictionManager.deleteFlow(flow_key);
    }
}

void printPredictions()
{
    if (!PRINT)
        return;

    const auto &flow_predictions = predictionManager.getPredictions();
    for (const auto &entry : flow_predictions)
    {
        const auto &flow_key = entry.first;
        const auto &flow_prediction = entry.second;

        std::cout << "Flow Key: " << flow_key << ", Prediction: " << classes[flow_prediction.getCurrentPredictionClass()] << std::endl;
    }
    std::cout << "--------------------------" << std::endl;
}

int process_packet(struct nfq_q_handle *qh, struct nfgenmsg *nfmsg, struct nfq_data *nfa, void *data)
{
    struct nfqnl_msg_packet_hdr *ph;
    ph = nfq_get_msg_packet_hdr(nfa);
    if (ph)
    {
        uint32_t id = ntohl(ph->packet_id);
        unsigned char *packet;
        int packet_len = nfq_get_payload(nfa, &packet);

        if (packet_len >= 0)
        {
            // Extract timestamp
            std::chrono::system_clock::time_point packet_timestamp = get_timestamp(nfa);

            // Parsing packet headers
            PacketHeaders packet_headers = parse_packet(packet, packet_len, packet_timestamp);
            packets.push_back(packet_headers);
            packet_count++;

            // Checking if packets exceed window duration
            if (!packets.empty())
            {
                auto duration = std::chrono::duration_cast<std::chrono::seconds>(
                                    packet_headers.timestamp - packets.front().timestamp)
                                    .count();

                if (duration >= WINDOW_DURATION)
                {
                    // Split the packets into flows
                    FlowManager flowManager = split_into_flows();

                    // Extract features from flows
                    std::map<std::string, Features> flowFeatures = flowManager.extractFeatures();

                    // Make predictions based on the features
                    predict(flowFeatures);

                    // Print the predictions
                    printPredictions();
                }
            }

            // Profiling the number of packets obtained
            auto current_time = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(current_time - last_profiling_time).count() >= PROFILING_INTERVAL)
            {
                std::cout << "Processed " << packet_count << " packets in the last " << PROFILING_INTERVAL << " seconds." << std::endl;
                last_profiling_time = current_time;
                packet_count = 0;
            }
        }

        return nfq_set_verdict(qh, id, NF_ACCEPT, 0, NULL);
    }
    return 0;
}

void start_packet_processing(int queue_num)
{
    struct nfq_handle *handle = nfq_open();
    if (!handle)
    {
        std::cerr << "Error opening NFQ" << std::endl;
        return;
    }

    if (nfq_unbind_pf(handle, AF_INET) < 0)
    {
        std::cerr << "Error unbinding existing NFQ handler for AF_INET" << std::endl;
        return;
    }

    if (nfq_bind_pf(handle, AF_INET) < 0)
    {
        std::cerr << "Error binding NFQ handler for AF_INET" << std::endl;
        return;
    }

    struct nfq_q_handle *qh = nfq_create_queue(handle, queue_num, &process_packet, nullptr);
    if (!qh)
    {
        std::cerr << "Error creating NFQ queue" << std::endl;
        return;
    }

    if (nfq_set_mode(qh, NFQNL_COPY_PACKET, 0xffff) < 0)
    {
        std::cerr << "Error setting packet mode" << std::endl;
        return;
    }

    int fd = nfq_fd(handle);
    char buf[4096];

    while (true)
    {
        int rv = recv(fd, buf, sizeof(buf), 0);
        nfq_handle_packet(handle, buf, rv);
        // std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Sleep to avoid busy waiting
    }

    nfq_destroy_queue(qh);
    nfq_unbind_pf(handle, AF_INET);
    nfq_close(handle);
}

int main()
{
    int queue_num = 1;
    std::cout << "Starting Packet Processing..." << std::endl;
    std::thread processing_thread(start_packet_processing, queue_num);

    processing_thread.join();
    return 0;
}
