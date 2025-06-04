#include <iostream>
#include <pcap.h>
#include <cstring>
#include <chrono>
#include "packet_parser.h"
#include "flow.h"
#include "feature_extraction.h"

int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <pcap file>" << std::endl;
        return 1;
    }

    char errbuf[PCAP_ERRBUF_SIZE];
    pcap_t *handle = pcap_open_offline(argv[1], errbuf);
    if (!handle) {
        std::cerr << "Failed to open pcap file: " << errbuf << std::endl;
        return 1;
    }

    std::vector<PacketHeaders> packets;
    struct pcap_pkthdr *header;
    const u_char *data;
    int res;

    while ((res = pcap_next_ex(handle, &header, &data)) >= 0) {
        if (res == 0) continue;  // Timeout
        auto timestamp = std::chrono::system_clock::from_time_t(header->ts.tv_sec)
                       + std::chrono::microseconds(header->ts.tv_usec);

        PacketHeaders parsed = parse_packet(data + 14, header->len - 14, timestamp); // skip Ethernet header (14 bytes)
        if (!parsed.src_ip.empty()) {
            packets.push_back(parsed);
            print_packet_headers(parsed); // Optional debug
        }
    }

    pcap_close(handle);

    // Flow processing
    FlowManager flowManager;
    for (const auto &pkt : packets)
        flowManager.addPacket(pkt);

    auto flowFeatures = flowManager.extractFeatures();
    for (const auto &[key, features] : flowFeatures) {
        std::cout << "\nFlow Key: " << key << std::endl;
        print_features(features);
    }

    std::cout << "✅ Finished parsing and feature extraction from PCAP.\n";
    return 0;
}
