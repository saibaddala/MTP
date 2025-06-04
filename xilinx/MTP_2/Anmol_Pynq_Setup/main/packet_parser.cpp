#include "packet_parser.h"
#include "flow.h"
#include <iostream>
#include <netinet/ip.h>  // For IPv4 headers
#include <netinet/ip6.h> // For IPv6 headers
#include <netinet/tcp.h> // For TCP headers
#include <netinet/udp.h> // For UDP headers
#include <arpa/inet.h>
#include <atomic>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <sys/time.h>

PacketHeaders parse_packet(const unsigned char *packet, int len, std::chrono::system_clock::time_point timestamp)
{
    uint8_t version = packet[0] >> 4; // Get the IP version (first 4 bits of the first byte)
    std::string flow_key;             // Define a unique key for the flow
    uint16_t src_port = 0;            // Initialize source port
    uint16_t dst_port = 0;            // Initialize destination port
    std::string src_ip;               // Variable to hold source IP
    std::string dst_ip;               // Variable to hold destination IP
    std::string protocol;             // Variable to hold protocol type (TCP/UDP)

    if (version == 4) // IPv4
    {
        struct ip *ip_hdr = (struct ip *)(packet);
        struct sockaddr_in src, dst;
        src.sin_addr = ip_hdr->ip_src;
        dst.sin_addr = ip_hdr->ip_dst;

        // Convert IP addresses to string format
        src_ip = inet_ntoa(src.sin_addr);
        dst_ip = inet_ntoa(dst.sin_addr);
        protocol = "TCP"; // Default to TCP

        if (ip_hdr->ip_p == IPPROTO_TCP)
        {
            struct tcphdr *tcp_hdr = (struct tcphdr *)(packet + sizeof(struct ip));
            src_port = ntohs(tcp_hdr->th_sport);
            dst_port = ntohs(tcp_hdr->th_dport);
        }
        else if (ip_hdr->ip_p == IPPROTO_UDP)
        {
            struct udphdr *udp_hdr = (struct udphdr *)(packet + sizeof(struct ip));
            src_port = ntohs(udp_hdr->uh_sport);
            dst_port = ntohs(udp_hdr->uh_dport);
            protocol = "UDP";
        }
        else
        {
            return PacketHeaders{};
        }
    }
    else if (version == 6) // IPv6
    {
        struct ip6_hdr *ip6_hdr = (struct ip6_hdr *)(packet);
        char src[INET6_ADDRSTRLEN], dst[INET6_ADDRSTRLEN];
        inet_ntop(AF_INET6, &(ip6_hdr->ip6_src), src, INET6_ADDRSTRLEN);
        inet_ntop(AF_INET6, &(ip6_hdr->ip6_dst), dst, INET6_ADDRSTRLEN);

        // Default to TCP
        src_ip = std::string(src);
        dst_ip = std::string(dst);
        protocol = "TCP"; // Default to TCP

        if (ip6_hdr->ip6_nxt == IPPROTO_TCP)
        {
            struct tcphdr *tcp_hdr = (struct tcphdr *)(packet + sizeof(struct ip6_hdr));
            src_port = ntohs(tcp_hdr->th_sport);
            dst_port = ntohs(tcp_hdr->th_dport);
        }
        else if (ip6_hdr->ip6_nxt == IPPROTO_UDP)
        {
            struct udphdr *udp_hdr = (struct udphdr *)(packet + sizeof(struct ip6_hdr));
            src_port = ntohs(udp_hdr->uh_sport);
            dst_port = ntohs(udp_hdr->uh_dport);
            protocol = "UDP";
        }
        else
        {
            return PacketHeaders{};
        }
    }
    else
    {
        std::cerr << "Unknown IP version" << std::endl;
        return PacketHeaders{};
    }

    PacketHeaders packet_headers = {src_ip, dst_ip, src_port, dst_port, protocol, len, timestamp};
    return packet_headers;
}