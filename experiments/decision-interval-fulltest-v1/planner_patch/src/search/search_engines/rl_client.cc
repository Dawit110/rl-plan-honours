#include "rl_client.h"
#include <iostream>
#include <cstdio>
#include <cerrno>
#include <stdexcept>

namespace rl_client {

RLClient::RLClient(int port, std::string ip_address) : port(port), ip_address(ip_address), sock(0) {}

bool RLClient::init_connection() {

    printf("\nPort: %i\n", port);
    
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        printf("\n Socket creation error \n");
        return false;
    }

    memset(&serv_addr, 0, sizeof(serv_addr));

    #ifdef __APPLE__
        serv_addr.sin_len = sizeof(serv_addr);
    #endif

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);

    // Convert IPv4 and IPv6 addresses from text to binary form 
    if(inet_pton(AF_INET, ip_address.c_str(), &serv_addr.sin_addr)<=0)
    {
        printf("\nInvalid address/ Address not supported \n");
        return false;
    }

    sockfd = connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr));

    if (sockfd < 0)
    {
        perror("RL connection failed");
        return false;
    }
    return true;
}

void RLClient::send_msg(const std::string& msg) const {
    size_t sent = 0;
    while (sent < msg.size()) {
        ssize_t n = send(sock, msg.data() + sent, msg.size() - sent, 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { perror("RL send failed"); std::exit(70); }
        sent += static_cast<size_t>(n);
    }
}

void RLClient::send_msg(const std::map<int, std::map<std::string, double>>& open_lists_stats, double reward, bool done) const {
    std::string py_dict = "{";
    for (auto& config_values : open_lists_stats) {
        int config = config_values.first;
        py_dict += "\"" + std::to_string(config) + "\" : {";

        for (auto& value_pair : config_values.second) {
            py_dict += "\"" + value_pair.first + "\" : " + std::to_string(value_pair.second) + ",";
        }
        py_dict = py_dict.substr(0, py_dict.size() - 1);
        py_dict += "},";
    }
    py_dict += "\"reward\" : " + std::to_string(reward);
    py_dict += ",\"done\" : ";
    py_dict += done ? std::to_string((double)1) : std::to_string((double)0);
    py_dict += "}";
    std::string msg = std::to_string(py_dict.size());
    for (size_t i = 0; i < 4 - msg.size(); i++) {
        msg = "0" + msg;
    }
    msg += py_dict;
    // std::cout << msg << std::endl;
    send_msg(msg);
}

std::string RLClient::read_msg() const {
    auto exact = [this](size_t count) {
        std::string out(count, '\0');
        size_t used = 0;
        while (used < count) {
            ssize_t n = recv(sock, &out[used], count - used, 0);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) { std::cerr << "RL socket closed or failed" << std::endl; std::exit(70); }
            used += static_cast<size_t>(n);
        }
        return out;
    };
    std::string header = exact(4);
    if (header.find_first_not_of(" 0123456789") != std::string::npos) std::exit(70);
    int count = std::stoi(header);
    if (count < 1 || count > 9999) std::exit(70);
    std::string body = exact(count);
    if (body == "END") { std::cout << "Termination due to RL agent" << std::endl; std::exit(0); }
    if (body.size() != 1 || body[0] < '0' || body[0] > '3') std::exit(70);
    return header + body;
}

std::string stats_json(const decision_interval::Stats &stats) {
    std::string text = "{";
    bool first_queue = true;
    for (const auto &queue : stats) {
        if (!first_queue) text += ",";
        first_queue = false;
        text += "\"" + std::to_string(queue.first) + "\":{";
        bool first = true;
        for (const auto &pair : queue.second) {
            if (!first) text += ",";
            first = false;
            text += "\"" + pair.first + "\":" + std::to_string(pair.second);
        }
        text += "}";
    }
    return text + "}";
}

void RLClient::send_interval(const decision_interval::Stats &current,
    const decision_interval::Stats &previous, const decision_interval::Control &control,
    long expanded, bool done, bool failed) const {
    std::string data = stats_json(current);
    data.pop_back();
    data += ",\"interval_previous\":" + stats_json(previous);
    data += ",\"interval_meta\":{\"expanded\":" + std::to_string(expanded);
    data += ",\"removals\":" + std::to_string(control.removals);
    data += ",\"decisions\":" + std::to_string(control.decisions);
    data += ",\"active_block_expansions\":" + std::to_string(expanded - control.start_expanded);
    data += ",\"reason\":\"" + control.reason + "\"";
    data += ",\"completed_interval_histogram\":{";
    bool first = true;
    for (const auto &h : control.histogram) {
        if (!first) data += ",";
        first = false;
        data += "\"" + std::to_string(h.first) + "\":" + std::to_string(h.second);
    }
    data += "}}";
    data += ",\"reward\":-1,\"done\":" + std::to_string(done ? 1 : 0);
    data += ",\"search_failed\":" + std::to_string(failed ? 1 : 0) + "}";
    if (data.size() > 9999) throw std::runtime_error("RL observation exceeds 4-digit framing");
    std::string length = std::to_string(data.size());
    send_msg(std::string(4-length.size(), '0') + length + data);
}

void RLClient::closeConn() const {
    shutdown(sock, 2);
    close(sock);
    shutdown(sockfd, 2);
    close(sockfd);
}

}
