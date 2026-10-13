#ifndef SEARCH_ENGINES_RL_CLIENT_H
#define SEARCH_ENGINES_RL_CLIENT_H

#include <stdio.h>
#include <sys/socket.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <string.h>
#include <unistd.h>
#include <string>
#include <map>
#include "interval_control.h"

namespace rl_client {
std::string stats_json(const decision_interval::Stats &stats);

class RLClient {
protected:
    int port;
    std::string ip_address;
    int sock;
    int sockfd;
    struct sockaddr_in address;
    struct sockaddr_in serv_addr;

public:
    RLClient(int port, std::string ip_address);

    bool init_connection();

    void send_msg(const std::string& msg) const;

    void closeConn() const;

    void send_msg(const std::map<int, std::map<std::string, double>>& open_lists_stats, double reward, bool done) const;

    void send_interval(const decision_interval::Stats &current,
        const decision_interval::Stats &previous, const decision_interval::Control &control,
        long expanded, bool done, bool failed) const;
    std::string read_msg() const;
};

}


#endif
