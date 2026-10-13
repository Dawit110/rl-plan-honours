#ifndef RL_INTERVAL_CONTROL_H
#define RL_INTERVAL_CONTROL_H
#include <map>
#include <string>
#include <cmath>
#include <stdexcept>

namespace decision_interval {
using Stats = std::map<int, std::map<std::string, double>>;
// Called before EVERY removal, including removals of already-closed entries.
// Only actual expansions advance the duration and patience counters.
class Control {
public:
    int mode, fixed, minimum, patience, maximum;
    int action = -1;
    long decisions = 0, removals = 0, start_expanded = 0, last_expanded = 0;
    long stale = 0;
    std::map<int, double> best;
    std::map<long, long> histogram;
    std::string reason = "initial";
    Control(int mode, int fixed, int minimum, int patience, int maximum)
        : mode(mode), fixed(fixed), minimum(minimum), patience(patience), maximum(maximum) {
        if (mode < 0 || mode > 2 || fixed < 1 || minimum < 1 || patience < 1 || maximum < minimum)
            throw std::invalid_argument("Invalid interval constants");
    }
    bool due(const Stats &stats, long expanded) {
        bool improved = false;
        for (const auto &queue : stats) {
            const auto &s = queue.second;
            const double value = s.at("Min Value");
            if (s.at("Open List Entries") > 0 && std::isfinite(value)) {
                auto old = best.find(queue.first);
                if (old == best.end() || value < old->second) {
                    best[queue.first] = value;
                    if (queue.first == action) improved = true;
                }
            }
        }
        stale = improved ? 0 : stale + expanded - last_expanded;
        last_expanded = expanded;
        const long age = expanded - start_expanded;
        if (decisions == 0) reason = "initial";
        else if (decisions == 1) reason = "warmup";
        else if (mode == 0) reason = "every_removal";
        else if (stats.at(action).at("Open List Entries") == 0) reason = "empty";
        else if (mode == 1 && age >= fixed) reason = "fixed";
        else if (mode == 2 && age >= maximum) reason = "maximum";
        else if (mode == 2 && age >= minimum && stale >= patience) reason = "stagnation";
        else return false;
        return true;
    }
    void choose(int selected, long expanded) {
        if (selected < 0 || selected > 3) throw std::invalid_argument("Invalid action");
        if (decisions > 0) ++histogram[expanded - start_expanded];
        action = selected;
        start_expanded = expanded;
        stale = 0;
        ++decisions;
    }
};
}
#endif
