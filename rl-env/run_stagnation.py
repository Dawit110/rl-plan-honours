import argparse
import json
import logging

from evaluate import eval_performance
from fd_env import FDEnvSelHeur


# The Fast Downward environment supplies five values per heuristic:
# 0 = Average Value
# 1 = Max Value
# 2 = Min Value
# 3 = Open List Entries
# 4 = Variance
FEATURES_PER_HEURISTIC = 5
MIN_VALUE_OFFSET = 2


class StagnationAgent:
    """
    Switch between two heuristics when the currently selected heuristic
    fails to achieve a new minimum heuristic value for a fixed number
    of controller decisions.
    """

    def __init__(self, first_action, second_action, threshold, verbose=False):
        self.first_action = first_action
        self.second_action = second_action
        self.threshold = threshold
        self.verbose = verbose

        self.current_action = first_action
        self.best_min = None
        self.stagnation = 0
        self.decision = 0
        self.switches = 0

        self.episode_switch_counts = []

    @staticmethod
    def min_value_index(action):
        return action * FEATURES_PER_HEURISTIC + MIN_VALUE_OFFSET

    def act(self, observation):
        self.decision += 1

        current_index = self.min_value_index(self.current_action)
        current_min = float(observation[current_index])

        # First observation after starting/switching establishes the baseline.
        if self.best_min is None:
            self.best_min = current_min
            self.stagnation = 0

        # A strict decrease is treated as search progress.
        elif current_min < self.best_min:
            if self.verbose:
                print(
                    "[STAG] decision {}: h{} improved {:.3f} -> {:.3f}".format(
                        self.decision,
                        self.current_action,
                        self.best_min,
                        current_min
                    )
                )

            self.best_min = current_min
            self.stagnation = 0

        # Otherwise the currently selected heuristic has not improved.
        else:
            self.stagnation += 1

        # Switch once the stagnation threshold has been reached.
        if self.stagnation >= self.threshold:
            old_action = self.current_action

            if self.current_action == self.first_action:
                self.current_action = self.second_action
            else:
                self.current_action = self.first_action

            self.switches += 1
            self.stagnation = 0

            # Establish a fresh baseline for the newly selected heuristic
            # using its value in the current observation.
            new_index = self.min_value_index(self.current_action)
            self.best_min = float(observation[new_index])

            if self.verbose:
                print(
                    "[STAG] decision {}: SWITCH h{} -> h{} "
                    "(new baseline min={:.3f})".format(
                        self.decision,
                        old_action,
                        self.current_action,
                        self.best_min
                    )
                )

        return int(self.current_action)

    def stop_episode(self):
        self.episode_switch_counts.append(self.switches)

        # Reset policy state before another planning instance.
        self.current_action = self.first_action
        self.best_min = None
        self.stagnation = 0
        self.decision = 0
        self.switches = 0

        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument(
        "--first-action",
        type=int,
        default=0,
        help="First heuristic/open-list action (0 = FF)"
    )

    parser.add_argument(
        "--second-action",
        type=int,
        default=1,
        help="Second heuristic/open-list action (1 = CG)"
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=50,
        help="Number of non-improving controller decisions before switching"
    )

    parser.add_argument(
        "--num-heuristics",
        type=int,
        default=4
    )

    parser.add_argument(
        "--eval-n-runs",
        type=int,
        default=1
    )

    parser.add_argument(
        "--time-step-limit",
        type=int,
        default=100000
    )

    parser.add_argument(
        "--port-file-id",
        type=int,
        default=None,
        dest="pfid"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=12345
    )

    parser.add_argument(
        "--save-eval-stats",
        default=None
    )

    parser.add_argument(
        "--verbose-policy",
        action="store_true"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    env = FDEnvSelHeur(
        host="",
        port=args.port,
        num_heuristics=args.num_heuristics,
        config_dir=".",
        port_file_id=args.pfid,
        use_general_state_info=False,
        time_step_limit=args.time_step_limit
    )

    env.seed(args.seed)

    agent = StagnationAgent(
        first_action=args.first_action,
        second_action=args.second_action,
        threshold=args.threshold,
        verbose=args.verbose_policy
    )

    eval_stats = eval_performance(
        env=env,
        agent=agent,
        n_steps=None,
        n_episodes=args.eval_n_runs,
        max_episode_len=args.time_step_limit
    )

    # Add policy-specific information to the result.
    eval_stats["policy"] = "stagnation"
    eval_stats["threshold"] = args.threshold
    eval_stats["actions"] = [
        args.first_action,
        args.second_action
    ]
    eval_stats["switches_per_episode"] = agent.episode_switch_counts

    print(
        "n_runs: {} mean: {} median: {} stdev: {}".format(
            args.eval_n_runs,
            eval_stats["mean"],
            eval_stats["median"],
            eval_stats["stdev"]
        )
    )

    for i, trajectory in enumerate(eval_stats["trajectories"]):
        print(
            "episode {}: decisions={}, switches={}, first actions={}".format(
                i,
                len(trajectory),
                eval_stats["switches_per_episode"][i],
                trajectory[:40]
            )
        )

    if args.save_eval_stats:
        filename = args.save_eval_stats
        if not filename.endswith(".json"):
            filename += ".json"

        with open(filename, "w") as outfile:
            json.dump(eval_stats, outfile)

        print("Saved:", filename)
