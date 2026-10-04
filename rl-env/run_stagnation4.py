import argparse
import json
import logging

from evaluate import eval_performance
from fd_env import FDEnvSelHeur


FEATURES_PER_HEURISTIC = 5
MIN_VALUE_OFFSET = 2


class StagnationAgent:
    """
    Cycle through a set of heuristics when the currently selected
    heuristic fails to achieve a new minimum heuristic value for
    a fixed number of controller decisions.
    """

    def __init__(self, actions, threshold, verbose=False):
        self.actions = actions
        self.threshold = threshold
        self.verbose = verbose

        self.action_index = 0
        self.current_action = actions[0]

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

        current_min = float(
            observation[self.min_value_index(self.current_action)]
        )

        # First observation establishes the baseline.
        if self.best_min is None:
            self.best_min = current_min
            self.stagnation = 0

        # Strict decrease = progress.
        elif current_min < self.best_min:
            if self.verbose:
                print(
                    "[STAG4] decision {}: h{} improved {:.3f} -> {:.3f}".format(
                        self.decision,
                        self.current_action,
                        self.best_min,
                        current_min
                    )
                )

            self.best_min = current_min
            self.stagnation = 0

        else:
            self.stagnation += 1

        # Current heuristic has stagnated: move to next heuristic.
        if self.stagnation >= self.threshold:
            old_action = self.current_action

            self.action_index = (
                self.action_index + 1
            ) % len(self.actions)

            self.current_action = self.actions[self.action_index]

            self.switches += 1
            self.stagnation = 0

            new_min = float(
                observation[self.min_value_index(self.current_action)]
            )

            self.best_min = new_min

            if self.verbose:
                print(
                    "[STAG4] decision {}: SWITCH h{} -> h{} "
                    "(baseline min={:.3f})".format(
                        self.decision,
                        old_action,
                        self.current_action,
                        new_min
                    )
                )

        return int(self.current_action)

    def stop_episode(self):
        self.episode_switch_counts.append(self.switches)

        self.action_index = 0
        self.current_action = self.actions[0]
        self.best_min = None
        self.stagnation = 0
        self.decision = 0
        self.switches = 0

        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument(
        "--actions",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3]
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=50
    )

    parser.add_argument("--num-heuristics", type=int, default=4)
    parser.add_argument("--eval-n-runs", type=int, default=1)
    parser.add_argument("--time-step-limit", type=int, default=100000)

    parser.add_argument(
        "--port-file-id",
        type=int,
        default=None,
        dest="pfid"
    )

    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--save-eval-stats", default=None)
    parser.add_argument("--verbose-policy", action="store_true")

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
        actions=args.actions,
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

    eval_stats["policy"] = "stagnation4"
    eval_stats["threshold"] = args.threshold
    eval_stats["actions"] = args.actions
    eval_stats["switches_per_episode"] = agent.episode_switch_counts

    for i, trajectory in enumerate(eval_stats["trajectories"]):
        print(
            "episode {}: decisions={}, switches={}, first actions={}".format(
                i,
                len(trajectory),
                eval_stats["switches_per_episode"][i],
                trajectory[:80]
            )
        )

    if args.save_eval_stats:
        filename = args.save_eval_stats

        if not filename.endswith(".json"):
            filename += ".json"

        with open(filename, "w") as outfile:
            json.dump(eval_stats, outfile)

        print("Saved:", filename)
