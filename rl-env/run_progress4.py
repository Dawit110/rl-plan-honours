import argparse
import json
import logging
from collections import deque

from evaluate import eval_performance
from fd_env import FDEnvSelHeur


FEATURES_PER_HEURISTIC = 5
MIN_VALUE_OFFSET = 2
OPEN_ENTRIES_OFFSET = 3


class NormalizedProgressAgent:

    def __init__(self, actions, window, initial_action=0, verbose=False):
        self.actions = actions
        self.window = window
        self.initial_action = initial_action
        self.verbose = verbose

        self.current_action = initial_action

        self.histories = {
            a: deque(maxlen=window + 1)
            for a in actions
        }

        self.decision = 0
        self.switches = 0
        self.episode_switch_counts = []

    @staticmethod
    def min_index(action):
        return action * FEATURES_PER_HEURISTIC + MIN_VALUE_OFFSET

    @staticmethod
    def entries_index(action):
        return action * FEATURES_PER_HEURISTIC + OPEN_ENTRIES_OFFSET

    def act(self, observation):
        self.decision += 1

        # Record h_min for all four heuristic open lists.
        for action in self.actions:
            h_min = float(observation[self.min_index(action)])
            self.histories[action].append(h_min)

        # Remain with initial/current heuristic until a complete
        # observation window is available.
        if any(
            len(self.histories[a]) < self.window + 1
            for a in self.actions
        ):
            return int(self.current_action)

        # Ignore currently empty open lists.
        available = [
            a for a in self.actions
            if float(observation[self.entries_index(a)]) > 0
        ]

        if not available:
            return int(self.current_action)

        scores = {}

        for action in available:
            history = self.histories[action]

            old_min = history[0]
            current_min = history[-1]

            denominator = max(abs(old_min), 1.0)

            scores[action] = (
                old_min - current_min
            ) / denominator

        best_score = max(scores.values())

        best_actions = [
            a for a, score in scores.items()
            if abs(score - best_score) < 1e-12
        ]

        # Stability-preserving tie break:
        # remain with current heuristic if it is tied for best.
        if self.current_action in best_actions:
            next_action = self.current_action
        else:
            next_action = best_actions[0]

        if next_action != self.current_action:
            old_action = self.current_action
            self.current_action = next_action
            self.switches += 1

            if self.verbose:
                print(
                    "[PROG4] decision {}: SWITCH h{} -> h{} | {}".format(
                        self.decision,
                        old_action,
                        self.current_action,
                        ", ".join(
                            "h{}={:.4f}".format(a, scores[a])
                            for a in sorted(scores)
                        )
                    )
                )

        elif self.verbose and self.decision % self.window == 0:
            print(
                "[PROG4] decision {}: stay h{} | {}".format(
                    self.decision,
                    self.current_action,
                    ", ".join(
                        "h{}={:.4f}".format(a, scores[a])
                        for a in sorted(scores)
                    )
                )
            )

        return int(self.current_action)

    def stop_episode(self):
        self.episode_switch_counts.append(self.switches)

        self.current_action = self.initial_action

        self.histories = {
            a: deque(maxlen=self.window + 1)
            for a in self.actions
        }

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
        "--window",
        type=int,
        default=50
    )

    parser.add_argument(
        "--initial-action",
        type=int,
        default=0
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

    agent = NormalizedProgressAgent(
        actions=args.actions,
        window=args.window,
        initial_action=args.initial_action,
        verbose=args.verbose_policy
    )

    eval_stats = eval_performance(
        env=env,
        agent=agent,
        n_steps=None,
        n_episodes=args.eval_n_runs,
        max_episode_len=args.time_step_limit
    )

    eval_stats["policy"] = "normalized_progress4"
    eval_stats["window"] = args.window
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
