import argparse
import json
import logging
from collections import deque

from evaluate import eval_performance
from fd_env import FDEnvSelHeur


FEATURES_PER_HEURISTIC = 5
MIN_VALUE_OFFSET = 2
OPEN_ENTRIES_OFFSET = 3


class RecentProgressAgent:
    """
    Select the heuristic whose minimum open-list heuristic value
    has decreased the most over a sliding observation window.
    """

    def __init__(self, actions, window, initial_action=0, verbose=False):
        self.actions = actions
        self.window = window
        self.initial_action = initial_action
        self.verbose = verbose

        self.current_action = initial_action
        self.histories = {
            action: deque(maxlen=window + 1)
            for action in actions
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

        # Record current minimum heuristic value for every candidate heuristic.
        for action in self.actions:
            h_min = float(observation[self.min_index(action)])
            self.histories[action].append(h_min)

        # During the warm-up period we do not yet have a full W-step window.
        if any(len(self.histories[a]) < self.window + 1 for a in self.actions):
            return int(self.current_action)

        # Do not deliberately select an empty open list.
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

            # Positive means the minimum heuristic value decreased.
            scores[action] = old_min - current_min

        best_score = max(scores.values())

        best_actions = [
            a for a, score in scores.items()
            if score == best_score
        ]

        # On ties, retain the current heuristic when possible.
        if self.current_action in best_actions:
            next_action = self.current_action
        else:
            next_action = best_actions[0]

        if next_action != self.current_action:
            old_action = self.current_action
            self.current_action = next_action
            self.switches += 1

            if self.verbose:
                score_text = ", ".join(
                    "h{}={:.3f}".format(a, scores[a])
                    for a in sorted(scores)
                )

                print(
                    "[PROG] decision {}: SWITCH h{} -> h{} "
                    "| progress: {}".format(
                        self.decision,
                        old_action,
                        self.current_action,
                        score_text
                    )
                )

        elif self.verbose and self.decision % self.window == 0:
            score_text = ", ".join(
                "h{}={:.3f}".format(a, scores[a])
                for a in sorted(scores)
            )

            print(
                "[PROG] decision {}: stay h{} | progress: {}".format(
                    self.decision,
                    self.current_action,
                    score_text
                )
            )

        return int(self.current_action)

    def stop_episode(self):
        self.episode_switch_counts.append(self.switches)

        self.current_action = self.initial_action
        self.histories = {
            action: deque(maxlen=self.window + 1)
            for action in self.actions
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
        default=[0, 1],
        help="Heuristic actions to compare; default: 0=FF, 1=CG"
    )

    parser.add_argument(
        "--window",
        type=int,
        default=50,
        help="Number of controller steps over which progress is measured"
    )

    parser.add_argument(
        "--initial-action",
        type=int,
        default=0
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

    agent = RecentProgressAgent(
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

    eval_stats["policy"] = "recent_progress"
    eval_stats["window"] = args.window
    eval_stats["actions"] = args.actions
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
                trajectory[:60]
            )
        )

    if args.save_eval_stats:
        filename = args.save_eval_stats

        if not filename.endswith(".json"):
            filename += ".json"

        with open(filename, "w") as outfile:
            json.dump(eval_stats, outfile)

        print("Saved:", filename)
