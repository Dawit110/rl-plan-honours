import argparse
import json
import logging
from collections import deque

from evaluate import eval_performance
from fd_env import FDEnvSelHeur


FEATURES_PER_HEURISTIC = 5
MIN_VALUE_OFFSET = 2
OPEN_ENTRIES_OFFSET = 3


class StagnationProgressAgent:
    """
    Hybrid policy:

    1. Keep using the current heuristic while its observed minimum
       heuristic value continues to reach new lows.

    2. If no new minimum is observed for `threshold` controller
       decisions, consider switching.

    3. Among the OTHER available heuristics, choose the heuristic
       with the greatest normalized progress over the most recent
       `window` observations.
    """

    def __init__(
        self,
        actions,
        threshold,
        window,
        initial_action=0,
        verbose=False
    ):
        self.actions = actions
        self.threshold = threshold
        self.window = window
        self.initial_action = initial_action
        self.verbose = verbose

        self.current_action = initial_action

        # Best minimum value observed while the current heuristic
        # has been active.
        self.best_min = None

        # Number of controller decisions since current heuristic
        # last reached a new minimum.
        self.stagnation = 0

        # Sliding h_min history for every heuristic.
        self.histories = {
            action: deque(maxlen=window + 1)
            for action in actions
        }

        self.decision = 0
        self.switches = 0
        self.episode_switch_counts = []

    @staticmethod
    def min_index(action):
        return (
            action * FEATURES_PER_HEURISTIC
            + MIN_VALUE_OFFSET
        )

    @staticmethod
    def entries_index(action):
        return (
            action * FEATURES_PER_HEURISTIC
            + OPEN_ENTRIES_OFFSET
        )

    def normalized_progress(self, action):
        """
        Return normalized progress over the current window:

             old_h_min - current_h_min
        P = ---------------------------
                  max(|old_h_min|, 1)
        """

        history = self.histories[action]

        old_min = history[0]
        current_min = history[-1]

        denominator = max(abs(old_min), 1.0)

        return (
            old_min - current_min
        ) / denominator

    def cyclic_tie_break(self, candidates):
        """
        Deterministic tie breaking.

        If several alternatives have the same progress score,
        prefer the first tied heuristic appearing after the current
        heuristic in cyclic action order.

        Example:
            current = 1
            action order = [0, 1, 2, 3]

        tie-breaking order becomes:
            2, 3, 0
        """

        current_index = self.actions.index(
            self.current_action
        )

        cyclic_order = (
            self.actions[current_index + 1:]
            + self.actions[:current_index]
        )

        for action in cyclic_order:
            if action in candidates:
                return action

        return candidates[0]

    def act(self, observation):
        self.decision += 1

        # ---------------------------------------------------------
        # 1. Record current h_min for ALL heuristics.
        # ---------------------------------------------------------
        for action in self.actions:
            h_min = float(
                observation[self.min_index(action)]
            )

            self.histories[action].append(h_min)

        # ---------------------------------------------------------
        # 2. Check whether CURRENT heuristic has improved.
        # ---------------------------------------------------------
        current_min = float(
            observation[
                self.min_index(self.current_action)
            ]
        )

        if self.best_min is None:
            # First observation establishes baseline.
            self.best_min = current_min
            self.stagnation = 0

        elif current_min < self.best_min:
            if self.verbose:
                print(
                    "[STAG+PROG4] decision {}: "
                    "h{} improved {:.3f} -> {:.3f}".format(
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

        # ---------------------------------------------------------
        # 3. If current heuristic has not stagnated, stay.
        # ---------------------------------------------------------
        if self.stagnation < self.threshold:
            return int(self.current_action)

        # ---------------------------------------------------------
        # 4. We have stagnated.
        #
        # We need a complete progress window before making an
        # informed switch.
        # ---------------------------------------------------------
        if any(
            len(self.histories[action])
            < self.window + 1
            for action in self.actions
        ):
            return int(self.current_action)

        # ---------------------------------------------------------
        # 5. Find AVAILABLE ALTERNATIVE heuristics.
        #
        # Exclude:
        #   - current heuristic
        #   - heuristics whose open list is empty
        # ---------------------------------------------------------
        candidates = [
            action
            for action in self.actions
            if action != self.current_action
            and float(
                observation[
                    self.entries_index(action)
                ]
            ) > 0
        ]

        # If no alternative open list is available,
        # remain with current heuristic.
        if not candidates:
            return int(self.current_action)

        # ---------------------------------------------------------
        # 6. Calculate normalized recent progress for each
        # alternative.
        # ---------------------------------------------------------
        scores = {
            action: self.normalized_progress(action)
            for action in candidates
        }

        best_score = max(scores.values())

        best_actions = [
            action
            for action, score in scores.items()
            if abs(score - best_score) < 1e-12
        ]

        # Deterministic tie-breaking.
        next_action = self.cyclic_tie_break(
            best_actions
        )

        # ---------------------------------------------------------
        # 7. Perform the switch.
        # ---------------------------------------------------------
        old_action = self.current_action

        self.current_action = next_action
        self.switches += 1

        # New heuristic begins with a fresh stagnation baseline.
        self.best_min = float(
            observation[
                self.min_index(self.current_action)
            ]
        )

        self.stagnation = 0

        if self.verbose:
            print(
                "[STAG+PROG4] decision {}: "
                "SWITCH h{} -> h{} | progress: {}".format(
                    self.decision,
                    old_action,
                    self.current_action,
                    ", ".join(
                        "h{}={:.4f}".format(
                            action,
                            scores[action]
                        )
                        for action in sorted(scores)
                    )
                )
            )

        return int(self.current_action)

    def stop_episode(self):
        self.episode_switch_counts.append(
            self.switches
        )

        self.current_action = self.initial_action

        self.best_min = None
        self.stagnation = 0

        self.histories = {
            action: deque(maxlen=self.window + 1)
            for action in self.actions
        }

        self.decision = 0
        self.switches = 0

        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=1
    )

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

    agent = StagnationProgressAgent(
        actions=args.actions,
        threshold=args.threshold,
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

    eval_stats["policy"] = "stagnation_progress4"
    eval_stats["threshold"] = args.threshold
    eval_stats["window"] = args.window
    eval_stats["actions"] = args.actions
    eval_stats[
        "switches_per_episode"
    ] = agent.episode_switch_counts

    for i, trajectory in enumerate(
        eval_stats["trajectories"]
    ):
        print(
            "episode {}: decisions={}, switches={}, "
            "first actions={}".format(
                i,
                len(trajectory),
                eval_stats[
                    "switches_per_episode"
                ][i],
                trajectory[:100]
            )
        )

    if args.save_eval_stats:
        filename = args.save_eval_stats

        if not filename.endswith(".json"):
            filename += ".json"

        with open(filename, "w") as outfile:
            json.dump(
                eval_stats,
                outfile
            )

        print("Saved:", filename)