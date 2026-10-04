import argparse
import json
import logging
from collections import deque

from evaluate import eval_performance
from fd_env import FDEnvSelHeur


# Fast Downward supplies five values per heuristic:
#
# 0 = Average Value
# 1 = Max Value
# 2 = Min Value
# 3 = Open List Entries
# 4 = Variance

FEATURES_PER_HEURISTIC = 5
MIN_VALUE_OFFSET = 2
OPEN_ENTRIES_OFFSET = 3


class ProgressQueueAgent:
    """
    Approach 4: Progress and Queue-Pressure Selection.

    For every heuristic, maintain a recent observation window.

    The controller computes:

        progress =
            (old_min - current_min)
            / max(abs(old_min), 1)

    and:

        queue_pressure =
            max(current_queue - old_queue, 0)
            / max(current_queue, 1)

    The final score is:

        score = progress - alpha * queue_pressure

    The available heuristic with the greatest score is selected.

    Positive heuristic progress is desirable.
    Positive queue growth is penalised.

    Queue shrinkage is not explicitly rewarded.
    """

    def __init__(
        self,
        actions,
        window,
        alpha,
        initial_action,
        verbose=False
    ):
        self.actions = list(actions)

        self.window = window
        self.alpha = alpha

        self.initial_action = initial_action
        self.current_action = initial_action

        self.verbose = verbose

        self.min_histories = {
            action: deque(
                maxlen=window + 1
            )
            for action in self.actions
        }

        self.queue_histories = {
            action: deque(
                maxlen=window + 1
            )
            for action in self.actions
        }

        self.decision = 0
        self.switches = 0

        self.episode_switch_counts = []

    # ---------------------------------------------------------
    # Observation indices
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Observation helpers
    # ---------------------------------------------------------

    def min_value(
        self,
        observation,
        action
    ):
        return float(
            observation[
                self.min_index(action)
            ]
        )

    def open_entries(
        self,
        observation,
        action
    ):
        return float(
            observation[
                self.entries_index(action)
            ]
        )

    # ---------------------------------------------------------
    # Scoring
    # ---------------------------------------------------------

    @staticmethod
    def normalized_progress(
        old_min,
        current_min
    ):
        return (
            old_min - current_min
        ) / max(
            abs(old_min),
            1.0
        )

    @staticmethod
    def queue_pressure(
        old_queue,
        current_queue
    ):
        growth = max(
            current_queue - old_queue,
            0.0
        )

        return growth / max(
            current_queue,
            1.0
        )

    # ---------------------------------------------------------
    # Store current observation
    # ---------------------------------------------------------

    def observe(self, observation):
        for action in self.actions:

            self.min_histories[
                action
            ].append(
                self.min_value(
                    observation,
                    action
                )
            )

            self.queue_histories[
                action
            ].append(
                self.open_entries(
                    observation,
                    action
                )
            )

    # ---------------------------------------------------------
    # Test whether full windows exist
    # ---------------------------------------------------------

    def histories_ready(self):
        return all(
            len(
                self.min_histories[action]
            ) >= self.window + 1
            and
            len(
                self.queue_histories[action]
            ) >= self.window + 1

            for action in self.actions
        )

    # ---------------------------------------------------------
    # Compute scores
    # ---------------------------------------------------------

    def calculate_scores(
        self,
        observation
    ):
        scores = {}

        details = {}

        for action in self.actions:

            current_queue = self.open_entries(
                observation,
                action
            )

            # fd_env maps +/-inf to zero.
            #
            # Therefore an empty open list must be explicitly
            # excluded rather than treating its h_min as 0.
            if current_queue <= 0:
                continue

            old_min = (
                self.min_histories[
                    action
                ][0]
            )

            current_min = (
                self.min_histories[
                    action
                ][-1]
            )

            old_queue = (
                self.queue_histories[
                    action
                ][0]
            )

            observed_queue = (
                self.queue_histories[
                    action
                ][-1]
            )

            progress = (
                self.normalized_progress(
                    old_min,
                    current_min
                )
            )

            pressure = (
                self.queue_pressure(
                    old_queue,
                    observed_queue
                )
            )

            score = (
                progress
                - self.alpha * pressure
            )

            scores[action] = score

            details[action] = {
                "progress": progress,
                "pressure": pressure,
                "score": score,
                "old_min": old_min,
                "current_min": current_min,
                "old_queue": old_queue,
                "current_queue": observed_queue,
            }

        return scores, details

    # ---------------------------------------------------------
    # Select action
    # ---------------------------------------------------------

    def select_action(
        self,
        scores
    ):
        if not scores:
            return self.current_action

        best_score = max(
            scores.values()
        )

        best_actions = [
            action
            for action, score
            in scores.items()
            if abs(
                score - best_score
            ) < 1e-12
        ]

        # Prefer the current heuristic in a tie.
        #
        # This avoids unnecessary switches when two heuristics
        # currently have equivalent scores.
        if self.current_action in best_actions:
            return self.current_action

        # Otherwise deterministic action-order tie break.
        for action in self.actions:
            if action in best_actions:
                return action

        return best_actions[0]

    # ---------------------------------------------------------
    # Controller step
    # ---------------------------------------------------------

    def act(self, observation):
        self.decision += 1

        self.observe(
            observation
        )

        # For the first W decisions there is not yet enough
        # history to calculate a W-step score.
        if not self.histories_ready():
            return int(
                self.current_action
            )

        scores, details = (
            self.calculate_scores(
                observation
            )
        )

        new_action = (
            self.select_action(
                scores
            )
        )

        if new_action != self.current_action:

            old_action = self.current_action

            self.current_action = new_action
            self.switches += 1

            if self.verbose:
                detail_parts = []

                for action in self.actions:

                    if action not in details:
                        detail_parts.append(
                            "h{}=EMPTY".format(
                                action
                            )
                        )
                        continue

                    d = details[action]

                    detail_parts.append(
                        "h{}:P={:.4f},Q={:.4f},S={:.4f}".format(
                            action,
                            d["progress"],
                            d["pressure"],
                            d["score"]
                        )
                    )

                print(
                    "[PROG+QUEUE] decision {}: "
                    "SWITCH h{} -> h{} | {}".format(
                        self.decision,
                        old_action,
                        new_action,
                        " | ".join(
                            detail_parts
                        )
                    )
                )

        return int(
            self.current_action
        )

    # ---------------------------------------------------------
    # Episode cleanup
    # ---------------------------------------------------------

    def stop_episode(self):

        self.episode_switch_counts.append(
            self.switches
        )

        self.current_action = (
            self.initial_action
        )

        self.decision = 0
        self.switches = 0

        self.min_histories = {
            action: deque(
                maxlen=self.window + 1
            )
            for action in self.actions
        }

        self.queue_histories = {
            action: deque(
                maxlen=self.window + 1
            )
            for action in self.actions
        }

        return True


# =============================================================
# Command line
# =============================================================

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
        "--window",
        type=int,
        default=50
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0
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

    logging.basicConfig(
        level=logging.INFO
    )

    env = FDEnvSelHeur(
        host="",
        port=args.port,
        num_heuristics=args.num_heuristics,
        config_dir=".",
        port_file_id=args.pfid,
        use_general_state_info=False,
        time_step_limit=args.time_step_limit
    )

    env.seed(
        args.seed
    )

    agent = ProgressQueueAgent(
        actions=args.actions,
        window=args.window,
        alpha=args.alpha,
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

    eval_stats[
        "policy"
    ] = "progress_queue4"

    eval_stats[
        "window"
    ] = args.window

    eval_stats[
        "alpha"
    ] = args.alpha

    eval_stats[
        "actions"
    ] = args.actions

    eval_stats[
        "switches_per_episode"
    ] = agent.episode_switch_counts

    for i, trajectory in enumerate(
        eval_stats["trajectories"]
    ):

        print(
            "episode {}: "
            "decisions={}, "
            "switches={}, "
            "first actions={}".format(
                i,
                len(trajectory),
                eval_stats[
                    "switches_per_episode"
                ][i],
                trajectory[:120]
            )
        )

    if args.save_eval_stats:

        filename = (
            args.save_eval_stats
        )

        if not filename.endswith(
            ".json"
        ):
            filename += ".json"

        with open(
            filename,
            "w"
        ) as outfile:

            json.dump(
                eval_stats,
                outfile
            )

        print(
            "Saved:",
            filename
        )