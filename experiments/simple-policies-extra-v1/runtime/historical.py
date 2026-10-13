"""Historical policy class bodies, copied verbatim; see source-provenance.json."""
from collections import deque
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

class ProbeThenExploitAgent:
    """
    Approach 3: Probe-Then-Exploit Selection.

    Each heuristic is given a short contiguous probing interval.

    During the probe, progress is measured using the change in that
    heuristic's observed minimum heuristic value.

    Once every heuristic has been probed, the heuristic with the
    greatest normalized progress is selected for a longer exploitation
    interval.

    After exploitation, the probing process repeats.
    """

    def __init__(
        self,
        actions,
        probe_length,
        exploit_length,
        verbose=False
    ):
        self.actions = list(actions)

        self.probe_length = probe_length
        self.exploit_length = exploit_length

        self.verbose = verbose

        # Saved across episodes for evaluation output.
        self.episode_switch_counts = []
        self.episode_cycle_counts = []

        self._reset_episode_state()

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
    # Episode/cycle setup
    # ---------------------------------------------------------

    def _reset_episode_state(self):
        self.decision = 0
        self.switches = 0

        self.last_action = None

        self.phase = "probe"

        # We rotate the first heuristic in each new probe cycle.
        #
        # Cycle 1:
        #   0, 1, 2, 3
        #
        # Cycle 2:
        #   1, 2, 3, 0
        #
        # Cycle 3:
        #   2, 3, 0, 1
        #
        # This reduces systematic bias caused by always probing
        # one heuristic first and another last.
        self.rotation = 0

        self.completed_cycles = 0

        self._start_probe_cycle()

    def _start_probe_cycle(self):
        rotation = self.rotation % len(self.actions)

        self.probe_order = (
            self.actions[rotation:]
            + self.actions[:rotation]
        )

        self.rotation += 1

        self.probe_index = 0
        self.probe_steps = 0
        self.probe_start_min = None

        self.probe_scores = {}

        self.phase = "probe"

        self.exploit_action = None
        self.exploit_steps = 0

        if self.verbose:
            print(
                "[PROBE+EXPLOIT] starting probe cycle {} "
                "| order={}".format(
                    self.completed_cycles + 1,
                    self.probe_order
                )
            )

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def _min_value(self, observation, action):
        return float(
            observation[
                self.min_index(action)
            ]
        )

    def _open_entries(self, observation, action):
        return float(
            observation[
                self.entries_index(action)
            ]
        )

    def _normalized_progress(
        self,
        start_min,
        end_min
    ):
        return (
            start_min - end_min
        ) / max(
            abs(start_min),
            1.0
        )

    def _return_action(self, action):
        """
        Register an action transition and return the action.
        """

        if (
            self.last_action is not None
            and action != self.last_action
        ):
            self.switches += 1

        self.last_action = action

        return int(action)

    # ---------------------------------------------------------
    # Finish one probe
    # ---------------------------------------------------------

    def _finish_probe(
        self,
        observation,
        action
    ):
        end_min = self._min_value(
            observation,
            action
        )

        open_entries = self._open_entries(
            observation,
            action
        )

        # fd_env converts +/- infinity to zero.
        #
        # Therefore an empty open list could otherwise look like
        # an excellent h_min value. Treat an empty queue as an
        # unusable probe instead.
        if open_entries <= 0:
            score = float("-inf")

        else:
            score = self._normalized_progress(
                self.probe_start_min,
                end_min
            )

        self.probe_scores[action] = score

        if self.verbose:
            if score == float("-inf"):
                score_text = "EMPTY"
            else:
                score_text = "{:.4f}".format(
                    score
                )

            print(
                "[PROBE+EXPLOIT] cycle {} "
                "h{} probe complete | "
                "start={:.3f} end={:.3f} "
                "score={}".format(
                    self.completed_cycles + 1,
                    action,
                    self.probe_start_min,
                    end_min,
                    score_text
                )
            )

        self.probe_index += 1
        self.probe_steps = 0
        self.probe_start_min = None

    # ---------------------------------------------------------
    # Choose exploitation heuristic
    # ---------------------------------------------------------

    def _choose_exploit_action(
        self,
        observation
    ):
        # Only exploit a heuristic whose open list currently
        # contains states.
        available = [
            action
            for action in self.actions
            if self._open_entries(
                observation,
                action
            ) > 0
        ]

        if not available:
            # Search should normally terminate before this matters,
            # but retain a deterministic fallback.
            return self.actions[0]

        scored_available = [
            action
            for action in available
            if self.probe_scores.get(
                action,
                float("-inf")
            ) != float("-inf")
        ]

        # If every probe became unavailable, simply use the first
        # currently available heuristic.
        if not scored_available:
            return available[0]

        best_score = max(
            self.probe_scores[action]
            for action in scored_available
        )

        best_actions = [
            action
            for action in scored_available
            if abs(
                self.probe_scores[action]
                - best_score
            ) < 1e-12
        ]

        # Deterministic tie-break using the original action order.
        for action in self.actions:
            if action in best_actions:
                return action

        return best_actions[0]

    # ---------------------------------------------------------
    # Controller decision
    # ---------------------------------------------------------

    def act(self, observation):
        self.decision += 1

        while True:

            # =================================================
            # PROBE PHASE
            # =================================================

            if self.phase == "probe":

                # All heuristics have now been probed.
                if self.probe_index >= len(
                    self.probe_order
                ):
                    self.exploit_action = (
                        self._choose_exploit_action(
                            observation
                        )
                    )

                    self.phase = "exploit"
                    self.exploit_steps = 0

                    self.completed_cycles += 1

                    if self.verbose:
                        score_parts = []

                        for action in self.actions:
                            score = (
                                self.probe_scores.get(
                                    action,
                                    float("-inf")
                                )
                            )

                            if score == float("-inf"):
                                text = "EMPTY"
                            else:
                                text = "{:.4f}".format(
                                    score
                                )

                            score_parts.append(
                                "h{}={}".format(
                                    action,
                                    text
                                )
                            )

                        print(
                            "[PROBE+EXPLOIT] cycle {} "
                            "EXPLOIT h{} for {} decisions "
                            "| {}".format(
                                self.completed_cycles,
                                self.exploit_action,
                                self.exploit_length,
                                ", ".join(
                                    score_parts
                                )
                            )
                        )

                    # Process exploitation immediately using
                    # this same observation.
                    continue

                action = self.probe_order[
                    self.probe_index
                ]

                # -------------------------------------------------
                # A completed probe is scored on the observation
                # AFTER its final selected action has executed.
                # -------------------------------------------------

                if (
                    self.probe_steps
                    >= self.probe_length
                ):
                    self._finish_probe(
                        observation,
                        action
                    )

                    continue

                open_entries = self._open_entries(
                    observation,
                    action
                )

                # -------------------------------------------------
                # Skip a heuristic whose queue is already empty.
                # -------------------------------------------------

                if (
                    self.probe_steps == 0
                    and open_entries <= 0
                ):
                    self.probe_start_min = (
                        self._min_value(
                            observation,
                            action
                        )
                    )

                    self.probe_scores[
                        action
                    ] = float("-inf")

                    if self.verbose:
                        print(
                            "[PROBE+EXPLOIT] cycle {} "
                            "h{} skipped: empty open list".format(
                                self.completed_cycles + 1,
                                action
                            )
                        )

                    self.probe_index += 1
                    self.probe_steps = 0
                    self.probe_start_min = None

                    continue

                # -------------------------------------------------
                # First decision of this probe:
                # record the initial minimum.
                # -------------------------------------------------

                if self.probe_steps == 0:
                    self.probe_start_min = (
                        self._min_value(
                            observation,
                            action
                        )
                    )

                    if self.verbose:
                        print(
                            "[PROBE+EXPLOIT] cycle {} "
                            "begin probe h{} "
                            "| start_min={:.3f}".format(
                                self.completed_cycles + 1,
                                action,
                                self.probe_start_min
                            )
                        )

                # Give this heuristic one controller decision.
                self.probe_steps += 1

                return self._return_action(
                    action
                )

            # =================================================
            # EXPLOIT PHASE
            # =================================================

            else:

                # Exploitation period completed.
                #
                # Start another complete round of probing.
                if (
                    self.exploit_steps
                    >= self.exploit_length
                ):
                    if self.verbose:
                        print(
                            "[PROBE+EXPLOIT] cycle {} "
                            "exploitation finished".format(
                                self.completed_cycles
                            )
                        )

                    self._start_probe_cycle()

                    continue

                # If the selected exploitation queue becomes
                # empty, terminate exploitation early and re-probe.
                if self._open_entries(
                    observation,
                    self.exploit_action
                ) <= 0:

                    if self.verbose:
                        print(
                            "[PROBE+EXPLOIT] cycle {} "
                            "h{} exploitation ended early: "
                            "empty open list".format(
                                self.completed_cycles,
                                self.exploit_action
                            )
                        )

                    self._start_probe_cycle()

                    continue

                self.exploit_steps += 1

                return self._return_action(
                    self.exploit_action
                )

    # ---------------------------------------------------------
    # Episode cleanup
    # ---------------------------------------------------------

    def stop_episode(self):
        self.episode_switch_counts.append(
            self.switches
        )

        self.episode_cycle_counts.append(
            self.completed_cycles
        )

        self._reset_episode_state()

        return True

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
