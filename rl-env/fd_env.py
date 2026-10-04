import socket
import time
import typing
from copy import deepcopy
from collections import deque
from enum import Enum
from os import remove
from os.path import join as joinpath
from typing import Union

import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
from gym.utils import seeding


class StateType(Enum):
    """Class to define numbers for state types"""
    RAW = 1
    DIFF = 2
    ABSDIFF = 3
    NORMAL = 4
    NORMDIFF = 5
    NORMABSDIFF = 6


class FDEnvSelHeur(Env):

    def __init__(self, num_heuristics: int, host: str = '', port: int = 12345,
                 num_steps=None, state_type: Union[int, StateType] = StateType.RAW,
                 seed: int = 12345, max_rand_steps: int = 0, config_dir: str = '.',
                 port_file_id=None, use_general_state_info: bool = True,
                 time_step_limit: int = -1, temporal_window: int = 0):
        """
        Initialize environment
        """

        self._heuristic_state_features = ['Average Value',  # 'Dead Ends Reliable',
                                          'Max Value', 'Min Value', 'Open List Entries',
                                          'Varianz']
        self.action_space = Discrete(num_heuristics)
        self._general_state_features = [#'evaluated_states', 'evaluations', 'expanded_states',
                                        # 'generated_ops',
                                        #'generated_states', 'num_variables',
                                        #'registered_states', 'reopened_states',
                                        #"cg_num_eff_to_eff", "cg_num_eff_to_pre", "cg_num_pre_to_eff"
                                        ]

        total_state_features = (num_heuristics * len(self._heuristic_state_features))
        self._use_gsi = use_general_state_info
        if use_general_state_info:
            total_state_features += len(self._general_state_features)
        if temporal_window < 0:
            raise ValueError("temporal_window must be non-negative")

        if temporal_window > 0:
            total_state_features += num_heuristics

        self.observation_space = Box(
            low=np.array([-np.inf for _ in range(total_state_features)]),
            high=np.array([np.inf for _ in range(total_state_features)]),
            dtype=np.float32
        )

        self.__skip_transform = [False for _ in range(total_state_features)]
        if use_general_state_info:
            self.__skip_transform[4] = True  # skip num_variables transform
            self.__skip_transform[7] = True
            self.__skip_transform[8] = True
            self.__skip_transform[9] = True

        self.__num_heuristics = num_heuristics
        self.host = host
        self.port = port

        self.socket = None
        self.conn = None

        self._prev_state = None
        self._last_observation = None

        self.temporal_window = temporal_window
        self._raw_history = deque(
            maxlen=max(1, temporal_window)
        )

        self.num_steps = num_steps
        self.time_step_limit = time_step_limit

        self.__state_type = StateType(state_type)
        self.__norm_vals = []
        self._config_dir = config_dir
        self._port_file_id = port_file_id

        self._transformation_func = None
        # create state transformation function with inputs (current state, previous state, normalization values)
        if self.__state_type == StateType.DIFF:
            self._transformation_func = lambda x, y, z, skip: x - y if not skip else x
        elif self.__state_type == StateType.ABSDIFF:
            self._transformation_func = lambda x, y, z, skip: abs(x - y) if not skip else x
        elif self.__state_type == StateType.NORMAL:
            self._transformation_func = lambda x, y, z, skip: FDEnvSelHeur._save_div(x, z) if not skip else x
        elif self.__state_type == StateType.NORMDIFF:
            self._transformation_func = lambda x, y, z, skip: \
                FDEnvSelHeur._save_div(x, z) - FDEnvSelHeur._save_div(y, z) if not skip else x
        elif self.__state_type == StateType.NORMABSDIFF:
            self._transformation_func = lambda x, y, z, skip:\
                abs(FDEnvSelHeur._save_div(x, z) - FDEnvSelHeur._save_div(y, z)) if not skip else x

        self.rng = np.random.RandomState(seed=seed)
        self.max_rand_steps = max_rand_steps
        self.__step = 0
        self.__start_time = None
        self.done = True  # Starts as true as the expected behavior is that before normal resets an episode was done.

    @staticmethod
    def _save_div(a, b):
        return np.divide(a, b, out=np.zeros_like(a), where=b != 0)

    def send_msg(self, msg: bytes):
        """
        Send message and prepend the message size

        Based on comment from SO see [1]
        [1] https://stackoverflow.com/a/17668009

        :param msg: The message as byte
        """
        # Prefix each message with a 4-byte length (network byte order)
        msg = str.encode("{:>04d}".format(len(msg))) + msg
        self.conn.sendall(msg)

    def recv_msg(self):
        """
        Recieve a whole message. The message has to be prepended with its total size
        Based on comment from SO see [1]
        """
        # Read message length and unpack it into an integer
        raw_msglen = self.recvall(4)
        if not raw_msglen:
            return None
        msglen = int(raw_msglen.decode())
        # Read the message data
        return self.recvall(msglen)

    def recvall(self, n: int):
        """
        Given we know the size we want to recieve, we can recieve that amount of bytes.
        Based on comment from SO see [1]

        :param n: Number of bytes to expect in the data
        """
        # Helper function to recv n bytes or return None if EOF is hit
        data = b''
        while len(data) < n:
            packet = self.conn.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    def _process_data(self):
        """
        Split received json into state reward and done
        :return:
        """
        packet = self.recv_msg()
        if packet is None:
            raise EOFError("Planner closed the socket")
        msg = packet.decode()
        #print("----------------------------")
        #print(msg)
        #print("=>")
        msg = msg.replace('-inf', '0')
        msg = msg.replace('inf', '0')
        #print(msg)
        data = eval(msg)
        r = data['reward']
        done = data['done']
        del data['reward']
        del data['done']

        state = []

        if self._use_gsi:
            for feature in self._general_state_features:
                state.append(data[feature])
        for heuristic_id in range(self.__num_heuristics):  # process heuristic data
            for feature in self._heuristic_state_features:
                state.append(data["%d" % heuristic_id][feature])

        raw_state = np.asarray(state, dtype=np.float32)

        if self._prev_state is None:
            self.__norm_vals = deepcopy(state)
            self._prev_state = deepcopy(state)

        if self.__state_type != StateType.RAW:
            tmp_state = state
            state = list(map(
                self._transformation_func,
                state,
                self._prev_state,
                self.__norm_vals,
                self.__skip_transform
            ))
            self._prev_state = tmp_state

        if self.temporal_window > 0:

            if self._raw_history:
                previous = self._raw_history[0]

                # Five original features per heuristic.
                # Minimum value is at offset 2.
                temporal_features = [
                    float(
                        raw_state[5 * h + 2]
                        - previous[5 * h + 2]
                    )
                    for h in range(self.__num_heuristics)
                ]
            else:
                temporal_features = [
                    0.0
                    for _ in range(self.__num_heuristics)
                ]

            state.extend(temporal_features)

            self._raw_history.append(
                raw_state.copy()
            )

        observation = np.array(state)
        self._last_observation = observation.copy()
        return observation, r, done

    def _interrupted_step(self, exc):
        # No final observation/reward arrives on EOF. This is an explicit fallback,
        # not evidence that a plan was found or that the planner timed out.
        import logging
        self.done = True
        self.kill_connection()
        if self._last_observation is None:
            raise RuntimeError("Planner disconnected before an observation") from exc
        logging.getLogger(__name__).warning(
            "Planner connection interrupted (%s); truncating episode. "
            "Reusing last observation with missing reward set to 0.",
            type(exc).__name__)
        return (self._last_observation.copy(), 0.0, False,
                {'needs_reset': True, 'planner_disconnected': True,
                 'interruption_reason': type(exc).__name__})

    def step(self, action: typing.Union[int, typing.List[int]]):
        self.__step += 1
        if not np.issubdtype(type(action), np.integer):
            action = action[0]
        if self.num_steps:
            msg = ','.join([str(action), str(self.num_steps)])
        else:
            msg = str(action)
        try:
            self.send_msg(str.encode(msg))
            s, r, d = self._process_data()
        except (EOFError, ConnectionResetError, ConnectionAbortedError,
                BrokenPipeError) as exc:
            return self._interrupted_step(exc)

        info = {}
        if d:
            self.done = True
            self.kill_connection()
        elif self.time_step_limit >= 0 and self.__step > self.time_step_limit:
            info['needs_reset'] = True
            try:
                self.send_msg(b"END")
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                pass
            finally:
                self.kill_connection()
                self.done = True
        return s, r, d, info

    def reset(self):
        import logging
        # A planner can also exit before its initial state or during warm-up.
        # Do not pass an ended warm-up episode to the learner as a fresh one.
        if not self.done and self.conn is not None:
            try:
                self.send_msg(b"END")
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                pass
        self.kill_connection()
        while True:
            self._prev_state = None
            self._raw_history.clear()
            self._last_observation = None
            self.__step = 0
            self.__start_time = time.time()
            self.done = False

            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen()
            if self._port_file_id:
                fp = joinpath(self._config_dir, 'port_{:d}.txt'.format(self._port_file_id))
            else:
                fp = joinpath(self._config_dir, 'port.txt')
            with open(fp, 'w') as portfh:
                portfh.write(str(self.port))
            print(fp, flush=True)
            self.conn, address = self.socket.accept()
            # The launcher runs one planner at a time. Remove this episode's
            # readiness file before another reset can create the next one.
            remove(fp)
            try:
                s, _, d = self._process_data()
            except (EOFError, ConnectionResetError, ConnectionAbortedError,
                    BrokenPipeError) as exc:
                logging.getLogger(__name__).warning(
                    "Planner disconnected during initial observation (%s); "
                    "waiting for the next planning problem.", type(exc).__name__)
                self.done = True
                self.kill_connection()
                continue

            if d:
                self.done = True
                self.kill_connection()
                logging.getLogger(__name__).warning(
                    "Planner ended before warm-up; waiting for next problem.")
                continue

            if self.max_rand_steps > 1:
                actions = range(self.rng.randint(1, self.max_rand_steps + 1))
            else:
                actions = range(1)
            ended = False
            for _ in actions:
                action = self.action_space.sample() if self.max_rand_steps > 1 else 0
                s, _, d, info = self.step(action)
                if d or info.get('needs_reset', False):
                    ended = True
                    break
            if ended:
                logging.getLogger(__name__).warning(
                    "Planner episode ended during warm-up; waiting for next problem.")
                continue
            return s

    def kill_connection(self):
        """Kill the connection"""
        if self.conn:
            try:
                self.conn.shutdown(2)
            except OSError:
                pass
            self.conn.close()
            self.conn = None
        if self.socket:
            self.socket.close()
            self.socket = None

    def close(self):
        """
        Needs to "kill" the environment
        :return:
        """
        self.kill_connection()

    def render(self, mode: str = 'human') -> None:
        """
        Required by gym.Env but not implemented
        :param mode:
        :return: None
        """
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


if __name__ == '__main__':
    """
    Only for debugging purposes
    """
    HOST = ''  # The server's hostname or IP address
    PORT = 54321  # The port used by the server

    env = FDEnvSelHeur(host=HOST, port=PORT, num_heuristics=2)
    env.reset()
    try:
        for i in range(100):
            done = False
            while not done:
                s, r, done, _ = env.step(np.random.randint(0, 2))
            print(i)
            if i < 99:
                env.reset()
    finally:
        print('Closing Env')
        env.close()
