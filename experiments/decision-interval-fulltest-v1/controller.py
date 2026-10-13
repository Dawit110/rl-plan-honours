#!/usr/bin/env python3
"""Run the frozen Chainer controller, adapting only the previous raw observation.

All inference and feature transforms remain in the supplied frozen modules.
No NumPy reimplementation is used for the Mac pilot.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import runpy
import signal
import sys
import time

FEATURES = ('Average Value', 'Max Value', 'Min Value', 'Open List Entries', 'Varianz')

def decode(packet):
    return ast.literal_eval(packet.decode().replace('-inf', '0').replace('inf', '0'))

def flatten(data):
    return [data[str(i)][key] for i in range(4) for key in FEATURES]

def adapt_previous(env, data):
    if 'interval_previous' not in data:
        raise RuntimeError('Extended planner observation is missing previous raw statistics')
    if env._prev_state is not None:
        env._prev_state = flatten(data['interval_previous'])

def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--source', required=True)
    p.add_argument('--extended', action='store_true')
    p.add_argument('--trace', action='store_true')
    p.add_argument('--profile', action='store_true')
    args, forwarded = p.parse_known_args()
    source = Path(args.source).resolve()
    sys.path.insert(0, str(source))
    import fd_env
    from chainerrl.agents.double_dqn import DoubleDQN
    original_receive = fd_env.FDEnvSelHeur.recv_msg
    original_act = DoubleDQN.act
    counts = dict(observations=0, learned_decisions=0, profiling_enabled=args.profile,
                  inference_cpu_seconds=0.0 if args.profile else None,
                  socket_receive_wall_seconds=0.0 if args.profile else None,
                  cpu_at_controller_ready=time.process_time(),
                  note='Periodic counters may be lower bounds for interrupted/limited cases.')
    trace = open('trace.jsonl', 'w', buffering=1) if args.trace else None
    def snapshot():
        counts['controller_process_cpu_seconds'] = time.process_time()
        temp = Path('controller-metrics.tmp')
        temp.write_text(json.dumps(counts, sort_keys=True) + '\n')
        temp.replace('controller-metrics.json')
    def receive(env):
        start = time.perf_counter() if args.profile else 0
        packet = original_receive(env)
        if args.profile:
            counts['socket_receive_wall_seconds'] += time.perf_counter() - start
        if packet is None: return None
        counts['observations'] += 1
        data = decode(packet) if (args.extended or args.trace) else None
        if args.extended:
            # Keep decimal-parsed Python numbers; the frozen code casts later.
            adapt_previous(env, data)
            counts['planner'] = data['interval_meta']
        if trace:
            trace.write(json.dumps(dict(kind='raw', state=flatten(data), done=data['done']))+'\n')
        if counts['observations'] % 1024 == 0:
            snapshot()
        return packet
    def act(agent, obs):
        start = time.process_time() if args.profile else 0
        action = original_act(agent, obs)
        if args.profile: counts['inference_cpu_seconds'] += time.process_time() - start
        counts['learned_decisions'] += 1
        if trace:
            trace.write(json.dumps(dict(kind='act', state=obs.tolist(), action=int(action)))+'\n')
        return action
    fd_env.FDEnvSelHeur.recv_msg = receive
    DoubleDQN.act = act
    def terminate(signum, frame):
        snapshot()
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    sys.argv = [str(source/'train_and_eval_chainer_agent.py')] + forwarded
    snapshot()
    try:
        runpy.run_path(sys.argv[0], run_name='__main__')
    finally:
        snapshot()
        if trace: trace.close()

if __name__ == '__main__': main()
