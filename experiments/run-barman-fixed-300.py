#!/usr/bin/env python3
"""Evaluate fixed FF, CG, CEA and ADD through the four-queue controller setup.
Prepare only by default. --run starts 80 cases, with no training.
Requires the existing corrected progress pilot and its prepared folder.
"""
import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

POLICIES = ['always_ff', 'always_cg', 'always_cea', 'always_add']
CONTROLLER = '''import argparse
import json
import chainerrl
from fd_env import FDEnvSelHeur
from evaluate import eval_performance

class FixedPolicy:
    def __init__(self, policy):
        self.policy = policy
        self.index = 0
    def act(self, observation):
        return {'always_ff': 0, 'always_cg': 1, 'always_cea': 2, 'always_add': 3}[self.policy]
    def stop_episode(self):
        self.index = 0

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--policy', choices=['always_ff', 'always_cg', 'always_cea', 'always_add'], required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    args = p.parse_args()
    chainerrl.misc.set_random_seed(args.seed)
    env = FDEnvSelHeur(num_heuristics=4, port=args.port, port_file_id=args.port,
        config_dir='.', state_type=2, temporal_window=0, use_general_state_info=False,
        max_rand_steps=0, time_step_limit=1000000)
    env.seed(2 ** 32 - 1 - args.seed)
    env = chainerrl.wrappers.CastObservationToFloat32(env)
    try:
        stats = eval_performance(env, FixedPolicy(args.policy), n_steps=None,
            n_episodes=1, max_episode_len=1000000)
        with open('eval.json', 'w') as f:
            json.dump(stats, f)
    finally:
        env.close()

if __name__ == '__main__':
    main()
'''


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup(root, output, args):
    pilot_dir = (root / args.pilot).resolve()
    pilot_protocol = pilot_dir / 'protocol.json'
    if not pilot_protocol.is_file():
        raise RuntimeError('Pilot protocol missing: ' + str(pilot_protocol))
    prior = json.loads(pilot_protocol.read_text())
    if prior.get('version') != 'barman-original-budget-v1':
        raise RuntimeError('Unexpected pilot protocol version.')
    if prior.get('wall_seconds', prior.get('eval_wall_seconds')) != 300:
        raise RuntimeError('Expected the reviewed 300-second pilot.')
    if sys.version != prior['python'] or sys.executable != prior['executable']:
        raise RuntimeError('Use the same Python environment as the pilot.')
    for name, expected in prior['hashes'].items():
        path = root / name
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError('Pilot file changed or missing: ' + name)
    pilot = load_module('baseline_pilot_helpers', root / 'experiments/run-barman-original-budget.py')
    for name, entry in pilot.SOURCES.items():
        path = pilot_dir / 'source' / name
        if not path.is_file() or path.read_text() != entry['text']:
            raise RuntimeError('Pilot isolated source differs: ' + str(path))
    train = prior['train']; dev = prior['development']
    if len(train) != 80 or len(dev) != 20 or set(train) & set(dev):
        raise RuntimeError('Invalid pilot training/development split.')
    protocol = dict(version='barman-fixed-300-v1', policies=POLICIES,
        seed=args.seed, wall_seconds=300, fd_time_limit=300, development=dev,
        action_mapping={'always_ff': 0, 'always_cg': 1, 'always_cea': 2, 'always_add': 3},
        warmup='One FF action in reset, identical to the RL evaluation environment.',
        timing='End-to-end; includes startup, translation and search. Fixed policies do not perform neural inference.',
        hashes=dict(prior['hashes'], **{str(Path(__file__).resolve()): sha(Path(__file__).resolve()),
                                      str(pilot_protocol): sha(pilot_protocol)}))
    target = output / 'protocol.json'
    if target.exists():
        if json.loads(target.read_text()) != protocol:
            raise RuntimeError('Baseline protocol changed; use a new output folder.')
    elif any(p.name != '.runner.lock' for p in output.iterdir()):
        raise RuntimeError('Nonempty output has no protocol.')
    else:
        save(target, protocol)
    source = output / 'source'; source.mkdir(exist_ok=True)
    files = {'controller.py': CONTROLLER}
    for name in ('fd_env.py', 'evaluate.py', 'evaluator.py'):
        files[name] = pilot.SOURCES[name]['text']
    for name, text in files.items():
        compile(text, name, 'exec')
        path = source / name
        if path.exists() and path.read_text() != text:
            raise RuntimeError('Baseline source changed: ' + str(path))
        if not path.exists():
            path.write_text(text)
    return pilot, protocol


def summary(output):
    rows = [json.loads(p.read_text()) for p in sorted((output / 'evaluation').glob('*/*/*/result.json'))]
    if not rows:
        return
    with (output / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader(); writer.writerows(rows)
    for policy in POLICIES:
        cases = [r for r in rows if r['configuration'] == policy]
        done = [r for r in cases if r['completed']]
        print('{}: {}/{} solved; 20 planned; {} incomplete/errors'.format(policy,
            sum(r['status'] == 'solved' for r in done), len(done), len(cases)-len(done)), flush=True)


def evaluate(root, output, args, pilot, protocol):
    helper = load_module('baseline_evaluator', output / 'source/evaluator.py')
    helper.WINDOWS = {policy: 0 for policy in POLICIES}
    helper.stop_group = pilot.stop
    helper.controller_command = lambda domain, policy, unused: [sys.executable, '-u',
        str(output / 'source/controller.py'), '--policy', policy, '--seed', str(args.seed),
        '--port', str(args.port)]
    helper.planner_command = lambda domain, problem, policy, unused: pilot.planner_command(
        root, problem, args.seed, args.port, 300)
    eval_args = SimpleNamespace(port=args.port, wall_seconds=300)
    try:
        for problem in protocol['development']:
            for policy in POLICIES:
                try:
                    helper.run_case('barman', problem, policy, eval_args, output / 'evaluation')
                except BaseException as exc:
                    path = output / 'evaluation/barman' / Path(problem).stem / policy / 'result.json'
                    if path.exists():
                        result = json.loads(path.read_text())
                        result.update(completed=False, error=str(exc))
                        if result.get('status') == 'solved':
                            result['status'] = 'runner_error'
                        save(path, result)
                    raise
            summary(output)
    finally:
        summary(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path.cwd())
    p.add_argument('--pilot', default='results-barman-original-budget-v1-seed1')
    p.add_argument('--output', default='results-barman-fixed-300-seed1')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--port', type=int, default=12631)
    p.add_argument('--run', action='store_true')
    args = p.parse_args()
    if not 1 <= args.port <= 65535 or not 0 <= args.seed < 2 ** 32:
        p.error('Invalid port or seed.')
    root = args.root.resolve(); output = (root / args.output).resolve()
    if root not in output.parents:
        p.error('Output must be a subdirectory of the project.')
    output.mkdir(parents=True, exist_ok=True)
    handles = []
    try:
        for path in (output / '.runner.lock', root / '.eval-port-12631.lock',
                     root / '.eval-port-12521.lock') + (() if args.port == 12631 else (root / '.eval-port-{}.lock'.format(args.port),)):
            f = path.open('a'); handles.append(f)
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Another pilot or evaluator is active; stop it first.')
        pilot, protocol = setup(root, output, args)
        print('Prepared:', output, flush=True)
        print('20 development problems; fixed FF, CG, CEA and ADD; 300 seconds; no training.', flush=True)
        print('All policies use all four queues and the controller connection. Warm-up is one FF action.', flush=True)
        if not args.run:
            print('Add --run to start the 80 evaluations.', flush=True)
            return
        stdout, stderr = pilot.environment_info()
        previous = (root / args.pilot / 'environment.txt').read_text()
        if stdout != previous:
            raise RuntimeError('Environment differs from pilot; inspect before comparing.')
        (output / 'environment.txt').write_text(stdout)
        (output / 'environment-warnings.txt').write_text(stderr)
        evaluate(root, output, args, pilot, protocol)
        print('BASELINES COMPLETE. Upload', output / 'summary.csv', flush=True)
    finally:
        for f in reversed(handles):
            f.close()


if __name__ == '__main__':
    main()
