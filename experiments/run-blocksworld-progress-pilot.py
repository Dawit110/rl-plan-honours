#!/usr/bin/env python3
"""Isolated Blocksworld progress-feature pilot v1. Python 3.8+.
Compares original (20 inputs) to progress (28 inputs).
Per heuristic, progress adds:
  relative improvement in best valid observed queue minimum over 4 observations;
  capped observations since a new best minimum, divided by 1000.
Relative improvement uses max(1, abs(first valid minimum)) as episode scale.
Both added features are in [0,1]. Empty queues emit [0,1], cannot update the
best minimum, and break the progress window. Episode reset clears history.
All four queues share insertions: progress is NOT causal credit to that queue.
Statistics are queue-key proxies, not true distances to a solution.
Original inputs, reward, action interval and cutoff handling remain unchanged.
Diagnostics are saved to progress/feature-diagnostics.json at normal trainer exit.
Defaults: 100000 interactions, 1000 episode allowance, seed 1, 30s dev deadline.
Pilot uses the same deterministic 80/20 split as the previous runner.
This tests a feature bundle; improvements would not isolate each added feature.
Synthetic wrapper tests and simulated runner tests passed; actual legacy
Gym/Chainer learning and Mac planner integration require the local run.

Default: prepare and show settings. Add --run to train and evaluate both models.
Existing test results and source files are never modified.
"""
import argparse
import base64
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import zlib

SOURCE_BUNDLE = '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'
SOURCES = json.loads(zlib.decompress(base64.b64decode(SOURCE_BUNDLE)).decode())
MODELS = {'original': 0, 'progress': 4}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def live_group(pgid):
    """Inspect the group created by start_new_session; ignore zombies."""
    if sys.platform.startswith('linux'):
        members = []
        for path in Path('/proc').glob('[0-9]*/stat'):
            try:
                data = path.read_text()
                fields = data[data.rfind(')') + 2:].split()
                if int(fields[2]) == pgid and fields[0] != 'Z':
                    members.append(int(path.parent.name))
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
        return members
    output = subprocess.check_output(['ps', '-axo', 'pid=,pgid=,stat='], text=True)
    members = []
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 3 and int(parts[1]) == pgid and not parts[2].startswith('Z'):
            members.append(int(parts[0]))
    return members


def stop(process):
    if process is None or getattr(process, '_cleanup_complete', False):
        return
    if process.pid == os.getpgrp():
        raise RuntimeError('Refusing to signal our own process group.')
    for sig in (signal.SIGTERM, signal.SIGKILL):
        members = live_group(process.pid)
        if members:
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
            except PermissionError:
                # Signal only members still verified as belonging to our group.
                for pid in members:
                    try:
                        if os.getpgid(pid) == process.pid:
                            os.kill(pid, sig)
                    except ProcessLookupError:
                        pass
                    except PermissionError:
                        pass
        try:
            process.wait(timeout=0.3 if sig == signal.SIGTERM else 2)
        except subprocess.TimeoutExpired:
            pass
    end = time.monotonic() + 2
    while live_group(process.pid) and time.monotonic() < end:
        time.sleep(0.05)
    if live_group(process.pid) or process.poll() is None:
        raise RuntimeError('Owned process group {} remains active. Inspect Activity Monitor before retrying.'.format(process.pid))
    process._cleanup_complete = True


def prepare(root, output, args):
    base = root / 'experiments/blocksworld/blocksworld'
    problems = sorted((base / 'train').glob('*.pddl'), key=lambda p: p.name)
    if len(problems) != 100:
        raise RuntimeError('Expected exactly 100 original Blocksworld training problems.')
    paths = [base / 'domain.pddl', root / 'fast-downward/fast-downward.py',
             root / 'fast-downward/builds/release/bin/downward', Path(__file__).resolve()]
    paths.extend(problems)
    for directory in ('fast-downward/driver', 'fast-downward/src/translate',
                      'fast-downward/builds/release/bin/translate'):
        paths.extend(sorted((root / directory).rglob('*.py')))
    for relative, expected in {'fast-downward/src/search/open_lists/best_first_open_list.cc': '991a55bbbfd68485757cea6620c3627c8dbc8df025fd407423cafbc1bccda1cc', 'fast-downward/src/search/open_lists/rl_open_list.cc': '4dc41fafd1e200ef28ab3c6f5a2a8e26d90055c639ef41b60cea1078f0ca8fad', 'fast-downward/src/search/search_engines/rl_eager_search.cc': 'f2234d89c59285a8559ae748bbb34fe2b46f91a18bde75460a460a015ff1edf7', 'fast-downward/src/search/search_engines/rl_client.cc': '6aab2fdc75f4cb1316f66d9d7b82a56ad8d312c1f3505a8a1c6217619fcddfc0'}.items():
        source = root / relative
        if not source.is_file() or digest(source) != expected:
            raise RuntimeError('Planner source changed since review: ' + str(source))
        paths.append(source)
    # Verify the installed controller sources match those reviewed in this chat.
    for name, entry in SOURCES.items():
        if name == 'evaluator.py':
            continue
        installed = root / 'rl-env' / name
        if not installed.is_file() or digest(installed) != entry['original_sha256']:
            raise RuntimeError('Source changed or missing: {}. Send the current file for review; do not bypass this check.'.format(installed))
        paths.append(installed)
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise RuntimeError('Missing files: ' + ', '.join(missing))
    shuffled = list(problems)
    random.Random(args.split_seed).shuffle(shuffled)
    dev = sorted(p.name for p in shuffled[:20])
    train = sorted(p.name for p in shuffled[20:])
    assert len(set(train) & set(dev)) == 0
    hashes = {str(p.relative_to(root)) if root in p.parents else str(p): digest(p) for p in paths}
    protocol = dict(version='blocksworld-progress-v1', root=str(root), python=sys.version,
                    executable=sys.executable, seed=args.seed, split_seed=args.split_seed,
                    steps=args.steps, episode_limit=args.episode_limit,
                    exploration_steps=args.steps // 2, train_time_limit=60,
                    eval_wall_seconds=args.wall_seconds, train=train, development=dev,
                    models=MODELS, hashes=hashes, feature_spec='best-observed queue minimum; relative window=4; age cap=1000; invalid=[0,1]',
                    scope='New models trained only on 80 instances; 20 held out from the original training collection. Designated test set unused.')
    manifest = output / 'protocol.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != protocol:
            raise RuntimeError('Pilot settings or files changed. Use a new --output directory.')
    elif any(p.name != '.runner.lock' for p in output.iterdir()):
        raise RuntimeError('Nonempty output has no protocol. Choose a fresh output folder.')
    else:
        save(manifest, protocol)
    source_dir = output / 'source'
    source_dir.mkdir(exist_ok=True)
    for name, entry in SOURCES.items():
        path = source_dir / name
        if path.exists() and path.read_text() != entry['text']:
            raise RuntimeError('Isolated source changed: ' + str(path))
        if not path.exists():
            path.write_text(entry['text'])
    save(output / 'split.json', {'training': train, 'development': dev, 'split_seed': args.split_seed})
    return protocol


def search(seed, port):
    return ('rl_eager(rl([single(ff()),single(cg()),single(cea()),single(add())],'
            'random_seed={},handle_empty_list=true),rl_control_interval=0,rl_client_port={})').format(seed, port)


def planner_command(root, problem, seed, port, limit):
    base = root / 'experiments/blocksworld/blocksworld'
    return [sys.executable, str(root / 'fast-downward/fast-downward.py'),
            '--overall-time-limit', str(limit), str(base / 'domain.pddl'),
            str(base / 'train' / problem), '--search', search(seed, port)]


def environment_info():
    script = ('import json,sys,numpy,chainer,chainerrl; from importlib.metadata import version; '
              'print(json.dumps(dict(python=sys.version,numpy=numpy.__version__, '
              'chainer=chainer.__version__,chainerrl=version("chainerrl")))); numpy.show_config()')
    result = subprocess.run([sys.executable, '-c', script], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError('Environment import check failed:\n' + result.stderr)
    return result.stdout, result.stderr


def train(root, output, args, protocol, model):
    model_dir = output / model
    marker = model_dir / 'training-complete.json'
    agent_dir = model_dir / 'agent'
    final = agent_dir / (str(args.steps) + '_finish')
    if marker.exists():
        record = json.loads(marker.read_text())
        for name, h in record['checkpoint_hashes'].items():
            if not (final / name).is_file() or digest(final / name) != h:
                raise RuntimeError('Completed checkpoint changed or missing: ' + str(final / name))
        print('SKIP completed training:', model, flush=True)
        return
    if model_dir.exists():
        archive = output / 'archived-incomplete'
        archive.mkdir(exist_ok=True)
        destination = archive / (model + '-' + str(time.time_ns()))
        model_dir.rename(destination)
        print('Preserved incomplete training:', destination, flush=True)
    model_dir.mkdir()
    command = [sys.executable, '-u', str(output / 'source/train_and_eval_chainer_agent.py'),
               '--outdir', str(agent_dir), '--outdir-time-suffix', 'none',
               '--seed', str(args.seed), '--steps', str(args.steps),
               '--replay-start-size', '100', '--minibatch-size', '32',
               '--checkpoint_frequency', str(args.steps),
               '--eval-interval', str(args.steps + 1),
               '--final-exploration-steps', str(max(1, args.steps // 2)),
               '--max-rand-step', '3', '--state', '2', '--temporal-window', str(MODELS[model]),
               '--num-heuristics', '4', '--n-hidden-channels', '75', '--n-hidden-layers', '2',
               '--port', str(args.port), '--port-file-id', str(args.port),
               '--time-step-limit', str(args.episode_limit)]
    save(model_dir / 'training-command.json', command)
    train_order = list(protocol['train'])
    random.Random(args.seed).shuffle(train_order)
    save(model_dir / 'training-order.json', train_order)
    ready = agent_dir / ('port_{}.txt'.format(args.port))
    start = time.monotonic()
    trainer = planner = None
    records = []
    print('TRAIN', model, 'steps', args.steps, 'episode limit', args.episode_limit, flush=True)
    with (model_dir / 'trainer.log').open('w') as log:
        try:
            train_env = os.environ.copy()
            train_env.pop('RL_FEATURE_DIAGNOSTICS', None)
            if model == 'progress':
                train_env['RL_FEATURE_DIAGNOSTICS'] = str(model_dir / 'feature-diagnostics.json')
            trainer = subprocess.Popen(command, env=train_env, cwd=model_dir, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while trainer.poll() is None and not final.is_dir():
                while not ready.exists():
                    if trainer.poll() is not None or final.is_dir():
                        break
                    if time.monotonic() - start > args.max_train_seconds:
                        raise RuntimeError('Training exceeded its safety limit. Inspect trainer.log.')
                    time.sleep(0.05)
                if trainer.poll() is not None or final.is_dir():
                    break
                if time.monotonic() - start > args.max_train_seconds:
                    raise RuntimeError('Training exceeded its safety limit.')
                problem = train_order[len(records) % len(train_order)]
                episode = model_dir / 'episodes' / '{:04d}'.format(len(records) + 1)
                episode.mkdir(parents=True)
                began = time.monotonic()
                with (episode / 'planner.log').open('w') as plog:
                    planner = subprocess.Popen(planner_command(root, problem, args.seed, args.port, 60),
                        cwd=episode, stdout=plog, stderr=subprocess.STDOUT, start_new_session=True)
                    while planner.poll() is None:
                        if time.monotonic() - began > 65:
                            raise RuntimeError('Training planner exceeded 65s wall safety limit; inspect its log.')
                        if time.monotonic() - start > args.max_train_seconds:
                            raise RuntimeError('Training exceeded its safety limit.')
                        if trainer.poll() not in (None, 0):
                            raise RuntimeError('Trainer failed; inspect trainer.log.')
                        time.sleep(0.02)
                code = planner.returncode
                stop(planner)
                planner = None
                records.append(dict(episode=len(records) + 1, problem=problem,
                                    planner_exit_code=code, wall_seconds=time.monotonic() - began))
                save(model_dir / 'exposure.json', {'episodes': records, 'distinct_problems_launched': len(set(r['problem'] for r in records))})
                if code not in (0, 10, 11, 12, 20, 21, 22, 23, 24, 70):
                    raise RuntimeError('Unexpected training planner exit {}; inspect episode log.'.format(code))
                # Ensure this episode consumed its readiness file before the next launch.
                # fd_env removes it immediately after accepting the connection.
                if ready.exists() and code in (10, 20, 21):
                    raise RuntimeError('Planner exited before connecting; inspect translator log.')
                if len(records) % 10 == 0:
                    print(' ', model, 'episodes', len(records), 'distinct problems', len(set(r['problem'] for r in records)), flush=True)
            trainer.wait(timeout=30)
            if trainer.returncode != 0 or not final.is_dir():
                raise RuntimeError('Training failed or final checkpoint missing; inspect trainer.log.')
        finally:
            try:
                stop(planner)
            finally:
                stop(trainer)
    checkpoints = {name: digest(final / name) for name in ('model.npz', 'target_model.npz', 'optimizer.npz')}
    save(marker, dict(wall_seconds=time.monotonic() - start, checkpoint_hashes=checkpoints,
                     episodes_launched=len(records), distinct_problems_launched=len(set(r['problem'] for r in records))))
    print('TRAIN COMPLETE', model, 'distinct problems launched', len(set(r['problem'] for r in records)), flush=True)


def write_summary(output):
    rows = [json.loads(p.read_text()) for p in sorted((output / 'evaluation').glob('*/*/*/result.json'))]
    if not rows:
        return
    with (output / 'summary.csv').open('w', newline='') as stream:
        columns = sorted({key for r in rows for key in r})
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    for model in MODELS:
        subset = [r for r in rows if r['configuration'] == model]
        done = [r for r in subset if r['completed']]
        print('{}: {}/{} solved; 20 planned; {} incomplete/errors'.format(model,
            sum(r['status'] == 'solved' for r in done), len(done), len(subset) - len(done)), flush=True)


def evaluate(root, output, args, protocol):
    spec = importlib.util.spec_from_file_location('pilot_evaluator', output / 'source/evaluator.py')
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    helper.stop_group = stop
    def controller_command(domain, model, unused):
        cp = output / model / 'agent' / (str(args.steps) + '_finish')
        return [sys.executable, '-u', str(output / 'source/train_and_eval_chainer_agent.py'),
                '--outdir', '.', '--load', str(cp), '--evaluate', '--save-eval-stats', 'eval.json',
                '--seed', str(args.seed), '--state', '2', '--temporal-window', str(MODELS[model]),
                '--num-heuristics', '4', '--n-hidden-channels', '75', '--n-hidden-layers', '2',
                '--port', str(args.port), '--port-file-id', str(args.port),
                '--eval-n-runs', '1', '--time-step-limit', '1000000']
    helper.controller_command = controller_command
    helper.planner_command = lambda domain, problem, model, unused: planner_command(root, problem, args.seed, args.port, args.wall_seconds)
    eval_args = SimpleNamespace(port=args.port, wall_seconds=args.wall_seconds)
    try:
        for problem in protocol['development']:
            for model in MODELS:
                try:
                    helper.run_case('blocksworld', problem, model, eval_args, output / 'evaluation')
                except BaseException as exc:
                    result_path = output / 'evaluation/blocksworld' / Path(problem).stem / model / 'result.json'
                    if result_path.exists():
                        result = json.loads(result_path.read_text())
                        result['completed'] = False
                        result['error'] = str(exc)
                        if result.get('status') == 'solved':
                            result['status'] = 'runner_error'
                        save(result_path, result)
                    raise
            write_summary(output)
    finally:
        write_summary(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', default='results-blocksworld-progress-v1-seed1')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--split-seed', type=int, default=20260926)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--episode-limit', type=int, default=1000)
    parser.add_argument('--wall-seconds', type=int, default=30)
    parser.add_argument('--port', type=int, default=12631)
    parser.add_argument('--max-train-seconds', type=int, default=7200)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.steps < 100 or args.episode_limit < 2 or args.wall_seconds < 1 or args.max_train_seconds < 1:
        parser.error('Invalid positive budgets; steps must be at least 100 and episode limit at least 2.')
    if not 1 <= args.port <= 65535:
        parser.error('Invalid TCP port.')
    root = args.root.resolve(); output = (root / args.output).resolve()
    if root not in output.parents:
        parser.error('Output must be a new subdirectory of the project.')
    output.mkdir(parents=True, exist_ok=True)
    # Also block the existing default full evaluator while this pilot runs.
    locks = []
    try:
        for path in (output / '.runner.lock', root / '.eval-port-{}.lock'.format(args.port), root / '.eval-port-12521.lock'):
            handle = path.open('a'); locks.append(handle)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Another pilot/evaluator is active. Stop it first.')
        protocol = prepare(root, output, args)
        print('Prepared:', output, flush=True)
        print('80 training / 20 development; designated test collection unused.', flush=True)
        print('Original versus progress v1; both trained from scratch. Progress adds 8 features.', flush=True)
        print('Steps/model:', args.steps, 'episode limit:', args.episode_limit, 'development seconds:', args.wall_seconds, flush=True)
        if not args.run:
            print('Preparation complete. Add --run to start training and evaluation.', flush=True)
            return
        stdout, stderr = environment_info()
        envfile = output / 'environment.txt'
        if envfile.exists() and envfile.read_text() != stdout:
            raise RuntimeError('Python package/backend configuration changed; use a new output directory.')
        envfile.write_text(stdout)
        (output / 'environment-warnings.txt').write_text(stderr)
        for model in MODELS:
            train(root, output, args, protocol, model)
        evaluate(root, output, args, protocol)
        print('PILOT COMPLETE. Upload', output / 'summary.csv', flush=True)
    finally:
        for handle in reversed(locks):
            handle.close()


if __name__ == '__main__':
    main()
