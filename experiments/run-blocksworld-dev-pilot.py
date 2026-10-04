#!/usr/bin/env python3
"""Isolated Blocksworld training/development pilot. Python 3.8+.
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
MODELS = {'original': 0, 'temporal': 4}


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
    protocol = dict(version='blocksworld-dev-pilot-v1', root=str(root), python=sys.version,
                    executable=sys.executable, seed=args.seed, split_seed=args.split_seed,
                    steps=args.steps, episode_limit=args.episode_limit,
                    exploration_steps=args.steps // 2, train_time_limit=60,
                    eval_wall_seconds=args.wall_seconds, train=train, development=dev,
                    models=MODELS, hashes=hashes,
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
              'chainer=chainer.__version__,chainerrl=version(\"chainerrl\")))); numpy.show_config()')
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
            trainer = subprocess.Popen(command, cwd=model_dir, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
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
    parser.add_argument('--output', default='results-blocksworld-dev-pilot-v1')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--split-seed', type=int, default=20260926)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--episode-limit', type=int, default=500)
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
        print('Models train from scratch; existing 30k checkpoints are not loaded.', flush=True)
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
