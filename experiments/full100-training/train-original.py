#!/usr/bin/env python3
"""Full-100 training only. Python 3.8+.
Prepare by default; --run trains one model and saves four checkpoints.
Uses all 100 supplied training instances; never reads the test collection.
Twenty training instances are preselected for later TRAINING-BASED checkpoint
selection (not held-out validation). Candidates: 100k, 300k, 600k, 1m.
Later selection: coverage at 300s, capped runtime, then earliest checkpoint.
This adapted protocol is not an exact reproduction of the paper.
Learning settings and embedded controller sources match the audited pilot.
Completed runs are verified and skipped; partial training is preserved.
Snapshots omit replay memory and are not exact training-resume checkpoints.
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
DOMAIN = 'barman'
MODELS = {'original': 0}
BUDGETS = (100000, 300000, 600000, 1000000)
EVAL_MODELS = ['original_{}'.format(n) for n in BUDGETS]


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
    base = root / 'experiments' / DOMAIN / DOMAIN
    problems = sorted((base / 'train').glob('*.pddl'), key=lambda p: p.name)
    if len(problems) != 100:
        raise RuntimeError('Expected exactly 100 training problems for the selected domain.')
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
    selection = sorted(p.name for p in shuffled[:20])
    train = sorted(p.name for p in problems)
    assert len(train) == 100 and set(selection).issubset(train)
    hashes = {str(p.relative_to(root)) if root in p.parents else str(p): digest(p) for p in paths}
    protocol = dict(version='full100-original-v1', domain=DOMAIN, root=str(root), python=sys.version,
                    executable=sys.executable, seed=args.seed, split_seed=args.split_seed,
                    steps=args.steps, episode_limit=args.episode_limit,
                    exploration_steps=args.steps // 2, train_time_limit=300, checkpoint_budgets=list(BUDGETS), selection_rule='coverage; capped runtime sum; earliest checkpoint',
                    eval_wall_seconds=args.wall_seconds, train=train, training_selection=selection, selection_data_role='training subset; not held out',
                    models=MODELS, hashes=hashes, feature_spec='original one-step differences only',
                    scope='All 100 instances used for training. Twenty of these reserved only as a selection LIST, not withheld from training. Test collection unused. No evaluations run by this script.')
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
    save(output / 'training-set.json', {'training': train, 'training_selection': selection, 'selection_seed': args.split_seed, 'selection_is_held_out': False})
    return protocol


def search(seed, port):
    return ('rl_eager(rl([single(ff()),single(cg()),single(cea()),single(add())],'
            'random_seed={},handle_empty_list=true),rl_control_interval=0,rl_client_port={})').format(seed, port)


def planner_command(root, problem, seed, port, limit):
    base = root / 'experiments' / DOMAIN / DOMAIN
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
        verify_snapshots(output)
        print('SKIP completed training:', model, flush=True)
        return
    if model_dir.exists():
        raise RuntimeError('Incomplete training preserved in ' + str(model_dir) +
                           '. Do not restart blindly; inspect trainer.log and snapshots first.')
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
                    planner = subprocess.Popen(planner_command(root, problem, args.seed, args.port, 300),
                        cwd=episode, stdout=plog, stderr=subprocess.STDOUT, start_new_session=True)
                    while planner.poll() is None:
                        if time.monotonic() - began > 315:
                            raise RuntimeError('Training planner exceeded 315s wall safety limit; inspect its log.')
                        if time.monotonic() - start > args.max_train_seconds:
                            raise RuntimeError('Training exceeded its safety limit.')
                        if trainer.poll() not in (None, 0):
                            raise RuntimeError('Trainer failed; inspect trainer.log.')
                        time.sleep(0.02)
                code = planner.returncode
                stop(planner)
                planner = None
                planner_text = (episode / 'planner.log').read_text(errors='replace')
                records.append(dict(solution_marker=('Solution found.' in planner_text),
                                    agent_cutoff_marker=('Termination due to RL agent' in planner_text),
                                    episode=len(records) + 1, problem=problem,
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
    record_snapshots(output)
    checkpoints = {name: digest(final / name) for name in ('model.npz', 'target_model.npz', 'optimizer.npz')}
    save(marker, dict(wall_seconds=time.monotonic() - start, checkpoint_hashes=checkpoints,
                     episodes_launched=len(records), distinct_problems_launched=len(set(r['problem'] for r in records))))
    print('TRAIN COMPLETE', model, 'distinct problems launched', len(set(r['problem'] for r in records)), flush=True)






def snapshot_path(output, budget):
    suffix = '_finish' if budget == 1000000 else '_snapshot'
    return output / 'original/agent' / (str(budget) + suffix)


def record_snapshots(output):
    manifest = {}
    for budget in BUDGETS:
        cp = snapshot_path(output, budget)
        for name in ('model.npz', 'target_model.npz', 'optimizer.npz'):
            path = cp / name
            if not path.is_file():
                raise RuntimeError('Missing retained checkpoint file: ' + str(path))
            manifest[str(path.relative_to(output))] = digest(path)
    save(output / 'snapshot-hashes.json', manifest)


def verify_snapshots(output):
    manifest = json.loads((output / 'snapshot-hashes.json').read_text())
    for relative, expected in manifest.items():
        path = output / relative
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError('Retained snapshot changed: ' + str(path))




def main():
    global DOMAIN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', default=None)
    parser.add_argument('--domain', required=True, choices=['barman', 'blocksworld', 'rovers', 'childsnack', 'sokoban', 'visitall'])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--split-seed', type=int, default=20260926)
    parser.add_argument('--steps', type=int, choices=[1000000], default=1000000)
    parser.add_argument('--episode-limit', type=int, default=7500)
    parser.add_argument('--wall-seconds', type=int, default=300)
    parser.add_argument('--port', type=int, default=12631)
    parser.add_argument('--max-train-seconds', type=int, default=86400)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    DOMAIN = args.domain
    if args.output is None:
        args.output = 'results-{}-original-full100-v1-seed{}'.format(DOMAIN, args.seed)
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
        print('100 training problems; 20 training-selection problems (NOT held out); test collection unused.', flush=True)
        print('TRAINING ONLY: retain 100k, 300k, 600k and 1m; no evaluations now.', flush=True)
        print('Steps/model:', args.steps, 'episode limit:', args.episode_limit, 'later selection seconds:', args.wall_seconds, flush=True)
        if not args.run:
            print('Preparation complete. Add --run to start training only.', flush=True)
            return
        stdout, stderr = environment_info()
        envfile = output / 'environment.txt'
        if envfile.exists() and envfile.read_text() != stdout:
            raise RuntimeError('Python package/backend configuration changed; use a new output directory.')
        envfile.write_text(stdout)
        (output / 'environment-warnings.txt').write_text(stderr)
        for model in MODELS:
            train(root, output, args, protocol, model)
        print('TRAINING ONLY COMPLETE:', output, flush=True)
    finally:
        for handle in reversed(locks):
            handle.close()


if __name__ == '__main__':
    main()
