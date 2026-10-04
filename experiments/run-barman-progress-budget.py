#!/usr/bin/env python3
"""Progress-only Barman large-budget diagnostic. Python 3.8+.
Prepare by default; --run trains and then evaluates retained checkpoints.
Unchanged progress v1 features; same schedule as original budget diagnostic.
One million learner interactions (approximately one million updates), maximum
7500 learner interactions per episode, 300s planner training time limit.
One continuous run; retained policy snapshots at 100k, 300k, 600k and final 1m.
Snapshots are for evaluation, not exact training resumption (no replay memory).
Epsilon decays 1.0 to 0.1 over 500k interactions. Other learner settings retained.
This is paper-informed, NOT an exact paper reproduction or a one-factor ablation.
Each snapshot is evaluated on the same 20 development tasks with a 300s deadline.
Selection: highest coverage, then smallest sum of solved runtimes plus 300s for
each unsolved task, then earliest checkpoint. Designated test set is never used.
No fixed-policy results at 300s are supplied: compare under matched limits later.
Interrupted training is preserved and is NOT silently restarted. Completed
training and completed evaluation cases can be reused by rerunning --run.
Synthetic tests cannot validate legacy Chainer learning on the user's Mac.
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
MODELS = {'progress': 4}
BUDGETS = (100000, 300000, 600000, 1000000)
EVAL_MODELS = ['progress_{}'.format(n) for n in BUDGETS]


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
    base = root / 'experiments/barman/barman'
    problems = sorted((base / 'train').glob('*.pddl'), key=lambda p: p.name)
    if len(problems) != 100:
        raise RuntimeError('Expected exactly 100 original Barman training problems.')
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
    protocol = dict(version='barman-progress-budget-v1', root=str(root), python=sys.version,
                    executable=sys.executable, seed=args.seed, split_seed=args.split_seed,
                    steps=args.steps, episode_limit=args.episode_limit,
                    exploration_steps=args.steps // 2, train_time_limit=300, checkpoint_budgets=list(BUDGETS), selection_rule='coverage; capped runtime sum; earliest checkpoint',
                    eval_wall_seconds=args.wall_seconds, train=train, development=dev,
                    models=MODELS, hashes=hashes, feature_spec='unchanged progress v1: relative new-best window=4, age cap=1000, invalid=[0,1]',
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
    base = root / 'experiments/barman/barman'
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


def write_summary(output):
    rows = [json.loads(p.read_text()) for p in sorted((output / 'evaluation').glob('*/*/*/result.json'))]
    if not rows:
        return
    with (output / 'summary.csv').open('w', newline='') as stream:
        columns = sorted({key for r in rows for key in r})
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    for model in EVAL_MODELS:
        subset = [r for r in rows if r['configuration'] == model]
        done = [r for r in subset if r['completed']]
        print('{}: {}/{} solved; 20 planned; {} incomplete/errors'.format(model,
            sum(r['status'] == 'solved' for r in done), len(done), len(subset) - len(done)), flush=True)


def evaluate(root, output, args, protocol):
    spec = importlib.util.spec_from_file_location('pilot_evaluator', output / 'source/evaluator.py')
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    helper.stop_group = stop
    helper.WINDOWS = {name: 4 for name in EVAL_MODELS}
    verify_snapshots(output)
    def controller_command(domain, model, unused):
        cp = snapshot_path(output, int(model.split('_')[-1]))
        return [sys.executable, '-u', str(output / 'source/train_and_eval_chainer_agent.py'),
                '--outdir', '.', '--load', str(cp), '--evaluate', '--save-eval-stats', 'eval.json',
                '--seed', str(args.seed), '--state', '2', '--temporal-window', '4',
                '--num-heuristics', '4', '--n-hidden-channels', '75', '--n-hidden-layers', '2',
                '--port', str(args.port), '--port-file-id', str(args.port),
                '--eval-n-runs', '1', '--time-step-limit', '1000000']
    helper.controller_command = controller_command
    helper.planner_command = lambda domain, problem, model, unused: planner_command(root, problem, args.seed, args.port, args.wall_seconds)
    eval_args = SimpleNamespace(port=args.port, wall_seconds=args.wall_seconds)
    try:
        for problem in protocol['development']:
            for model in EVAL_MODELS:
                try:
                    helper.run_case('barman', problem, model, eval_args, output / 'evaluation')
                except BaseException as exc:
                    result_path = output / 'evaluation/barman' / Path(problem).stem / model / 'result.json'
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


def snapshot_path(output, budget):
    suffix = '_finish' if budget == 1000000 else '_snapshot'
    return output / 'progress/agent' / (str(budget) + suffix)


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


def select_checkpoint(output, args):
    rows = list(csv.DictReader((output / 'summary.csv').open()))
    curve = []
    for budget, model in zip(BUDGETS, EVAL_MODELS):
        cases = [r for r in rows if r['configuration'] == model]
        if len(cases) != 20 or any(r['completed'] != 'True' for r in cases):
            raise RuntimeError('Checkpoint selection requires 20 completed cases per checkpoint.')
        solved = [r for r in cases if r['status'] == 'solved']
        curve.append(dict(interactions=budget, solved=len(solved), total=20,
            capped_runtime_sum=sum(min(args.wall_seconds, float(r['end_to_end_wall_seconds']))
                if r['status']=='solved' else args.wall_seconds for r in cases)))
    winner = min(curve, key=lambda r: (-r['solved'], r['capped_runtime_sum'], r['interactions']))
    save(output / 'learning-curve.json', dict(checkpoints=curve,
        selected_interactions=winner['interactions'],
        selected_checkpoint=str(snapshot_path(output, winner['interactions'])),
        selection='Development coverage, then capped runtime sum, then earliest checkpoint.',
        scope='Development selection only. No independent test performance claimed.'))
    print('Selected development checkpoint:', winner['interactions'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', default='results-barman-progress-budget-v1-seed1')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--split-seed', type=int, default=20260926)
    parser.add_argument('--steps', type=int, choices=[1000000], default=1000000)
    parser.add_argument('--episode-limit', type=int, default=7500)
    parser.add_argument('--wall-seconds', type=int, default=300)
    parser.add_argument('--port', type=int, default=12631)
    parser.add_argument('--max-train-seconds', type=int, default=86400)
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
        print('Progress only: snapshots at 100k, 300k, 600k and 1m; 80 development cases.', flush=True)
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
        select_checkpoint(output, args)
        print('PILOT COMPLETE. Upload', output / 'summary.csv', flush=True)
    finally:
        for handle in reversed(locks):
            handle.close()


if __name__ == '__main__':
    main()
