#!/usr/bin/env python3
"""Evaluate frozen 30k controllers and fixed heuristics on designated test sets.

Install in rl-plan/experiments. Python 3.8+, standard library only.
A solved result requires planner exit 0, 'Solution found.', and completion
within the end-to-end deadline. This does not independently validate plans.
Controller startup, translation and search count toward the deadline; cleanup
and optional trajectory collection do not. Missing action data stays null.
Run serially. Ctrl-C preserves completed results; repeat command to resume.
--limit selects a lexical prefix per domain, for checking the runner only.
Do not tune the method using these designated test results.
"""
import argparse
from collections import Counter
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ['barman', 'blocksworld', 'childsnack', 'rovers', 'sokoban', 'visitall']
CONFIGS = ['original', 'temporal', 'ff', 'cg', 'cea', 'add']
WINDOWS = {'original': 0, 'temporal': 4}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def checkpoint(domain, model, args):
    return (ROOT / args.experiments / '{}-{}-seed{}-steps{}'.format(
        domain, model, args.seed, args.steps) / 'agent' / '{}_finish'.format(args.steps))


def prepare(args, output):
    files = [Path(__file__).resolve(), ROOT / 'rl-env/fd_env.py',
             ROOT / 'rl-env/evaluate.py', ROOT / 'rl-env/train_and_eval_chainer_agent.py',
             ROOT / 'fast-downward/fast-downward.py',
             ROOT / 'fast-downward/builds/release/bin/downward']
    # Capture driver and translator source too, when present.
    for directory in ('fast-downward/driver', 'fast-downward/src/translate'):
        files.extend(sorted((ROOT / directory).rglob('*.py')))
    problems = {}
    for domain in DOMAINS:
        base = ROOT / 'experiments' / domain / domain
        files.append(base / 'domain.pddl')
        tests = sorted((base / 'test').glob('*.pddl'), key=lambda p: p.name)
        if len(tests) != 100:
            raise RuntimeError('Expected 100 test problems in {}; found {}'.format(base / 'test', len(tests)))
        problems[domain] = [p.name for p in tests]
        files.extend(tests)
        for model in WINDOWS:
            cp = checkpoint(domain, model, args)
            for name in ('model.npz', 'target_model.npz', 'optimizer.npz'):
                files.append(cp / name)
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise RuntimeError('Missing required files:\n' + '\n'.join(missing))
    manifest = dict(protocol='six-domain-wall-v1', seed=args.seed, steps=args.steps,
                    wall_seconds=args.wall_seconds, fd_time_limit=args.time_limit,
                    state=2, hidden_layers=2, hidden_channels=75,
                    temporal_windows=WINDOWS, configurations=CONFIGS,
                    split='test', test_order=problems, python=sys.version,
                    executable=sys.executable,
                    hashes={str(p.relative_to(ROOT)) if ROOT in p.parents else str(p): digest(p)
                            for p in files})
    manifest_file = output / 'protocol.json'
    if manifest_file.exists():
        if json.loads(manifest_file.read_text()) != manifest:
            raise RuntimeError('Protocol/files changed. Use a new output directory; existing results were not changed.')
    else:
        if any(output.glob('*/*/*/result.json')):
            raise RuntimeError('Results exist without protocol.json; refusing to mix results.')
        atomic_json(manifest_file, manifest)
    return problems


def controller_command(domain, model, args):
    return [sys.executable, '-u', str(ROOT / 'rl-env/train_and_eval_chainer_agent.py'),
            '--outdir', '.', '--load', str(checkpoint(domain, model, args)),
            '--evaluate', '--save-eval-stats', 'eval.json', '--seed', str(args.seed),
            '--state', '2', '--temporal-window', str(WINDOWS[model]),
            '--num-heuristics', '4', '--n-hidden-channels', '75', '--n-hidden-layers', '2',
            '--port', str(args.port), '--port-file-id', str(args.port),
            '--eval-n-runs', '1', '--time-step-limit', '1000000']


def planner_command(domain, problem, configuration, args):
    base = ROOT / 'experiments' / domain / domain
    if configuration in WINDOWS:
        search = ('rl_eager(rl([single(ff()),single(cg()),single(cea()),single(add())],'
                  'random_seed={},handle_empty_list=true),rl_control_interval=0,'
                  'rl_client_port={})').format(args.seed, args.port)
    else:
        search = 'eager(single({}()))'.format(configuration)
    return [sys.executable, str(ROOT / 'fast-downward/fast-downward.py'),
            '--overall-time-limit', str(args.time_limit), str(base / 'domain.pddl'),
            str(base / 'test' / problem), '--search', search]


def stop_group(process):
    if process is None or getattr(process, '_cleanup_complete', False):
        return

    def signal_owned_process(sig):
        # A denied group signal does not mean a still-running child is gone.
        if process.poll() is not None:
            return
        try:
            if sig == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            if process.poll() is None:
                raise RuntimeError(
                    'Cannot stop owned subprocess PID {}. '
                    'Inspect it in Activity Monitor before resuming.'
                    .format(process.pid)) from exc

    def signal_group(sig):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            signal_owned_process(sig)
        except PermissionError:
            # macOS may deny a group signal after group members have exited.
            # Fall back only to the Popen child that this runner owns.
            print('Cleanup: group signal denied for PID {}; '
                  'checking owned subprocess.'.format(process.pid), flush=True)
            signal_owned_process(sig)

    signal_group(signal.SIGTERM)
    try:
        process.wait(timeout=0.2)
    except subprocess.TimeoutExpired:
        pass
    # Clean up any descendants even if the wrapper has already exited.
    signal_group(signal.SIGKILL)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('Subprocess PID {} did not stop.'.format(process.pid)) from exc
    process._cleanup_complete = True


def classify(code, text, elapsed, budget, timed_out):
    if timed_out or elapsed > budget:
        return 'external_wall_timeout'
    if code == 0 and 'Solution found.' in text:
        return 'solved'
    if code == 23:
        return 'fd_time_limit'
    if code == 12:
        return 'search_unsolved_incomplete'
    # Unknown planner failures must be inspected, not silently counted as timeouts.
    return 'planner_error'


def parse_metric(pattern, text, cast=int):
    matches = re.findall(pattern, text)
    return cast(matches[-1]) if matches else None


def run_case(domain, problem, configuration, args, output):
    case = output / domain / Path(problem).stem / configuration
    result_file = case / 'result.json'
    if result_file.exists():
        previous = json.loads(result_file.read_text())
        if previous.get('completed'):
            print('SKIP', domain, problem, configuration, flush=True)
            return
    case.mkdir(parents=True, exist_ok=True)
    # Preserve incomplete attempts, including stale readiness/statistics files.
    existing = list(case.iterdir())
    if existing:
        archive = case / ('attempt-' + str(time.time_ns()))
        archive.mkdir()
        for p in existing:
            if not p.name.startswith('attempt-'):
                p.rename(archive / p.name)
    port_file = case / 'port_{}.txt'.format(args.port)
    result = dict(domain=domain, problem=problem, configuration=configuration,
                  completed=False, status='running', planner_exit_code=None,
                  controller_exit_code=None, end_to_end_wall_seconds=None,
                  planner_wall_seconds=None, total_wall_seconds=None,
                  expanded=None, generated=None, plan_length=None, fd_total_seconds=None,
                  action_data_available=False, action_data_note='Unavailable',
                  **{h + '_selections': None for h in CONFIGS[2:]})
    controller = planner = None
    start = time.monotonic()
    deadline = start + args.wall_seconds
    print('RUN ', domain, problem, configuration, flush=True)
    try:
        with (case / 'controller.log').open('w') as clog, (case / 'planner.log').open('w') as plog:
            if configuration in WINDOWS:
                controller = subprocess.Popen(controller_command(domain, configuration, args),
                    cwd=case, stdout=clog, stderr=subprocess.STDOUT, start_new_session=True)
                while not port_file.exists():
                    if controller.poll() is not None:
                        raise RuntimeError('Controller exited before readiness; inspect controller.log')
                    if time.monotonic() >= deadline:
                        raise TimeoutError('End-to-end deadline during controller startup')
                    time.sleep(0.02)
            if time.monotonic() >= deadline:
                raise TimeoutError('End-to-end deadline before planner startup')
            planner_start = time.monotonic()
            planner = subprocess.Popen(planner_command(domain, problem, configuration, args),
                cwd=case, stdout=plog, stderr=subprocess.STDOUT, start_new_session=True)
            timed_out = False
            while planner.poll() is None:
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                if controller is not None and controller.poll() not in (None, 0):
                    raise RuntimeError('Controller crashed; inspect controller.log')
                time.sleep(0.01)
            observed_end = time.monotonic()
            result['end_to_end_wall_seconds'] = observed_end - start
            result['planner_wall_seconds'] = observed_end - planner_start
            if timed_out:
                stop_group(planner)
            result['planner_exit_code'] = planner.poll()
            text = (case / 'planner.log').read_text(errors='replace')
            result['status'] = classify(planner.returncode, text, observed_end-start,
                                        args.wall_seconds, timed_out)
            result['expanded'] = parse_metric(r'Expanded (\d+) state', text)
            result['generated'] = parse_metric(r'Generated (\d+) state', text)
            result['plan_length'] = parse_metric(r'Plan length: (\d+)', text)
            result['fd_total_seconds'] = parse_metric(r'Total time:\s*([\d.]+)s', text, float)
            if result['status'] == 'planner_error':
                raise RuntimeError('Unclassified planner exit {}; inspect planner.log'.format(planner.returncode))
            if controller is not None:
                # Optional statistics only; not part of the success criterion.
                try:
                    controller.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
                result['controller_exit_code'] = controller.poll()
                stats = case / 'eval.json'
                if stats.exists():
                    try:
                        trajectories = json.loads(stats.read_text()).get('trajectories', [])
                        if trajectories:
                            counts = Counter(int(a) for a in trajectories[0])
                            for i, h in enumerate(CONFIGS[2:]):
                                result[h + '_selections'] = counts[i]
                            result['action_data_available'] = True
                            result['action_data_note'] = ('Recorded controller decisions; excludes warm-up; '
                                'may include an action whose reply was lost on disconnect.')
                    except (ValueError, TypeError):
                        result['action_data_note'] = 'Unreadable or incomplete eval.json'
            result['completed'] = True
    except TimeoutError as exc:
        result.update(status='external_wall_timeout', completed=True, error=str(exc))
        result['end_to_end_wall_seconds'] = time.monotonic() - start
    except KeyboardInterrupt:
        result.update(status='interrupted', completed=False)
        raise
    except Exception as exc:
        if result['status'] != 'planner_error':
            result['status'] = 'runner_error'
        result['error'] = str(exc)
        raise
    finally:
        try:
            stop_group(planner)
        finally:
            try:
                stop_group(controller)
            finally:
                if controller is not None:
                    result['controller_exit_code'] = controller.poll()
                result['total_wall_seconds'] = time.monotonic() - start
                atomic_json(result_file, result)
    print('   {} ({:.2f}s)'.format(result['status'], result['end_to_end_wall_seconds']), flush=True)


def summary(output):
    rows = [json.loads(p.read_text()) for p in sorted(output.glob('*/*/*/result.json'))]
    if not rows:
        return
    columns = sorted({k for r in rows for k in r})
    with (output / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    coverage = []
    for domain in DOMAINS:
        for config in CONFIGS:
            subset = [r for r in rows if r['domain'] == domain and r['configuration'] == config]
            completed = [r for r in subset if r['completed']]
            solved = sum(r['status'] == 'solved' for r in completed)
            coverage.append(dict(domain=domain, configuration=config, solved=solved,
                                 completed=len(completed), planned=100,
                                 errors_or_interrupted=len(subset)-len(completed)))
    with (output / 'coverage.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(coverage[0]))
        writer.writeheader()
        writer.writerows(coverage)
    print('\nCoverage so far (completed cases only):', flush=True)
    for row in coverage:
        if row['completed'] or row['errors_or_interrupted']:
            print('{domain:12} {configuration:9} {solved}/{completed} solved; '
                  '{planned} planned; {errors_or_interrupted} errors/interrupted'.format(**row))
    print('Summary:', output / 'summary.csv', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', choices=DOMAINS + ['all'], default='all')
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--wall-seconds', type=float, default=60)
    parser.add_argument('--time-limit', type=int, default=60)
    parser.add_argument('--port', type=int, default=12521)
    parser.add_argument('--experiments', default='results-cross-domain-auto')
    parser.add_argument('--output', default='results-six-domain-test-30k-seed1-wall60')
    args = parser.parse_args()
    if not 1 <= args.limit <= 100 or args.wall_seconds <= 0 or args.time_limit < 1:
        parser.error('limit must be 1..100 and time budgets positive')
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Lock the port across output folders, preventing competing local runners.
    with (ROOT / '.eval-port-{}.lock'.format(args.port)).open('a') as portlock, \
            (output / '.runner.lock').open('a') as lock:
        for handle in (portlock, lock):
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('An evaluator already holds this port or output directory.')
        problems = prepare(args, output)
        domains = DOMAINS if args.domain == 'all' else [args.domain]
        print('Designated TEST evaluation; six configurations; limit per domain:', args.limit, flush=True)
        try:
            for domain in domains:
                for problem in problems[domain][:args.limit]:
                    for config in CONFIGS:
                        run_case(domain, problem, config, args, output)
                    summary(output)
        finally:
            summary(output)


if __name__ == '__main__':
    main()
