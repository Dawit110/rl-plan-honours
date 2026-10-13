#!/usr/bin/env python3
"""Resumable full100 evaluation: 4 workers, 300 aggregate CPU seconds per case."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import contextlib
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import preflight as pf

HERE = Path(__file__).resolve().parent
CPU = 300
ACCEPTED = ('solved', 'cpu_timeout', 'search_unsolved')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def configuration(task):
    return ('%s-seed%d-%d' % (task['model'], task['seed'], task['budget'])
            if task['model'] in pf.MODELS else task['model'])


def case_dir(output, task):
    return output / task['phase'] / Path(task['problem']).stem / configuration(task)


def config_for(root, task, port, args):
    domain, model, problem = task['domain'], task['model'], task['problem']
    base = root / 'experiments' / domain / domain
    controller = None
    if model in pf.MODELS:
        folder = pf.trained(root, domain, model, task['seed'])
        controller = [sys.executable, '-u', str(folder / 'source/train_and_eval_chainer_agent.py'),
            '--outdir', '.', '--load', str(pf.checkpoint(folder, model, task['budget'])),
            '--evaluate', '--save-eval-stats', 'eval.json', '--seed', str(task['seed']), '--state', '2',
            '--temporal-window', '0' if model == 'original' else '4', '--num-heuristics', '4',
            '--n-hidden-channels', '75', '--n-hidden-layers', '2', '--port', str(port),
            '--port-file-id', str(port), '--eval-n-runs', '1', '--time-step-limit', '1000000']
        search = ('rl_eager(rl([single(ff()),single(cg()),single(cea()),single(add())],'
            'random_seed=%d,handle_empty_list=true),rl_control_interval=0,rl_client_port=%d)' % (task['seed'], port))
    else:
        search = 'eager(single(%s()))' % model
    planner = [sys.executable, str(root / 'fast-downward/fast-downward.py'),
        str(base / 'domain.pddl'), str(base / ('train' if task['phase'] == 'selection' else 'test') / problem),
        '--search', search]
    return dict(task=task, controller=controller, planner=planner, port=port,
        cpu_seconds=CPU, wall_safeguard=args.wall_safeguard)


def valid(row, task, args):
    pf.require(row.get('config', {}).get('task') == task, 'Result has a different task/protocol')
    pf.require(row['config']['cpu_seconds'] == CPU and row['config']['wall_safeguard'] == args.wall_safeguard,
               'Result has a different resource budget')
    if not row.get('completed') or row.get('status') not in ACCEPTED:
        return False
    cpu = row.get('cpu_seconds')
    pf.require(isinstance(cpu, (int, float)) and math.isfinite(cpu) and cpu >= 0, 'Invalid measured CPU time')
    pf.require(row.get('cpu_overshoot_seconds', 0) <= max(.5, CPU * .02), 'Excessive CPU-budget overshoot')
    if row['status'] == 'solved':
        pf.require(cpu < CPU and row['planner_exit_code'] == 0, 'Invalid success outside the CPU budget')
    if row['status'] == 'cpu_timeout':
        pf.require(cpu >= CPU, 'Premature CPU timeout')
    return True


def summarize(output):
    rows = []
    for phase in ('selection', 'test'):
        for path in sorted((output / phase).glob('*/*/result.json')):
            record = pf.read(path)
            task = record.get('config', {}).get('task', {})
            row = {k: v for k, v in record.items() if k not in ('config', 'traceback')}
            row.update(task)
            if task:
                row['configuration'] = configuration(task)
            row['phase'] = phase
            row['completed'] = bool(row.get('completed') and row.get('status') in ACCEPTED)
            rows.append(row)
    if not rows:
        return
    groups = {}
    for row in rows:
        key = (row['phase'], row.get('configuration', 'unknown'))
        group = groups.setdefault(key, dict(domain=output.name, phase=key[0], configuration=key[1],
            planned=20 if key[0] == 'selection' else 100, completed=0, solved=0,
            cpu_timeouts=0, errors_or_interrupted=0, capped_cpu_sum=0.0))
        if row['completed']:
            group['completed'] += 1
            group['solved'] += row['status'] == 'solved'
            group['cpu_timeouts'] += row['status'] == 'cpu_timeout'
            group['capped_cpu_sum'] += min(CPU, row['cpu_seconds']) if row['status'] == 'solved' else CPU
        else:
            group['errors_or_interrupted'] += 1
    for name, data in [('results', rows), ('coverage', list(groups.values()))]:
        temp = output / (name + '.tmp')
        with temp.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted({k for row in data for k in row}))
            writer.writeheader()
            writer.writerows(data)
        temp.replace(output / (name + '.csv'))


def command_output(command):
    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True, timeout=4).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return 'unavailable: ' + str(exc)


def memory_state():
    return dict(timestamp=time.time(), swap=command_output(['sysctl', 'vm.swapusage']),
        vm_stat=command_output(['vm_stat']),
        pressure=command_output(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level']))


class Batch:
    def __init__(self, root, output, args):
        self.root, self.output, self.args = root, output, args
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.active = {}
        self.ports = queue.Queue()
        for port in range(args.port, args.port + args.workers):
            self.ports.put(port)

    def execute(self, task):
        if self.stop.is_set():
            return None
        port = self.ports.get()
        try:
            directory = case_dir(self.output, task)
            directory.mkdir(parents=True, exist_ok=True)
            existing = [p for p in directory.iterdir() if not p.name.startswith('attempt-')]
            if existing:
                archive = directory / ('attempt-' + str(time.time_ns()))
                archive.mkdir()
                for path in existing:
                    path.rename(archive / path.name)
            config = config_for(self.root, task, port, self.args)
            save(directory / 'config.json', config)
            print('RUN  %s %s %s %s' % (task['phase'], task['domain'], task['problem'], configuration(task)), flush=True)
            with (directory / 'supervisor.log').open('w') as log:
                with self.lock:
                    if self.stop.is_set():
                        return None
                    process = subprocess.Popen([sys.executable, '-u', str(HERE / 'case.py'), str(directory / 'config.json')],
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    self.active[process.pid] = process
                try:
                    exit_code = process.wait(timeout=self.args.wall_safeguard + 30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise RuntimeError('Case supervisor hung: ' + str(directory))
                finally:
                    with self.lock:
                        self.active.pop(process.pid, None)
            path = directory / 'result.json'
            row = pf.read(path)
            try:
                pf.require(exit_code == 0 and valid(row, task, self.args), 'Case failed: ' + str(path))
            except BaseException as exc:
                row.update(completed=False, validation_error=str(exc))
                save(path, row)
                raise
            print('DONE %s %s %s: %s; %.2fs CPU / %.2fs elapsed' % (task['domain'], task['problem'],
                configuration(task), row['status'], row['cpu_seconds'], row['wall_seconds']), flush=True)
            return row
        except BaseException:
            self.stop.set()
            raise
        finally:
            self.ports.put(port)

    def cancel(self):
        self.stop.set()
        with self.lock:
            processes = list(self.active.values())
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

    def run(self, tasks):
        keys = [str(case_dir(self.output, task)) for task in tasks]
        pf.require(len(keys) == len(set(keys)), 'Duplicate scheduled case')
        todo = []
        for task in tasks:
            path = case_dir(self.output, task) / 'result.json'
            if path.exists() and valid(pf.read(path), task, self.args):
                continue
            todo.append(task)
        print('BATCH: %d planned, %d already complete, %d to run; %d workers' %
              (len(tasks), len(tasks) - len(todo), len(todo), self.args.workers), flush=True)
        summarize(self.output)
        if not todo:
            return
        pool = ThreadPoolExecutor(max_workers=self.args.workers)
        iterator = iter(todo)
        pending = set()
        next_memory = 0
        critical_since = None
        completed = len(tasks) - len(todo)
        def submit():
            if not self.stop.is_set():
                task = next(iterator, None)
                if task is not None:
                    pending.add(pool.submit(self.execute, task))
        try:
            for _ in range(self.args.workers):
                submit()
            while pending:
                ready, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in ready:
                    row = future.result()
                    pf.require(row is not None, 'Batch interrupted')
                    completed += 1
                    submit()
                if ready:
                    summarize(self.output)
                    save(self.output / 'progress.json', dict(domain=self.output.name,
                        phase=tasks[0]['phase'], seed=tasks[0]['seed'],
                        batch_completed=completed, batch_planned=len(tasks), updated=time.time(), workers=self.args.workers))
                now = time.monotonic()
                if now >= next_memory:
                    sample = memory_state()
                    with (self.output / 'memory.jsonl').open('a') as stream:
                        stream.write(json.dumps(sample) + '\n')
                    if sample['pressure'] == '4':
                        critical_since = critical_since or now
                        if now - critical_since >= 10:
                            raise RuntimeError('Sustained critical memory pressure. Completed results saved; resume after resolving the pressure.')
                    else:
                        critical_since = None
                    next_memory = now + 10
        except BaseException:
            self.cancel()
            raise
        finally:
            pool.shutdown(wait=True)
            summarize(self.output)


def task(domain, phase, problem, model, seed, budget):
    return dict(domain=domain, phase=phase, problem=problem, model=model, seed=seed, budget=budget)


def selection_tasks(manifest, seed):
    return [task(manifest['domain'], 'selection', problem, model, seed, budget)
        for problem in manifest['training_selection'][str(seed) + '-original']
        for budget in pf.BUDGETS for model in pf.MODELS]


def rank(rows, budget):
    pf.require(len(rows) == 20 and all(r['completed'] and r['status'] in ACCEPTED for r in rows), 'Selection incomplete')
    return (-sum(r['status'] == 'solved' for r in rows),
        sum(min(CPU, r['cpu_seconds']) if r['status'] == 'solved' else CPU for r in rows), budget)


def select(output, manifest, seed, args):
    selected = {}
    for model in pf.MODELS:
        ranks = []
        for budget in pf.BUDGETS:
            rows = []
            for problem in manifest['training_selection'][str(seed) + '-' + model]:
                request = task(manifest['domain'], 'selection', problem, model, seed, budget)
                row = pf.read(case_dir(output, request) / 'result.json')
                pf.require(valid(row, request, args), 'Selection result invalid')
                rows.append(row)
            ranks.append(rank(rows, budget))
        best = min(ranks)[2]
        decision = dict(domain=manifest['domain'], model=model, seed=seed, selected_budget=best,
            selection_role='training subset; not held out', cpu_budget=CPU,
            rule='coverage; capped CPU-time sum; earliest checkpoint',
            candidates=[dict(budget=b, solved=-n, capped_cpu_sum=t) for n, t, b in ranks])
        path = output / ('selected-%s-seed%d.json' % (model, seed))
        if path.exists():
            pf.require(pf.read(path) == decision, 'Saved checkpoint decision changed: ' + str(path))
        else:
            save(path, decision)
        selected[model] = best
        print('SELECTED %s %s seed%d: %d' % (manifest['domain'], model, seed, best), flush=True)
    return selected


def test_tasks(manifest, seed, selected):
    tasks = []
    for problem in manifest['test']:
        for model in pf.MODELS:
            tasks.append(task(manifest['domain'], 'test', problem, model, seed, selected[model]))
        if seed == 1:
            for model in pf.FIXED:
                tasks.append(task(manifest['domain'], 'test', problem, model, 1, None))
    return tasks


def run_domain(root, output, manifest, args):
    for seed in pf.SEEDS:
        print('\nCHECKPOINT SELECTION %s seed%d -- 300 CPU seconds' % (manifest['domain'], seed), flush=True)
        Batch(root, output, args).run(selection_tasks(manifest, seed))
        selected = select(output, manifest, seed, args)
        print('\nTEST %s seed%d -- all 100 test problems' % (manifest['domain'], seed), flush=True)
        Batch(root, output, args).run(test_tasks(manifest, seed, selected))
        print('TEST SEED COMPLETE %s seed%d -- %s' % (manifest['domain'], seed, output / 'coverage.csv'), flush=True)
    for relative, expected in manifest['hashes'].items():
        pf.require(pf.sha(root / relative) == expected, 'Input changed during evaluation: ' + relative)
    save(output / 'evaluation-complete.json', dict(domain=manifest['domain'], selection_cases=480,
        test_cases=1000, seeds=list(pf.SEEDS), workers=args.workers, cpu_seconds=CPU,
        protocol_sha256=pf.sha(output / 'protocol.json')))
    print('DOMAIN COMPLETE:', manifest['domain'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path.cwd())
    p.add_argument('--domains', nargs='+', choices=pf.DOMAINS, required=True)
    p.add_argument('--workers', type=int, choices=(4,), default=4)
    p.add_argument('--port', type=int, default=13041)
    p.add_argument('--wall-safeguard', type=int, default=1800)
    p.add_argument('--output', type=Path, default=Path('results-full100-cpu-v1'))
    p.add_argument('--run', action='store_true')
    args = p.parse_args()
    pf.require(sys.platform == 'darwin', 'This launch script is for macOS')
    pf.require(sys.version_info[:2] == (3, 8), 'Activate the Python 3.8 rl-plan environment')
    pf.require(len(args.domains) == len(set(args.domains)), 'Duplicate domains')
    pf.require(1024 <= args.port <= 65531, 'Invalid first port')
    pf.require(args.wall_safeguard >= 1200, 'Wall safeguard must be at least 1200 seconds')
    for key in pf.THREADS:
        os.environ[key] = '1'
    os.environ.pop('RL_FEATURE_DIAGNOSTICS', None)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    root = args.root.resolve()
    output = (root / args.output).resolve()
    ps = command_output(['ps', '-axo', 'pid=,command='])
    pf.require(not ps.startswith('unavailable:'), 'Cannot inspect existing planner jobs')
    conflicts = [line for line in ps.splitlines() if any(s in line for s in (
        'train_and_eval_chainer_agent.py', '/bin/downward', 'fast-downward.py'))]
    pf.require(not conflicts, 'Existing planner/controller is running; stop it normally first:\n' + '\n'.join(conflicts))
    with contextlib.ExitStack() as stack:
        for name in ['.full100-evaluation.lock', '.cpu-worker-check.lock'] + [
                '.eval-port-%d.lock' % port for port in range(args.port, args.port + args.workers)]:
            handle = stack.enter_context((root / name).open('a'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for port in range(args.port, args.port + args.workers):
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', port))
        bundles = pf.trainers(root)
        manifests = {}
        for domain in args.domains:
            manifest = pf.check(root, domain, bundles, args)
            existing = output / domain / 'protocol.json'
            if existing.exists():
                pf.require(pf.read(existing) == manifest, 'Recorded CPU protocol/files/environment differ: ' + str(existing))
            else:
                pf.require(not existing.parent.exists() or not any(existing.parent.iterdir()),
                           'Nonempty output without a CPU protocol: ' + str(existing.parent))
            manifests[domain] = manifest
            print('PREFLIGHT OK %s: seeds 1/2/3; 24 checkpoints; 4 workers; 300 aggregate CPU seconds' % domain, flush=True)
        if not args.run:
            print('Preflight complete. Add --run to launch full evaluation.', flush=True)
            return
        output.mkdir(parents=True, exist_ok=True)
        print('FULL CPU EVALUATION STARTING -- no benchmark stages', flush=True)
        try:
            for domain in args.domains:
                destination = output / domain
                manifest = manifests[domain]
                pf.require(pf.check(root, domain, bundles, args) == manifest, 'Inputs changed while waiting: ' + domain)
                if not (destination / 'protocol.json').exists():
                    save(destination / 'protocol.json', manifest)
                run_domain(root, destination, manifest, args)
            print('ALL ASSIGNED DOMAINS COMPLETE:', output, flush=True)
        except BaseException as exc:
            path = output / ('stopped-' + time.strftime('%Y%m%d-%H%M%S') + '.json')
            save(path, dict(error=str(exc), traceback=traceback.format_exc()))
            print('STOPPED; completed cases preserved. Details:', path, flush=True)
            raise


if __name__ == '__main__':
    main()
