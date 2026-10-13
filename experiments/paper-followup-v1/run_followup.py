#!/usr/bin/env python3
"""Separate follow-up: fixed four-list policies, faithful learning settings, replacement inputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import contextlib
import csv
import fcntl
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import queue
import signal
import socket
import subprocess
import sys
import threading
import time

import engine
import preflight as pf

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / 'runtime'
MODELS = ('original', 'replacement')
FIXED = ('ff', 'cg', 'cea', 'add')
CAP = 4 * 1024**3
ACTIVE = {}
ACTIVE_LOCK = threading.Lock()
STOP = threading.Event()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def immutable(path, value):
    if path.exists():
        check(pf.read(path) == value, 'Recorded inputs/protocol differ: ' + str(path))
    else:
        engine.save(path, value)


def search(seed, port, fixed=False):
    return ('rl_eager(rl([single(ff()),single(cg()),single(cea()),single(add())],'
            'random_seed=%d,handle_empty_list=%s),rl_control_interval=0,rl_client_port=%d)' % (
                seed, 'false' if fixed else 'true', port))


def model_dir(root, domain, model, seed):
    return root / 'results-paper-followup-v1' / 'training' / domain / ('%s-seed%d' % (model, seed))


def training_command(folder, model, seed, port):
    return [sys.executable, '-u', str(RUNTIME / 'train_and_eval_chainer_agent.py'),
        '--outdir', str(folder / 'agent'), '--outdir-time-suffix', 'none',
        '--seed', str(seed), '--steps', '1000000', '--time-step-limit', '7500',
        '--state', '2', '--temporal-window', '0' if model == 'original' else '4',
        '--num-heuristics', '4', '--max-rand-step', '3',
        '--n-hidden-channels', '75', '--n-hidden-layers', '2',
        '--final-exploration-steps', '500000', '--start-epsilon', '1', '--end-epsilon', '0.1',
        '--replay-start-size', '1000', '--minibatch-size', '32',
        '--gamma', '0.99', '--update-interval', '1', '--target-update-interval', '500',
        '--target-update-method', 'hard', '--eval-interval', '30000', '--eval-n-runs', '100',
        '--checkpoint_frequency', '100000', '--port', str(port), '--port-file-id', str(port)]


def stop_process(process):
    if process is None:
        return
    # All registered children are private sessions created by this runner.
    for sig, delay in ((signal.SIGTERM, 2), (signal.SIGKILL, 2)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            members = subprocess.check_output(['ps', '-axo', 'pid=,pgid='], text=True)
            for line in members.splitlines():
                pid, pgid = map(int, line.split())
                if pgid == process.pid:
                    try:
                        if os.getpgid(pid) == process.pid:
                            os.kill(pid, sig)
                    except ProcessLookupError:
                        pass
        try:
            process.wait(timeout=delay)
        except subprocess.TimeoutExpired:
            continue
        # A reaped driver can still have live descendants, so also perform KILL.
    members = subprocess.check_output(['ps', '-axo', 'pid=,pgid=,stat='], text=True)
    live = [line for line in members.splitlines()
            if len(line.split()) == 3 and int(line.split()[1]) == process.pid and
            not line.split()[2].startswith('Z')]
    check(not live, 'Owned process group survived cleanup: ' + str(process.pid))


def registered(command, **kwargs):
    with ACTIVE_LOCK:
        check(not STOP.is_set(), 'Run cancelled')
        p = subprocess.Popen(command, start_new_session=True, **kwargs)
        ACTIVE[p.pid] = p
    return p


def release(process):
    if process is not None:
        try:
            stop_process(process)
        finally:
            with ACTIVE_LOCK:
                ACTIVE.pop(process.pid, None)


def train_one(root, domain, model, seed, port, manifest):
    folder = model_dir(root, domain, model, seed)
    complete = folder / 'training-complete.json'
    if complete.exists():
        for name, h in pf.read(complete)['hashes'].items():
            check(digest(folder / name) == h, 'Changed trained model: ' + str(folder / name))
        print('SKIP completed training', domain, model, seed, flush=True)
        return
    check(not folder.exists(), 'Incomplete training preserved: %s. See trainer.log; do not overwrite it.' % folder)
    folder.mkdir(parents=True)
    immutable(folder / 'protocol.json', dict(manifest_sha256=digest(manifest), domain=domain,
        model=model, seed=seed, inputs=20, selection='100 training episodes every 30000 interactions; highest mean reward',
        order='lexical cycle across both training and periodic selection episodes', replay_start_size=1000))
    command = training_command(folder, model, seed, port)
    engine.save(folder / 'training-command.json', command)
    problems = sorted((root / 'experiments' / domain / domain / 'train').glob('*.pddl'))
    ready = folder / 'agent' / ('port_%d.txt' % port)
    request = folder / 'agent' / 'episode-request.json'
    env = os.environ.copy()
    env['RL_STRICT_DISCONNECT'] = '1'
    if model == 'replacement':
        env['RL_FEATURE_DIAGNOSTICS'] = str(folder / 'feature-diagnostics.json')
    trainer = planner = None
    start = time.monotonic()
    episode = 0
    print('TRAIN', domain, model, 'seed', seed, '-- periodic selection enabled', flush=True)
    try:
        with (folder / 'trainer.log').open('w') as log, (folder / 'episodes.jsonl').open('w') as exposure:
            trainer = registered(command, cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT)
            while trainer.poll() is None:
                check(not STOP.is_set(), 'Training cancelled')
                # A failed/hung episode stops the job; it is never accepted as training data.
                check(time.monotonic() - start < 4 * 86400, 'Training exceeded four-day safety deadline')
                if not ready.exists():
                    time.sleep(.05)
                    continue
                problem = problems[episode % len(problems)]
                episode += 1
                phase = pf.read(request)['phase']
                dest = folder / 'episodes' / ('%05d-%s' % (episode, phase))
                dest.mkdir(parents=True)
                fd = [sys.executable, str(root / 'fast-downward/fast-downward.py'),
                    str(problem.parent.parent / 'domain.pddl'), str(problem), '--search', search(seed, port)]
                began = time.monotonic()
                with (dest / 'planner.log').open('w') as plog:
                    planner = registered(fd, cwd=dest, stdout=plog, stderr=subprocess.STDOUT)
                    while planner.poll() is None:
                        check(not STOP.is_set(), 'Training cancelled')
                        if trainer.poll() is not None:
                            break
                        check(time.monotonic() - began < 7200, 'Training episode exceeded two-hour hang safeguard')
                        time.sleep(.05)
                code = planner.poll()
                text = (dest / 'planner.log').read_text(errors='replace')
                trainer_done = trainer.poll() is not None
                release(planner)
                planner = None
                if not trainer_done:
                    check(code in (0, 11, 12, 70), 'Unexpected planner exit %r: %s' % (code, dest))
                record = dict(episode=episode, phase=phase, problem=problem.name,
                    planner_exit_code=code, wall_seconds=time.monotonic()-began,
                    solution_marker='Solution found.' in text,
                    agent_cutoff_marker='Termination due to RL agent' in text)
                exposure.write(json.dumps(record)+'\n'); exposure.flush()
                if episode % 25 == 0:
                    print('TRAIN PROGRESS', domain, model, seed, episode, 'episodes; phase', phase, flush=True)
            check(trainer.wait() == 0, 'Training failed: ' + str(folder / 'trainer.log'))
        required = [folder / 'agent' / cp / name for cp in ('best', '1000000_finish')
                    for name in ('model.npz', 'target_model.npz', 'optimizer.npz')]
        check(all(p.is_file() for p in required), 'Missing best/final model: ' + str(folder))
        scores = folder / 'agent/scores.txt'
        check(scores.is_file(), 'Periodic evaluation scores missing')
        with scores.open() as score_stream:
            score_rows = list(csv.DictReader(score_stream, delimiter='\t'))
        check(len(score_rows) >= 32, 'Too few periodic evaluations; inspect scores.txt')
        required.append(scores)
        engine.save(complete, dict(domain=domain, model=model, seed=seed, episodes=episode,
            wall_seconds=time.monotonic()-start, selection_rounds=len(score_rows),
            hashes={str(p.relative_to(folder)): digest(p) for p in required}))
        print('TRAIN COMPLETE', domain, model, seed, flush=True)
    finally:
        try:
            release(planner)
        finally:
            release(trainer)


def train_all(root, args, manifest):
    ports = queue.Queue()
    for port in range(args.port, args.port + args.workers):
        ports.put(port)
    jobs = [(d, m, s) for s in args.seeds for d in args.domains for m in MODELS]
    def job(data):
        port = ports.get()
        try:
            train_one(root, *data, port, manifest)
        except BaseException:
            STOP.set()
            raise
        finally:
            ports.put(port)
    pool = ThreadPoolExecutor(max_workers=args.workers)
    pending = set()
    iterator = iter(jobs)
    def submit():
        item = next(iterator, None)
        if item is not None:
            pending.add(pool.submit(job, item))
    try:
        for _ in range(args.workers):
            submit()
        critical = None
        last_check = 0
        while pending:
            ready, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in ready:
                future.result()
                submit()
            if time.monotonic() - last_check > 10:
                sample = engine.memory_state()
                with (manifest.parent / 'training-memory.jsonl').open('a') as f:
                    f.write(json.dumps(sample)+'\n')
                if sample['pressure'] == '4':
                    critical = critical or time.monotonic()
                    check(time.monotonic()-critical < 10, 'Sustained critical memory pressure; training stopped safely')
                else:
                    critical = None
                last_check = time.monotonic()
    except BaseException:
        STOP.set()
        raise
    finally:
        # Each worker owns and cleans its trainer and planner on STOP.
        pool.shutdown(wait=True)


def configuration(task):
    if task['model'] in FIXED:
        return 'fixed4-' + task['model']
    return '%s-seed%d-repeat%d-best' % (task['model'], task['seed'], task['repeat'])


def config_for(root, task, port, args):
    model = task['model']
    fixed = model in FIXED
    # Repeat seeds vary evaluation stochasticity without changing training seeds.
    seed = 1 if fixed else task['repeat']
    if fixed:
        controller = [sys.executable, '-u', str(RUNTIME / 'fixed_policy.py'),
            '--action', str(FIXED.index(model)), '--port', str(port), '--seed', str(seed)]
    else:
        checkpoint = model_dir(root, task['domain'], model, task['seed']) / 'agent/best'
        controller = [sys.executable, '-u', str(RUNTIME / 'train_and_eval_chainer_agent.py'),
            '--outdir', '.', '--load', str(checkpoint), '--evaluate', '--save-eval-stats', 'eval.json',
            '--seed', str(seed), '--state', '2', '--temporal-window', '0' if model == 'original' else '4',
            '--num-heuristics', '4', '--n-hidden-channels', '75', '--n-hidden-layers', '2',
            '--port', str(port), '--port-file-id', str(port), '--eval-n-runs', '1',
            '--time-step-limit', str(2**63-1)]
    base = root / 'experiments' / task['domain'] / task['domain']
    planner = [sys.executable, str(root / 'fast-downward/fast-downward.py'),
        str(base / 'domain.pddl'), str(base / 'test' / task['problem']),
        '--search', search(seed, port, fixed)]
    return dict(task=task, controller=controller, planner=planner, port=port,
        cpu_seconds=300, wall_safeguard=args.wall_safeguard, resident_limit_bytes=CAP)


def evaluate(root, args, manifest, fixed):
    for domain in args.domains:
        output = manifest.parent / ('fixed-four-list' if fixed else 'learned') / domain
        hashes = {}
        if not fixed:
            for seed in args.seeds:
                for model in MODELS:
                    folder = model_dir(root, domain, model, seed)
                    record = pf.read(folder / 'training-complete.json')
                    for relative, h in record['hashes'].items():
                        check(digest(folder / relative) == h, 'Changed model after training')
                        hashes[str((folder / relative).relative_to(root))] = h
        protocol = dict(manifest_sha256=digest(manifest), domain=domain, model_hashes=hashes,
            fixed_four_lists=fixed, cpu_seconds=300, sampled_aggregate_resident_limit_bytes=CAP)
        immutable(output / 'protocol.json', protocol)
        names = sorted(p.name for p in (root / 'experiments' / domain / domain / 'test').glob('*.pddl'))
        if fixed:
            tasks = [dict(domain=domain, phase='test', problem=name, model=m, seed=1, repeat=1,
                          budget=None) for name in names for m in FIXED]
            engine.Batch(root, output, args).run(tasks)
        else:
            # All five policies receive repetition one before additional repetitions.
            for repeat in range(1, args.eval_repeats+1):
                tasks = [dict(domain=domain, phase='test', problem=name, model=m, seed=s,
                    repeat=repeat, budget=None) for name in names for s in args.seeds for m in MODELS]
                engine.Batch(root, output, args).run(tasks)
        engine.save(output / 'evaluation-complete.json', dict(domain=domain, fixed=fixed,
            cases=400 if fixed else 100*len(MODELS)*len(args.seeds)*args.eval_repeats))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--domains', nargs='+', required=True, choices=pf.DOMAINS)
    p.add_argument('--mode', choices=('all', 'fixed', 'train', 'evaluate'), default='all')
    p.add_argument('--workers', type=int, choices=range(1,5), default=4)
    p.add_argument('--seeds', type=int, nargs='+', default=[1,2,3,4,5])
    p.add_argument('--eval-repeats', type=int, choices=(1,25), default=1)
    p.add_argument('--port', type=int, default=13241)
    p.add_argument('--wall-safeguard', type=int, default=1800)
    p.add_argument('--run', action='store_true')
    args = p.parse_args()
    check(sys.platform == 'darwin' and sys.version_info[:2] == (3,8), 'Use macOS with the Python 3.8 rl-plan environment')
    check(len(set(args.domains)) == len(args.domains), 'Duplicate domains')
    check(args.seeds == [1,2,3,4,5], 'This protocol requires training seeds 1 through 5')
    check(1024 <= args.port <= 65531 and args.wall_safeguard >= 1200, 'Invalid ports or wall safeguard')
    root = Path.cwd().resolve()
    for key in pf.THREADS:
        os.environ[key] = '1'
    os.environ.pop('RL_FEATURE_DIAGNOSTICS', None)
    os.environ.pop('RL_STRICT_DISCONNECT', None)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    existing = subprocess.check_output(['ps','-axo','pid=,command='], text=True)
    check(not any(s in existing for s in ('train_and_eval_chainer_agent.py', '/bin/downward', 'fast-downward.py')),
          'Another planner/controller is active; wait for it to finish first')
    with contextlib.ExitStack() as stack:
        for name in ['.full100-evaluation.lock', '.cpu-worker-check.lock', '.paper-followup.lock'] + [
                '.eval-port-%d.lock' % n for n in range(args.port,args.port+args.workers)]:
            handle = stack.enter_context((root/name).open('a'))
            fcntl.flock(handle, fcntl.LOCK_EX|fcntl.LOCK_NB)
        for port in range(args.port,args.port+args.workers):
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',port))
        imports = subprocess.run([sys.executable,'-c',
            'import numpy,chainer,chainerrl,gym; print("Learning dependencies import successfully")'],check=True)
        hashes = {str(path.relative_to(HERE)): digest(path) for path in HERE.rglob('*.py')}
        problems = {}
        for domain in args.domains:
            previous = root/'results-full100-cpu-v1'/domain/'protocol.json'
            old = pf.read(previous)
            marker = pf.read(previous.parent/'evaluation-complete.json')
            check(digest(previous)==marker['protocol_sha256'], 'Old evaluation protocol changed')
            for relative, expected in old['hashes'].items():
                if relative.startswith(('fast-downward/','experiments/'+domain+'/')):
                    check(digest(root/relative)==expected, 'Planner or dataset changed: '+relative)
                    hashes['project/'+relative] = expected
            base = root/'experiments'/domain/domain
            check(len(list((base/'train').glob('*.pddl')))==100 and len(old['test'])==100, 'Expected 100/100 problems')
            problems[domain] = dict(test=old['test'], train=sorted(p.name for p in (base/'train').glob('*.pddl')),
                byte_identical_pairs=old.get('train_test_byte_identical_pairs', []))
        protocol = dict(version='paper-followup-v1', root=str(root), platform=platform.platform(),
            python=sys.version, executable=sys.executable, seeds=args.seeds,
            evaluation_repetitions=args.eval_repeats, models=list(MODELS), problems=problems,
            training_steps=1000000, episode_limit=7500, replay_start_size=1000,
            exploration_steps=500000, periodic_selection_interval=30000,
            periodic_selection_episodes=100, selection_rule='highest mean training-episode reward',
            ordering='lexical cycle across training and selection episodes, as released launcher',
            evaluation_cpu_seconds=300, evaluation_decision_limit='effectively unlimited',
            evaluation_memory='sampled aggregate resident bytes, 4 GiB; NOT Linux RLIMIT_AS',
            fixed_policy='four open lists; TCP controller; handle_empty_list=false; one run per problem',
            original_features='20 one-step differences',
            replacement_features='12 differences: mean/minimum/queue size; 8 normalized progress and age signals',
            packages={n:metadata.version(n) for n in ('numpy','scipy','chainer','chainerrl','gym')},
            thread_limits={k:os.environ[k] for k in pf.THREADS}, hashes=hashes,
            departures=['macOS and local hardware', 'aggregate process CPU accounting',
                'sampled resident memory instead of Linux address-space cap',
                'installed ChainerRL 0.8.0 matches released requirements; paper describes 0.7.0',
                '1 evaluation repetition default; paper averages 25 per trained policy',
                'socket reliability patches retained; unexpected training disconnects fail',
                'no replay persistence at intermediate checkpoints; interrupted training cannot exactly resume'],
            development='Replacement proposed after seeing first experiment; test-aware follow-up, not fresh confirmatory evidence')
        manifest = root/'results-paper-followup-v1/protocol.json'
        if manifest.exists():
            check(pf.read(manifest)==protocol, 'Follow-up protocol changed; do not mix results')
        print('PREFLIGHT OK:', ', '.join(args.domains), 'five seeds; original + replacement; mode', args.mode, flush=True)
        if not args.run:
            print('Add --run to begin. Default final evaluation: ONE repetition, not the paper\'s 25.')
            return
        immutable(manifest,protocol)
        engine.save(manifest.parent / ('launch-%s.json' % time.strftime('%Y%m%d-%H%M%S')),
            dict(mode=args.mode, workers=args.workers, first_port=args.port,
                 wall_safeguard=args.wall_safeguard, domains=args.domains))
        engine.configuration = configuration
        engine.config_for = config_for
        engine.ACCEPTED = ('solved','cpu_timeout','search_unsolved','memory_limit')
        engine.HERE = HERE
        if args.mode in ('all','fixed'):
            evaluate(root,args,manifest,True)
        if args.mode in ('all','train'):
            train_all(root,args,manifest)
        if args.mode in ('all','evaluate'):
            evaluate(root,args,manifest,False)
        print('REQUESTED FOLLOW-UP PHASES COMPLETE:',manifest.parent,flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        STOP.set()
        print('Stopped. Completed cases/models preserved. Incomplete training is not exactly resumable.',flush=True)
        sys.exit(130)
