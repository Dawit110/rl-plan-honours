#!/usr/bin/env python3
"""Frozen exploratory evaluation of four deterministic simple policies."""
import argparse
import contextlib
import csv
import fcntl
from importlib import metadata
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import engine
import preflight as pf

HERE = Path(__file__).resolve().parent
POLICIES = ('recent50','stagprog50','probe25x100','queue50a1')
CAP = 4 * 1024**3
PROTOCOLS = {}
ENGINE_VALID = engine.valid


def configuration(task):
    return task['model']


def prepare(root, domain, args, check_packages=True):
    previous = root/'results-paper-followup-v1/protocol.json'
    reference = pf.read(previous)
    hashes = {'reference_protocol': pf.sha(previous)}
    for name in ('case.py','cpu_meter.py','runtime/fd_env.py','runtime/evaluate.py'):
        pf.require(pf.sha(HERE/name) == reference['hashes'][name],
                   'Resource supervisor/runtime differs from corrected baseline: '+name)
    for path in sorted(list(HERE.rglob('*.py'))+[HERE/'source-provenance.json']):
        hashes['package/'+str(path.relative_to(HERE))] = pf.sha(path)
    base = root/'experiments'/domain/domain
    train = sorted((base/'train').glob('*.pddl'))
    test = sorted((base/'test').glob('*.pddl'))
    pf.require(len(train) == len(test) == 100, 'Expected 100 train/test files: '+domain)
    inputs = [base/'domain.pddl'] + train + test
    inputs += [root/k[len('project/'):] for k in reference['hashes']
               if k.startswith('project/fast-downward/')]
    pf.require(any(str(p).endswith('/bin/downward') for p in inputs), 'Missing planner provenance')
    for path in inputs:
        relative = str(path.relative_to(root))
        pf.require(path.is_file(), 'Missing input: '+str(path))
        value = pf.sha(path)
        pf.require(value == reference['hashes'].get('project/'+relative),
                   'Input changed since corrected baseline evaluation: '+relative)
        hashes['project/'+relative] = value
    packages = {n:metadata.version(n) for n in ('numpy','scipy','chainer','chainerrl','gym')} if check_packages else {}
    if check_packages:
        pf.require(packages == reference['packages'], 'Dependency versions differ from corrected baselines')
    train_hashes = {pf.sha(p) for p in train}
    overlap = [p.name for p in test if pf.sha(p) in train_hashes]
    return dict(version='simple-policies-extra-v1',domain=domain,root=str(root),hashes=hashes,
        policies=list(POLICIES),test=[p.name for p in test],known_train_test_overlap=overlap,
        seed=1,python=sys.version,platform=platform.platform(),packages=packages,
        cpu_seconds=300,resident_limit_bytes=CAP,wall_safeguard=args.wall_safeguard,
        decision_limit=2**63-1,threads={k:os.environ.get(k) for k in pf.THREADS},
        heuristic_order=['ff','cg','cea','add'],initial_action='ff (or first available)',
        warmup='The unchanged environment performs one FF warm-up action before the policy sees its first observation, as in corrected fixed baselines.',
        recent50='Raw queue-minimum change over 50 intervals, normalized by max(abs(old_min),1), reconsidered every decision after full history. Current-action tie preference.',
        stagprog50='After 50 nonimproving observations, choose OTHER available heuristic with largest raw normalized 50-interval progress; cyclic tie preference.',
        probe25x100='Probe each heuristic for 25 actions, score after its 25th action; exploit best available for 100 actions; rotate probe order between cycles.',
        queue50a1='Raw normalized 50-interval progress minus max(new_queue-old_queue,0)/max(new_queue,1), alpha=1; reconsider every decision.',
        historical_cores='Class bodies copied verbatim from earlier run_progress4.py, run_stag_progress4.py, run_probe_exploit4.py and run_progress_queue4.py. Adapter validates shape/nonempty finite entries and returns deterministic fallback if all queues empty to prevent historical probe infinite loop.',
        empty_queue_handling='Historical cores retain their queue handling, including possible empty-queue requests during warmup/ongoing probe. Planner handle_empty_list=true provides fallback as for learned controllers; corrected constant baseline used false.',
        overhead='Controller, imports, raw observation processing, translation and planner all charged by unchanged aggregate CPU supervisor.',
        comparison='Same 300 CPU seconds and sampled 4GiB RSS cap as corrected fixed baselines. No practical decision cap, matching corrected fixed; prior learned evaluation had a 1m-decision cap and no RSS cap.',
        interpretation='Exploratory follow-up after prior test inspection. No tuning or per-domain policy selection using these test results. Report all four policies.',
        progress_caveat='Shared-queue progress is a proxy; it is neither distance to the goal nor causal credit to a chosen heuristic.')


def tasks_for(protocol):
    tasks = []
    for i, name in enumerate(protocol['test']):
        k = i % len(POLICIES)
        for policy in POLICIES[k:]+POLICIES[:k]:
            tasks.append(dict(domain=protocol['domain'],phase='test',problem=name,
                              model=policy,seed=1,budget=None))
    return tasks


def config_for(root, task, port, args):
    protocol = PROTOCOLS[task['domain']]
    pf.require(task in tasks_for(protocol), 'Unknown evaluation task')
    controller = [sys.executable,'-u',str(HERE/'runtime/simple_policy.py'),
                  '--policy',task['model'],'--port',str(port),'--seed','1']
    base = root/'experiments'/task['domain']/task['domain']
    search = ('rl_eager(rl([single(ff()),single(cg()),single(cea()),single(add())],'
              'random_seed=1,handle_empty_list=true),rl_control_interval=0,rl_client_port=%d)') % port
    planner = [sys.executable,str(root/'fast-downward/fast-downward.py'),
               str(base/'domain.pddl'),str(base/'test'/task['problem']),'--search',search]
    return dict(task=task,controller=controller,planner=planner,port=port,
                cpu_seconds=300,wall_safeguard=args.wall_safeguard,resident_limit_bytes=CAP)


def valid(row, task, args):
    expected = config_for(Path(PROTOCOLS[task['domain']]['root']),task,row['config']['port'],args)
    pf.require(row['config'] == expected, 'Saved case configuration differs from frozen protocol')
    return ENGINE_VALID(row,task,args)


def restricted_summary(output, protocol, args):
    rows = []
    for policy in POLICIES:
        eligible = [t for t in tasks_for(protocol) if t['model'] == policy
                    and t['problem'] not in protocol['known_train_test_overlap']]
        done = []
        for task in eligible:
            path = engine.case_dir(output,task)/'result.json'
            if path.exists():
                result = pf.read(path)
                if valid(result,task,args):
                    done.append(result)
        rows.append(dict(domain=protocol['domain'],policy=policy,planned=len(eligible),
            completed=len(done),solved=sum(r['status']=='solved' for r in done),
            capped_cpu_sum=sum(r['cpu_seconds'] if r['status']=='solved' else 300 for r in done)))
    with (output/'coverage-nonoverlap.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def verify_inputs(root, protocol):
    for name, expected in protocol['hashes'].items():
        if name == 'reference_protocol':
            path = root/'results-paper-followup-v1/protocol.json'
        elif name.startswith('package/'):
            path = HERE/name[len('package/'):]
        else:
            path = root/name[len('project/'):]
        pf.require(pf.sha(path) == expected, 'Input changed during evaluation: '+name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domains',nargs='+',required=True,choices=pf.DOMAINS)
    parser.add_argument('--workers',type=int,choices=range(1,5),default=4)
    parser.add_argument('--port',type=int,default=13641)
    parser.add_argument('--wall-safeguard',type=int,default=1800)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    pf.require(sys.platform=='darwin' and sys.version_info[:2]==(3,8), 'Use macOS and rl-plan Python 3.8')
    pf.require(len(set(args.domains))==len(args.domains),'Duplicate domains')
    pf.require(1024<=args.port<=65531 and args.wall_safeguard>=1200,'Invalid port or safeguard')
    root=Path.cwd().resolve()
    for key in pf.THREADS:os.environ[key]='1'
    for key in ('RL_FEATURE_DIAGNOSTICS','RL_STRICT_DISCONNECT'):os.environ.pop(key,None)
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    existing=subprocess.check_output(['ps','-axo','pid=,command='],text=True)
    conflicts=[s for s in existing.splitlines() if any(x in s for x in
        ('train_and_eval_chainer_agent.py','fixed_policy.py','simple_policy.py','/bin/downward','fast-downward.py'))]
    pf.require(not conflicts,'Another planner/controller is still running:\n'+'\n'.join(conflicts))
    engine.configuration=configuration;engine.config_for=config_for;engine.valid=valid
    engine.ACCEPTED=('solved','cpu_timeout','search_unsolved','memory_limit')
    with contextlib.ExitStack() as stack:
        for name in ['.simple-policies.lock','.selector-data.lock','.full100-evaluation.lock',
                     '.paper-followup.lock','.cpu-worker-check.lock']+[
                     '.eval-port-%d.lock'%p for p in range(args.port,args.port+args.workers)]:
            handle=stack.enter_context((root/name).open('a'))
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for port in range(args.port,args.port+args.workers):
            with socket.socket() as sock:sock.bind(('127.0.0.1',port))
        subprocess.run([sys.executable,'-c','import numpy,chainer,chainerrl,gym; print("Dependencies import successfully")'],check=True)
        for domain in args.domains:
            protocol=prepare(root,domain,args)
            path=root/'results-simple-policies-extra-v1'/domain/'protocol.json'
            if path.exists():
                pf.require(pf.read(path)==protocol,'Saved protocol differs: '+str(path))
            else:
                pf.require(not path.parent.exists() or not any(path.parent.iterdir()),'Nonempty output without protocol: '+str(path.parent))
            PROTOCOLS[domain]=protocol
            print('PREFLIGHT OK %s: 100 test problems x 4 simple policies; 300 CPU seconds; %d workers' % (domain,args.workers),flush=True)
        if not args.run:
            print('Preflight complete. Add --run to evaluate. No training required.')
            return
        for domain in args.domains:
            protocol=PROTOCOLS[domain];verify_inputs(root,protocol)
            output=root/'results-simple-policies-extra-v1'/domain
            if not (output/'protocol.json').exists():engine.save(output/'protocol.json',protocol)
            engine.save(output/('launch-%d.json'%time.time_ns()),dict(workers=args.workers,hostname=socket.gethostname(),port=args.port))
            try:
                engine.Batch(root,output,args).run(tasks_for(protocol))
            finally:
                restricted_summary(output,protocol,args)
            verify_inputs(root,protocol)
            engine.save(output/'evaluation-complete.json',dict(domain=domain,cases=400,
                        protocol_sha256=pf.sha(output/'protocol.json')))
            print('DOMAIN COMPLETE:',output/'coverage.csv',flush=True)
        print('ALL ASSIGNED SIMPLE-POLICY EVALUATIONS COMPLETE.',flush=True)


if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Stopped. Completed cases saved; resume with the same command.',flush=True)
        sys.exit(130)
