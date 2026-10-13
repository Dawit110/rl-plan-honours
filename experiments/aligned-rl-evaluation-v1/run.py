#!/usr/bin/env python3
"""Reevaluate frozen Original/Progress checkpoints under aligned resource limits."""
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
MODELS = ('original','progress')
CAP = 4 * 1024**3
PROTOCOLS = {}
ENGINE_VALID = engine.valid


def configuration(task):
    return '%s-seed%d-%d' % (task['model'],task['seed'],task['budget'])


def checkpoint_choice(selected, domain, model, seed):
    pf.require(selected['domain'] == domain and selected['model'] == model
            and selected['seed'] == seed, 'Checkpoint selection identity differs')
    pf.require(selected['selection_role'] == 'training subset; not held out',
            'Only previously TRAINING-selected checkpoints are permitted')
    pf.require(selected['cpu_budget'] == 300, 'Unexpected selection CPU budget')
    candidates = selected['candidates']
    pf.require(sorted(c['budget'] for c in candidates) == list(pf.BUDGETS),
            'Expected four checkpoint candidates')
    best = min(candidates, key=lambda c: (-c['solved'], c['capped_cpu_sum'], c['budget']))
    pf.require(best['budget'] == selected['selected_budget'], 'Checkpoint selection rank differs')
    return best['budget']

def prepare(root, domain, args, check_packages=True):
    previous = root/'results-paper-followup-v1/protocol.json'
    reference = pf.read(previous)
    hashes = {'reference_protocol': pf.sha(previous)}
    for name in ('case.py','cpu_meter.py'):
        pf.require(pf.sha(HERE/name) == reference['hashes'][name],
                   'Resource supervisor/runtime differs from corrected baseline: '+name)
    for path in sorted(HERE.rglob('*.py')):
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
    old_path=root/'results-full100-cpu-v1'/domain/'protocol.json'
    old=pf.read(old_path)
    marker=pf.read(old_path.parent/'evaluation-complete.json')
    pf.require(marker['protocol_sha256']==pf.sha(old_path),'Previous evaluation protocol hash differs')
    hashes['project/'+str(old_path.relative_to(root))]=pf.sha(old_path)
    candidates=[]
    for model in MODELS:
        for seed in (1,2,3):
            selected_path=old_path.parent/('selected-%s-seed%d.json'%(model,seed))
            budget=checkpoint_choice(pf.read(selected_path),domain,model,seed)
            hashes['project/'+str(selected_path.relative_to(root))]=pf.sha(selected_path)
            folder=pf.trained(root,domain,model,seed)
            checkpoint=pf.checkpoint(folder,model,budget)
            prefixes=[str(p.relative_to(root))+'/' for p in (folder/'source',checkpoint)]
            for relative,expected in old['hashes'].items():
                if any(relative.startswith(prefix) for prefix in prefixes):
                    path=root/relative
                    pf.require(pf.sha(path)==expected,'Frozen model/runtime changed: '+relative)
                    hashes['project/'+relative]=expected
            for path in [folder/'source/train_and_eval_chainer_agent.py',folder/'source/fd_env.py',checkpoint/'model.npz']:
                pf.require('project/'+str(path.relative_to(root)) in hashes,'Missing model/runtime provenance: '+str(path))
            for path in (folder/'source').glob('*.py'):
                pf.require('project/'+str(path.relative_to(root)) in hashes,'Unrecorded runtime source: '+str(path))
            candidates.append(dict(model=model,seed=seed,budget=budget))
    return dict(version='aligned-rl-evaluation-v1',domain=domain,root=str(root),hashes=hashes,
        candidates=candidates,test=[p.name for p in test],known_train_test_overlap=overlap,
        python=sys.version,platform=platform.platform(),packages=packages,
        cpu_seconds=300,resident_limit_bytes=CAP,wall_safeguard=args.wall_safeguard,
        decision_limit=2**63-1,threads={k:os.environ.get(k) for k in pf.THREADS},
        changes='Add sampled aggregate 4 GiB RSS cap; replace 1m decision limit by 2**63-1. Same aggregate 300 CPU seconds.',
        preserved='Original frozen 20/28-feature model runtimes and previously training-selected checkpoints, test problems and per-model seeds. No retraining or reselection.',
        selection_limitation='Checkpoint choices were made under the earlier selection protocol. This evaluates the same trained systems under aligned TEST limits, not a newly selected or paper-identical pipeline.',
        search='Four queues, external controller, handle_empty_list=true, matching prior learned evaluation.',
        interpretation='Exploratory protocol-alignment rerun. Preserve old/new results separately and report differences; improvement is not assumed.')


def tasks_for(protocol):
    tasks=[]
    candidates=protocol['candidates']
    for i,name in enumerate(protocol['test']):
        k=i%len(candidates)
        for candidate in candidates[k:]+candidates[:k]:
            tasks.append(dict(domain=protocol['domain'],phase='test',problem=name,**candidate))
    return tasks


def config_for(root,task,port,args):
    pf.require(task in tasks_for(PROTOCOLS[task['domain']]),'Unknown evaluation task')
    folder=pf.trained(root,task['domain'],task['model'],task['seed'])
    controller=[sys.executable,'-u',str(folder/'source/train_and_eval_chainer_agent.py'),
        '--outdir','.', '--load',str(pf.checkpoint(folder,task['model'],task['budget'])),
        '--evaluate','--save-eval-stats','eval.json','--seed',str(task['seed']),
        '--state','2','--temporal-window','0' if task['model']=='original' else '4',
        '--num-heuristics','4','--n-hidden-channels','75','--n-hidden-layers','2',
        '--port',str(port),'--port-file-id',str(port),'--eval-n-runs','1',
        '--time-step-limit',str(2**63-1)]
    base=root/'experiments'/task['domain']/task['domain']
    search=('rl_eager(rl([single(ff()),single(cg()),single(cea()),single(add())],'
            'random_seed=%d,handle_empty_list=true),rl_control_interval=0,rl_client_port=%d)')%(task['seed'],port)
    planner=[sys.executable,str(root/'fast-downward/fast-downward.py'),
        str(base/'domain.pddl'),str(base/'test'/task['problem']),'--search',search]
    return dict(task=task,controller=controller,planner=planner,port=port,
        cpu_seconds=300,wall_safeguard=args.wall_safeguard,resident_limit_bytes=CAP)


def valid(row, task, args):
    expected = config_for(Path(PROTOCOLS[task['domain']]['root']),task,row['config']['port'],args)
    pf.require(row['config'] == expected, 'Saved case configuration differs from frozen protocol')
    return ENGINE_VALID(row,task,args)


def restricted_summary(output, protocol, args):
    rows = []
    for candidate in protocol['candidates']:
        policy=configuration(candidate)
        eligible = [t for t in tasks_for(protocol) if configuration(t) == policy
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
    parser.add_argument('--port',type=int,default=13841)
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
            path=root/'results-aligned-rl-evaluation-v1'/domain/'protocol.json'
            if path.exists():
                pf.require(pf.read(path)==protocol,'Saved protocol differs: '+str(path))
            else:
                pf.require(not path.parent.exists() or not any(path.parent.iterdir()),'Nonempty output without protocol: '+str(path.parent))
            PROTOCOLS[domain]=protocol
            print('PREFLIGHT OK %s: 100 test problems x 2 models x 3 seeds; 300 CPU seconds; %d workers' % (domain,args.workers),flush=True)
        if not args.run:
            print('Preflight complete. Add --run to evaluate. No training required.')
            return
        for domain in args.domains:
            protocol=PROTOCOLS[domain];verify_inputs(root,protocol)
            output=root/'results-aligned-rl-evaluation-v1'/domain
            if not (output/'protocol.json').exists():engine.save(output/'protocol.json',protocol)
            engine.save(output/('launch-%d.json'%time.time_ns()),dict(workers=args.workers,hostname=socket.gethostname(),port=args.port))
            try:
                engine.Batch(root,output,args).run(tasks_for(protocol))
            finally:
                restricted_summary(output,protocol,args)
            verify_inputs(root,protocol)
            engine.save(output/'evaluation-complete.json',dict(domain=domain,cases=600,
                        protocol_sha256=pf.sha(output/'protocol.json')))
            print('DOMAIN COMPLETE:',output/'coverage.csv',flush=True)
        print('ALL ALIGNED RL EVALUATIONS COMPLETE.',flush=True)


if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Stopped. Completed cases saved; resume with the same command.',flush=True)
        sys.exit(130)
