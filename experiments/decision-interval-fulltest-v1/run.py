#!/usr/bin/env python3
"""Frozen, full-test-set Original vs fixed/adaptive decision-interval evaluation."""
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
CAP = 4*1024**3
CONDITIONS = ('original', 'fixed16', 'adaptive4p8m64')
PROTOCOLS = {}
ENGINE_VALID = engine.valid

def configuration(task):
    return '%s-seed%d-%d' % (task['condition'], task['seed'], task['budget'])

def package_hashes():
    paths = list(HERE.glob('*.py')) + list((HERE/'planner_patch').rglob('*'))
    paths += [HERE/'input-provenance.json']
    return {str(p.relative_to(HERE)):pf.sha(p) for p in paths if p.is_file()}

def prepare(root, domain, args):
    source = pf.read(HERE/'input-provenance.json')
    selected = source['domains'][domain]
    pf.require(len(selected['test_sha256']) == 100, 'Expected 100 frozen test files per domain')
    hashes = dict(source['planner'], **selected['hashes'])
    for name, digest in hashes.items():
        pf.require(pf.sha(root/name) == digest, 'Input differs from supplied bundle: '+name)
    reference = pf.read(root/'results-aligned-rl-evaluation-v1'/domain/'protocol.json')
    for name in ('case.py','cpu_meter.py'):
        pf.require(pf.sha(HERE/name) == reference['hashes']['package/'+name], 'Resource supervisor differs: '+name)
    # Validate the actual, already-evaluated legacy executable, built translator,
    # all frozen loader inputs (including optimizer), and the full current split.
    for name, digest in reference['hashes'].items():
        if name.startswith('project/fast-downward/') or name.startswith('project/experiments/'+domain+'/') or (
                name.startswith('project/results-'+domain+'-original-')):
            relative = name[len('project/'):]
            pf.require(pf.sha(root/relative) == digest, 'Aligned input changed: '+relative)
            hashes[relative] = digest
    packages = {n:metadata.version(n) for n in ('numpy','scipy','chainer','chainerrl','gym')}
    pf.require(packages == reference['packages'], 'Dependency versions changed since aligned evaluation')
    build = pf.read(HERE/'build.json')
    for name, digest in dict(build['source_hashes'], **build['artifact_hashes']).items():
        pf.require(pf.sha(HERE/'planner'/name) == digest, 'Isolated build changed: '+name)
    expected_patched = dict(source['planner'])
    # Require that the build is exactly the declared original plus this patch.
    expected_patched = {k[len('fast-downward/'):]:v for k,v in expected_patched.items()}
    expected_patched.update({str(p.relative_to(HERE/'planner_patch')):pf.sha(p)
                            for p in (HERE/'planner_patch').rglob('*') if p.is_file()})
    pf.require(build['source_hashes'] == expected_patched, 'Build does not match the approved patch')
    return dict(version='decision-interval-fulltest-v1', domain=domain, root=str(root),
        hashes=hashes, package_hashes=package_hashes(), build_sha256=pf.sha(HERE/'build.json'),
        packages=packages, python=sys.version, platform=platform.platform(),
        threads={k:os.environ.get(k) for k in pf.THREADS},
        candidates=selected['candidates'], problems=sorted(selected['test_sha256']), conditions=list(CONDITIONS),
        verification_problem=selected['problems'][0],
        known_train_test_overlap=sorted(n for n,h in selected['test_sha256'].items() if h in set(selected['train_sha256'].values())),
        selection='All 100 existing test files per domain. Training-only diagnostic problem retained from pilot design.', cpu_seconds=300, resident_limit_bytes=CAP,
        wall_safeguard=args.wall_safeguard, decision_limit=2**63-1,
        interval=dict(fixed=16,minimum=4,patience=8,maximum=64,unit='successful expansions'),
        baseline='Frozen Original model and original planner; common lightweight controller recording.',
        features='Original one-removal DIFF, not block differences. FF warmup exactly one removal.',
        limitations='Exploratory follow-up on previously inspected test sets; not fresh confirmation. Known exact train/test overlaps are flagged. No checkpoint reselection. Runtime comparisons include interval bookkeeping and recording. Only one fixed interval tested.')

def tasks_for(protocol):
    tasks=[]
    for i, problem in enumerate(protocol['problems']):
        for candidate in protocol['candidates']:
            # Rotate condition order, including across seeds, without outcomes.
            j=(i+candidate['seed']-1)%len(CONDITIONS)
            for condition in CONDITIONS[j:]+CONDITIONS[:j]:
                tasks.append(dict(candidate,domain=protocol['domain'],phase='test', problem=problem,
                                  condition=condition))
    return tasks

def config_for(root, task, port, args, condition=None, trace=False, profile=False, decision_limit=2**63-1):
    chosen = condition or task['condition']
    folder=pf.trained(root,task['domain'],'original',task['seed'])
    controller=[sys.executable,'-u',str(HERE/'controller.py'),'--source',str(folder/'source')]
    if chosen != 'original': controller += ['--extended']
    if trace: controller += ['--trace']
    if profile: controller += ['--profile']
    controller += ['--outdir','.', '--load',str(pf.checkpoint(folder,'original',task['budget'])),
        '--evaluate','--save-eval-stats','eval.json','--seed',str(task['seed']),
        '--state','2','--temporal-window','0','--num-heuristics','4',
        '--n-hidden-channels','75','--n-hidden-layers','2','--port',str(port),
        '--port-file-id',str(port),'--eval-n-runs','1','--time-step-limit',str(decision_limit)]
    search=('rl_eager(rl([single(ff()),single(cg()),single(cea()),single(add())],'
            'random_seed=%d,handle_empty_list=true),rl_control_interval=0,rl_client_port=%d')%(task['seed'],port)
    if chosen != 'original':
        mode={'compat':0,'fixed16':1,'adaptive4p8m64':2}[chosen]
        search += ',interval_mode=%d,interval_fixed=16,interval_min=4,interval_patience=8,interval_max=64'%mode
    search += ')'
    planner_root = root/'fast-downward' if chosen == 'original' else HERE/'planner'
    base=root/'experiments'/task['domain']/task['domain']
    planner=[sys.executable,str(planner_root/'fast-downward.py'),str(base/'domain.pddl'),
             str(base/('train' if task['phase']=='verification' else 'test')/task['problem']),'--search',search]
    return dict(task=task,controller=controller,planner=planner,port=port,
                cpu_seconds=300,wall_safeguard=args.wall_safeguard,resident_limit_bytes=CAP)

def valid(row, task, args):
    expected=config_for(Path(PROTOCOLS[task['domain']]['root']),task,row['config']['port'],args)
    pf.require(row['config']==expected,'Saved case configuration differs')
    return ENGINE_VALID(row,task,args)

def summarize(output):
    protocol=PROTOCOLS[output.name]
    rows=[];groups={}
    for task in tasks_for(protocol):
        path=engine.case_dir(output,task)/'result.json'
        key=configuration(task)
        group=groups.setdefault(key,dict(domain=output.name,configuration=key,planned=len(protocol['problems']),
            completed=0,solved=0,cpu_timeouts=0,memory_limits=0,errors_or_interrupted=0,capped_cpu_sum=0.0))
        if not path.exists():continue
        result=pf.read(path);complete=valid(result,task, type('Args',(),dict(wall_safeguard=protocol['wall_safeguard']))())
        row=dict(task,configuration=key,**{k:v for k,v in result.items() if k not in ('config','traceback')})
        metric=path.parent/'controller-metrics.json'
        if metric.exists():
            counts=pf.read(metric)
            row['learned_decisions']=counts['learned_decisions']
            row['controller_cpu_at_last_snapshot']=counts.get('controller_process_cpu_seconds')
            data=counts.get('planner',{})
            for k in ('expanded','removals','decisions'):row['interval_'+k]=data.get(k)
            row['completed_interval_histogram']=data.get('completed_interval_histogram')
        row['known_train_test_overlap']=task['problem'] in protocol['known_train_test_overlap']
        row['completed']=complete;rows.append(row)
        if complete:
            group['completed']+=1;group['solved']+=result['status']=='solved'
            group['cpu_timeouts']+=result['status']=='cpu_timeout'
            group['memory_limits']+=result['status']=='memory_limit'
            group['capped_cpu_sum']+=result['cpu_seconds'] if result['status']=='solved' else 300
        else:group['errors_or_interrupted']+=1
    nonoverlap={}
    for task in tasks_for(protocol):
        if task['problem'] in protocol['known_train_test_overlap']: continue
        key=configuration(task)
        g=nonoverlap.setdefault(key,dict(domain=output.name,configuration=key,planned=0,completed=0,solved=0,capped_cpu_sum=0.0))
        g['planned']+=1
    for row in rows:
        if row['known_train_test_overlap']: continue
        g=nonoverlap[row['configuration']]
        if row['completed']:
            g['completed']+=1; g['solved']+=row['status']=='solved'
            g['capped_cpu_sum']+=row['cpu_seconds'] if row['status']=='solved' else 300
    for name,data in [('results',rows),('coverage',list(groups.values())),('coverage-nonoverlap',list(nonoverlap.values()))]:
        if not data:continue
        temp=output/(name+'.tmp')
        with temp.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=sorted({k for r in data for k in r}));w.writeheader();w.writerows(data)
        temp.replace(output/(name+'.csv'))

def verify_inputs(root, protocol):
    for name,digest in protocol['hashes'].items():pf.require(pf.sha(root/name)==digest,'Input changed during pilot: '+name)
    pf.require(package_hashes()==protocol['package_hashes'],'Package changed during pilot')
    pf.require(pf.sha(HERE/'build.json')==protocol['build_sha256'],'Build manifest changed')
    build=pf.read(HERE/'build.json')
    for name,digest in build['artifact_hashes'].items():pf.require(pf.sha(HERE/'planner'/name)==digest,'Built executable/translator changed')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--domains',nargs='+',required=True,choices=pf.DOMAINS)
    p.add_argument('--workers',type=int,choices=range(1,5),default=4)
    p.add_argument('--port',type=int,default=14841)
    p.add_argument('--wall-safeguard',type=int,default=1800)
    p.add_argument('--run',action='store_true')
    args=p.parse_args();root=Path.cwd().resolve()
    pf.require(sys.platform=='darwin' and sys.version_info[:2]==(3,8),'Use the Mac rl-plan Python 3.8 environment')
    pf.require(len(set(args.domains))==len(args.domains),'Duplicate domains')
    pf.require(1024<=args.port<=65531 and args.wall_safeguard>=1200,'Invalid port or wall safeguard')
    for key in pf.THREADS:os.environ[key]='1'
    for key in ('RL_FEATURE_DIAGNOSTICS','RL_STRICT_DISCONNECT','RL_INTERVAL_TRACE'):os.environ.pop(key,None)
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    active=subprocess.check_output(['ps','-axo','pid=,command='],text=True)
    conflicts=[s for s in active.splitlines() if any(x in s for x in (
        'train_and_eval_chainer_agent.py','fixed_policy.py','simple_policy.py','/bin/downward','fast-downward.py',str(HERE/'controller.py')))]
    pf.require(not conflicts,'Another planner/controller is running. Finish or pause it first:\n'+'\n'.join(conflicts))
    engine.configuration=configuration;engine.config_for=config_for;engine.valid=valid;engine.summarize=summarize
    engine.ACCEPTED=('solved','cpu_timeout','search_unsolved','memory_limit')
    with contextlib.ExitStack() as stack:
        for name in ['.simple-policies.lock','.selector-data.lock','.full100-evaluation.lock',
                     '.paper-followup.lock','.cpu-worker-check.lock','.decision-interval.lock']+[
                     '.eval-port-%d.lock'%i for i in range(args.port,args.port+args.workers)]:
            f=stack.enter_context((root/name).open('a'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for port in range(args.port,args.port+args.workers):
            with socket.socket() as sock:sock.bind(('127.0.0.1',port))
        subprocess.run([sys.executable,'-c','import numpy,chainer,chainerrl,gym; print("Dependencies import successfully")'],check=True)
        for domain in args.domains:
            protocol=prepare(root,domain,args);PROTOCOLS[domain]=protocol
            output=root/'results-decision-interval-fulltest-v1'/domain
            path=output/'protocol.json'
            if path.exists():pf.require(pf.read(path)==protocol,'Saved protocol differs: '+str(path))
            else:pf.require(not output.exists() or not any(output.iterdir()),'Nonempty output without protocol')
            print('PREFLIGHT OK %s: 100 test problems x 3 seeds x 3 conditions; 900 cases'%domain,flush=True)
        if not args.run:
            print('Add --run to verify compatibility and execute the full test evaluation.');return
        import verify
        for domain in args.domains:
            protocol=PROTOCOLS[domain];verify_inputs(root,protocol)
            output=root/'results-decision-interval-fulltest-v1'/domain
            if not (output/'protocol.json').exists():engine.save(output/'protocol.json',protocol)
            engine.save(output/('launch-%d.json'%time.time_ns()),dict(workers=args.workers,host=socket.gethostname(),port=args.port))
            verify.check(root,output,protocol,args)
            engine.Batch(root,output,args).run(tasks_for(protocol))
            verify_inputs(root,protocol)
            engine.save(output/'evaluation-complete.json',dict(cases=900,protocol_sha256=pf.sha(output/'protocol.json')))
            print('DOMAIN COMPLETE:',output/'coverage.csv',flush=True)
    print('FULL TEST EVALUATION COMPLETE. Export both Macs for paired analysis.',flush=True)

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:
        print('Stopped. Completed test cases saved; rerun the same command to resume.',flush=True)
        sys.exit(130)
