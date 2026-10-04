#!/usr/bin/env python3
"""Read-only checkpoint benchmark. Python 3.8+, macOS/Linux."""
import argparse, concurrent.futures, fcntl, hashlib, importlib.util, json
import os, platform, queue, socket, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace

THREAD_VARS = ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(p, data):
    tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(data,indent=2)+'\n'); tmp.replace(p)

def system_state():
    result={}
    commands=[['vm_stat'],['sysctl','vm.swapusage'],['sysctl','hw.memsize'],['sysctl','hw.physicalcpu']]
    for c in commands:
        try: result[' '.join(c)]=subprocess.check_output(c,stderr=subprocess.STDOUT,text=True)
        except Exception as e: result[' '.join(c)]=str(e)
    return result

def module():
    p=Path(__file__).with_name('eval_helper.py')
    spec=importlib.util.spec_from_file_location('isolated_helper',str(p))
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

def run_case(root, output, task, port, seconds):
    domain, model, problem=task
    trained=root/('results-%s-%s-full100-v1-seed1'%(domain,model))
    checkpoint=trained/model/'agent/1000000_finish'
    h=module()  # independent globals for every worker
    h.WINDOWS={model:0}
    h.controller_command=lambda d,m,a:[sys.executable,'-u',str(trained/'source/train_and_eval_chainer_agent.py'),
        '--outdir','.', '--load',str(checkpoint),'--evaluate','--save-eval-stats','eval.json',
        '--seed','1','--state','2','--temporal-window','0' if model=='original' else '4',
        '--num-heuristics','4','--n-hidden-channels','75','--n-hidden-layers','2',
        '--port',str(port),'--port-file-id',str(port),'--eval-n-runs','1','--time-step-limit','1000000']
    search=('rl_eager(rl([single(ff()),single(cg()),single(cea()),single(add())],'
            'random_seed=1,handle_empty_list=true),rl_control_interval=0,rl_client_port=%d)'%port)
    h.planner_command=lambda d,p,m,a:[sys.executable,str(root/'fast-downward/fast-downward.py'),
        '--overall-time-limit',str(seconds),str(root/'experiments'/domain/domain/'domain.pddl'),
        str(root/'experiments'/domain/domain/'train'/problem),'--search',search]
    h.run_case(domain,problem,model,SimpleNamespace(port=port,wall_seconds=seconds),output)
    return json.loads((output/domain/Path(problem).stem/model/'result.json').read_text())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--workers',nargs='+',type=int,choices=[1,2,3],default=[1,2,1])
    p.add_argument('--seconds',type=int,default=60)
    a=p.parse_args(); root=a.root.resolve()
    if a.workers[0]!=1: p.error('First stage must use one worker as reference.')
    if not 10<=a.seconds<=300: p.error('seconds must be 10..300')
    for key in THREAD_VARS: os.environ[key]='1'
    os.environ.pop('RL_FEATURE_DIAGNOSTICS',None)
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    tasks=[]; hashes={}
    for domain in ('blocksworld','barman'):
        reference=None
        for model in ('original','progress'):
            trained=root/('results-%s-%s-full100-v1-seed1'%(domain,model))
            protocol=json.loads((trained/'protocol.json').read_text())
            names=protocol['train']
            if len(names)!=100: raise RuntimeError('Expected 100 training instances')
            if reference is None: reference=names
            if names!=reference: raise RuntimeError('Training lists differ')
            if not (trained/model/'training-complete.json').is_file(): raise RuntimeError('Training incomplete: '+str(trained))
            for rel,expected in json.loads((trained/'snapshot-hashes.json').read_text()).items():
                f=trained/rel
                if sha(f)!=expected: raise RuntimeError('Checkpoint hash mismatch: '+str(f))
            for f in sorted((trained/'source').glob('*.py')): hashes[str(f)]=sha(f)
            for idx in (0,49,99):
                name=names[idx]; f=root/'experiments'/domain/domain/'train'/name
                hashes[str(f)]=sha(f); tasks.append((domain,model,name))
    for f in [root/'fast-downward/fast-downward.py',root/'fast-downward/builds/release/bin/downward']:
        hashes[str(f)]=sha(f)
    out=root/('results-concurrency-'+time.strftime('%Y%m%d-%H%M%S'))
    out.mkdir(exist_ok=False)
    save(out/'protocol.json',dict(purpose='Training-only hardware benchmark; not checkpoint selection or test evaluation',
         tasks=tasks,workers=a.workers,seconds=a.seconds,hashes=hashes,python=sys.version,
         platform=platform.platform(),threads={k:os.environ[k] for k in THREAD_VARS}))
    handles=[]
    try:
        for port in range(12841,12844):
            h=(root/('.eval-port-%d.lock'%port)).open('a'); handles.append(h)
            fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with socket.socket() as sock: sock.bind(('127.0.0.1',port))
        summaries=[]; baseline=None
        for stage,workers in enumerate(a.workers,1):
            stageout=out/('stage-%d-workers-%d'%(stage,workers)); stageout.mkdir()
            save(stageout/'system-before.json',system_state())
            ports=queue.Queue()
            for port in range(12841,12841+workers): ports.put(port)
            def execute(task):
                port=ports.get()
                try: return run_case(root,stageout,task,port,a.seconds)
                finally: ports.put(port)
            print('\nSTAGE %d: %d workers, %d cases'%(stage,workers,len(tasks)),flush=True)
            started=time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                rows=list(pool.map(execute,tasks))
            elapsed=time.monotonic()-started
            save(stageout/'system-after.json',system_state())
            save(stageout/'cases.json',rows)
            indexed={(r['domain'],r['configuration'],r['problem']):r for r in rows}
            if baseline is None: baseline=indexed
            paired=[]; changes=[]
            for key,r in indexed.items():
                b=baseline[key]
                if r['status']!=b['status']: changes.append(dict(case=key,baseline=b['status'],current=r['status']))
                if r['status']==b['status']=='solved':
                    paired.append(r['end_to_end_wall_seconds']/b['end_to_end_wall_seconds'])
            import statistics
            summary=dict(stage=stage,workers=workers,cases=len(rows),elapsed_seconds=elapsed,
                 solved=sum(r['status']=='solved' for r in rows),
                 throughput_speedup=(summaries[0]['elapsed_seconds']/elapsed if summaries else 1),
                 median_paired_solved_slowdown=statistics.median(paired) if paired else None,
                 status_changes=changes)
            summaries.append(summary); save(out/'summary.json',summaries)
            print(json.dumps(summary,indent=2),flush=True)
        for f,expected in hashes.items():
            if sha(Path(f))!=expected: raise RuntimeError('Source/problem changed during benchmark: '+f)
        print('\nBENCHMARK COMPLETE. Send:',out/'summary.json',flush=True)
    finally:
        for h in handles: h.close()

if __name__=='__main__':
    main()
