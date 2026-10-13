"""Compare bounded original/compatibility traces using actual frozen networks."""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import engine
import preflight as pf

HERE=Path(__file__).resolve().parent

def check(root,output,protocol,args):
    # Import the live module so the same task/config constructor is used when
    # run.py is executed as __main__.
    run=sys.modules['__main__']
    marker=output/'compatibility-passed.json'
    identity=dict(protocol_sha256=pf.sha(output/'protocol.json'),decision_prefix_limit=256)
    if marker.exists():
        old=pf.read(marker);pf.require(old['identity']==identity,'Compatibility record differs')
        for name,digest in old['evidence_hashes'].items():
            pf.require(pf.sha(output/name)==digest,'Compatibility evidence changed: '+name)
        print('COMPATIBILITY CHECK already passed:',protocol['domain'],flush=True);return
    evidence={};comparisons=[]
    for candidate in protocol['candidates']:
        task=dict(candidate,domain=protocol['domain'],phase='verification',
                  problem=protocol['verification_problem'],condition='original')
        records=[];counts=[];outcomes=[]
        for condition in ('original','compat'):
            parent=output/'verification'/('seed%d'%candidate['seed'])/condition
            directory=parent/str(time.time_ns());directory.mkdir(parents=True)
            config=run.config_for(root,task,args.port,args,condition=condition,trace=True,profile=True,decision_limit=256)
            engine.save(directory/'config.json',config)
            with (directory/'supervisor.log').open('w') as log:
                process=subprocess.Popen([sys.executable,str(HERE/'verify_case.py'),str(directory/'config.json')],
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    code=process.wait(timeout=620)
                except BaseException:
                    process.terminate()
                    try:process.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL);process.wait()
                    raise
            pf.require(code==0,'Compatibility check failed; inspect '+str(directory))
            trace=directory/'trace.jsonl';lines=[json.loads(s) for s in trace.read_text().splitlines()]
            pf.require(lines,'Empty compatibility trace')
            records.append(lines)
            log=(directory/'planner.log').read_text(errors='replace')
            counts.append({key:re.findall(pattern,log) for key,pattern in (
                ('expanded',r'Expanded (\d+) state'),('plan_cost',r'Plan cost: (\d+)'),('plan_length',r'Plan length: (\d+)'))})
            outcomes.append(pf.read(directory/'diagnostic.json')['solved'])
            for name in ('trace.jsonl','diagnostic.json','planner.log','controller-metrics.json'):
                path=directory/name;evidence[str(path.relative_to(output))]=pf.sha(path)
        pf.require(records[0]==records[1],'Raw/features/actions differ between Original and compatibility build')
        pf.require(counts[0]==counts[1] and outcomes[0]==outcomes[1],'Compatibility outcome/expansion/plan mismatch')
        decisions=sum(row['kind']=='act' for row in records[0])
        pf.require(decisions>0,'Selected compatibility problem solved during warm-up; no learned actions checked')
        comparisons.append(dict(seed=candidate['seed'],learned_decisions=decisions,problem=task['problem'],
                                solved=outcomes[0],counts=counts[0]))
        print('COMPATIBILITY PASS %s seed%d: %d learned decisions match exactly'%(
            protocol['domain'],candidate['seed'],decisions),flush=True)
    engine.save(marker,dict(identity=identity,comparisons=comparisons,evidence_hashes=evidence,
        scope='Identical raw observations, float32 model inputs, greedy actions; expansion/plan statistics when solved. Bounded prefixes are not full-search proofs.'))
