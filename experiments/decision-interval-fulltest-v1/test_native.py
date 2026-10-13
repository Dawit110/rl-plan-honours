#!/usr/bin/env python3
"""Native development tests. NumPy inference here is ONLY a test action source.
The shipped pilot uses the actual frozen Chainer implementation.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
import numpy as np
from controller import decode, flatten

def read_exact(conn,n):
    out=b''
    while len(out)<n:
        part=conn.recv(n-len(out))
        if not part:raise EOFError('Planner disconnected')
        out+=part
    return out

def act(weights,obs):
    x=np.asarray(obs,dtype=np.float32)
    for layer in ('hidden_layers/0','hidden_layers/1','output'):
        x=weights['model/'+layer+'/W']@x+weights['model/'+layer+'/b']
        if layer!='output':x=np.maximum(x,0)
    return int(x.argmax())

def session(binary,sas,weights,mode,limit=384,fragment=False,trace=False):
    records=[];previous=None;packets=[]
    with tempfile.TemporaryDirectory() as tmp,socket.socket() as server:
        server.bind(('127.0.0.1',0));server.listen();server.settimeout(30)
        port=server.getsockname()[1]
        search=('rl_eager(rl([single(ff()),single(cg()),single(cea()),single(add())],random_seed=1,'
            'handle_empty_list=true),rl_control_interval=0,rl_client_port=%d'%port)
        if mode is not None:search+=',interval_mode=%d'%mode
        search+=')'
        env=dict(os.environ)
        if trace:env['RL_INTERVAL_TRACE']='1'
        with open(sas,'rb') as inp,open(Path(tmp)/'log','w') as log:
            proc=subprocess.Popen([str(binary),'--search',search],stdin=inp,stdout=log,stderr=subprocess.STDOUT,cwd=tmp,env=env)
            try:
                try:
                    conn,_=server.accept();conn.settimeout(30)
                except TimeoutError:
                    raise RuntimeError('Native connection failed: '+Path(tmp,'log').read_text())
                with conn:
                    for i in range(limit+1):
                        count=int(read_exact(conn,4));data=decode(read_exact(conn,count));packets.append(data)
                        current=np.asarray(flatten(data),dtype=np.float64)
                        prev=flatten(data['interval_previous']) if mode is not None else previous
                        if prev is None:prev=current
                        obs=(current-np.asarray(prev)).astype(np.float32)
                        previous=current
                        action=0 if i==0 else act(weights,obs)
                        records.append((current.tolist(),obs.tolist(),action,int(data['done'])))
                        if data['done']:break
                        message=b'END' if i==limit else str(action).encode()
                        wire=('%4d'%len(message)).encode()+message
                        if fragment:
                            for byte in wire:conn.sendall(bytes([byte]));time.sleep(.0001)
                        else:conn.sendall(wire)
                    proc.wait(timeout=10)
                assert proc.returncode==0,Path(tmp,'log').read_text()
            finally:
                if proc.poll() is None:proc.kill();proc.wait()
        log=Path(tmp,'log').read_text()
        stats={k:re.findall(p,log) for k,p in [('expanded',r'Expanded (\d+) state'),('cost',r'Plan cost: (\d+)')]}
        if trace:
            states=[ast.literal_eval(s.replace('-inf','0').replace('inf','0')) for s in Path(tmp,'interval-trace.jsonl').read_text().splitlines()]
            for p in packets[1:]:
                removals=p['interval_meta']['removals']
                assert p['interval_previous']==states[removals-1]['stats'],'Previous stats are not one removal ago'
                if not p['done'] and removals < len(states):
                    assert flatten(p)==flatten(states[removals]['stats'])
            assert len(states)>len(packets) or mode==0 or len(states)<16
        return records,stats,packets

def main():
    p=argparse.ArgumentParser();p.add_argument('--legacy',type=Path,required=True);p.add_argument('--patched',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True);args=p.parse_args()
    provenance=json.loads((Path(__file__).parent/'input-provenance.json').read_text())
    outcomes=[]
    for domain,design in provenance['domains'].items():
        root=args.inputs/('primary' if domain in ('blocksworld','rovers','visitall') else 'secondary')
        base=root/'experiments'/domain/domain
        with tempfile.TemporaryDirectory() as tmp:
            command=[sys.executable,str(args.legacy/'fast-downward.py'),'--translate',str(base/'domain.pddl'),str(base/'train'/design['problems'][0])]
            with open(Path(tmp)/'translate.log','w') as log:
                subprocess.run(command,cwd=tmp,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
            sas=Path(tmp)/'output.sas'
            for c in design['candidates']:
                suffix='_finish' if c['budget']==1000000 else '_snapshot'
                model=root/('results-%s-original-full100-v1-seed%d'%(domain,c['seed']))/'original/agent'/(str(c['budget'])+suffix)/'model.npz'
                weights=np.load(model)
                legacy=session(args.legacy/'builds/release/bin/downward',sas,weights,None)
                compat=session(args.patched/'builds/release/bin/downward',sas,weights,0,fragment=True)
                assert legacy[:2]==compat[:2],(domain,c['seed'],'compatibility mismatch')
                print('MATCH',domain,c['seed'],len(legacy[0]),legacy[1],flush=True)
                outcomes.append(dict(domain=domain,seed=c['seed'],observations=len(legacy[0]),statistics=legacy[1]))
                if c['seed']==1:
                    for mode in (1,2):
                        result=session(args.patched/'builds/release/bin/downward',sas,weights,mode,limit=64,trace=True)
                        print('INTERVAL',domain,mode,len(result[0]),result[2][-1]['interval_meta']['removals'],flush=True)
    print(json.dumps(outcomes,indent=2))

if __name__=='__main__':main()
