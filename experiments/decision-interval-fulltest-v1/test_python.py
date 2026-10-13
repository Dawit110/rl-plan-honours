#!/usr/bin/env python3
import ast
from copy import deepcopy
from enum import Enum
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import controller
import run

class StateType(Enum):
    RAW=1
    DIFF=2

def frozen_processor(path):
    tree=ast.parse(path.read_text())
    definition=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FDEnvSelHeur')
    method=next(n for n in definition.body if isinstance(n,ast.FunctionDef) and n.name=='_process_data')
    definition.bases=[];definition.body=[method];definition.decorator_list=[]
    module=ast.Module(body=[definition],type_ignores=[])
    namespace=dict(np=np,deepcopy=deepcopy,StateType=StateType)
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),namespace)
    env=namespace['FDEnvSelHeur']()
    env._use_gsi=False;env._FDEnvSelHeur__num_heuristics=4
    env._heuristic_state_features=controller.FEATURES
    env._prev_state=None;env._FDEnvSelHeur__state_type=StateType.DIFF
    env._FDEnvSelHeur__skip_transform=[False]*20
    env._transformation_func=lambda x,y,z,skip:x-y
    env.temporal_window=0
    return env

class Tests(unittest.TestCase):
    def test_plan_is_frozen_and_unique(self):
        source=json.loads((run.HERE/'input-provenance.json').read_text())
        total=0
        for name,d in source['domains'].items():
            selected=[d['train_sha256'][p] for p in d['problems']]
            self.assertEqual(len(selected),len(set(selected)))
            self.assertFalse(set(selected)&set(d['test_sha256'].values()))
            protocol=dict(domain=name,candidates=d['candidates'],problems=sorted(d['test_sha256']))
            tasks=run.tasks_for(protocol);self.assertEqual(len(tasks),900)
            self.assertEqual(len({(t['problem'],run.configuration(t)) for t in tasks}),900)
            self.assertEqual({t['seed'] for t in tasks},{1,2,3});total+=len(tasks)
        self.assertEqual(total,5400)

    def test_real_frozen_feature_code(self):
        inputs=Path(sys.argv[1]) if len(sys.argv)>1 and not sys.argv[1].startswith('-') else None
        # Source roots are supplied through the module global by __main__.
        for source in INPUTS.glob('*/results-*-original-full100-v1-seed*/source/fd_env.py'):
            env=frozen_processor(source)
            def payload(n):
                data={str(i):{key:100000000.123456 + n*.123456 + j*.555555
                              for j,key in enumerate(controller.FEATURES)} for i in range(4)}
                data.update(reward=-1,done=0);return data
            initial=payload(0);env.recv_msg=lambda:json.dumps(initial).encode()
            first,_,_=env._process_data();np.testing.assert_array_equal(first,np.zeros(20))
            previous=payload(15);current=payload(16)
            current['interval_previous']={str(i):previous[str(i)] for i in range(4)}
            controller.adapt_previous(env,current)
            env.recv_msg=lambda:json.dumps(current).encode()
            actual,_,_=env._process_data()
            expected=np.asarray(controller.flatten(current))-np.asarray(controller.flatten(previous))
            np.testing.assert_array_equal(actual,expected)
            self.assertFalse(np.array_equal(actual,np.asarray(controller.flatten(current))-np.asarray(controller.flatten(initial))))

    def test_resource_supervisor_bytes_unchanged(self):
        import hashlib
        for name in ('case.py','cpu_meter.py'):
            local=(run.HERE/name).read_bytes()
            original=(INPUTS/'primary/experiments/aligned-rl-evaluation-v1'/name).read_bytes()
            self.assertEqual(hashlib.sha256(local).digest(),hashlib.sha256(original).digest())

    def test_command_semantics(self):
        source=json.loads((run.HERE/'input-provenance.json').read_text())
        d=source['domains']['barman'];protocol=dict(d,domain='barman',problems=sorted(d['test_sha256']))
        task=run.tasks_for(protocol)[0]
        args=type('Args',(),dict(wall_safeguard=1800))()
        for cond in run.CONDITIONS:
            c=run.config_for(Path('/repo'),task,14841,args,condition=cond)
            self.assertEqual(c['cpu_seconds'],300);self.assertEqual(c['resident_limit_bytes'],4*1024**3)
            self.assertIn('/test/',c['planner'][-3])
            diagnostic=dict(task,phase='verification',problem=d['problems'][0])
            self.assertIn('/train/',run.config_for(Path('/repo'),diagnostic,14841,args)['planner'][-3])
            self.assertEqual('--extended' in c['controller'],cond!='original')
            self.assertIn(str(2**63-1),c['controller'])

if __name__=='__main__':
    INPUTS=Path(sys.argv.pop(1)).resolve()
    unittest.main()
