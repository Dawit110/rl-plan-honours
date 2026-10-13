import json
from pathlib import Path
import tempfile
import types
import unittest
import run as r


class AlignedTests(unittest.TestCase):
    def fixture(self,root):
        base=root/'experiments/barman/barman'
        for split in ('train','test'):
            (base/split).mkdir(parents=True)
            for i in range(100):(base/split/('p%d.pddl'%i)).write_text(split+str(i))
        (base/'domain.pddl').write_text('domain')
        fd=root/'fast-downward/builds/release/bin/downward';fd.parent.mkdir(parents=True);fd.write_text('binary')
        fdpy=root/'fast-downward/fast-downward.py';fdpy.write_text('driver')
        hashes={str(p.relative_to(root)):r.pf.sha(p) for p in list(base.rglob('*.pddl'))+[fd,fdpy]}
        for model in r.MODELS:
            for seed in (1,2,3):
                folder=r.pf.trained(root,'barman',model,seed)
                cp=r.pf.checkpoint(folder,model,300000);cp.mkdir(parents=True)
                (folder/'source').mkdir()
                for path in [cp/'model.npz',folder/'source/train_and_eval_chainer_agent.py',folder/'source/fd_env.py',folder/'source/evaluate.py']:
                    path.write_text('fixture '+str(path));hashes[str(path.relative_to(root))]=r.pf.sha(path)
        old=root/'results-full100-cpu-v1/barman';old.mkdir(parents=True)
        (old/'protocol.json').write_text(json.dumps(dict(hashes=hashes)))
        (old/'evaluation-complete.json').write_text(json.dumps(dict(protocol_sha256=r.pf.sha(old/'protocol.json'))))
        for model in r.MODELS:
            for seed in (1,2,3):
                (old/('selected-%s-seed%d.json'%(model,seed))).write_text(json.dumps(dict(
                    domain='barman',model=model,seed=seed,selection_role='training subset; not held out',cpu_budget=300,
                    selected_budget=300000,candidates=[dict(budget=b,solved=99 if b==300000 else 90,capped_cpu_sum=1000) for b in r.pf.BUDGETS])))
        corrected=root/'results-paper-followup-v1/protocol.json';corrected.parent.mkdir()
        ch={'project/'+k:v for k,v in hashes.items() if k.startswith(('experiments/','fast-downward/'))}
        ch.update({n:r.pf.sha(r.HERE/n) for n in ('case.py','cpu_meter.py')})
        corrected.write_text(json.dumps(dict(hashes=ch)))

    def test_full_prepare_and_frozen_model_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.fixture(root)
            args=types.SimpleNamespace(wall_safeguard=1800)
            p=r.prepare(root,'barman',args,False)
            self.assertEqual(len(p['candidates']),6)
            self.assertEqual(p['decision_limit'],2**63-1)
            r.verify_inputs(root,p)
            path=r.pf.trained(root,'barman','progress',2)/'source/fd_env.py'
            path.write_text('modified')
            with self.assertRaises(RuntimeError):r.prepare(root,'barman',args,False)
            with self.assertRaises(RuntimeError):r.verify_inputs(root,p)

    def test_arguments_all_seeds_and_distinct_output_cases(self):
        p=dict(root='/fixture',domain='barman',test=['p%d.pddl'%i for i in range(100)],
               candidates=[dict(model=m,seed=s,budget=300000) for m in r.MODELS for s in (1,2,3)])
        r.PROTOCOLS['barman']=p;tasks=r.tasks_for(p)
        self.assertEqual(len(tasks),600)
        r.engine.configuration=r.configuration
        self.assertEqual(len({str(r.engine.case_dir(Path('/out'),t)) for t in tasks}),600)
        args=types.SimpleNamespace(wall_safeguard=1800)
        for t in tasks[:6]:
            c=r.config_for(Path('/fixture'),t,13841,args);cmd=c['controller']
            self.assertEqual(cmd[cmd.index('--time-step-limit')+1],str(2**63-1))
            self.assertEqual(cmd[cmd.index('--seed')+1],str(t['seed']))
            self.assertEqual(cmd[cmd.index('--temporal-window')+1],'0' if t['model']=='original' else '4')
            self.assertIn('/source/train_and_eval_chainer_agent.py',cmd[2])
            self.assertIn('/300000_snapshot',cmd[cmd.index('--load')+1])
            self.assertEqual(c['cpu_seconds'],300);self.assertEqual(c['resident_limit_bytes'],4*1024**3)
            self.assertIn('/test/',c['planner'][-3])
            row=dict(config=c,completed=True,status='solved',cpu_seconds=2,planner_exit_code=0)
            self.assertTrue(r.valid(row,t,args))
            row['config']['resident_limit_bytes']=8*1024**3
            with self.assertRaises(RuntimeError):r.valid(row,t,args)

    def test_selection_cannot_be_changed_to_test_choice(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.fixture(root)
            p=root/'results-full100-cpu-v1/barman/selected-original-seed1.json'
            choice=r.pf.read(p);choice['selected_budget']=100000;p.write_text(json.dumps(choice))
            with self.assertRaisesRegex(RuntimeError,'rank differs'):
                r.prepare(root,'barman',types.SimpleNamespace(wall_safeguard=1800),False)


if __name__=='__main__':unittest.main()
