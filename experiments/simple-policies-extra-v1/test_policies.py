"""Behavioral tests of the historical rules and add-on task isolation."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import types
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'runtime'))
from policies import SimpleAgent, POLICIES
import run as runner


def obs(minima=(100,100,100,100), queues=(10,10,10,10)):
    return [x for h in range(4) for x in (200,300,minima[h],queues[h],0)]


class Rules(unittest.TestCase):
    def test_recent_waits_full_window_then_can_switch_each_step(self):
        a=SimpleAgent('recent50')
        for _ in range(50):self.assertEqual(a.act(obs()),0)
        self.assertEqual(a.act(obs((100,90,100,100))),1)
        self.assertEqual(a.act(obs((100,90,80,100))),2)

    def test_recent_zero_progress_tie_stays(self):
        a=SimpleAgent('recent50')
        for _ in range(100):self.assertEqual(a.act(obs()),0)

    def test_stag_progress_waits_then_selects_alternative(self):
        a=SimpleAgent('stagprog50')
        for _ in range(50):self.assertEqual(a.act(obs()),0)
        self.assertEqual(a.act(obs((100,99,80,100))),2)
        # A strict new minimum resets the timer.
        for _ in range(49):self.assertEqual(a.act(obs((100,99,80,100))),2)
        self.assertEqual(a.act(obs((100,99,79,100))),2)
        self.assertEqual(a.agent.stagnation,0)

    def test_stag_progress_ties_rotate(self):
        a=SimpleAgent('stagprog50')
        for _ in range(50):a.act(obs())
        self.assertEqual(a.act(obs()),1)
        for _ in range(49):a.act(obs())
        self.assertEqual(a.act(obs()),2)

    def test_probe_exact_durations_post_action_scores_and_rotation(self):
        a=SimpleAgent('probe25x100');mins=[100]*4;actions=[]
        for i in range(201):
            action=a.act(obs(mins));actions.append(action)
            mins[action]-=action
        self.assertEqual(actions,[h for h in range(4) for _ in range(25)]+[3]*100+[1])

    def test_probe_skips_empty_queue_and_all_empty_terminates_call(self):
        a=SimpleAgent('probe25x100')
        self.assertEqual(a.act(obs(queues=(0,10,10,10))),1)
        self.assertEqual(a.act(obs(queues=(0,0,0,0))),0)

    def test_queue_pressure_changes_choice_and_shrink_is_not_bonus(self):
        a=SimpleAgent('queue50a1')
        for _ in range(50):a.act(obs())
        self.assertEqual(a.act(obs((50,80,100,100),(20,10,10,10))),1)
        self.assertEqual(a.agent.queue_pressure(20,10),0)
        self.assertEqual(a.agent.queue_pressure(10,20),.5)

    def test_queue_filter_after_warmup(self):
        a=SimpleAgent('queue50a1')
        for _ in range(50):a.act(obs())
        self.assertEqual(a.act(obs((0,90,100,100),(0,10,10,10))),1)

    def test_episode_reset_and_bad_input(self):
        for name in POLICIES:
            a=SimpleAgent(name)
            for _ in range(80):a.act(obs())
            a.stop_episode();self.assertEqual(a.act(obs()),0)
            with self.assertRaises(ValueError):a.act([0]*28)
            with self.assertRaises(ValueError):a.act(obs(queues=(float('nan'),1,1,1)))

    def test_class_bodies_match_recorded_earlier_sources(self):
        text=(HERE/'runtime/historical.py').read_text();tree=ast.parse(text)
        nodes={n.name:n for n in tree.body if isinstance(n,ast.ClassDef)}
        for entry in json.loads((HERE/'source-provenance.json').read_text()):
            n=nodes[entry['class_name']]
            body='\n'.join(text.splitlines()[n.lineno-1:n.end_lineno])
            self.assertEqual(hashlib.sha256(body.encode()).hexdigest(),entry['class_body_sha256'])

    def test_independent_task_paths_and_configuration(self):
        p=dict(domain='barman',root='/fixture',test=['p%d.pddl'%i for i in range(100)])
        runner.PROTOCOLS['barman']=p
        tasks=runner.tasks_for(p);self.assertEqual(len(tasks),400)
        self.assertEqual({t['model'] for t in tasks},set(POLICIES))
        self.assertFalse(set(POLICIES)&{'rr1','rr50','stag50','progress50'})
        args=types.SimpleNamespace(wall_safeguard=1800)
        for task in tasks[:4]:
            c=runner.config_for(Path('/fixture'),task,13641,args)
            self.assertEqual(c['cpu_seconds'],300)
            self.assertEqual(c['resident_limit_bytes'],4*1024**3)
            self.assertEqual(c['planner'][-1].count('single('),4)
            self.assertIn('handle_empty_list=true',c['planner'][-1])
            self.assertIn('simple-policies-extra-v1',c['controller'][2])
            row=dict(config=c,completed=True,status='solved',cpu_seconds=2,planner_exit_code=0)
            self.assertTrue(runner.valid(row,task,args))


if __name__=='__main__':unittest.main()
