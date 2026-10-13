#!/usr/bin/env python3
"""Bounded diagnostic session; deliberately separate from performance results."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from cpu_meter import Meter
from case import cleanup, save

def main(config_path):
    config_path=Path(config_path).resolve();directory=config_path.parent
    config=json.loads(config_path.read_text())
    if os.getpid()!=os.getpgrp():raise RuntimeError('Private diagnostic group required')
    meter=Meter(os.getpid());procs=[];start=time.monotonic();result={}
    def interrupt(signum,frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupt)
    def check():
        sample=meter.snapshot()
        result.update(cpu_seconds=sample['cpu_seconds'],wall_seconds=time.monotonic()-start,
                      peak_sampled_rss=max(result.get('peak_sampled_rss',0),sample['resident_bytes']))
        if sample['cpu_seconds']>=300 or sample['resident_bytes']>=4*1024**3 or time.monotonic()-start>=600:
            raise RuntimeError('Diagnostic resource limit reached; compatibility has not been established')
    try:
        with (directory/'controller.log').open('w') as clog,(directory/'planner.log').open('w') as plog:
            controller=subprocess.Popen(config['controller'],cwd=directory,stdout=clog,stderr=subprocess.STDOUT)
            procs.append(controller)
            while not (directory/('port_%d.txt'%config['port'])).exists():
                check()
                if controller.poll() is not None:raise RuntimeError('Diagnostic controller failed before readiness')
                time.sleep(.02)
            planner=subprocess.Popen(config['planner'],cwd=directory,stdout=plog,stderr=subprocess.STDOUT)
            procs.append(planner)
            while controller.poll() is None or planner.poll() is None:
                check()
                if controller.poll() not in (None,0):raise RuntimeError('Diagnostic controller crashed')
                # A trivial warm-up solution cannot return through the legacy reset loop.
                if planner.poll() is not None and controller.poll() is None:
                    if 'Solution found.' in (directory/'planner.log').read_text(errors='replace'):
                        break
                time.sleep(.02)
            check()
            log=(directory/'planner.log').read_text(errors='replace')
            if planner.poll()!=0:raise RuntimeError('Diagnostic planner failed: inspect '+str(directory/'planner.log'))
            if 'Solution found.' not in log and 'Termination due to RL agent' not in log:
                raise RuntimeError('Diagnostic ended without a solution or the deliberate decision cutoff')
            result.update(ok=True,solved='Solution found.' in log,
                          truncated='Termination due to RL agent' in log)
    except BaseException as exc:
        result.update(ok=False,error=str(exc))
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        cleanup(meter,procs);save(directory/'diagnostic.json',result)
    return 0 if result.get('ok') else 2

if __name__=='__main__':sys.exit(main(sys.argv[1]))
