#!/usr/bin/env python3
"""One isolated case supervisor, launched in a new session by benchmark.py."""
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time
import traceback
from cpu_meter import Meter


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def child_cpu():
    u = resource.getrusage(resource.RUSAGE_CHILDREN)
    return u.ru_utime + u.ru_stime


def cleanup(meter, processes):
    # Signal only processes verified as still in this case's private group.
    for sig, seconds in ((signal.SIGTERM, 1.0), (signal.SIGKILL, 2.0)):
        owned = sorted(set(meter.members()) | meter.descendants(os.getpid()))
        for pid in reversed(owned):
            if pid == os.getpid():
                continue
            try:
                belongs = os.getpgid(pid) == meter.pgid or pid in meter.descendants(os.getpid())
                if belongs and not meter.usage(pid)['exited']:
                    os.kill(pid, sig)
            except (ProcessLookupError, FileNotFoundError):
                pass
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            for p in processes:
                p.poll()
            remaining = []
            for pid in sorted(set(meter.members()) | meter.descendants(os.getpid())):
                if pid == os.getpid():
                    continue
                try:
                    if not meter.usage(pid)['exited']:
                        remaining.append(pid)
                except (ProcessLookupError, FileNotFoundError):
                    pass
            if not remaining:
                for p in processes:
                    p.wait(timeout=2)
                return
            time.sleep(0.02)
    raise RuntimeError('Owned descendants survived cleanup: %r' % remaining)


def run(config, directory):
    if os.getpid() != os.getpgrp():
        raise RuntimeError('Case must be launched with start_new_session=True')
    meter = Meter(os.getpid())
    result = dict(config=config, completed=False, status='starting',
        cpu_seconds=None, wall_seconds=None, peak_sampled_resident_bytes=0,
        max_case_processes=0, planner_exit_code=None, controller_exit_code=None)
    procs = []
    controller = planner = None
    start = time.monotonic()
    base_waited = child_cpu()
    decision = None
    def check():
        nonlocal decision
        sample = meter.snapshot()
        result['peak_sampled_resident_bytes'] = max(result['peak_sampled_resident_bytes'], sample['resident_bytes'])
        result['max_case_processes'] = max(result['max_case_processes'], sample['process_count'])
        result['snapshot_retries'] = sample['snapshot_retries']
        result['cpu_seconds'] = sample['cpu_seconds']
        result['wall_seconds'] = time.monotonic() - start
        if sample['resident_bytes'] >= config.get('resident_limit_bytes', float('inf')):
            decision = 'memory_limit'
        elif result['cpu_seconds'] >= config['cpu_seconds']:
            decision = 'cpu_timeout'
        elif result['wall_seconds'] >= config['wall_safeguard']:
            decision = 'wall_safeguard'
        return decision
    def interrupt(signum, frame):
        raise KeyboardInterrupt('Case cancelled')
    signal.signal(signal.SIGTERM, interrupt)
    try:
        with (directory / 'controller.log').open('w') as clog, (directory / 'planner.log').open('w') as plog:
            if config.get('controller'):
                controller = subprocess.Popen(config['controller'], cwd=directory, stdout=clog, stderr=subprocess.STDOUT)
                procs.append(controller)
                port_file = directory / ('port_%d.txt' % config['port'])
                while not port_file.exists():
                    if check():
                        break
                    if controller.poll() is not None:
                        raise RuntimeError('Controller exited before readiness; see controller.log')
                    time.sleep(0.02)
            if not check():
                planner = subprocess.Popen(config['planner'], cwd=directory, stdout=plog, stderr=subprocess.STDOUT)
                procs.append(planner)
                while True:
                    code = planner.poll()
                    if check():
                        break
                    if code is not None:
                        break
                    if controller is not None and controller.poll() is not None:
                        # Normal controller exit can precede the planner's final output.
                        if controller.returncode != 0:
                            raise RuntimeError('Controller crashed; see controller.log')
                    time.sleep(0.02)
            if decision:
                result['status'] = decision
            else:
                text = (directory / 'planner.log').read_text(errors='replace')
                if planner.returncode == 0 and 'Solution found.' in text:
                    result['status'] = 'solved'
                elif planner.returncode in (11, 12):
                    result['status'] = 'search_unsolved'
                else:
                    raise RuntimeError('Planner exit %r; inspect planner.log' % planner.returncode)
                for key, pattern in [('expanded', r'Expanded (\d+) state'),
                        ('generated', r'Generated (\d+) state'), ('plan_length', r'Plan length: (\d+)')]:
                    values = re.findall(pattern, text)
                    result[key] = int(values[-1]) if values else None
                if controller is not None and controller.poll() not in (None, 0):
                    raise RuntimeError('Controller failed after planner exit; inspect controller.log')
            result['completed'] = True
    except KeyboardInterrupt as exc:
        result.update(status='interrupted', error=str(exc))
    except Exception as exc:
        result.update(status='runner_error', error=str(exc), traceback=traceback.format_exc())
    finally:
        # The decision CPU sample precedes cleanup; cleanup is outside the budget.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            cleanup(meter, procs)
        except Exception as exc:
            result.update(completed=False, status='cleanup_error', error=str(exc), traceback=traceback.format_exc())
        result['waited_child_cpu_after_cleanup'] = child_cpu() - base_waited
        result['total_wall_seconds'] = time.monotonic() - start
        if planner is not None:
            result['planner_exit_code'] = planner.poll()
        if controller is not None:
            result['controller_exit_code'] = controller.poll()
        result['cpu_overshoot_seconds'] = (max(0, result['cpu_seconds'] - config['cpu_seconds'])
            if result['cpu_seconds'] is not None else None)
        save(directory / 'result.json', result)
    return 0 if result['completed'] and result['status'] != 'wall_safeguard' else 2


if __name__ == '__main__':
    path = Path(sys.argv[1]).resolve()
    sys.exit(run(json.loads(path.read_text()), path.parent))
