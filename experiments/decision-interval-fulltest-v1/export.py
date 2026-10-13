#!/usr/bin/env python3
"""Export only this fulltest's provenance, diagnostics and results."""
import argparse
from pathlib import Path
import tarfile
import time

HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label',required=True,choices=('primary','secondary'))
    args=p.parse_args();root=Path.cwd().resolve()
    result=root/'results-decision-interval-fulltest-v1'
    if not result.is_dir():raise SystemExit('No fulltest results found in this project')
    dest=Path.home()/'Downloads'/('rl-plan-decision-interval-fulltest-%s-%s.tar.gz'%(
        args.label,time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())))
    with tarfile.open(dest,'x:gz') as t:
        for path in sorted(result.rglob('*')):
            if path.is_file() and not path.is_symlink() and path.suffix in ('.json','.jsonl','.csv','.log'):
                t.add(path,arcname=str(path.relative_to(root)),recursive=False)
        for path in sorted(HERE.rglob('*')):
            if path.is_file() and not path.is_symlink() and 'planner' not in path.relative_to(HERE).parts and '__pycache__' not in path.parts:
                t.add(path,arcname='package/'+str(path.relative_to(HERE)),recursive=False)
    print('Upload this file:\n'+str(dest))
