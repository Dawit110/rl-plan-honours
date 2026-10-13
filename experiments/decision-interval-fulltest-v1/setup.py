#!/usr/bin/env python3
"""Build an isolated planner from hash-checked local sources."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import platform
import preflight as pf

HERE = Path(__file__).resolve().parent
PLANNER = HERE / 'planner'

def build(args):
    root = Path.cwd().resolve()
    provenance = pf.read(HERE/'input-provenance.json')
    for relative, expected in provenance['planner'].items():
        pf.require(pf.sha(root/relative) == expected, 'Planner source changed: '+relative)
    replacements = {str(p.relative_to(HERE/'planner_patch')):p
                    for p in (HERE/'planner_patch').rglob('*') if p.is_file()}
    expected = {k[len('fast-downward/'):]:v for k,v in provenance['planner'].items()}
    expected.update({k:pf.sha(p) for k,p in replacements.items()})
    if (HERE/'build.json').exists():
        previous = pf.read(HERE/'build.json')
        pf.require(previous['source_hashes'] == expected, 'Existing isolated build uses different source')
        for relative, digest in previous['artifact_hashes'].items():
            pf.require(pf.sha(PLANNER/relative) == digest, 'Built artifact changed: '+relative)
        for relative, digest in expected.items():
            pf.require(pf.sha(PLANNER/relative) == digest, 'Isolated source changed: '+relative)
        print('Existing isolated build verified:', PLANNER, flush=True)
        return
    pf.require(shutil.which(args.cmake), 'cmake is unavailable in this environment. Activate rl-plan and install cmake if needed.')
    # Resume an incomplete build, but never overwrite unexpected source bytes.
    for relative in expected:
        target = PLANNER/relative
        original = root/'fast-downward'/relative
        desired = replacements.get(relative, original).read_bytes()
        if target.exists():
            pf.require(target.read_bytes() == desired, 'Unexpected isolated source: '+str(target))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(desired)
    build_dir = PLANNER/'builds/release'
    commands = [
        [args.cmake, '-S', str(PLANNER/'src'), '-B', str(build_dir),
         '-DCMAKE_BUILD_TYPE=Release', '-DUSE_LP=NO', '-DCMAKE_POLICY_VERSION_MINIMUM=3.5'],
        [args.cmake, '--build', str(build_dir), '--parallel', str(args.jobs)]]
    with (HERE/'build.log').open('a') as log:
        for command in commands:
            print('Building isolated planner; log:', HERE/'build.log', flush=True)
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    artifacts = [build_dir/'bin/downward'] + list((build_dir/'bin/translate').rglob('*.py'))
    result = dict(source_hashes=expected, artifact_hashes={str(p.relative_to(PLANNER)):pf.sha(p) for p in artifacts},
                  commands=commands, platform=platform.platform(), python=sys.version,
                  cmake_version=subprocess.check_output([args.cmake,'--version'], text=True),
                  cmake_cache=(build_dir/'CMakeCache.txt').read_text())
    (HERE/'build.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print('BUILD COMPLETE:', build_dir/'bin/downward', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, choices=range(1,5), default=2)
    parser.add_argument('--cmake', default='cmake')
    build(parser.parse_args())
