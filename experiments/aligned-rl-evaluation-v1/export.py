"""Export compact auditable results without trajectory-heavy eval.json files."""
import argparse
from pathlib import Path
import tarfile


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--label',choices=('primary','secondary'),required=True)
    args=p.parse_args()
    root=Path.cwd().resolve()
    results=root/'results-aligned-rl-evaluation-v1'
    if not results.is_dir():raise SystemExit('No results-aligned-rl-evaluation-v1 folder in current project')
    package=Path(__file__).resolve().parent
    files=[]
    for path in results.rglob('*'):
        if not path.is_file() or any(part.startswith('attempt-') for part in path.parts):continue
        if path.parent.parent == results and path.suffix in ('.json','.csv'):
            files.append(path)
        elif path.name in ('result.json','config.json'):
            files.append(path)
    files+=list(package.glob('*.py'))+list(package.glob('*.txt'))+list(package.glob('*.md'))
    files+=list((package/'runtime').glob('*.py'))
    destination=Path.home()/'Downloads'/('rl-plan-aligned-rl-%s.tar.gz'%args.label)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(destination,'w:gz') as archive:
        for path in sorted(set(files)):
            archive.add(path,arcname=str(path.relative_to(root)),recursive=False)
    print('Send this file:',destination)
    print('Includes partial results if evaluation is still running; completion markers identify finished domains.')


if __name__=='__main__':main()
