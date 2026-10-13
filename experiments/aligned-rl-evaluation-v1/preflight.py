"""Read-only input checks for the full100 CPU evaluation."""
import hashlib
import importlib.util
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import socket
import sys

MODELS = ('original', 'progress')
SEEDS = (1, 2, 3)
BUDGETS = (100000, 300000, 600000, 1000000)
FIXED = ('ff', 'cg', 'cea', 'add')
DOMAINS = ('blocksworld', 'rovers', 'visitall', 'barman', 'childsnack', 'sokoban')
THREADS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')
HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trained(root, domain, model, seed):
    return root / ('results-%s-%s-full100-v1-seed%d' % (domain, model, seed))


def checkpoint(folder, model, budget):
    return folder / model / 'agent' / (str(budget) + ('_finish' if budget == 1000000 else '_snapshot'))


def trainers(root):
    result = {}
    for model in MODELS:
        path = root / 'experiments/full100-training' / ('train-%s.py' % model)
        spec = importlib.util.spec_from_file_location('training_' + model, str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result[model] = module
    return result


def check(root, domain, bundles, args):
    base = root / 'experiments' / domain / domain
    hashes = {}
    def capture(path):
        require(path.is_file(), 'Missing file: ' + str(path))
        key = str(path.relative_to(root)) if root in path.parents else str(path)
        hashes[key] = sha(path)
    train_names = sorted(p.name for p in (base / 'train').glob('*.pddl'))
    test_names = sorted(p.name for p in (base / 'test').glob('*.pddl'))
    require(len(train_names) == len(test_names) == 100, 'Expected 100 train/100 test: ' + domain)
    for path in [base / 'domain.pddl'] + list((base / 'train').glob('*.pddl')) + list((base / 'test').glob('*.pddl')):
        capture(path)
    require(not {sha(p) for p in (base / 'train').glob('*.pddl')} &
            {sha(p) for p in (base / 'test').glob('*.pddl')}, 'Identical train/test file bytes: ' + domain)
    reference = None
    selections = {}
    for seed in SEEDS:
        for model in MODELS:
            folder = trained(root, domain, model, seed)
            p = read(folder / 'protocol.json')
            expected = dict(domain=domain, seed=seed, steps=1000000, episode_limit=7500,
                exploration_steps=500000, checkpoint_budgets=list(BUDGETS),
                eval_wall_seconds=300, models={model: 0 if model == 'original' else 4},
                selection_rule='coverage; capped runtime sum; earliest checkpoint',
                selection_data_role='training subset; not held out')
            for key, value in expected.items():
                require(p.get(key) == value, 'Unexpected %s in %s' % (key, folder))
            require(p['train'] == train_names, 'Training list mismatch: ' + str(folder))
            chosen = p['training_selection']
            require(len(chosen) == len(set(chosen)) == 20 and set(chosen) <= set(train_names),
                    'Invalid selection list: ' + str(folder))
            signature = (p['split_seed'], chosen)
            if reference is None:
                reference = signature
            require(signature == reference, 'Selection lists/split seeds differ: ' + str(folder))
            selections[str(seed) + '-' + model] = chosen
            for path in (folder / 'protocol.json', folder / model / 'training-complete.json', folder / 'snapshot-hashes.json'):
                capture(path)
            saved = read(folder / 'snapshot-hashes.json')
            for budget in BUDGETS:
                for name in ('model.npz', 'target_model.npz', 'optimizer.npz'):
                    path = checkpoint(folder, model, budget) / name
                    capture(path)
                    require(sha(path) == saved.get(str(path.relative_to(folder))), 'Checkpoint hash mismatch: ' + str(path))
            for name, entry in bundles[model].SOURCES.items():
                path = folder / 'source' / name
                capture(path)
                require(path.read_bytes() == entry['text'].encode(), 'Changed controller source: ' + str(path))
            for relative, expected_hash in p['hashes'].items():
                if relative.startswith(('experiments/' + domain + '/', 'fast-downward/driver/', 'fast-downward/src/')) or relative == 'fast-downward/fast-downward.py':
                    path = root / relative
                    capture(path)
                    require(sha(path) == expected_hash, 'Training source/data mismatch: ' + str(path))
    for path in list(HERE.glob('*.py')) + [root / 'fast-downward/fast-downward.py', root / 'fast-downward/builds/release/bin/downward']:
        capture(path)
    for model in MODELS:
        capture(root / 'experiments/full100-training' / ('train-%s.py' % model))
    translator = root / 'fast-downward/builds/release/bin/translate'
    require(translator.is_dir(), 'Missing built translator: ' + str(translator))
    for path in sorted(translator.rglob('*.py')):
        capture(path)
        source = root / 'fast-downward/src/translate' / path.relative_to(translator)
        require(source.is_file() and sha(source) == sha(path), 'Built translator differs: ' + str(path))
    return dict(version='full100-cpu-evaluation-v1', domain=domain, root=str(root),
        hostname=socket.gethostname(), platform=platform.platform(), python=sys.version,
        executable=sys.executable, first_port=args.port, seeds=list(SEEDS), budgets=list(BUDGETS),
        packages={name: metadata.version(name) for name in ('numpy', 'scipy', 'chainer', 'chainerrl', 'gym')},
        threads={key: os.environ[key] for key in THREADS}, cpu_seconds=300,
        wall_safeguard=args.wall_safeguard, workers=args.workers,
        timing='aggregate user+system CPU for controller, translation, driver, search and descendants; supervisor and cleanup excluded',
        decision_limit=1000000, memory_limit='not enforced', cpu_poll_seconds=0.02,
        training_selection=selections, test=test_names, fixed=list(FIXED),
        fixed_runs='once per domain; deterministic default fixed-search settings',
        selection_rule='coverage; capped CPU-time sum; earliest checkpoint',
        protocol_change='User chose aggregate CPU budgets instead of the originally recorded wall-time evaluation budget. Training/checkpoints unchanged.',
        worker_evidence='1/2/3/4/1 workers at 60 CPU seconds on both Macs; user chose to proceed without a 300-second benchmark',
        hashes=hashes)
