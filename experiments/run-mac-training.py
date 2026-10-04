import argparse
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument("--port-file", required=True)
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--max-runs", type=int, default=1000)
parser.add_argument("--time-limit", type=int, default=60)
parser.add_argument("--finish-dir", required=True)
parser.add_argument("--log-dir", required=True)

args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = (
    ROOT
    / "experiments"
    / "blocksworld"
    / "blocksworld"
    / "train"
)

DOMAIN = (
    ROOT
    / "experiments"
    / "blocksworld"
    / "blocksworld"
    / "domain.pddl"
)

if not DOMAIN.exists():
    DOMAIN = ROOT / "gen-benchmarks" / "blocksworld" / "domain.pddl"

problems = sorted(
    TRAIN_DIR.glob("prob*.pddl"),
    key=lambda p: int(p.stem[4:])
)

if not problems:
    raise RuntimeError("No training problems found.")

port_file = Path(args.port_file)
finish_dir = Path(args.finish_dir)
log_dir = Path(args.log_dir)

log_dir.mkdir(parents=True, exist_ok=True)

print(f"Training instances: {len(problems)}")
print(f"Port file: {port_file}")
print()

run = 0

while run < args.max_runs:

    # The RL environment writes the port file when it is ready
    # for the next planning episode.
    while not port_file.exists():
        if finish_dir.is_dir():
            print("Training finished. Launcher stopping.", flush=True)
            raise SystemExit(0)

        time.sleep(0.1)

    if finish_dir.is_dir():
        print("Training finished. Launcher stopping.", flush=True)
        raise SystemExit(0)

    problem = problems[run % len(problems)]

    print(
        f"[episode {run + 1}] "
        f"{problem.name}",
        flush=True
    )

    search = (
        "rl_eager("
        "rl([single(ff()),single(cg()),single(cea()),single(add())],"
        f"random_seed={args.seed},"
        "handle_empty_list=true),"
        "rl_control_interval=0,"
        f"rl_client_port={args.port})"
    )

    command = [
        "python",
        str(ROOT / "fast-downward" / "fast-downward.py"),
        "--overall-time-limit",
        str(args.time_limit),
        str(DOMAIN),
        str(problem),
        "--search",
        search,
    ]

    log_file = log_dir / (
        f"episode_{run + 1:04d}_{problem.stem}.log"
    )

    with log_file.open("w") as output:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
        )

    print(
        f"    planner exit code: {result.returncode}",
        flush=True
    )

    run += 1

print("Launcher reached max-runs.")
