import argparse
import random
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DOMAINS = [
    "barman",
    "blocksworld",
    "childsnack",
    "rovers",
    "sokoban",
    "visitall",
]

parser = argparse.ArgumentParser()

parser.add_argument(
    "--domain",
    choices=DOMAINS,
    required=True
)

parser.add_argument(
    "--shuffle-seed",
    type=int,
    default=1
)

parser.add_argument(
    "--list-only",
    action="store_true"
)

parser.add_argument("--port-file")
parser.add_argument("--port", type=int)

parser.add_argument("--finish-dir")
parser.add_argument("--log-dir")

parser.add_argument(
    "--max-runs",
    type=int,
    default=1000
)

parser.add_argument(
    "--time-limit",
    type=int,
    default=60
)

args = parser.parse_args()

# --------------------------------------------------
# Find the domain's original training collection.
# --------------------------------------------------

domain_dir = (
    ROOT
    / "experiments"
    / args.domain
    / args.domain
)

domain_file = domain_dir / "domain.pddl"

train_dir = domain_dir / "train"

if not domain_file.is_file():
    raise FileNotFoundError(domain_file)

if not train_dir.is_dir():
    raise FileNotFoundError(train_dir)

problems = sorted(
    train_dir.glob("*.pddl"),
    key=lambda p: p.name
)

if len(problems) != 100:
    raise RuntimeError(
        f"Expected 100 training instances; "
        f"found {len(problems)}."
    )

# --------------------------------------------------
# Generate a reproducible training sequence.
# --------------------------------------------------

rng = random.Random(args.shuffle_seed)

rng.shuffle(problems)

print()
print("Domain:", args.domain)
print("Training instances:", len(problems))
print("Shuffle seed:", args.shuffle_seed)

print()
print("First 10 training problems:")

for i, problem in enumerate(problems[:10], 1):
    print(f"  {i}: {problem.name}")

# Allow checking the problem ordering
# without starting Fast Downward.
if args.list_only:
    raise SystemExit(0)

# --------------------------------------------------
# Validate training-run arguments.
# --------------------------------------------------

required = {
    "--port-file": args.port_file,
    "--port": args.port,
    "--finish-dir": args.finish_dir,
    "--log-dir": args.log_dir,
}

missing = [
    key
    for key, value in required.items()
    if value is None
]

if missing:
    parser.error(
        "Missing arguments: " + ", ".join(missing)
    )

port_file = (
    ROOT / args.port_file
).resolve()

finish_dir = (
    ROOT / args.finish_dir
).resolve()

log_dir = (
    ROOT / args.log_dir
).resolve()

if finish_dir.exists():
    raise RuntimeError(
        f"Final checkpoint already exists: {finish_dir}"
    )

log_dir.mkdir(
    parents=True,
    exist_ok=True
)

# Record the full training order.
manifest = (
    log_dir
    / f"training-order-seed-{args.shuffle_seed}.txt"
)

manifest.write_text(
    "\n".join(
        problem.name
        for problem in problems
    ) + "\n"
)

print()
print("Saved training order:", manifest)

# --------------------------------------------------
# Launch successive planning episodes.
# --------------------------------------------------

run = 0

while run < args.max_runs:

    while not port_file.exists():

        if finish_dir.is_dir():
            print(
                "Training completed. "
                "Launcher stopping.",
                flush=True
            )
            raise SystemExit(0)

        time.sleep(0.1)

    if finish_dir.is_dir():
        print(
            "Training completed. "
            "Launcher stopping.",
            flush=True
        )
        raise SystemExit(0)

    problem = problems[run % len(problems)]

    print(
        f"[episode {run + 1}] "
        f"{problem.name}",
        flush=True
    )

    search = (
        "rl_eager("
        "rl([single(ff()),single(cg()),"
        "single(cea()),single(add())],"
        f"random_seed={args.shuffle_seed},"
        "handle_empty_list=true),"
        "rl_control_interval=0,"
        f"rl_client_port={args.port})"
    )

    command = [
        "python",
        str(
            ROOT
            / "fast-downward"
            / "fast-downward.py"
        ),
        "--overall-time-limit",
        str(args.time_limit),
        str(domain_file),
        str(problem),
        "--search",
        search,
    ]

    log_file = log_dir / (
        f"episode_{run + 1:04d}_"
        f"{problem.stem}.log"
    )

    with log_file.open("w") as output:

        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=360,
            check=False
        )

    print(
        "    Planner exit code:",
        result.returncode,
        flush=True
    )

    run += 1

print("Launcher reached max-runs.")
