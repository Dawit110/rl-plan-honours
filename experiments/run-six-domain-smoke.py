import argparse
import os
import re
import signal
import subprocess
import sys
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

PORT = 12370

OUTPUT = ROOT / "results-six-domain-smoke"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--domain",
    choices=DOMAINS + ["all"],
    default="all",
)
args = parser.parse_args()

selected = DOMAINS if args.domain == "all" else [args.domain]


def stop_process(process):
    if process.poll() is not None:
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=3)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


for domain in selected:

    print("\n" + "=" * 65)
    print("DOMAIN:", domain.upper(), flush=True)
    print("=" * 65)

    domain_dir = ROOT / "experiments" / domain / domain

    domain_file = domain_dir / "domain.pddl"

    problems = sorted(
        (domain_dir / "train").glob("*.pddl")
    )

    if not problems:
        print("ERROR: No training problems found.")
        continue

    problem = problems[0]

    checkpoint = (
        ROOT
        / "experiments"
        / domain
        / f"{domain}_full_train_set_2-75-1"
        / "best"
    )

    run_dir = OUTPUT / domain / "rl1"

    run_dir.mkdir(parents=True, exist_ok=True)

    port_file = run_dir / f"port_{PORT}.txt"

    port_file.unlink(missing_ok=True)

    print("Problem:", problem.name, flush=True)
    print("Checkpoint:", checkpoint, flush=True)

    controller_command = [
        sys.executable,
        "-u",
        str(ROOT / "rl-env/train_and_eval_chainer_agent.py"),

        "--outdir", ".",
        "--load", str(checkpoint),
        "--evaluate",
        "--save-eval-stats", "eval.json",

        "--seed", "1",
        "--state", "2",
        "--num-heuristics", "4",

        "--n-hidden-channels", "75",
        "--n-hidden-layers", "2",

        "--eval-n-runs", "1",

        "--port", str(PORT),
        "--port-file-id", str(PORT),

        "--time-step-limit", "1000000",
    ]

    search = (
        "rl_eager("
        "rl([single(ff()),single(cg()),"
        "single(cea()),single(add())],"
        "random_seed=1,"
        "handle_empty_list=true),"
        "rl_control_interval=0,"
        f"rl_client_port={PORT})"
    )

    planner_command = [
        sys.executable,
        str(ROOT / "fast-downward/fast-downward.py"),

        "--overall-time-limit", "30",

        str(domain_file),
        str(problem),

        "--search",
        search,
    ]

    controller = None
    planner = None

    try:

        with (run_dir / "controller.log").open("w") as controller_log:

            controller = subprocess.Popen(
                controller_command,
                cwd=run_dir,
                stdout=controller_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

            deadline = time.monotonic() + 30

            while not port_file.exists():

                if controller.poll() is not None:
                    raise RuntimeError(
                        "Controller exited before opening port."
                    )

                if time.monotonic() > deadline:
                    raise RuntimeError(
                        "Controller port startup timeout."
                    )

                time.sleep(0.1)

            with (run_dir / "planner.log").open("w") as planner_log:

                start = time.monotonic()

                planner = subprocess.Popen(
                    planner_command,
                    cwd=run_dir,
                    stdout=planner_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )

                try:
                    planner.wait(timeout=240)

                    elapsed = (
                        time.monotonic() - start
                    )

                except subprocess.TimeoutExpired:

                    elapsed = (
                        time.monotonic() - start
                    )

                    stop_process(planner)

                    print(
                        "OUTER SAFETY TIMEOUT",
                        flush=True,
                    )

                    continue

        log = (
            run_dir / "planner.log"
        ).read_text(errors="replace")

        solved = (
            "Solution found." in log
            and "search exit code: 0" in log
        )

        timed_out = (
            "Time limit has been reached." in log
            and "search exit code: 23" in log
        )

        if solved:
            status = "SOLVED"
        elif timed_out:
            status = "PLANNER TIME LIMIT"
        else:
            status = "ERROR / OTHER EXIT"

        print("Status:", status, flush=True)

        print(
            "Planner return code:",
            planner.returncode,
            flush=True,
        )

        print(
            "End-to-end planner elapsed time:",
            round(elapsed, 3),
            "seconds",
            flush=True,
        )

        match = re.search(
            r"Expanded (\d+) state",
            log,
        )

        if match:
            print(
                "Expanded states:",
                match.group(1),
                flush=True,
            )

        print("Logs:", run_dir, flush=True)

    except Exception as error:

        print(
            "ERROR:",
            repr(error),
            flush=True,
        )

    finally:

        if planner is not None:
            stop_process(planner)

        if controller is not None:
            stop_process(controller)


print("\nSix-domain smoke test finished.")
