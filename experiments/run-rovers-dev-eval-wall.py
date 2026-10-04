import argparse
import os
import signal
import csv
import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DOMAIN_DIR = ROOT / "experiments" / "rovers" / "rovers"
DOMAIN_FILE = DOMAIN_DIR / "domain.pddl"
TRAIN_DIR = DOMAIN_DIR / "train"

EXPERIMENT_ROOT = ROOT / "results-cross-domain-auto"
OUTPUT = ROOT / "results-rovers-dev-30k-wall60"

PORT = 12521

HEURISTICS = ["ff", "cg", "cea", "add"]

RL_MODELS = {
    "original": {
        "checkpoint": (
            EXPERIMENT_ROOT
            / "rovers-original-seed1-steps30000"
            / "agent"
            / "30000_finish"
        ),
        "temporal_window": 0,
    },
    "temporal": {
        "checkpoint": (
            EXPERIMENT_ROOT
            / "rovers-temporal-seed1-steps30000"
            / "agent"
            / "30000_finish"
        ),
        "temporal_window": 4,
    },
}

CONFIGURATIONS = [
    "original",
    "temporal",
    "ff",
    "cg",
    "cea",
    "add",
]

SEARCH_RL = (
    "rl_eager("
    "rl([single(ff()),single(cg()),single(cea()),single(add())],"
    "random_seed=1,handle_empty_list=true),"
    "rl_control_interval=0,"
    f"rl_client_port={PORT})"
)


def load_development_problems():
    original_manifest = (
        EXPERIMENT_ROOT
        / "rovers-original-seed1-steps30000"
        / "planner-logs"
        / "training-order-seed-1.txt"
    )

    temporal_manifest = (
        EXPERIMENT_ROOT
        / "rovers-temporal-seed1-steps30000"
        / "planner-logs"
        / "training-order-seed-1.txt"
    )

    original_order = [
        line.strip()
        for line in original_manifest.read_text().splitlines()
        if line.strip()
    ]

    temporal_order = [
        line.strip()
        for line in temporal_manifest.read_text().splitlines()
        if line.strip()
    ]

    if original_order != temporal_order:
        raise RuntimeError(
            "Original and temporal training manifests differ."
        )

    if len(original_order) != 100:
        raise RuntimeError(
            "Expected 100 ordered Rovers training problems."
        )

    problems = original_order[19:29]

    expected = [
        "prob156.pddl",
        "prob140.pddl",
        "prob180.pddl",
        "prob85.pddl",
        "prob99.pddl",
        "prob30.pddl",
        "prob106.pddl",
        "prob197.pddl",
        "prob49.pddl",
        "prob163.pddl",
    ]

    if problems != expected:
        raise RuntimeError(
            "Development problems do not match the "
            "previously recorded positions 20-29."
        )

    for name in problems:
        if not (TRAIN_DIR / name).is_file():
            raise FileNotFoundError(TRAIN_DIR / name)

    return problems


def controller_command(model):
    config = RL_MODELS[model]
    checkpoint = config["checkpoint"]

    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)

    return [
        sys.executable,
        "-u",
        str(
            ROOT
            / "rl-env"
            / "train_and_eval_chainer_agent.py"
        ),

        "--outdir", ".",
        "--load", str(checkpoint),
        "--evaluate",
        "--save-eval-stats", "eval.json",

        "--seed", "1",
        "--state", "2",

        "--temporal-window",
        str(config["temporal_window"]),

        "--num-heuristics", "4",
        "--n-hidden-channels", "75",
        "--n-hidden-layers", "2",

        "--port", str(PORT),
        "--port-file-id", str(PORT),

        "--eval-n-runs", "1",
        "--time-step-limit", "1000000",
    ]


def planner_command(problem, configuration, time_limit):
    if configuration in RL_MODELS:
        search = SEARCH_RL
    else:
        search = (
            f"eager(single({configuration}()))"
        )

    return [
        sys.executable,
        str(
            ROOT
            / "fast-downward"
            / "fast-downward.py"
        ),
        "--overall-time-limit",
        str(time_limit),

        str(DOMAIN_FILE),
        str(TRAIN_DIR / problem),

        "--search",
        search,
    ]


def stop_process(process):
    if process is None or process.poll() is not None:
        return

    try:
        process.terminate()
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    except ProcessLookupError:
        pass


def last_match(pattern, text):
    matches = re.findall(pattern, text)

    if not matches:
        return None

    return matches[-1]


def parse_actions(eval_file):
    counts = Counter()

    if not eval_file.exists():
        return None

    data = json.loads(eval_file.read_text())

    trajectories = data.get("trajectories", [])

    if trajectories:
        counts.update(trajectories[0])

    return counts


def run_case(problem, configuration, args):
    problem_stem = Path(problem).stem

    run_dir = (
        OUTPUT
        / problem_stem
        / configuration
    )

    result_file = run_dir / "result.json"

    if result_file.exists():
        previous = json.loads(result_file.read_text())

        if previous.get("completed"):
            print(
                f"SKIP {problem_stem} / {configuration}",
                flush=True,
            )
            return previous

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    port_file = run_dir / f"port_{PORT}.txt"

    if port_file.exists():
        port_file.unlink()

    result = {
        "problem": problem,
        "configuration": configuration,
        "completed": False,
        "status": "not_started",
        "planner_exit_code": None,
        "controller_exit_code": None,
        "total_wall_seconds": None,
        "planner_wall_seconds": None,
        "fd_total_seconds": None,
        "expanded": None,
        "generated": None,
        "plan_length": None,
        "ff_selections": None,
        "cg_selections": None,
        "cea_selections": None,
        "add_selections": None,
    }

    controller = None

    print(
        f"RUN  {problem_stem} / {configuration}",
        flush=True,
    )

    start = time.monotonic()

    try:
        with (
            run_dir / "controller.log"
        ).open("w") as controller_log:

            if configuration in RL_MODELS:
                controller = subprocess.Popen(
                    controller_command(
                        configuration
                    ),
                    cwd=run_dir,
                    stdout=controller_log,
                    stderr=subprocess.STDOUT,
                )

                deadline = min(
                    time.monotonic() + 30,
                    start + args.max_wall_seconds,
                )

                while not port_file.is_file():
                    if controller.poll() is not None:
                        raise RuntimeError(
                            "Controller exited before "
                            "opening its port."
                        )

                    if time.monotonic() > deadline:
                        raise RuntimeError(
                            "Controller did not open "
                            "its port in 30 seconds."
                        )

                    time.sleep(0.1)

            command = planner_command(
                problem,
                configuration,
                args.time_limit,
            )

            planner_start = time.monotonic()

            with (
                run_dir / "planner.log"
            ).open("w") as planner_log:

                planner = subprocess.Popen(
                    command,
                    cwd=run_dir,
                    stdout=planner_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )

                remaining = max(
                    0.001,
                    start
                    + args.max_wall_seconds
                    - time.monotonic(),
                )

                try:
                    planner.wait(timeout=remaining)

                except subprocess.TimeoutExpired:
                    result["status"] = (
                        "external_wall_timeout"
                    )

                    # Terminate the Fast Downward wrapper
                    # and its child processes together.
                    for sig in (
                        signal.SIGTERM,
                        signal.SIGKILL,
                    ):
                        try:
                            os.killpg(planner.pid, sig)
                        except ProcessLookupError:
                            pass
                        except PermissionError:
                            if planner.poll() is None:
                                if sig == signal.SIGTERM:
                                    planner.terminate()
                                else:
                                    planner.kill()

                        try:
                            planner.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            continue

                        break

                result["planner_exit_code"] = (
                    planner.returncode
                )

            result["planner_wall_seconds"] = round(
                time.monotonic()
                - planner_start,
                3,
            )

            # Official elapsed measurement:
            # controller startup + translation + search.
            # Excludes post-planner controller cleanup.
            result["end_to_end_wall_seconds"] = round(
                time.monotonic() - start,
                3,
            )

            if controller is not None:
                try:
                    controller.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass

                result["controller_exit_code"] = (
                    controller.poll()
                )

        planner_file = (
            run_dir / "planner.log"
        )

        text = planner_file.read_text(
            errors="replace"
        )

        solved = (
            "Solution found." in text
        )

        if result["status"] == (
            "external_wall_timeout"
        ):
            pass

        elif (
            solved
            and result["planner_exit_code"] == 0
        ):
            result["status"] = "solved"

        elif result["planner_exit_code"] == 23:
            result["status"] = (
                "fd_time_limit"
            )

        else:
            result["status"] = (
                "not_solved_or_error"
            )

        result["expanded"] = last_match(
            r"Expanded (\d+) state",
            text,
        )

        result["generated"] = last_match(
            r"Generated (\d+) state",
            text,
        )

        result["plan_length"] = last_match(
            r"Plan length: (\d+)",
            text,
        )

        result["fd_total_seconds"] = (
            last_match(
                r"Total time:\s*([\d.]+)s",
                text,
            )
        )

        if configuration in RL_MODELS:
            counts = parse_actions(
                run_dir / "eval.json"
            )

            for index, name in enumerate(
                HEURISTICS
            ):
                result[
                    f"{name}_selections"
                ] = (
                    None if counts is None
                    else counts[index]
                )

        result["completed"] = True

    except Exception as error:
        result["status"] = (
            "runner_error"
        )

        result["error"] = str(error)

        print(
            "ERROR:",
            error,
            flush=True,
        )

        raise

    finally:
        stop_process(controller)

        result["total_wall_seconds"] = round(
            time.monotonic() - start,
            3,
        )

        result_file.write_text(
            json.dumps(
                result,
                indent=2,
            )
        )

    print(
        "  Status:",
        result["status"],
        flush=True,
    )

    print(
        "  Expanded:",
        result["expanded"],
        flush=True,
    )

    print(
        "  Planner wall:",
        result["planner_wall_seconds"],
        "seconds",
        flush=True,
    )

    return result


def write_summary(problems):
    rows = []

    for problem in problems:
        stem = Path(problem).stem

        for configuration in CONFIGURATIONS:
            result_file = (
                OUTPUT
                / stem
                / configuration
                / "result.json"
            )

            if result_file.exists():
                rows.append(
                    json.loads(
                        result_file.read_text()
                    )
                )

    if not rows:
        return

    columns = [
        "problem",
        "configuration",
        "completed",
        "status",
        "planner_exit_code",
        "controller_exit_code",
        "total_wall_seconds",
        "end_to_end_wall_seconds",
        "planner_wall_seconds",
        "fd_total_seconds",
        "expanded",
        "generated",
        "plan_length",
        "ff_selections",
        "cg_selections",
        "cea_selections",
        "add_selections",
    ]

    summary_file = (
        OUTPUT / "summary.csv"
    )

    with summary_file.open(
        "w",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(
        "Summary saved:",
        summary_file,
    )

    for configuration in CONFIGURATIONS:
        subset = [
            row
            for row in rows
            if row["configuration"]
            == configuration
        ]

        solved = sum(
            row["status"] == "solved"
            for row in subset
        )

        print(
            f"{configuration:<10}"
            f" solved {solved}/{len(subset)}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help=(
            "Number of development "
            "problems, from 1 to 10."
        ),
    )

    parser.add_argument(
        "--time-limit",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--max-wall-seconds",
        type=int,
        default=600,
    )

    parser.add_argument(
        "--only",
        default=None,
        help="Evaluate one problem stem, e.g. prob49.",
    )

    args = parser.parse_args()

    if not 1 <= args.limit <= 10:
        parser.error(
            "--limit must be between 1 and 10"
        )

    if args.time_limit < 1:
        parser.error(
            "--time-limit must be positive"
        )

    if args.max_wall_seconds < 1:
        parser.error(
            "--max-wall-seconds must be positive"
        )

    if not DOMAIN_FILE.is_file():
        raise FileNotFoundError(DOMAIN_FILE)

    all_problems = load_development_problems()

    if args.only is not None:
        problems = [
            name
            for name in all_problems
            if Path(name).stem == args.only
        ]

        if not problems:
            parser.error(
                "Requested problem is not in "
                "the development collection."
            )
    else:
        problems = all_problems[:args.limit]

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Rovers development evaluation")
    print("Problems:", problems)
    print(
        "Configurations:",
        CONFIGURATIONS,
    )
    print(
        "Fast Downward time limit:",
        args.time_limit,
    )
    print()

    try:
        for problem in problems:
            for configuration in CONFIGURATIONS:
                run_case(
                    problem,
                    configuration,
                    args,
                )

    finally:
        write_summary(problems)


if __name__ == "__main__":
    main()
