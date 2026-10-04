import argparse
import os
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

MODELS = {
    "original": 0,
    "temporal": 4,
}

parser = argparse.ArgumentParser()

parser.add_argument(
    "--domain",
    choices=DOMAINS + ["all"],
    default="all",
)

parser.add_argument(
    "--model",
    choices=["original", "temporal", "both"],
    default="both",
)

parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--port", type=int, default=12400)

parser.add_argument(
    "--time-limit",
    type=int,
    default=60,
)

parser.add_argument(
    "--max-wall-seconds",
    type=int,
    default=3600,
)

parser.add_argument(
    "--output",
    default="results-cross-domain-auto",
)

args = parser.parse_args()

if args.steps < 1:
    parser.error("--steps must be positive")

selected_domains = (
    DOMAINS
    if args.domain == "all"
    else [args.domain]
)

selected_models = (
    list(MODELS)
    if args.model == "both"
    else [args.model]
)

output_root = (ROOT / args.output).resolve()


def stop_process(process):
    if process is None or process.poll() is not None:
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)

    except ProcessLookupError:
        return

    except PermissionError:
        try:
            process.terminate()
        except (ProcessLookupError, PermissionError):
            print(
                "WARNING: Could not terminate process:",
                process.pid,
                flush=True,
            )
            return

    try:
        process.wait(timeout=3)

    except subprocess.TimeoutExpired:

        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)

            except ProcessLookupError:
                return

            except PermissionError:
                try:
                    process.kill()
                except (ProcessLookupError, PermissionError):
                    print(
                        "WARNING: Could not kill process:",
                        process.pid,
                        flush=True,
                    )
                    return

        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            print(
                "WARNING: Process did not exit:",
                process.pid,
                flush=True,
            )


def run_experiment(domain, model):

    window = MODELS[model]

    name = (
        f"{domain}-{model}"
        f"-seed{args.seed}"
        f"-steps{args.steps}"
    )

    outdir = output_root / name

    agent_dir = outdir / "agent"

    finish_dir = (
        agent_dir
        / f"{args.steps}_finish"
    )

    if finish_dir.is_dir():
        print(
            f"SKIPPING completed experiment: {name}",
            flush=True,
        )
        return

    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(
            "Incomplete experiment directory already exists: "
            + str(outdir)
        )

    outdir.mkdir(parents=True, exist_ok=True)

    planner_logs = outdir / "planner-logs"

    checkpoint_frequency = min(
        5000,
        max(1, args.steps // 2),
    )

    exploration_steps = max(
        1,
        args.steps // 2,
    )

    trainer_command = [
        sys.executable,
        "-u",
        str(
            ROOT
            / "rl-env"
            / "train_and_eval_chainer_agent.py"
        ),

        "--outdir", str(agent_dir),
        "--outdir-time-suffix", "none",

        "--seed", str(args.seed),
        "--steps", str(args.steps),

        "--replay-start-size", "100",
        "--minibatch-size", "32",

        "--checkpoint_frequency",
        str(checkpoint_frequency),

        "--eval-interval",
        str(max(1000000, args.steps + 1)),

        "--final-exploration-steps",
        str(exploration_steps),

        "--max-rand-step", "3",

        "--state", "2",
        "--temporal-window", str(window),

        "--num-heuristics", "4",

        "--n-hidden-channels", "75",
        "--n-hidden-layers", "2",

        "--port", str(args.port),
        "--port-file-id", str(args.port),

        "--time-step-limit", "7500",
    ]

    launcher_command = [
        sys.executable,
        "-u",
        str(
            ROOT
            / "experiments"
            / "run-domain-training.py"
        ),

        "--domain", domain,
        "--shuffle-seed", str(args.seed),

        "--port-file",
        str(agent_dir / f"port_{args.port}.txt"),

        "--port", str(args.port),

        "--finish-dir",
        str(finish_dir),

        "--log-dir",
        str(planner_logs),

        "--max-runs", "1000",

        "--time-limit",
        str(args.time_limit),
    ]

    print()
    print("=" * 65)
    print("EXPERIMENT:", name)
    print("=" * 65)

    print("Domain:", domain)
    print("Model:", model)
    print("Temporal window:", window)
    print("Training steps:", args.steps)
    print("Seed:", args.seed)
    print("Output:", outdir)
    print(flush=True)

    trainer = None
    launcher = None

    start = time.monotonic()

    try:

        with (
            (outdir / "trainer.log").open("w")
        ) as trainer_log, (
            (outdir / "launcher.log").open("w")
        ) as launcher_log:

            trainer = subprocess.Popen(
                trainer_command,
                cwd=ROOT,
                stdout=trainer_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

            launcher = subprocess.Popen(
                launcher_command,
                cwd=ROOT,
                stdout=launcher_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

            while True:

                trainer_status = trainer.poll()
                launcher_status = launcher.poll()

                if trainer_status is not None:
                    break

                if launcher_status is not None:

                    if (
                        launcher_status == 0
                        and finish_dir.is_dir()
                    ):
                        # The launcher may detect the completed
                        # checkpoint before ChainerRL exits.
                        try:
                            trainer.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            raise RuntimeError(
                                "Final checkpoint exists, but learner "
                                "did not exit within 30 seconds."
                            )

                        break

                    raise RuntimeError(
                        "Training launcher stopped before "
                        "the learner finished. "
                        f"Launcher exit code: {launcher_status}. "
                        f"Final checkpoint exists: "
                        f"{finish_dir.is_dir()}"
                    )

                elapsed = (
                    time.monotonic() - start
                )

                if elapsed > args.max_wall_seconds:
                    raise RuntimeError(
                        "Experiment exceeded its "
                        "wall-clock safety limit."
                    )

                time.sleep(0.5)

            if trainer.returncode != 0:
                raise RuntimeError(
                    "Learner exited with code "
                    + str(trainer.returncode)
                )

            if not finish_dir.is_dir():
                raise RuntimeError(
                    "Learner exited without "
                    "saving its final checkpoint."
                )

            try:
                launcher.wait(timeout=15)
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    "Launcher did not stop "
                    "after learner completion."
                )

            if launcher.returncode != 0:
                raise RuntimeError(
                    "Launcher exited with code "
                    + str(launcher.returncode)
                )

        elapsed = time.monotonic() - start

        print(
            "SUCCESS:",
            name,
            flush=True,
        )

        print(
            "Elapsed:",
            round(elapsed, 2),
            "seconds",
            flush=True,
        )

        print(
            "Final checkpoint:",
            finish_dir,
            flush=True,
        )

        print(
            "Trainer log:",
            outdir / "trainer.log",
            flush=True,
        )

        print(
            "Launcher log:",
            outdir / "launcher.log",
            flush=True,
        )

    finally:
        stop_process(launcher)
        stop_process(trainer)


for domain in selected_domains:
    for model in selected_models:
        run_experiment(domain, model)

print()
print("Selected experiments finished.")
