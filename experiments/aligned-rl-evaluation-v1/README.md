# Aligned evaluation of the existing Original and Progress controllers

This reruns the previously selected policies under the resource limits used
for corrected fixed baselines and the simple-policy follow-ups. It does not
retrain models, change features, or reselect checkpoints. Earlier results are
preserved in results-full100-cpu-v1; new results use
results-aligned-rl-evaluation-v1.

## What changes

* 300 aggregate user+system CPU seconds remains the budget, including controller
  startup/imports, translation and search across the supervised process tree.
* A sampled aggregate resident-memory cap of 4 GiB is now enforced by the same
  case supervisor as corrected fixed baselines. This is not an instantaneous
  hard RSS limit; sampling can permit brief overshoot.
* The controller's 1,000,000-decision limit becomes 2**63-1 (no practical cap),
  matching corrected fixed and simple-policy evaluations.
* The 1,800-second wall safeguard remains a hang/error guard, not an accepted
  planning timeout. Critical system memory pressure stops the batch; completed
  valid cases remain reusable.

The old and new scores may or may not differ. Missing caps are protocol
mismatches, not proof that past coverage was affected. This rerun provides the
comparison; it should not be described as guaranteed to improve RL coverage.

## What stays fixed

Original and Progress each use training seeds 1, 2 and 3, their original frozen
20/28-input source snapshots, the same training-selected checkpoints and test
problems, and their previous per-model evaluation seeds. Search still uses all
four heuristic queues with handle_empty_list=true. Fixed constant controllers
used false as required by their policy implementation.

Preflight verifies hashes against the earlier learned and corrected-fixed
protocols, checks checkpoint ranking from TRAINING selection records, and
freezes the inputs. It does not select a checkpoint from final-test outcomes.
No replacement-feature model or newer runtime is substituted for Progress.

Checkpoint selection itself remains the earlier procedure and resource limits.
This is a controlled reevaluation of the SAME trained systems under aligned
test limits. It is not a full paper replication or newly aligned selection
experiment; training, selection, hardware, repetitions and accounting details
still need to be documented. Prior inspection of these test sets also makes
this a follow-up study, not independent confirmatory evidence.

## Scope and runtime

100 test problems x 2 approaches x 3 seeds x 6 domains = 3,600 cases.
At 300 seconds for every case, the allocation is 300 CPU-hours, or 37.5 hours
at ideal eight-worker throughput across two Macs. Early solves reduce work;
contention and uneven domain runtimes increase elapsed time. This is not a
guaranteed wall-time bound. No fixed-baseline runs or checkpoint-selection runs
are repeated here.

## Launch

Run only one batch on each Mac. To prioritize this rerun now, press Ctrl+C once
in the active simple-policy terminal and wait for cleanup and the shell prompt.
The saved simple-policy results can later be resumed with their original command.
Alternatively, let that batch finish before launching this one.

Download rl-plan-aligned-rl-evaluation-v1.zip into Downloads on both Macs.

Primary:

```bash
conda activate rl-plan
cd ~/Downloads/ANU/Honours/rl-plan
unzip -n ~/Downloads/rl-plan-aligned-rl-evaluation-v1.zip -d experiments &&
caffeinate -i python -u experiments/aligned-rl-evaluation-v1/run.py \
  --domains blocksworld rovers visitall --workers 4 --run
```

Secondary:

```bash
conda activate rl-plan
cd ~/Downloads/Dawit_honours_project/rl-plan-honours
unzip -n ~/Downloads/rl-plan-aligned-rl-evaluation-v1.zip -d experiments &&
caffeinate -i python -u experiments/aligned-rl-evaluation-v1/run.py \
  --domains barman childsnack sokoban --workers 4 --run
```

No package installation is needed. Omitting --run checks preflight only. If
memory pressure recurs, close unnecessary applications and resume with
--workers 2. Worker changes are recorded per launch; CPU accounting does not
guarantee identical outcomes under different system loads. Do not edit source
files or model folders during the run.

## Results and export

Each domain has 600 cases, interleaved across the six model/seed configurations.
coverage.csv updates as cases finish:

    results-aligned-rl-evaluation-v1/DOMAIN/coverage.csv

coverage-nonoverlap.csv additionally excludes known exact train/test overlap
instances (90 retained for Childsnack). This does not undo prior model exposure.
Check completed=planned before reporting final coverage. A valid unsolved case
receives capped loss 300; errors are not silently accepted as failures.

On the primary Mac:

```bash
python experiments/aligned-rl-evaluation-v1/export.py --label primary
```

On the secondary, use --label secondary. Upload
rl-plan-aligned-rl-primary.tar.gz and rl-plan-aligned-rl-secondary.tar.gz from
Downloads. Archives contain summaries, protocols, per-case configs/results and
runner source. Preserve full logs locally. Compare paired per-instance outcomes
with the earlier run, and report protocol changes even if scores stay identical.

## Verification

The supplied tests check all six seed/model configurations, 600 unique tasks,
the no-practical-cap argument, the 4 GiB memory setting, retained temporal
feature flags, model/runtime hash checks, resume configuration checks, and
rejection of a checkpoint choice that differs from recorded training ranking.
The case supervisor and CPU meter are copied unchanged from the already-used
corrected-baseline package. No full Fast Downward/Chainer run has been performed
in the development container, which lacks the Macs' build and environment.
Inspect the first PREFLIGHT and DONE lines on both Macs before leaving it running.
