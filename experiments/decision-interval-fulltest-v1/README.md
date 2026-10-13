# Decision intervals: full test evaluation v1

This replaces the small performance pilot at the user's request. It uses all
100 existing test files per domain, six domains, all three frozen Original
seeds and three conditions. There is no retraining or checkpoint reselection.

| Condition | When Original chooses again |
|---|---|
| Original | Every queue removal, using the existing original executable. |
| fixed16 | After 16 successful expansions. |
| adaptive4p8m64 | Minimum 4 expansions, patience 8 without a new running-best selected-queue minimum, maximum 64. |

An empty selected queue ends a held block immediately; the existing seeded
random fallback remains available. The single-removal FF warm-up and original
one-removal feature differences are retained. The neural network chooses the
heuristic; progress/stagnation determine when to query it again. This does not
combine the earlier eight simple policies or use the trained Progress network.
No performance improvement is established yet.

## Workload and limits

900 cases/domain; 2,700/Mac; **5,400 total**, with four workers on each Mac.
Every case retains the aligned supervisor's 300 aggregate CPU-second budget,
sampled aggregate 4 GiB RSS cap, 1,800-second wall safeguard, and no practical
decision cap (2**63 - 1). Controller, driver, translation and search CPU and
their descendants are charged; supervisor and cleanup CPU are excluded.

Full nominal allocation is **450 CPU-hours**, or **56.25 hours at ideal
eight-worker throughput**, plus builds and diagnostics. This is a workload
calculation, not an elapsed-time forecast or guarantee. Early solves reduce
work; contention and imbalance increase elapsed time.

Original is rerun as a contemporaneous baseline with common lightweight
controller recording. All new bookkeeping and payload costs are charged.
Historical aligned results remain separate.

## Run

Finish or pause the current simple-policy or pilot evaluation before starting.
To pause, press Ctrl+C once and wait for cleanup and the prompt. Completed
cases remain saved. Do not run two batches concurrently on the same Mac.

Download `rl-plan-decision-interval-fulltest-v1.zip` into Downloads on both Macs.

Primary:

```bash
conda activate rl-plan
cd ~/Downloads/ANU/Honours/rl-plan

unzip -n ~/Downloads/rl-plan-decision-interval-fulltest-v1.zip -d experiments &&
caffeinate -i python experiments/decision-interval-fulltest-v1/setup.py --jobs 2 &&
caffeinate -i python -u experiments/decision-interval-fulltest-v1/run.py \
  --domains blocksworld rovers visitall --workers 4 --run
```

Secondary:

```bash
conda activate rl-plan
cd ~/Downloads/Dawit_honours_project/rl-plan-honours

unzip -n ~/Downloads/rl-plan-decision-interval-fulltest-v1.zip -d experiments &&
caffeinate -i python experiments/decision-interval-fulltest-v1/setup.py --jobs 2 &&
caffeinate -i python -u experiments/decision-interval-fulltest-v1/run.py \
  --domains barman childsnack sokoban --workers 4 --run
```

`--jobs 2` controls compilation only; the evaluation uses **four workers**.
Setup requires CMake and a C++ compiler. It builds hash-checked copies in its
own `planner` folder, without editing original sources, models or results.
The isolated Release build disables LP, which these four heuristics do not
require. Build settings and binary hashes are recorded; differences from the
older original build remain a possible runtime confound.

The small performance pilot is skipped, but the short correctness gate remains.
On one frozen training problem per domain and all three seeds, Original and
patched every-removal mode must match raw observations, actual Chainer inputs
and actions, and available expansion/plan statistics. Each side stops at a
solution or a deliberate 256-decision cutoff. These diagnostics are not test
scores. They retain 300 CPU seconds and 4 GiB RSS, with a 600-second diagnostic
wall safeguard. The 36 sessions allow another 3 CPU-hours at full limits;
short prefixes and early solutions should usually finish much sooner.

A failed gate stops the run. Send the error rather than removing the check.
The critical-system-memory safeguard remains active. If four workers repeatedly
trigger it, resume with two; this does not change per-case budgets.

## Results, resume and export

Rerun the same `run.py` command to resume. Valid completed cases and passed
gates are skipped; interrupted attempts are archived and retried. Code, models,
problem lists and limits are checked. Worker count may change and is recorded.

```text
results-decision-interval-fulltest-v1/DOMAIN/coverage.csv
results-decision-interval-fulltest-v1/DOMAIN/coverage-nonoverlap.csv
results-decision-interval-fulltest-v1/DOMAIN/results.csv
```

All 100 existing test filenames are included for comparison with the aligned
main table. Known exact train/test duplicate files are flagged in results and
excluded in the nonoverlap table. That exclusion does not establish independence
of the remaining test set.

After completion, from the appropriate project root:

```bash
# Primary
python experiments/decision-interval-fulltest-v1/export.py --label primary

# Secondary
python experiments/decision-interval-fulltest-v1/export.py --label secondary
```

Upload both timestamped archives printed by the exporter. They include logs,
provenance, resource data, diagnostic profiles and per-case counters. Limited
or interrupted cases may have lower-bound counters. Completed-interval
histograms include warm-up; the final active interval is reported separately.

## Reporting

Treat this as the final predefined extension if time is tight. Write the
methodology and report completed Original/Progress findings while it runs.
Tell the supervisor that this extension is implemented and evaluation is
ongoing, not that it has already improved performance.

Report paired gains and losses, coverage, capped CPU, common-solved runtime,
expansions, decisions and all seeds/domains. Totals across seeds count repeated
problem/seed evaluations, not distinct planning tasks. Do not select only
favourable seeds or domains.

The existing test sets have already been inspected during development, so
this remains an **exploratory full-test follow-up**, not fresh confirmatory
validation. A better result would support improvement under this protocol;
it would not alone establish generalisation, novelty, or superiority over
all fixed intervals. There is only one fixed-interval control. Keep its
constants frozen and report the outcome rather than tuning on test results.
