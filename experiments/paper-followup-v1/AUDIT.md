# Follow-up experiment: verified changes and remaining differences

Prepared 6 October 2026. The completed full100 CPU experiment remains the primary
experiment. This package writes only to `results-paper-followup-v1`.

## What was checked

The ICAPS 2021 paper *Learning Heuristic Selection with Dynamic Algorithm
Configuration*, and the released `aaai2021-code` branch's `experiments/job.sh`,
`experiments/fd-rl-train-auto-port.sh`, README, controller and environment sources.
Downloaded reference hashes are retained in `upstream-hashes.json`.

Sources:
- https://speckdavid.github.io/assets/pdf/speck-etal-icaps2021b.pdf
- https://github.com/speckdavid/rl-plan/tree/aaai2021-code
- https://raw.githubusercontent.com/speckdavid/rl-plan/aaai2021-code/experiments/job.sh
- https://raw.githubusercontent.com/speckdavid/rl-plan/aaai2021-code/experiments/fd-rl-train-auto-port.sh
- https://raw.githubusercontent.com/speckdavid/rl-plan/aaai2021-code/rl-env/train_and_eval_chainer_agent.py

## Training audit

| Setting | Completed experiment | New original and replacement | Released job/code |
|---|---|---|---|
| Training problems | 100/domain | 100/domain | 100/domain |
| Interaction steps | 1 million | 1 million | 1 million in CLI |
| Episode decision cutoff | 7,500 | 7,500 | 7,500 |
| Hidden layers | 2 x 75 | 2 x 75 | 2 x 75 |
| Replay learning starts | 100 transitions | 1,000 transitions | 1,000 default |
| Replay capacity | 500,000 | 500,000 | 500,000 |
| Batch / gamma / target update | 32 / .99 / 500 | unchanged | same defaults |
| Exploration | 1 to .1 over 500k | unchanged | same |
| Training order | seed-shuffled cycle | lexical cycle, shared with periodic evaluation | shell glob cycle |
| Seeds | 1,2,3 | 1,2,3,4,5 | SLURM array 1-5 |
| Policy selection | four snapshots, 20 training problems, coverage/CPU ranking | 100 training episodes about every 30k interactions; highest mean return | same selection through ChainerRL evaluator |
| Planner time limit during learning | 300 seconds + wall safeguard | no research time cutoff, 7,500-decision episode cutoff; hang abort fails the job | no planner time limit in shell launcher |

The paper calls the budget update steps, but the released job passes `--steps
1000000` to ChainerRL's environment-interaction loop. Replay warm-up means these
are not exactly one million gradient updates. This package follows the code.
Periodic evaluation occurs at episode boundaries after the interval is reached,
as in ChainerRL; it is not necessarily exactly at steps 30,000, 60,000, etc.
The next 100 lexical-cycle episodes cover all training files when no reset skips
occur. Episode logs record warm-up/reset retries and the actual sequence.

The Python learner's optimization and architecture match the released script.
Environment connection reliability patches remain. Unexpected mid-episode EOF
now fails training rather than manufacturing a zero-reward transition. Snapshot
saving omits replay serialization: this does not change successful-run learning,
but incomplete training is not exactly resumable. The runtime saves both `best`
and final models; final evaluation loads `best`, not the final weights.

## Feature experiment

Original has 20 inputs: one-step changes in mean, maximum, minimum, queue size,
and variance for each of four heuristics. Replacement also has 20 inputs:
12 one-step changes (mean, minimum, queue size), then eight history features
(normalized four-observation best-minimum improvement and age since improvement,
capped/scaled at 1,000 observations). It replaces maximum and variance changes.
Both networks therefore have the same number of trainable parameters.
This is a prespecified hypothesis, not a claim that the removed features have
been proven useless. The existing additive 28-input variant is preserved in the
first experiment but is not retrained here; this follow-up directly compares
original versus replacement under matched new training settings.

## Evaluation audit

The paper's main table maintains all four open lists even for fixed policies.
The previous standalone `eager(single(...))` baselines maintained only one list.
The new fixed controller selects a constant action through TCP while Fast
Downward maintains all four lists. `handle_empty_list=false` follows the released
README for static policies. Learned policies retain `handle_empty_list=true`.
The normal initial FF warm-up decision follows the released environment.

This is a closer implementation of the paper's design, NOT an exact reproduction:
- User's agreed budget remains 300 aggregate user+system CPU seconds, including
  controller, driver, translator and search. Paper/released commands use their
  Linux timing setup. CPU seconds do not normalize processor speed.
- A sampled 4 GiB aggregate resident-memory cap is enforced at approximately
  20 ms intervals. It is not identical to the released Linux 4G address-space
  limit. Memory-limit cases are reported separately and count as unsolved.
- macOS, compiler and hardware differ. Dependencies remain those already tested
  on the Macs. ChainerRL 0.8.0 matches released requirements; the paper describes
  0.7.0. No environment downgrade is attempted during this follow-up.
- There is no one-million-decision final-evaluation cutoff in this package.
  Only CPU/memory budgets and a separately identified 1,800-second hang safeguard
  govern final cases. The previous evaluator had that extra decision cutoff.
- Default final evaluation is one evaluation seed per trained policy to obtain
  a first result sooner. The paper averages 25 runs per policy. `--eval-repeats 25`
  selects that repetition count BEFORE the run; it produces 150,000 learned
  test cases across six domains rather than 6,000. Do not change the option in
  an existing output directory. Repetitions are not independent training seeds.
- RND, ALT, and oracle-selector comparisons are not included in this package.

Fixed baselines run once per problem because constant actions and disabled
empty-list random fallback do not introduce policy sampling. Final learned
evaluation seeds are 1..R for each trained policy, independent of training seed.

## Data and interpretation

No test problems are used for training or periodic checkpoint selection.
Childsnack's ten known byte-identical train/test pairs remain documented;
report both the 100-file benchmark and the 90-file nonidentical subset.
No assertion of structural disjointness is made.

The replacement was proposed after observing the first test results. Describe
this as a test-aware follow-up experiment, not a new confirmatory test on unseen
research data. Preserve all seeds, including poor ones, and retain negative
results. Matching methodology does not guarantee matching published numbers or
an improvement from replacement features.

## Validation limits

Local tests exercise feature values/dimensions, strict disconnect handling,
training CLI settings, four-list fixed-policy commands, selected-model loading,
CPU and memory budget termination, and protection of completed/incomplete runs.
The local runtime does not have the Mac's Chainer/Gym environment or its compiled
Fast Downward executable. A complete learning run has not been executed here.
Startup checks on each Mac validate installed imports and hash the planner/data
against the completed experiment before launching actual work.
