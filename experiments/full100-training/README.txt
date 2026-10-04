FULL-100 TRAINING ONLY, v1

Install this folder as rl-plan/experiments/full100-training.
Activate conda environment rl-plan, then from the repository root run:
  caffeinate -i bash experiments/full100-training/train-all.sh

This trains original and progress sequentially on all 100 supplied training
instances for each domain. Order: seed 1 across six domains, then seeds 2 and 3.
Within each domain original precedes progress. Models use the same shuffled
problem order for a given seed. Stopping early may leave partial training that
cannot be resumed exactly; the runner refuses to silently overwrite it.

One million learner interactions, episode limit 7500, 300s planner training
limit, 315s wall safety check, four retained checkpoints: 100k/300k/600k/1m.
Original has 20 inputs; progress has 28. All embedded learning/environment
code and hyperparameters are unchanged from the audited budget-v1 pilots.
This includes replay-start-size 100 and Adam eps=1e-2. It is not a paper-exact
reproduction. No optimizer-setting change is bundled into this comparison.

No evaluation is launched by these scripts. Before training, a deterministic
20-problem selection list is recorded, using selection seed 20260926. ALL of
these problems also participate in training. Later compare all four retained
checkpoints on this training subset at a 300-second end-to-end wall limit.
Choose by coverage, then capped runtime sum, then earliest checkpoint.
This is training-based selection, not held-out validation. The 100-problem
designated test collection remains separate and is never read by this runner.
An evaluation runner is a separate next step; it is not included in this ZIP.

Results: results-DOMAIN-MODEL-full100-v1-seedN under the repository root.
Launcher logs: results-full100-v1-logs. Detailed trainer/episode logs and
exposure.json are inside each results folder. Existing pilot folders and
installed rl-env files are preserved. Source/planner hash gates are retained.
If a hash check fails, report it; do not bypass the check.

The launcher preflights all 36 configurations before training. To preflight
only, use: bash experiments/full100-training/train-all.sh --prepare-only
Use the same Python environment for preparation and execution. Keep this
folder unchanged after preparing runs; manifests enforce source consistency.
Completed training is verified and skipped when the launcher is rerun.

Validation: syntax, unchanged embedded source bundles, 100-instance manifests,
matched selection lists, changed-file rejection and simulated train-only
orchestration checked locally. Real Chainer training and the planner were not
executed in the assistant's environment; the user's Mac provides those checks.
