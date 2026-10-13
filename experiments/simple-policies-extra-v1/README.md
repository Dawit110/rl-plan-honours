# Four additional historical simple policies

This is an ADD-ON to simple-policies-v1. It adds four named configurations and
does not overwrite that package or reuse its result folders. Run one evaluation
batch at a time on each Mac; do not run two four-worker batches together.
If the first batch is already running, let it finish before launching this one.
If it has not started, run the first package followed by this add-on.

| Label | Fixed rule |
|---|---|
| recent50 | Historical Recent Progress: compare raw queue minima 50 observation intervals apart, normalized by max(abs(old minimum),1), and reconsider every decision after the first full window. Prefer current action on ties. |
| stagprog50 | Historical Stagnation + Progress: after 50 observations without a new minimum during this visit, select the OTHER available heuristic with greatest normalized raw progress over 50 intervals. Break ties cyclically. |
| probe25x100 | Historical Probe-then-exploit: probe each heuristic for 25 actions, then use the highest-scoring available heuristic for 100 actions. Probe scores use start/end minima, measured after the final probing action. Rotate the first probed heuristic each cycle. |
| queue50a1 | Historical Progress + Queue Pressure: normalized raw progress over 50 intervals minus alpha times positive relative queue growth, with alpha=1. Reconsider every decision after a full history window. |

The queue penalty is max(new_queue-old_queue,0)/max(new_queue,1). Queue shrinkage
is not rewarded. The history-based policies use 51 observations to span 50
intervals. Parameters and heuristic order (FF,CG,CEA,ADD) are fixed for all six
domains. There is no policy training or parameter sweep.

RECENT50 is intentionally distinct from the previous package's PROGRESS50.
The previous one scores normalized running-best decreases, acts in 50-action
blocks, and rotates when all scores are zero. RECENT50 scores raw endpoints,
can switch every decision and stays on a zero-score tie. Keep their names and
results separate. The combined two packages therefore contain EIGHT policies.

## Fidelity to earlier implementations

runtime/historical.py contains verbatim class bodies from the available earlier
run_progress4.py, run_stag_progress4.py, run_probe_exploit4.py and
run_progress_queue4.py. source-provenance.json records source and class hashes.
The old CLIs have been replaced by the same evaluation entry point used in the
new simple-policy package. This is reevaluation under the later protocol, not
a claim to reproduce the timing or coverage of the nine-instance pilot.

An adapter checks RAW observation shape and finite data and returns action 0
when every queue is empty. This avoids an infinite retry loop in the historical
probe controller and returns control to the planner to report exhaustion.
The class bodies' other empty-queue behavior is retained: during warmup or an
ongoing probe, they can request an empty queue. The planner's seeded fallback
is enabled with handle_empty_list=true, as for learned controllers. Corrected
constant controllers used false. Histories are not newly reset when individual
queues empty; this also preserves historical behavior.

Progress in shared queues is a proxy, not distance to the goal or causal credit
for an action. Sequential probes share and change the same search state; they
are not independent experiments or planner restarts.

## Evaluation protocol

* 100 test instances x 4 additional policies x 6 domains = 2,400 NEW cases.
* Four workers per Mac; seed 1; 300 aggregate CPU seconds including controller,
  translation, feature processing and search; sampled aggregate 4 GiB RSS cap.
* No practical decision cap (2**63-1), matching corrected constant controllers.
  Earlier learned evaluation had a 1m-decision cap and no RSS cap.
* The same environment, evaluator, CPU meter and case supervisor as the first
  simple-policy package and corrected fixed baselines. The environment performs
  one FF warm-up action before handing observations to each policy.
* A 1,800-second wall hang safeguard is treated as an error, not valid failure.
* Source, problem and planner hashes plus dependencies are checked and frozen.
  Valid completed cases are skipped on resume; changed protocols are rejected.
* coverage.csv includes all 100 test instances. coverage-nonoverlap.csv also
  excludes known train/test duplicates (90 retained for Childsnack).
* The test sets were previously inspected: this is exploratory follow-up.
  Report all policies, including negative results, without tuning to the test.

The new worst-case allocated workload is another 200 CPU-hours: 25 hours at
ideal eight-worker throughput. That is not an elapsed-time guarantee. Early
solves reduce work; contention, uneven work distribution and stops add elapsed
time. CPU accounting reduces waiting-time penalties but does not guarantee
outcome invariance under concurrency.

## Primary Mac, after its current evaluation finishes

Download rl-plan-simple-policies-extra-v1.zip to Downloads, then:

```bash
conda activate rl-plan
cd ~/Downloads/ANU/Honours/rl-plan
unzip -n ~/Downloads/rl-plan-simple-policies-extra-v1.zip -d experiments &&
caffeinate -i python -u experiments/simple-policies-extra-v1/run.py \
  --domains blocksworld rovers visitall --workers 4 --run
```

## Secondary Mac, after its current evaluation finishes

```bash
conda activate rl-plan
cd ~/Downloads/Dawit_honours_project/rl-plan-honours
unzip -n ~/Downloads/rl-plan-simple-policies-extra-v1.zip -d experiments &&
caffeinate -i python -u experiments/simple-policies-extra-v1/run.py \
  --domains barman childsnack sokoban --workers 4 --run
```

No new dependencies or training required. Omitting --run performs preflight
only. If stopped, rerun the same command; lower --workers to 2 if necessary
after sustained memory pressure. Do not modify code during a run.

Results: results-simple-policies-extra-v1/DOMAIN/coverage.csv

Export from each project folder when done:

```bash
python experiments/simple-policies-extra-v1/export.py --label primary
```

Use --label secondary on the secondary Mac. Upload the resulting
rl-plan-simple-policies-extra-primary.tar.gz and
rl-plan-simple-policies-extra-secondary.tar.gz from Downloads. Keep the first
package's archives too; export.py does not combine them.

## Verification

11 tests check history window boundaries, decision cadence, stagnation reset,
alternative/tie selection, exact probe and exploitation lengths, probe order
rotation, empty-queue handling, queue penalty, episode reset, copied class
hashes and distinct task configurations. Run test_policies.py to repeat them.
Shared resource supervision was already tested in the original package. The
development container lacks the Macs' Fast Downward build/legacy dependencies;
full planning execution remains to be checked on the Macs. These tests make
no claim about coverage improvement or exact replication of the paper.
