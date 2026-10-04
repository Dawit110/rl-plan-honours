PRIMARY MAC CONCURRENCY CHECK
Install folder under rl-plan/experiments/concurrency-check.
From project root with conda rl-plan activated:
  caffeinate -i python -u experiments/concurrency-check/benchmark.py --workers 1 2 1

12 cases per stage: Barman and Blocksworld; original and progress seed 1;
final 1m checkpoint; positions 0,49,99 from each saved 100-problem training list.
60s end-to-end deadline includes controller startup and translation. No test
problems read. These results are hardware diagnostics, not final evaluation.
One worker repeated at end helps reveal changing background load/thermal effects.
Three-worker follow-up, only if memory remains comfortable:
  caffeinate -i python -u experiments/concurrency-check/benchmark.py --workers 1 3 1
Each invocation creates a new timestamped results-concurrency-* folder. Existing
models and results are not overwritten. Saved checkpoint hashes are checked.
Each case has separate logs/result.json; overall comparison is summary.json.
Unexpected errors stop the benchmark; completed case results survive.
Ctrl+C may wait for bounded in-flight cases before exiting (up to around 60s
plus process cleanup). Do not force-kill processes or run simultaneous benchmarks.

Monitor Activity Monitor Memory during the run. Record memory pressure and swap
before/at peak/after (OS snapshots are also saved before and after stages).
Light report work is allowed, but keep workload similar across stages and record
what was open. A small 60s test cannot certify all domains at a 300s timeout.
No memory cap is imposed by this diagnostic; do not try 3 workers if 2 causes
memory pressure or appreciable swap growth. No automatic concurrency recommendation
is made: inspect throughput, paired solved-case slowdown, changed outcomes, and
memory together. A timeout in both stages hides potential slowdown; longer cases
may be needed before selecting final concurrency. Fast results near startup time
are weak evidence of sustained throughput.

The runner restricts BLAS/OpenMP threads to one and uses ports 12841-12843.
Checkpoint source copies are used with the original observation settings and
progress window 4. The included helper is the existing audited wall-time evaluator.
Local validation: syntax plus real subprocess orchestration with synthetic
controllers/planners at 1,2,3 workers. Real Chainer/Fast Downward execution has
not been run in the assistant environment; the primary Mac run is that test.
