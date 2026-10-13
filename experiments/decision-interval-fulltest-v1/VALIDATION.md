# Full-test package validation — 12 October 2026

The planner patch, controller adapter, setup builder and resource supervisor are
byte-identical to the validated pilot package. Native tests below were performed
on that identical implementation and have not been repeated for the scope change.
The full-test runner now schedules 5,400 unique test problem/seed/condition cases
(900/domain); its short compatibility diagnostics still use training inputs.
Four updated Python test groups pass for the full-test configuration.


## Completed locally

- Verified all 964 manifest-listed primary files and all 959 secondary files.
  The active planner sources agree; three extra primary backup files are
  excluded from the isolated build.
- Compiled both the original supplied C++ source and patched C++ source on
  Linux using GCC 13 and CMake 3.31.10, Release, LP disabled.
- Nine standalone C++ control checks passed: original removal cadence,
  expansion-counted fixed boundaries, FF warm-up, stagnation patience,
  maximum duration with continuing progress, queue-minimum oscillation,
  genuine progress resetting patience, empty queues, retaining inactive-queue
  episode bests, and rejection of invalid constants (some checks cover
  multiple assertions).
- Four Python test groups passed: frozen sample/task construction, precise
  feature arithmetic through the actual `_process_data` method from all 18
  supplied Original runtimes, supervisor byte identity, and pilot command
  construction. The original pilot sample is distinct and excludes test-identical bytes.
  The updated full-test scheduler has 5,400 unique task keys and explicitly
  uses test paths, retaining separate training paths for correctness checks.
- Eighteen native compatibility comparisons passed across six domains and
  all 18 selected weight files. Original and patched every-removal mode
  produced identical raw observations, float32 difference vectors, actions,
  and expansion/plan-cost statistics where completed. Each session ran to a
  solution or a bounded 384-action test prefix. Fragmented action frames were
  exercised. See `validation/native-tests.log`.
- Twelve native interval sessions passed (fixed and adaptive, six domains).
  Traces check that each transmitted previous observation is exactly the
  pre-removal observation from one removal ago, including discarded closed
  entries. These include both successful termination within a held block
  and deliberately stopped prefixes.
- All Python files compile successfully.

The native development action source uses NumPy and the uploaded weights,
with planner RNG seed 1 in each paired test. It is a deterministic test driver,
not a replacement for Chainer or evidence that framework inference is
identical on a different platform. Tests containing intentional truncation
must not be counted as planning failures or performance measurements.

## Mac validation still required

The complete Chainer/controller/supervisor pipeline has not been executed
here in the users' macOS Python 3.8 environments. The shipped runner therefore
requires its own compatibility gate before any timed cases in each domain.
It uses each actual selected checkpoint and seed, the existing legacy
executable, and the freshly built compatibility executable. It compares
raw observations, real Chainer inputs/actions and available planning outcomes.
A mismatch or failed diagnostic stops the run.

The timed `case.py` and `cpu_meter.py` are byte-for-byte copies of the aligned
evaluator. Its previously established accounting and resource semantics are
reused; this package does not independently prove Darwin accounting accuracy.

The patch preserves queue-statistic calculations and the frozen controller's
feature transform. It additionally gives its new protocol complete socket
framing and a structured terminal notification for exhausted search. Search
exhaustion is still unsolved; the supervisor determines success from planner
exit/output, not from a controller reward.

No speedup, coverage gain, robustness gain, or literature novelty has been
established. The pilot is the next measurement, and the compatibility gate is
an engineering check rather than a proof of full-search equivalence on every
possible instance.

Full-test summary check passed: nine 100-case configuration groups per domain; known-overlap successes are excluded only from the nonoverlap table. Childsnack has ten flagged test filenames.
