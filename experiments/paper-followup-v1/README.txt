PAPER FOLLOW-UP v1

Read AUDIT.md for the exact experimental changes and limitations. This package
does not modify previous source files, trained models, or evaluation results.

Install on BOTH Macs, with the ZIP in Downloads:
  unzip -n ~/Downloads/rl-plan-paper-followup-v1.zip -d experiments

From each Mac's project root with conda environment rl-plan active:

Primary:
  caffeinate -i python -u experiments/paper-followup-v1/run_followup.py --domains blocksworld rovers visitall --workers 4 --mode all --eval-repeats 1 --run

Secondary:
  caffeinate -i python -u experiments/paper-followup-v1/run_followup.py --domains barman childsnack sokoban --workers 4 --mode all --eval-repeats 1 --run

The primary root is ~/Downloads/ANU/Honours/rl-plan
The secondary root is ~/Downloads/Dawit_honours_project/rl-plan-honours

Each command performs quick preflight, then:
1. Four-list fixed FF/CG/CEA/ADD on all 100 test problems, once per domain.
2. Original and replacement training, five seeds per domain, four independent
   model jobs at a time. Periodic evaluation is part of training, not extra jobs.
3. Evaluation of each selected best model on all 100 test problems.

There are 30 training jobs per Mac, 60 total. The same four-worker limit applies
to the current phase; do not launch additional runners at the same time.
The first completed training jobs will give a meaningful estimate of the
remaining training time. Do not assume the earlier two-hour training duration:
each job now includes about 33 evaluations of 100 training episodes.

Results:
  results-paper-followup-v1/protocol.json
  results-paper-followup-v1/fixed-four-list/DOMAIN/coverage.csv
  results-paper-followup-v1/fixed-four-list/DOMAIN/results.csv
  results-paper-followup-v1/training/DOMAIN/MODEL-seedN/trainer.log
  results-paper-followup-v1/training/DOMAIN/MODEL-seedN/agent/scores.txt
  results-paper-followup-v1/training/DOMAIN/MODEL-seedN/agent/best/
  results-paper-followup-v1/learned/DOMAIN/coverage.csv
  results-paper-followup-v1/learned/DOMAIN/results.csv

The prior results-full100-cpu-v1 directories must remain present. Their saved
hashes are used for preflight. All training problems and test problems are read
from the same supplied dataset. Childsnack overlap is carried into the protocol.

Separate phases are available using --mode fixed, train, or evaluate. Keep the
same domain list, seed list and repetition count in each command. Default seeds
are 1 2 3 4 5. Five training seeds are required by this follow-up protocol.

To choose the paper's 25 final evaluation repetitions, use --eval-repeats 25 FROM
THE FIRST LAUNCH. This greatly increases evaluation time. The default 1 is an
explicit deadline compromise, not an exact reproduction of the published table.
Do not switch between 1 and 25 inside an existing output directory.

Resume:
- Evaluation cases resume by running the same command again.
- Completed training jobs are verified and skipped.
- INTERRUPTED TRAINING IS PRESERVED BUT CANNOT EXACTLY RESUME: replay memory and
  complete RNG/environment state are not saved. The runner refuses to overwrite
  an incomplete model. Send its trainer.log for diagnosis rather than deleting
  it or loading partial weights as if they were a full-budget run.
- Ctrl-C cleans up child processes. Keep Macs plugged in, lids open, and allow
  caffeinate to continue until work finishes. Do not edit the new scripts while
  a run is active. Host critical memory pressure stops scheduling safely.

Tested locally: numerical feature behavior, command settings, process budget
handling and completion protections. Full Chainer/Fast Downward training can
only be verified in the installed Mac environment. No claim is made that the
replacement model will outperform original, or that coverage will equal the
paper's results on different hardware and resource accounting.
