# Corrected honours experiments — frozen 13 October 2026

This is the selected evidence set for the report. No experiments were rerun during consolidation. Both source repositories were at dc0faab when exported. Raw results retain their original machine paths and metadata for provenance; use cases.csv and summary.csv for analysis.

## Included evaluations

| Family | Cases | Authoritative source | Replication |
|---|---:|---|---|
| Fixed four-queue FF, CG, CEA, ADD | 2400 | results-paper-followup-v1/fixed-four-list | One run per problem and heuristic |
| Original and Progress | 3600 | results-aligned-rl-evaluation-v1 | Three trained model seeds |
| recent50, stagprog50, probe25x100, queue50a1 | 2400 | results-simple-policies-extra-v1 | One run per problem and policy |
| Original, fixed16, adaptive4p8m64 | 5400 | results-decision-interval-fulltest-v1 | Three frozen Original model seeds |

All 13,800 final cases use 300 aggregate process CPU seconds, a sampled aggregate 4 GiB resident-memory limit, a 1800-second wall safeguard, and the same 100 test filenames per domain. Fixed controllers have no learned-decision cutoff; aligned and interval RL use an effectively unlimited cutoff. Timeouts and legitimate search-unsolved outcomes are distinct from errors. Superseded interrupted attempts are excluded.

## Comparability and interpretation

The corrected fixed baselines use four queues and an external constant controller, with handle_empty_list=false. Simple and learned selectors use handle_empty_list=true. These are matching resource budgets, not identical search configurations. Native eager(single(...)) baselines in results-full100-cpu-v1 did not enforce memory and are excluded from the main evidence set. Older Original/Progress results from that folder are also excluded. Its protocols are retained only as checkpoint-selection provenance.

Use aligned Original for comparisons against Progress and the simple policies. Use the contemporaneous interval-run Original for fixed16/adaptive comparisons. Do not pool the two Original evaluations as independent training seeds. Each seed label represents a collection of domain-specific trained models, not one cross-domain model.

The interval comparison solves 1370/1800 (Original), 1405/1800 (fixed16), and 1392/1800 (adaptive). Capped CPU totals are 171039.6103, 148303.9882, and 153908.9212 seconds respectively. Fixed16 gains 49 paired solves and loses 14, net +35; coverage improves 1.94 percentage points, capped CPU falls 13.3%. Adaptive improves coverage by 22 cases and lowers capped CPU by 10.0%. Capped CPU charges unsolved cases 300 seconds. These are problem–seed cases over 600 unique test problems.

Simple stagprog50 solves 483/600, compared with aligned Original seed totals 472, 477, 421. This is a post-evaluation comparison among policies, not an independently validated policy-selection procedure. Aligned Progress solves 1356/1800 versus Original 1370/1800.

Childsnack has ten exact train/test duplicates: prob5, prob22, prob27, prob37, prob40, prob48, prob49, prob59, prob60, prob62 (.pddl). Report this limitation and a non-overlap sensitivity analysis. Interval methods tie at 228/270 on the remaining Childsnack cases. Test sets had been inspected during prior development; these are exploratory follow-ups. Three model seeds do not make the same problem three independent benchmark instances. Runtime changes combine altered search and controller overhead, and direct inference profiling was disabled. Do not claim the entire speedup is overhead removal or claim statistical significance from these summaries.

## Contents and scope

- cases.csv: one row per final case, including path to raw result.
- summary.csv: per-domain, condition and seed totals.
- raw/: original final result records, protocols, counters and interval compatibility evidence from both machines.
- provenance/: export hashes, available checkpoint-selection records, and explicitly superseded protocols.
- audit.json: final counts and status inventory.

Source packages are installed at experiments/{paper-followup-v1,aligned-rl-evaluation-v1,simple-policies-extra-v1,decision-interval-fulltest-v1}. Machine-specific build.json and binaries are excluded; build records/protocol hashes remain in the original uploaded archives. Existing training code is already in the base Git commit. Model checkpoints, PDDL inputs and full planner logs must remain backed up on the Macs and in the source archives: this compact Git snapshot alone is not a self-contained rerun environment.

The older RR1/RR50/STAG50/PROGRESS50 policy family and the static selector are not included: no corrected final evaluations for those policies were present in these consolidation exports. They are optional historical context, not necessary additional experiments. No files on either Mac are deleted by the update.
