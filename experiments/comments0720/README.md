# Comments0720 Evidence Revision

This directory contains the frozen experiment matrices and compact evidence
used to address the July 20 review. Raw round records remain on the WSL ext4
filesystem and are excluded from Git; summaries, ranking tables, manifests,
and audits are retained for independent verification.

## Fixed-Profile Discovery

`fixed-arm-discovery.yaml` evaluates every one of the 24 registered recovery
profiles for 20,000 rounds under the three selection seeds `9103`, `9113`, and
`9127`. The 12 cells cover balanced 3+3 through 6+6 Wi-Fi/NR-U topologies,
Poisson rates from 0.015 to 0.045 packets/ms, saturated controls, periodic
external occupancy, and slow active-set turnover. The controller, action
grid, timing, and reward remain unchanged.

The discovery data may select a pair for confirmation only when the two
scenario-optimal profiles differ, the paired bootstrap lower bound for each
profile over the other scenario's profile is positive, and the corresponding
Jain-fairness loss is at most 0.01. To support the later repeated-phase test,
both cells must also be exactly expressible by the explicit phase schema:
Poisson traffic with no within-phase periodic interferer or join/lifetime
turnover. This eligibility rule is applied before any formal seed is run.
Selection seeds are never reused for the formal 32-seed confirmation.

The eligible pair selected by the discovery run is
`poisson-n04-p025` versus `poisson-n06-p045`. Their best profile families are
represented by arm 4, `(alpha,kappa,beta,m,b_init)=(11,5,2,10,15)`, and arm
20, `(11,7,3,6,15)`. On the three discovery seeds, arm 4 exceeds arm 20 by
0.019250 utility in the first cell (95% lower bound 0.016593), while arm 20
exceeds arm 4 by 0.000354 in the second cell (lower bound 0.000091). The Jain
losses are 0.000002 and 0.000009, both below 0.01. Exact values and input
hashes are stored in `results/discovery-analysis/advancement-selection.csv`.

The unrestricted strongest pair contains `turnover-n04-p030-j40-l400`.
It is retained in the conflict table but is not advanced because its
within-phase join/lifetime process cannot be represented by the frozen phase
schema. Replacing it with a static 4+4 phase would change the selected regime.

Run from the isolated worktree on the Linux filesystem:

```bash
PYTHONPATH="$PWD/src" ../../.venv/bin/dblbt-fcn sweep \
  --matrix experiments/comments0720/fixed-arm-discovery.yaml \
  --workers 24 \
  --output-dir experiments/comments0720/results/fixed-arm-discovery-runs

PYTHONPATH="$PWD/src" ../../.venv/bin/dblbt-fcn summarize \
  --manifest-dir experiments/comments0720/results/fixed-arm-discovery-runs/manifests \
  --output experiments/comments0720/results/fixed-arm-discovery-summary.csv

PYTHONPATH="$PWD/src" ../../.venv/bin/dblbt-fcn regime-rank \
  experiments/comments0720/results/fixed-arm-discovery-summary.csv \
  --output-dir experiments/comments0720/results/discovery-analysis
```

## Formal Confirmation

The four confirmation matrices use the same 32 untouched seeds. The two
stationary matrices run 100,000 rounds. The phase matrices alternate 4,096
round low/high cells four times, yielding seven measured transitions and a
32,768-round job. Fixed arms are separated from ordinary policies because the
matrix schema applies `arm_ids` only to `pretrain_arm` jobs.

```bash
PYTHONPATH="$PWD/src" ../../.venv/bin/dblbt-fcn sweep \
  --matrix experiments/comments0720/fixed-arm-confirmation.yaml \
  --workers 24 \
  --output-dir experiments/comments0720/results/fixed-arm-confirmation-runs

PYTHONPATH="$PWD/src" ../../.venv/bin/dblbt-fcn sweep \
  --matrix experiments/comments0720/adaptive-confirmation.yaml \
  --workers 24 --model models/linucb-initial.npz \
  --output-dir experiments/comments0720/results/adaptive-confirmation-runs

PYTHONPATH="$PWD/src" ../../.venv/bin/dblbt-fcn sweep \
  --matrix experiments/comments0720/multiphase-confirmation.yaml \
  --workers 24 --model models/linucb-initial.npz \
  --output-dir experiments/comments0720/results/multiphase-confirmation-runs

PYTHONPATH="$PWD/src" ../../.venv/bin/dblbt-fcn sweep \
  --matrix experiments/comments0720/multiphase-fixed-confirmation.yaml \
  --workers 24 \
  --output-dir experiments/comments0720/results/multiphase-fixed-confirmation-runs

PYTHONPATH="$PWD/src" ../../.venv/bin/dblbt-fcn adaptation-report \
  experiments/comments0720/results/multiphase-confirmation-runs/manifests \
  --output-dir experiments/comments0720/results/adaptation-analysis
```

## Restricted-Profile LinUCB Confirmation

The final adaptive condition separates offline profile discovery from online
control.  The 24 protocol-valid DB-LBT profiles are screened only on discovery
seeds `9103`, `9113`, and `9127`.  Deployment then restricts each node's
LinUCB instance to arm 4, `(kappa,beta,m,b_init)=(5,2,10,15)`, and arm 20,
`(7,3,6,15)`.  The warm start contains 73,541 discovery samples and uses only
each node's 11-dimensional local context and local reward.  The global
evaluation utility is not supplied to LinUCB.

Fit the auditable warm start and run the paired frozen matrices:

```bash
PYTHONPATH="$PWD/src" ../../.venv/bin/python \
  experiments/comments0720/fit_restricted_model.py \
  --summary experiments/comments0720/results/fixed-arm-discovery-summary.csv \
  --run-root experiments/comments0720/results/fixed-arm-discovery-runs \
  --model-output models/linucb-restricted-regime.npz \
  --audit-output experiments/comments0720/results/restricted-model-audit.json

PYTHONPATH="$PWD/src" ../../.venv/bin/dblbt-fcn sweep \
  --matrix experiments/comments0720/restricted-profile-confirmation.yaml \
  --workers 24 --model models/linucb-restricted-regime.npz \
  --output-dir experiments/comments0720/results/restricted-profile-confirmation-runs

PYTHONPATH="$PWD/src" ../../.venv/bin/dblbt-fcn sweep \
  --matrix experiments/comments0720/restricted-profile-confirmation-tmc.yaml \
  --workers 24 \
  --output-dir experiments/comments0720/results/restricted-profile-confirmation-tmc-runs

PYTHONPATH="$PWD/src" ../../.venv/bin/dblbt-fcn summarize \
  --manifest-dir experiments/comments0720/results/restricted-profile-confirmation-runs/manifests \
  --output experiments/comments0720/results/restricted-profile-confirmation-summary.csv

PYTHONPATH="$PWD/src" ../../.venv/bin/dblbt-fcn summarize \
  --manifest-dir experiments/comments0720/results/restricted-profile-confirmation-tmc-runs/manifests \
  --output experiments/comments0720/results/restricted-profile-confirmation-tmc-summary.csv
```

The primary estimator recomputes utility within each contiguous 4,096-round
phase and then averages phases within a seed.  This avoids applying a single
active-population denominator to both 4+4 and 6+6 phases.  The ordinary
whole-run aggregate is retained as a secondary diagnostic, so both aggregation
choices remain visible.

```bash
PYTHONPATH="$PWD/src" ../../.venv/bin/python \
  experiments/comments0720/analyze_phase_pair.py \
  --candidate-run-root experiments/comments0720/results/restricted-profile-confirmation-runs \
  --baseline-run-root experiments/comments0720/results/restricted-profile-confirmation-tmc-runs \
  --candidate-summary experiments/comments0720/results/restricted-profile-confirmation-summary.csv \
  --baseline-summary experiments/comments0720/results/restricted-profile-confirmation-tmc-summary.csv \
  --output-dir experiments/comments0720/results/restricted-profile-confirmation-analysis

PYTHONPATH="$PWD/src" ../../.venv/bin/dblbt-fcn adaptation-report \
  experiments/comments0720/results/restricted-profile-confirmation-runs/manifests \
  --output-dir experiments/comments0720/results/restricted-profile-adaptation-analysis

PYTHONPATH="$PWD/src" ../../.venv/bin/python \
  experiments/comments0720/plot_restricted_confirmation.py \
  --phase-effects experiments/comments0720/results/restricted-profile-confirmation-analysis/phase-effects.csv \
  --whole-run-effect experiments/comments0720/results/restricted-profile-confirmation-analysis/whole-run-effect.csv \
  --decisions experiments/comments0720/results/restricted-profile-confirmation-analysis/decision-arm-membership.csv \
  --output experiments/comments0720/results/restricted-profile-confirmation-analysis/restricted-confirmation.pdf
```

Across the 32 untouched seeds, the phase-averaged Adaptive-minus-TMC utility is
`+0.007338` (95% bootstrap CI `[+0.007093,+0.007584]`; 32/32 positive).  The
low phase contributes `+0.015735`, whereas the high phase contributes
`-0.001058`.  The mixed-active-set whole-run diagnostic is `-0.000668`.  All
224 transitions recover; median and 95th-percentile adaptation times are 736
and 928 rounds, below the 4,096-round phase dwell time.
