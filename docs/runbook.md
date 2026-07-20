# Event-Level Formal Runbook

## Frozen Scope

The formal event batch uses the versioned matrices without seed, threshold, or
round-count overrides. Pretraining contains 792 jobs in its own run root. The
shared formal root contains exactly 360 reproduction, 500 held-out, and 80
ablation jobs, for 940 reportable jobs. Smoke remains a separate three-job
gate.

The frozen algorithm constraints are `alpha=11`, the registered 24-arm grid,
an eight-own-attempt cold start, and a 32-contention-round decision interval.
Runtime learning is distributed and uses local observations only.

## Command

From WSL2 Ubuntu 24.04 at the repository root:

```bash
DBLBT_WORKERS=8 bash scripts/run_overnight.sh 2>&1 | tee logs/overnight.log
```

`DBLBT_WORKERS` defaults to 8 and must be in `1..24`. The lower default keeps
peak memory bounded on the 16 GB host. The script refuses to start with tracked
source changes, untracked source/config/script files, or less than 40 GB free
on the project drive.

## Checkpoints And Resume

Every job owns an immutable canonical raw record, completion marker, config
sidecar, and manifest. Re-running the command skips valid completed jobs and
re-runs only missing or invalid jobs. Do not delete `runs/pretrain` or
`runs/formal` when resuming.

The checkpoints are:

1. `bash scripts/run_smoke.sh` passes its full tests and three-job audit.
2. `runs/pretrain` reaches 792 complete manifests.
3. `models/linucb-initial.npz`, `models/fixed-oracle-arm.json`, and
   `models/event-formal-inputs.sha256` are frozen.
4. `runs/formal` reaches 940 complete manifests across all three matrices.
5. `results/tables/per-seed.csv` contains 940 rows.
6. `results/figures/formal` contains 16 figures and `tables/` contains eight
   evidence files.
7. The final read-only audit exits zero.

## Failure Policy

The script is fail-fast. A failed command stops later stages and preserves all
valid completed artifacts for resume. Do not edit source, matrices, the model,
or the Oracle artifact between the frozen tag and completion. Do not remove
failed, inconclusive, or negative hypothesis rows from the final evidence.

## Rehearsal Gate

The exact 100,000-round rehearsal ran on 2026-07-18 with eight workers and the
first registered seed from each matrix. The model and Oracle used for timing
were isolated rehearsal inputs and are not formal evidence.

| Matrix | Seed | Jobs | Elapsed seconds | Output bytes | Coordinator max RSS KiB |
|---|---:|---:|---:|---:|---:|
| pretrain | 1103 | 264 | 3801.10 | 1,324,977,311 | 133,580 |
| reproduction | 410 | 36 | 388.16 | 122,218,619 | 197,576 |
| heldout | 410 | 50 | 606.74 | 173,872,986 | 165,152 |
| ablation | 410 | 8 | 174.80 | 43,599,035 | 94,632 |

Scaling each row by its frozen seed count projects 23,100.3 seconds (6.42
hours) and 7,371,838,333 bytes (6.87 GiB) for 792 pretraining plus 940 formal
jobs. A 10 percent runtime contingency gives 7.06 hours, below the ten-hour
gate. The D drive had 194,747,932,672 bytes (181.37 GiB) free after rehearsal,
so the projected output leaves more than 174 GiB free and passes the 40 GB
gate. The RSS values cover the coordinator process and are diagnostic rather
than an aggregate of all worker processes.

## Completed Formal Evidence

The event simulator and all formal raw records remain bound to frozen revision
`0fa60428c163c41defa9b12e83dc034ed3ed6229` (`event-formal-v1`). The 792
pretraining jobs ran from `2026-07-17T21:02:07Z` through
`2026-07-18T00:05:14Z` (3.052 hours of wall time). The 940 formal jobs ran
from `2026-07-18T05:43:23Z` through `2026-07-18T08:13:33Z` (2.503 hours of
wall time). Their retained directories occupy 3.702 GiB and 3.160 GiB,
respectively. After final reporting, the D drive retained 174.15 GiB free.

The original sequential reporting path was replaced only after all 940 raw
manifests existed. Derived-only revision `72904af` uses ordered 16-process
read-only reduction while retaining the same metric functions, row order,
atomic publication, model checks, and audit rules. Fresh regression evidence
for that revision is `1147 passed, 1 skipped`. The formal derived stages took:

| Stage | Workers | Elapsed seconds | Accepted output |
|---|---:|---:|---|
| summarize | 16 | 1386.7 | 940 CSV data rows |
| plot and evidence tables | 16 | 1280.7 | 16 figures and 8 tables |
| independent read-only audit | 16 | 1251.5 | `audited=940` |

The frozen evidence identities are:

| Artifact | SHA-256 |
|---|---|
| `models/linucb-initial.npz` | `70611e9712e3a8e4b0f35fc8e0e616f4fde9a20b17c844f1b6406da2833301e6` |
| `models/fixed-oracle-arm.json` | `621c5a050078c26d93306e43690a041b09f0a69fb4f1c680ceb02da550725ba5` |
| `results/tables/per-seed.csv` | `0aa2868005267f0b63feda74f8113d6d21a90267066907949f7c7dafb1c4c5eb` |

The fixed Oracle arm is 16. All 940 summary rows contain finite numeric
metrics. The event-only hypothesis table records H1 `pass`, H2 `fail`, H3
`pass`, H4 `pass`, and H5 `not_evaluated`. After the pinned ns-3 validation and
the exact 18-job event match, `results/tables/final-hypotheses.csv` records H5
as `inconclusive`. The failed H2 and inconclusive H5 results are retained
without filtering or relabeling.
