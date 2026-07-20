# Reproducibility Artifact Map

This document separates retained evidence from outputs that must be regenerated.

## Retained in the Repository

- All Python source, tests, experiment matrices, and environment locks.
- The frozen LinUCB model and fixed-oracle selection record under `models/`.
- The 940-row formal event summary in `results/tables/per-seed.csv`.
- Registered hypothesis, cross-model, and audit tables under `results/tables/`.
- Publication PDF/PNG figures and plot-ready CSV tables under
  `results/figures/formal/`.
- The ns-3 scenario, three integration patches, pinned dependency revisions,
  27 formal SQLite databases, manifests, reduced metrics, and audit metadata.
- The six-page manuscript under `paper/fcn6/` and the online experimental
  appendix in `docs/experimental-appendix.md`.

## Regenerated Locally

The event-level raw JSONL records are not stored in Git because the complete set
occupies approximately 6.87 GiB. They are deterministically regenerated from
the committed matrices and seeds:

```bash
DBLBT_WORKERS=8 bash scripts/run_overnight.sh
```

Completed jobs use immutable configuration sidecars, compressed records,
completion markers, and manifests. Interrupted sweeps resume without replacing
valid records.

## Evidence Boundaries

Event results and ns-3 packet results are separate metric families. Event rows
report contention utility, effective airtime, collision probability, access
delay, and Jain fairness. ns-3 rows report application throughput, end-to-end
delay, packet-loss ratio, MAC collision evidence, and occupancy. Cross-model
validation compares scenario directions only and never pools their values.

The fixed oracle is selected using the three training seeds and is not an
oracle fitted on the ten formal seeds. The pretrained model and action grid are
bound to their SHA-256 identities in the committed provenance files.

## Minimal Verification

```bash
bash scripts/run_smoke.sh
sha256sum -c results/tables/final-artifacts.sha256
```

For a raw-backed audit after completing the formal run, use the command in
`docs/reproduction.md`.
