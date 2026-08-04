# LinUCB Regime-Discovery Experiments

This directory searches for reproducible conditions in which the frozen
distributed LinUCB controller materially improves over fixed TMC DB-LBT. It
does not modify the paper or the registered result set.

## Frozen Inputs

- Simulator source revision:
  `ce94329139ae30fd7a23344b068fc1e4e94b4a3f`
- LinUCB model SHA-256:
  `70611e9712e3a8e4b0f35fc8e0e616f4fde9a20b17c844f1b6406da2833301e6`
- Actions: the existing 24 legal recovery profiles
- Context: the existing 11 local MAC features
- Search seeds: `1709`, `1871`, `1999`
- Confirmation seeds: `4001`, `4003`, `4007`, `4013`, `4019`, `4021`,
  `4027`, `4049`, `4051`, `4057`

The pilot search varies offered load, periodic channel occupancy, sensing
perturbation, active-set turnover, and selected combinations. It does not add
extra spectrum or cost-free channel switching.

## Selection Rule

At most one scenario is selected from each of the `load`, `occupancy`,
`turnover`, `sensing`, and `combined` families. A pilot scenario is eligible
only when all three paired utility differences are positive, its mean utility
difference is at least `0.002`, and its mean Jain-fairness difference is at
least `-0.01`.

A confirmation result is called materially positive only when:

- the paired-bootstrap 95% lower bound is above zero;
- mean utility improvement is at least `0.005`;
- at least eight of ten paired seeds improve;
- mean Jain-fairness difference is at least `-0.01`.

## Runtime Location

Run the experiment in the WSL Linux filesystem under
`/root/codex-work/linucb-regime-discovery-20260722`. Raw run artifacts remain
there for performance. Compact summaries, plots, and findings are copied to
the Windows delivery directory after verification.

## Completed Outputs

- `FINDINGS.md`: interpretation, confirmed gains, and operating boundaries.
- `confirmation.yaml`: independently seeded 100,000-round matrix.
- `results/pilot-summary.csv`: 216 validated pilot rows.
- `results/pilot-effects.csv`: 36 paired pilot effects.
- `results/confirmation-summary.csv`: 150 validated confirmation rows.
- `results/confirmation-effects.csv`: five ten-seed paired effects.
- `results/figures/`: PNG and vector-PDF result figures.

Raw JSONL records and manifests remain under `results/*-runs/` in the WSL
filesystem. They are deliberately excluded from the compact delivery archive;
the matrices, frozen model, source revision, and commands above regenerate
them.
