# Comments0720 Evidence Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the weak aggregate-gain story with verified protocol-preserving robustness evidence, per-scenario fixed-arm tests, empirical adaptation time, and correctly separated packet-level metrics while retaining a six-page IEEE manuscript.

**Architecture:** The event simulator gains an optional explicit phase schedule without changing existing scenario semantics. Separate analysis code ranks fixed arms, detects incompatible per-scenario optima, and measures reward recovery after known phase boundaries. The ns-3 reducer exposes end-to-end packet loss and simultaneous-access overlap as different metrics; packet results become an A1--A3 boundary diagnostic.

**Tech Stack:** Python 3.12, Pydantic, NumPy/SciPy, pytest/Hypothesis, YAML experiment matrices, ns-3.35/5G-LENA/NR-U C++, SQLite, IEEEtran LaTeX.

---

### Task 1: Add Explicit Event Regime Phases

**Files:**
- Modify: `src/dblbt_fcn/experiment.py`
- Modify: `src/dblbt_fcn/simulation.py`
- Modify: `src/dblbt_fcn/traffic.py`
- Test: `tests/test_experiment.py`
- Test: `tests/test_simulation.py`

- [ ] **Step 1: Add failing schema tests**

Parse a Poisson scenario with two phase records containing `id`,
`duration_rounds`, `active_wifi_nodes`, `active_nru_nodes`, and
`poisson_rate_packets_ms`. Assert rejection when a phase exceeds declared node
counts, has fewer than 32 rounds, contains no active node, or is combined with
legacy join/lifetime turnover.

- [ ] **Step 2: Run the schema tests and confirm failure**

```bash
PYTHONPATH="$PWD/src" ../../.venv/bin/python -m pytest \
  tests/test_experiment.py -k phase -q
```

Expected: failure because `RegimePhaseSpec` and `ScenarioSpec.phases` do not
exist.

- [ ] **Step 3: Implement the frozen phase schema**

```python
class RegimePhaseSpec(_StrictModel):
    id: str
    duration_rounds: int = Field(ge=32)
    active_wifi_nodes: int = Field(ge=0)
    active_nru_nodes: int = Field(ge=0)
    poisson_rate_packets_ms: float = Field(gt=0, allow_inf_nan=False)


class ScenarioSpec(_StrictModel):
    # Existing fields remain unchanged.
    phases: tuple[RegimePhaseSpec, ...] = ()
    phase_repetitions: int = Field(default=1, ge=1)
```

Validate unique phase IDs, at least one active node, counts within the declared
topology, Poisson traffic, and mutual exclusion with legacy turnover fields.

- [ ] **Step 4: Add failing deterministic-transition tests**

Create a 128-round two-phase scenario, repeat it twice, and assert exact phase
IDs, change-point rounds, active node IDs, and policy-independent exogenous
arrival streams for paired TMC/adaptive jobs.

- [ ] **Step 5: Implement phase transitions**

At round zero and each boundary, activate the first declared Wi-Fi/NR-U nodes
through existing channel/controller APIs. Build one `PoissonTraffic` source
per `(phase_index, node_id)` using
`derive_stream_seed(job.exogenous_seed, node_id, f"arrivals_phase_{index}")`.
Write `phase_id`, `phase_index`, `phase_round`, and `change_point` into each
round record; unphased scenarios retain neutral values.

- [ ] **Step 6: Run targeted and full tests**

```bash
PYTHONPATH="$PWD/src" ../../.venv/bin/python -m pytest \
  tests/test_experiment.py tests/test_simulation.py -q
PYTHONPATH="$PWD/src" ../../.venv/bin/python -m pytest -q
```

Expected: all tests pass with the existing Windows-only skip.

- [ ] **Step 7: Commit**

```bash
git add src/dblbt_fcn/experiment.py src/dblbt_fcn/simulation.py \
  src/dblbt_fcn/traffic.py tests/test_experiment.py tests/test_simulation.py
git commit -m "feat: add explicit event regime phases"
```

### Task 2: Add Fixed-Arm Conflict and Adaptation-Time Analysis

**Files:**
- Create: `src/dblbt_fcn/regime_evidence.py`
- Modify: `src/dblbt_fcn/cli.py`
- Test: `tests/test_regime_evidence.py`

- [ ] **Step 1: Write failing ranking tests**

Use synthetic seed-level rows for three scenarios and four arms. Assert each
scenario's best arm, runner-up, paired margin, best-mean fixed arm, minimax
fixed arm, and pairs with conflicting optima. Reject duplicates and unpaired
rows; aggregate seeds before ranking.

- [ ] **Step 2: Write failing recovery-time tests**

Create synthetic rewards around a known change point. With an eight-window
rolling mean and three-window persistence, assert the first interval reaching
90% of the immediate-to-steady-state reward gap and a censored non-recovery.

- [ ] **Step 3: Implement immutable records and analyzers**

```python
@dataclass(frozen=True, slots=True)
class ScenarioArmRanking:
    scenario_id: str
    best_arm: int
    runner_up_arm: int
    best_mean: float
    paired_margin: float
    lower_95: float


@dataclass(frozen=True, slots=True)
class AdaptationTransition:
    run_id: str
    phase_id: str
    change_round: int
    recovery_rounds: int | None
    dwell_rounds: int
    censored: bool
```

Use 10,000 deterministic paired-bootstrap resamples. Define final-quarter mean
reward as steady state, the first eight decision windows as the immediate
level, and require three consecutive rolling windows above 90% recovery.

- [ ] **Step 4: Add deterministic CLI commands**

Add `regime-rank` for fixed-arm summary CSV and `adaptation-report` for a
manifest directory. Write CSV plus JSON audits containing input hashes,
thresholds, seed counts, and revision.

- [ ] **Step 5: Test and commit**

```bash
PYTHONPATH="$PWD/src" ../../.venv/bin/python -m pytest \
  tests/test_regime_evidence.py -q
PYTHONPATH="$PWD/src" ../../.venv/bin/python -m pytest -q
git add src/dblbt_fcn/regime_evidence.py src/dblbt_fcn/cli.py \
  tests/test_regime_evidence.py
git commit -m "feat: analyze fixed-arm conflict and adaptation time"
```

### Task 3: Separate ns-3 Packet Loss and Access Collision

**Files:**
- Modify: `ns3/scenarios/dblbt-nru-wifi-validation.cc`
- Modify: `src/dblbt_fcn/ns3_validation.py`
- Modify: `src/dblbt_fcn/cross_validation.py`
- Modify: `tests/test_ns3_outputs.py`
- Modify: `tests/test_ns3_validation_runner.py`
- Modify: `tests/test_cross_model_validation.py`

- [ ] **Step 1: Write failing database/reducer tests**

Require `packet_loss_ratio` and `simultaneous_access_collision_rate`. Populate
official `simultaneous_tx_*` rows and assert IP Tx/Rx loss differs from access
overlap and cannot substitute for it.

- [ ] **Step 2: Update C++ schema and metrics**

Rename the IP field to `packet_loss_ratio`. Aggregate the official overlap
callback at `ObserveOccupancy`: increment the union counter once when either
same- or cross-technology activity is already present, even when both are
true. Store that union divided by transmission count as
`simultaneous_access_collision_rate`. Bump schema version and retain the
official raw tables for provenance auditing; do not reconstruct the union by
adding their two aggregate columns because that can double-count one access.

- [ ] **Step 3: Update Python validation and reduction**

```python
Ns3MetricName = Literal[
    "throughput_mbps",
    "mean_delay_us",
    "packet_loss_ratio",
    "simultaneous_access_collision_rate",
    "channel_occupancy",
]
```

Only `simultaneous_access_collision_rate` may be compared with event
`collision_probability`; packet loss stays separate and outside collision
direction voting.

- [ ] **Step 4: Add shadowing and duration metadata**

Expose `--shadowingEnabled`, preserve `--simTime`, record both in metadata, and
set the formal packet matrix to shadowing enabled. Seed/run ID supply
independent channel realizations.

- [ ] **Step 5: Test and run one ns-3 smoke benchmark**

```bash
PYTHONPATH="$PWD/src" ../../.venv/bin/python -m pytest \
  tests/test_ns3_outputs.py tests/test_ns3_validation_runner.py \
  tests/test_cross_model_validation.py -q
bash scripts/run_ns3_validation.sh --smoke
```

- [ ] **Step 6: Commit**

```bash
git add ns3/scenarios/dblbt-nru-wifi-validation.cc \
  src/dblbt_fcn/ns3_validation.py src/dblbt_fcn/cross_validation.py \
  tests/test_ns3_outputs.py tests/test_ns3_validation_runner.py \
  tests/test_cross_model_validation.py
git commit -m "fix: separate packet loss from access collision"
```

### Task 4: Run Fixed-Profile Conflict Discovery

**Files:**
- Create: `experiments/comments0720/fixed-arm-discovery.yaml`
- Create: `experiments/comments0720/README.md`
- Generate: `experiments/comments0720/results/fixed-arm-discovery-runs/`
- Generate: `experiments/comments0720/results/discovery-analysis/`

- [ ] **Step 1: Freeze discovery matrix**

Use all 24 arms, seeds `9103`, `9113`, `9127`, and 20,000 rounds. Include 12
protocol-motivated cells spanning 3+3 through 6+6 nodes, Poisson rates 0.015
through 0.045, saturated controls, periodic occupancy, and slow turnover.

- [ ] **Step 2: Run, summarize, and rank**

```bash
PYTHONPATH="$PWD/src" ../../.venv/bin/python -m dblbt_fcn.cli sweep \
  --matrix experiments/comments0720/fixed-arm-discovery.yaml \
  --workers 24 \
  --output-dir experiments/comments0720/results/fixed-arm-discovery-runs
PYTHONPATH="$PWD/src" ../../.venv/bin/python -m dblbt_fcn.cli summarize \
  experiments/comments0720/results/fixed-arm-discovery-runs \
  --output experiments/comments0720/results/fixed-arm-discovery-summary.csv
PYTHONPATH="$PWD/src" ../../.venv/bin/python -m dblbt_fcn.cli regime-rank \
  experiments/comments0720/results/fixed-arm-discovery-summary.csv \
  --output-dir experiments/comments0720/results/discovery-analysis
```

- [ ] **Step 3: Apply frozen advancement rule**

Advance the largest cross-gap pair only if best arms differ, each paired lower
bound over the other regime's arm is positive, and Jain loss is no worse than
0.01. Otherwise record `no_conflicting_pair`, skip switching-superiority
claims, and continue with robustness/timescale-boundary reporting.

- [ ] **Step 4: Commit compact evidence**

Commit matrix, README, ranking, seed summary, and audit. Keep raw rounds ignored.

### Task 5: Confirm Per-Scenario Oracles and Measure T_adapt

**Files:**
- Create: `experiments/comments0720/fixed-arm-confirmation.yaml`
- Create: `experiments/comments0720/adaptive-confirmation.yaml`
- Create: `experiments/comments0720/multiphase-confirmation.yaml`
- Generate: `experiments/comments0720/results/confirmation-*`

- [ ] **Step 1: Freeze 32 untouched formal seeds**

```text
10103, 10111, 10133, 10139, 10141, 10151, 10159, 10163,
10169, 10177, 10181, 10193, 10211, 10223, 10243, 10247,
10253, 10259, 10267, 10271, 10273, 10289, 10301, 10303,
10313, 10321, 10331, 10333, 10337, 10343, 10357, 10369
```

- [ ] **Step 2: Confirm selected stationary regimes**

Run all 24 arms for 100,000 rounds. Report discovery-selected per-scenario
oracles, confirmation ranks, best-global-fixed, published TMC, and adaptive
LinUCB with paired effects on utility, P95 delay, collision, airtime, Jain
fairness, and worst-scenario utility.

- [ ] **Step 3: Run repeated multi-phase matrix**

Alternate the selected pair for four repetitions with at least 2,048 rounds
per phase. Run adaptive, TMC, and discovery-selected fixed arms under identical
phase schedules and all 32 seeds.

- [ ] **Step 4: Measure adaptation time**

Produce median/P95 `T_adapt`, censored fraction, `T_dwell`, and
`P95(T_adapt) < T_dwell`. Adaptation is ineffective if more than 10% of
transitions are censored.

- [ ] **Step 5: Commit compact evidence**

Commit matrices, manifests, summaries, effects, adaptation report, and audit
hashes; leave raw WSL files ignored.

### Task 6: Expand Packet-Level Diagnostic

**Files:**
- Modify: `src/dblbt_fcn/ns3_validation.py`
- Modify: `scripts/run_ns3_validation.sh`
- Generate: `ns3/validation-results-comments0720/`

- [ ] **Step 1: Benchmark one five-second shadowed TMC/adaptive pair**

If it completes within 30 minutes, use five seconds. Otherwise select the
longest duration that keeps the formal two-policy run below eight hours and
record the benchmark decision.

- [ ] **Step 2: Run 15 independent paired seeds**

Use `1201, 1213, 1223, 1231, 1237, 1249, 1259, 1277, 1283, 1289, 1291,
1301, 1303, 1307, 1319`, shadowing enabled, fixed positions, and independent
channel realizations.

- [ ] **Step 3: Audit and reduce**

Require provenance, raw-table, seed, schema, and metric checks. Produce
separate paired tables for throughput, delay, packet loss, simultaneous-access
collision, and occupancy. Never merge packet metrics into event utility.

### Task 7: Generate Figures and Rewrite the Six-Page Paper

**Files:**
- Create: `paper/fcn6/figures/comments0720-summary.pdf`
- Modify: `paper/fcn6/main.tex`
- Modify: `paper/fcn6/sections/*.tex`
- Modify: `paper/fcn6/references.bib`

- [ ] **Step 1: Build one evidence-dense two-panel figure**

Panel (a) shows per-scenario fixed-arm ranks and conflict status. Panel (b)
aligns phase reward/utility, change points, 90% threshold, and `T_adapt`. Use
color plus line style for grayscale readability.

- [ ] **Step 2: Rewrite claim path and title**

Use `Protocol-Preserving LinUCB Adaptation of DB-LBT with Local Observations
for Wi-Fi/NR-U Coexistence` unless results justify a more specific accurate
title. Lead with adaptive robustness. Contributions are protocol-constrained
adaptation, fixed-profile diagnosis, and measured timescale evidence.

- [ ] **Step 3: Add conditional theory statement**

Under bounded contexts, sub-Gaussian rewards, and neighbor drift at most
`epsilon_nbr` in a dwell window `H`, state:

```latex
\mathcal R_k(H)=\widetilde O\!\left(d\sqrt{|\mathcal Q|H}\right)
                 +O(H\epsilon_{\rm nbr}).
```

Cite a verified linear-bandit confidence source and label this a conditional
single-node window bound, not a Nash/adversarial guarantee.

- [ ] **Step 4: Correct packet wording and scope**

Use packet-loss ratio and simultaneous-access collision rate consistently.
Limit claims to A1--A3 event regimes; describe ns-3 as a diagnostic of effects
that break the abstraction.

- [ ] **Step 5: Keep artifacts visible without an appendix**

Use one unnumbered `Reproducibility Artifacts` block before references. Keep
extended matrices, seeds, diagnostics, and commands online.

### Task 8: Compile, Inspect, and Package

**Files:**
- Generate: `paper/fcn6/final/protocol-preserving-linucb-dblbt.pdf`
- Generate: `paper/fcn6/final/protocol-preserving-linucb-dblbt-overleaf.zip`
- Create: `experiments/comments0720/COMMENTS_RESPONSE.md`

- [ ] **Step 1: Run scientific/source checks**

```bash
PYTHONPATH="$PWD/src" ../../.venv/bin/python -m pytest -q
git diff --check
grep -RIn -e TODO -e TBD -e placeholder paper/fcn6 \
  experiments/comments0720 || true
```

- [ ] **Step 2: Compile until references stabilize**

```bash
cd paper/fcn6
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Expected: resolved citations/references and exactly six pages.

- [ ] **Step 3: Render and inspect all pages**

Render at 180 DPI. Check title/authors, legends, overflow, equation breaks,
reference readability, artifact URL, and figure/text placement. Fix every
overlap or clipped label.

- [ ] **Step 4: Write comment-response map and package**

Map comments 1--7 to exact artifacts and paper sections, including negative
outcomes. Package only source, bibliography, required figures, README, and the
compiled PDF; exclude multi-gigabyte raw runs.
