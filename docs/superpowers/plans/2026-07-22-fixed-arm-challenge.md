# Fixed-Arm Challenge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Independently test whether unchanged LinUCB beats the strongest fixed profile from the same 24-action grid in the two confirmed large-effect regimes.

**Architecture:** Use the existing `pretrain_arm` policy as a fixed-profile controller. Screen all arms with three independent short seeds, select by scenario-level mean utility, then create one fixed-profile matrix per scenario and one Adaptive matrix on ten untouched seeds. Reuse canonical summaries and paired-effect analysis.

**Tech Stack:** Python 3.12, Pydantic experiment matrices, existing `dblbt-fcn` event simulator, CSV summaries, pytest.

---

### Task 1: Freeze the fixed-arm pilot

**Files:**
- Create: `experiments/linucb-large-effect-search/fixed-arm-pilot.yaml`
- Modify: `experiments/linucb-large-effect-search/README.md`

- [ ] Add the two exact large-effect scenarios from `pilot.yaml`, 20,000
  rounds, seeds `7103`, `7117`, `7121`, policy `pretrain_arm`, and arm IDs
  `0..23`.
- [ ] Run
  `.venv/bin/dblbt-fcn validate-config experiments/linucb-large-effect-search/fixed-arm-pilot.yaml`
  and require exactly 144 jobs.
- [ ] Commit the frozen matrix and README contract.

### Task 2: Screen all fixed profiles

- [ ] Run the 144-job matrix with 24 workers into
  `results/fixed-arm-pilot-runs`.
- [ ] Summarize exactly 144 manifests into
  `results/fixed-arm-pilot-summary.csv`.
- [ ] For each scenario, group by `arm_id`, average `evaluation_utility` over
  all three seeds, and write every ranked arm to
  `results/fixed-arm-ranking.csv`.
- [ ] Write the top arm for each scenario to
  `results/selected-fixed-arms.csv`, including its profile parameters from
  `adaptive_arms()`.

### Task 3: Run independent fixed-versus-Adaptive confirmation

**Files:**
- Create: `experiments/linucb-large-effect-search/fixed-confirmation-<scenario>.yaml`
- Create: `experiments/linucb-large-effect-search/adaptive-fixed-challenge.yaml`

- [ ] Create one `pretrain_arm` matrix per selected scenario with its selected
  arm only, 100,000 rounds, and seeds `8101`, `8111`, `8117`, `8123`, `8147`,
  `8161`, `8171`, `8179`, `8191`, `8209`.
- [ ] Create one Adaptive matrix with both scenarios and the same rounds and
  seeds.
- [ ] Validate exactly 10 jobs per fixed matrix and 20 Adaptive jobs.
- [ ] Run all 40 jobs with 24 workers and the unchanged frozen model for the
  Adaptive matrix.
- [ ] Summarize the fixed and Adaptive manifests into canonical CSV files.

### Task 4: Analyze and report

- [ ] Join fixed and Adaptive rows by scenario and seed, then compute paired
  utility, collision, airtime, P95-delay, and Jain differences with paired
  95% intervals.
- [ ] Apply the frozen rule: utility lower bound above zero, at least 8/10
  positive seeds, and Jain difference at least `-0.01`.
- [ ] Update `experiments/linucb-large-effect-search/FINDINGS.md` with both
  outcomes: Adaptive-versus-TMC and Adaptive-versus-best-fixed.
- [ ] Regenerate the compact figures and visually inspect both PNGs.

### Task 5: Verify and deliver

- [ ] Run `git diff --check`, `pytest tests/test_regime.py -q`, and the full
  test suite.
- [ ] Verify manifest and summary counts, frozen model SHA-256, and all matrix
  hashes.
- [ ] Commit compact inputs/results while excluding raw records.
- [ ] Copy the compact delivery directory to `D:/Codex-work` and verify that
  all CSV, PNG, PDF, YAML, README, and findings files open from Windows.
