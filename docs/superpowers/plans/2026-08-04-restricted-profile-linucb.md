# Restricted-Profile LinUCB Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an auditable LinUCB condition that selects only fixed-arm-conflict profiles 4 and 20, then evaluate it on newly frozen seeds.

**Architecture:** Candidate filtering belongs in `LinUCB.select`; the adaptive controller owns the allowed set and the experiment condition activates it. Existing behavior remains the default. Separate Adaptive and TMC matrices preserve policy pairing and development/confirmation seed separation.

**Tech Stack:** Python 3.12, NumPy, Pydantic, pytest, YAML experiment matrices.

---

### Task 1: Candidate-Aware LinUCB Selection

**Files:**
- Modify: `tests/test_linucb.py`
- Modify: `src/dblbt_fcn/linucb.py`

- [ ] **Step 1: Write failing selection tests**

Add tests that call `agent.select(context, candidate_arms=(4, 20))`, assert the
result belongs to that tuple, assert lower-index tie breaking within the tuple,
and reject empty, duplicate, boolean, negative, or out-of-range candidates.

- [ ] **Step 2: Verify the tests fail for the missing keyword**

Run:

```bash
PYTHONPATH=src ../../.venv/bin/pytest tests/test_linucb.py -q
```

Expected: candidate-aware tests fail because `select` has no
`candidate_arms` parameter.

- [ ] **Step 3: Implement minimal candidate validation and scoring**

Change the public signature to:

```python
def select(
    self,
    context: object,
    *,
    candidate_arms: object = None,
) -> int:
```

Normalize `None` to `range(self.num_arms)`. Otherwise require a nonempty,
unique sequence of exact integers in `[0, self.num_arms)`, preserve its input
order for scoring, and retain lowest-global-arm tie breaking by iterating
sorted candidates.

- [ ] **Step 4: Verify LinUCB tests pass**

Run the command from Step 2. Expected: all tests pass.

### Task 2: Controller And Experiment Condition

**Files:**
- Modify: `tests/test_adaptive_lifecycle.py`
- Modify: `tests/test_experiment.py`
- Modify: `src/dblbt_fcn/adaptive.py`
- Modify: `src/dblbt_fcn/experiment.py`
- Modify: `src/dblbt_fcn/simulation.py`

- [ ] **Step 1: Write failing controller and matrix tests**

Add a controller test using `allowed_arms=(4, 20)` and assert every emitted
decision selects 4 or 20. Add an experiment test asserting
`conditions: [restricted_profiles]` expands to an adaptive job with that
ablation name.

- [ ] **Step 2: Verify failures**

Run:

```bash
PYTHONPATH=src ../../.venv/bin/pytest \
  tests/test_adaptive_lifecycle.py tests/test_experiment.py -q
```

Expected: failures report the missing controller argument and invalid literal.

- [ ] **Step 3: Implement forwarding and condition activation**

Add `restricted_profiles` to `AblationName`. Add optional `allowed_arms` to
`AdaptiveController`, validate it for LinUCB, and forward it from
`_select_agent`. In `simulate_job_records`, pass `(4, 20)` exactly when
`job.ablation == "restricted_profiles"`.

- [ ] **Step 4: Verify focused tests and smoke compatibility**

Run the Step 2 command, then:

```bash
PYTHONPATH=src ../../.venv/bin/pytest tests/test_simulation.py tests/test_cli.py -q
```

Expected: all tests pass.

### Task 3: Freeze And Run Independent Evidence

**Files:**
- Create: `experiments/comments0720/restricted-profile-pilot.yaml`
- Create: `experiments/comments0720/restricted-profile-pilot-tmc.yaml`
- Create: `experiments/comments0720/restricted-profile-confirmation.yaml`
- Create: `experiments/comments0720/restricted-profile-confirmation-tmc.yaml`
- Modify: `experiments/comments0720/README.md`

- [ ] **Step 1: Freeze both matrices before running the pilot**

Use three development seeds in the pilot and 32 distinct unseen seeds in the
confirmation matrices. Every matrix contains the existing repeated low/high
phase schedule. Adaptive matrices use only `adaptive_db_lbt` with condition
`restricted_profiles`; paired baseline matrices use only `tmc_db_lbt` and no
condition.

- [ ] **Step 2: Commit code and frozen matrices**

```bash
git add src tests experiments/comments0720/*.yaml docs/superpowers
git commit -m "feat: restrict LinUCB to validated DB-LBT profiles"
```

- [ ] **Step 3: Run pilot and apply the advancement gate**

Run the six pilot jobs, summarize them, and pair Adaptive against TMC using the
same phase schedule and seeds. Advance only when mean time-averaged phase
utility is positive, phase-averaged Jain loss is no more than 0.01, and all
decisions are arms 4/20. Retain the whole-run aggregate as a diagnostic.

- [ ] **Step 4: Run the frozen 32-seed confirmation if advanced**

Run with 24 workers, summarize manifests, and retain raw data outside Git.

### Task 4: Reduce And Verify Results

**Files:**
- Modify: `experiments/comments0720/analyze_confirmation.py`
- Create: compact CSV/PDF/JSON outputs under
  `experiments/comments0720/results/restricted-profile-analysis/`

- [ ] **Step 1: Add the new matrices to the existing audited reducer**

Validate expected job counts, paired seed sets, candidate-arm membership, and
input hashes. Recompute metrics in each contiguous phase and average within a
seed. Emit utility, collision, airtime, P95 delay, and Jain effects together
with the secondary whole-run aggregate.

- [ ] **Step 2: Run focused and full tests**

```bash
PYTHONPATH=src ../../.venv/bin/pytest tests/test_linucb.py \
  tests/test_adaptive_lifecycle.py tests/test_experiment.py \
  tests/test_simulation.py -q
PYTHONPATH=src ../../.venv/bin/pytest -q
```

Expected: focused and full suites pass.

- [ ] **Step 3: Commit compact evidence**

```bash
git add experiments/comments0720
git commit -m "exp: confirm restricted-profile LinUCB"
```
