# LinUCB Regime Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find and independently confirm traffic, channel-occupancy, and active-set regimes in which the frozen distributed LinUCB controller materially outperforms fixed TMC DB-LBT.

**Architecture:** Add one isolated experiment package containing a pilot matrix and a confirmation-matrix generator. Add a small package module that computes paired per-scenario effects from the canonical summary schema, selects at most one candidate per preregistered factor family, and writes machine-readable reports. Reuse the existing simulator, frozen model, manifest system, and summary command without modifying the paper or the registered experiment artifacts.

**Tech Stack:** Python 3.12, Pydantic, NumPy, PyYAML, pytest, existing `dblbt-fcn` event simulator.

---

## File Structure

- Create `experiments/linucb-regime-discovery/pilot.yaml`: fixed 20,000-round search matrix.
- Create `experiments/linucb-regime-discovery/README.md`: exact run and interpretation contract.
- Create `src/dblbt_fcn/regime.py`: paired-effect analysis, family selection, confirmation matrix generation, and CSV writing.
- Modify `src/dblbt_fcn/cli.py`: expose `regime-report` and `regime-confirmation` commands.
- Create `tests/test_regime.py`: unit tests for pairing, intervals, selection, and generated matrix validity.
- Create `tests/test_regime_experiment.py`: enforce pilot factors, seeds, job count, and policy scope.
- Generate under `experiments/linucb-regime-discovery/results/`: raw pilot/confirmation runs, canonical summaries, effect tables, and the final findings report.

### Task 1: Create an isolated experiment worktree

**Files:**
- Verify: `pyproject.toml`
- Verify: `models/linucb-initial.npz`

- [ ] **Step 1: Create the experiment worktree**

Run the repository's worktree workflow to create sibling directory
`D:/Codex-work/linucb-regime-discovery-20260722` on branch
`experiment/linucb-regime-discovery`.

- [ ] **Step 2: Verify the baseline**

Run: `pytest -q`

Expected: all existing tests pass before experiment changes.

- [ ] **Step 3: Record the baseline commit and model hash**

Run: `git rev-parse HEAD` and
`Get-FileHash models/linucb-initial.npz -Algorithm SHA256`.

Expected: one commit SHA and one SHA-256 value saved in the experiment README.

### Task 2: Define and validate the pilot matrix

**Files:**
- Create: `experiments/linucb-regime-discovery/pilot.yaml`
- Create: `experiments/linucb-regime-discovery/README.md`
- Test: `tests/test_regime_experiment.py`

- [ ] **Step 1: Write the failing matrix-contract test**

```python
def test_regime_pilot_matrix_is_isolated_and_factor_complete() -> None:
    matrix = load_matrix(Path("experiments/linucb-regime-discovery/pilot.yaml"))
    assert matrix.rounds == 20_000
    assert matrix.seeds == (1709, 1871, 1999)
    assert matrix.policies == ("tmc_db_lbt", "adaptive_db_lbt")
    assert not set(matrix.seeds) & set(PRETRAINING_SEEDS)
    assert not set(matrix.seeds) & set(HELD_OUT_SEEDS)
    prefixes = {scenario.id.split("-", 1)[0] for scenario in matrix.scenarios}
    assert prefixes == {"load", "occupancy", "turnover", "sensing", "combined"}
    assert len(expand_matrix(matrix)) == len(matrix.scenarios) * 6
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `pytest tests/test_regime_experiment.py -q`

Expected: FAIL because the pilot matrix does not exist.

- [ ] **Step 3: Add the matrix and experiment README**

The matrix contains 36 scenarios: eight load points, eight occupancy points,
six turnover points, six sensing points, and eight fractional combined points.
Every scenario uses supported `ScenarioSpec` fields only. The README records
the fixed search seeds, confirmation seeds
`4001,4003,4007,4013,4019,4021,4027,4049,4051,4057`, practical threshold
`0.005`, fairness tolerance `-0.01`, and the rule that pilot seeds cannot be
used for confirmatory claims.

- [ ] **Step 4: Validate and test**

Run: `dblbt-fcn validate-config experiments/linucb-regime-discovery/pilot.yaml`

Expected: `type=matrix name=linucb-regime-pilot jobs=216`.

Run: `pytest tests/test_regime_experiment.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the pilot contract**

```text
git add experiments/linucb-regime-discovery tests/test_regime_experiment.py
git commit -m "add LinUCB regime search matrix"
```

### Task 3: Implement paired regime analysis

**Files:**
- Create: `src/dblbt_fcn/regime.py`
- Modify: `src/dblbt_fcn/cli.py`
- Test: `tests/test_regime.py`

- [ ] **Step 1: Write failing tests for strict pairing and effect direction**

```python
def test_scenario_effects_require_matching_policy_seeds() -> None:
    rows = synthetic_rows(baseline_seeds=(1, 2), candidate_seeds=(1,))
    with pytest.raises(ValueError, match="paired seeds"):
        scenario_effects(rows)


def test_scenario_effects_decompose_candidate_minus_baseline() -> None:
    rows = synthetic_rows(
        baseline_utility=(0.70, 0.72, 0.71),
        candidate_utility=(0.72, 0.74, 0.73),
    )
    effect = scenario_effects(rows)[0]
    assert effect.utility_difference == pytest.approx(0.02)
    assert effect.positive_seeds == 3
    assert effect.lower_95 > 0
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `pytest tests/test_regime.py -q`

Expected: FAIL because `dblbt_fcn.regime` does not exist.

- [ ] **Step 3: Implement `ScenarioEffect` and generic paired bootstrap**

`ScenarioEffect` contains scenario ID, family, seed count, baseline and
candidate means, utility difference, relative difference, 95% interval,
positive-seed count, and candidate-minus-baseline differences for collision,
effective airtime, P95 delay, and Jain fairness. Bootstrap uses 10,000 paired
resamples with seed `20260722` and accepts the matrix's exact seed set rather
than the paper's hard-coded held-out seeds.

- [ ] **Step 4: Implement preregistered family selection**

```python
def select_confirmation_scenarios(effects: Sequence[ScenarioEffect]) -> tuple[str, ...]:
    eligible = [
        effect for effect in effects
        if effect.positive_seeds == effect.seed_count
        and effect.utility_difference >= 0.002
        and effect.fairness_difference >= -0.01
    ]
    selected: dict[str, ScenarioEffect] = {}
    for effect in eligible:
        current = selected.get(effect.family)
        if current is None or (
            effect.utility_difference,
            effect.scenario_id,
        ) > (
            current.utility_difference,
            current.scenario_id,
        ):
            selected[effect.family] = effect
    return tuple(selected[family].scenario_id for family in sorted(selected))
```

The declared families are `load`, `occupancy`, `turnover`, `sensing`, and
`combined`; selection is deterministic with scenario ID as the tie breaker.

- [ ] **Step 5: Implement confirmation-matrix generation**

Generate a validated `MatrixSpec` with 100,000 rounds, the ten fixed
confirmation seeds, policies `primary_db_lbt`, `tmc_db_lbt`, and
`adaptive_db_lbt`, and exact selected scenarios copied from the pilot matrix.
Reject selections not present in the pilot matrix.

- [ ] **Step 6: Expose two CLI commands**

`regime-report` loads a canonical summary and writes `effects.csv` plus
`selected-scenarios.txt`. `regime-confirmation` reads the pilot matrix and
selection file, then writes canonical YAML for the confirmation sweep.

- [ ] **Step 7: Run focused and full tests**

Run: `pytest tests/test_regime.py tests/test_regime_experiment.py -q`

Expected: PASS.

Run: `pytest -q`

Expected: all tests pass.

- [ ] **Step 8: Commit analysis tooling**

```text
git add src/dblbt_fcn/regime.py src/dblbt_fcn/cli.py tests/test_regime.py
git commit -m "add paired LinUCB regime analysis"
```

### Task 4: Run and summarize the pilot search

**Files:**
- Generate: `experiments/linucb-regime-discovery/results/pilot-runs/`
- Generate: `experiments/linucb-regime-discovery/results/pilot-summary.csv`
- Generate: `experiments/linucb-regime-discovery/results/pilot-effects.csv`
- Generate: `experiments/linucb-regime-discovery/results/selected-scenarios.txt`
- Generate: `experiments/linucb-regime-discovery/confirmation.yaml`

- [ ] **Step 1: Run the pilot with the frozen model**

```text
dblbt-fcn sweep --matrix experiments/linucb-regime-discovery/pilot.yaml --workers 24 --output-dir experiments/linucb-regime-discovery/results/pilot-runs --model models/linucb-initial.npz
```

Expected: `completed=216` on a fresh run, or only the remaining count after a resume.

- [ ] **Step 2: Build the canonical summary**

```text
dblbt-fcn summarize --manifest-dir experiments/linucb-regime-discovery/results/pilot-runs/manifests --output experiments/linucb-regime-discovery/results/pilot-summary.csv --workers 24
```

Expected: `rows=216`.

- [ ] **Step 3: Produce effects and selection**

Run `regime-report`, then `regime-confirmation` with the files above.

Expected: one effect row per scenario and no more than one selected scenario
per factor family. The generated confirmation matrix validates successfully.

- [ ] **Step 4: Inspect negative and positive families**

Confirm that rankings use scenario-level paired means, every effect has three
pairs, and no candidate is selected from an incomplete or unfair comparison.

### Task 5: Run independent confirmation and write findings

**Files:**
- Generate: `experiments/linucb-regime-discovery/results/confirmation-runs/`
- Generate: `experiments/linucb-regime-discovery/results/confirmation-summary.csv`
- Generate: `experiments/linucb-regime-discovery/results/confirmation-effects.csv`
- Create: `experiments/linucb-regime-discovery/FINDINGS.md`

- [ ] **Step 1: Run the confirmation matrix**

Run the generated matrix with 24 workers and the frozen LinUCB model.

Expected: `selected scenario count * 30` completed jobs.

- [ ] **Step 2: Summarize and compute confirmatory intervals**

Run the canonical summarizer and `regime-report` without candidate selection.
Each adaptive-versus-TMC effect must contain ten paired seeds.

- [ ] **Step 3: Apply the fixed material-gain rule**

Label a scenario materially positive only when lower 95% utility bound is
above zero, mean difference is at least 0.005, at least eight seeds improve,
and fairness difference is at least -0.01.

- [ ] **Step 4: Write `FINDINGS.md`**

Report all confirmed candidates, failures, effect components, the strongest
regime, and any load/occupancy boundary. State explicitly that the event model
does not establish packet-level gain.

- [ ] **Step 5: Verify artifacts and commit reproducible inputs/tooling**

Run: `git diff --check`, focused tests, full tests, and a manifest/summary row
count check. Commit configuration, analysis tooling, compact effect tables,
and findings. Do not commit large raw run files unless repository policy
already tracks equivalent artifacts.
