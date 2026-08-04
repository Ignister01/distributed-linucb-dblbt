# LinUCB Large-Effect Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find and independently confirm a larger LinUCB effect without changing the algorithm or comparison capability.

**Architecture:** Add one isolated experiment matrix and reuse the existing paired-effect pipeline. Parameterize confirmation-matrix identity and seeds so the second search has untouched confirmation randomness. Run pilot selection, full confirmation, mechanism analysis, and compact reporting without changing the paper.

**Tech Stack:** Python 3.12, Pydantic, NumPy, PyYAML, pytest, existing `dblbt-fcn` event simulator.

---

### Task 1: Freeze and validate the pilot contract

**Files:**
- Create: `experiments/linucb-large-effect-search/pilot.yaml`
- Create: `experiments/linucb-large-effect-search/README.md`
- Create: `tests/test_large_effect_experiment.py`

- [ ] Write a test that requires 20,000 rounds, seeds `(5101, 5107, 5113)`,
  policies `tmc_db_lbt` and `adaptive_db_lbt`, family counts
  `{load: 16, turnover: 24, combined: 16}`, and 336 expanded jobs.
- [ ] Run `pytest tests/test_large_effect_experiment.py -q` and verify failure
  because the matrix is absent.
- [ ] Add the explicit 56-scenario YAML matrix and experiment README.
- [ ] Run the focused test and `dblbt-fcn validate-config`; require 336 jobs.
- [ ] Commit the isolated pilot contract.

### Task 2: Support untouched confirmation seeds

**Files:**
- Modify: `src/dblbt_fcn/regime.py`
- Modify: `src/dblbt_fcn/cli.py`
- Modify: `tests/test_regime.py`
- Modify: `tests/test_cli.py`

- [ ] Add failing tests for configurable confirmation name and exact seed tuple.
- [ ] Verify the focused tests fail because the current generator is fixed.
- [ ] Add keyword-only `name`, `seeds`, and `rounds` inputs with strict
  nonempty/unique/nonnegative validation while retaining current defaults.
- [ ] Add repeatable `--seed`, `--name`, and `--rounds` CLI options.
- [ ] Run focused and full tests, then commit.

### Task 3: Run the pilot and select candidates

- [ ] Run the 336-job pilot with 24 workers and the frozen model.
- [ ] Summarize exactly 336 manifests to the canonical CSV.
- [ ] Produce paired effects and select at most one candidate for each of the
  `load`, `turnover`, and `combined` families.
- [ ] Record every family maximum and verify each selected effect contains
  exactly three paired seeds.

### Task 4: Run independent confirmation

- [ ] Generate a 100,000-round confirmation matrix with seeds `6101`, `6113`,
  `6121`, `6131`, `6133`, `6143`, `6151`, `6163`, `6173`, and `6197`.
- [ ] Validate the matrix job count equals selected scenarios multiplied by 30.
- [ ] Run with 24 workers, summarize all manifests, and compute ten-seed
  Adaptive-minus-TMC effects.
- [ ] Apply the fixed strong-effect and secondary-mechanism rules.

### Task 5: Report and deliver

- [ ] Write `FINDINGS.md` with confirmed and failed targets, component metrics,
  and comparison against `+0.020555`.
- [ ] Generate compact PNG/PDF figures and inspect them visually.
- [ ] Run `git diff --check`, focused tests, the full suite, manifest counts,
  summary counts, and model-hash verification.
- [ ] Commit compact inputs/results, retain raw records only in WSL, and copy
  the delivery package to `D:/Codex-work`.
