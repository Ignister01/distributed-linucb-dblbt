# Restricted-Profile LinUCB Design

## Objective

The current 24-arm controller improves the low-load phase but loses utility in
the high-load phase because its online decisions concentrate on profiles that
were not selected by the fixed-arm conflict experiment. The revision retains
distributed LinUCB and the same eleven local features, while restricting its
actions to the two protocol-valid profiles advanced before confirmation:
arm 4 `(kappa,beta,m,b_init)=(5,2,10,15)` and arm 20 `(7,3,6,15)`.
Their LinUCB states are warm-started only from local contexts and local rewards
recorded under the three discovery seeds for the two selected regimes.

## Alternatives

1. Restrict the action library to arms 4 and 20. This has the smallest control
   surface, uses the already established regime conflict, and avoids learning
   among nearly duplicate profiles.
2. Add discounted or sliding-window LinUCB. This could track drift but adds an
   unvalidated forgetting parameter and requires new pretraining.
3. Reset LinUCB after local change detection. This adds a detector threshold,
   false-alarm behavior, and cold-start cost.

Option 1 is selected for the targeted experiment. Options 2 and 3 remain
future work unless restriction fails the independent pilot.

## Interface And Behavior

- `LinUCB.select` accepts an optional candidate-arm sequence and scores only
  those arms. The default remains all registered arms.
- `AdaptiveController` accepts an optional allowed-arm sequence and forwards
  it only to LinUCB. Fixed-arm and context-free selectors are unchanged.
- The experiment condition `restricted_profiles` activates candidate arms
  `(4, 20)` for `adaptive_db_lbt`; all existing matrices retain their behavior.
- Raw decision records continue to store the selected global arm ID and full
  recovery profile, so existing provenance and readers remain compatible.

## Dynamic Metric

Each 4,096-round phase has a different active-node set. The primary dynamic
utility recomputes airtime, delay, collision probability, and Jain fairness
within each contiguous phase, then averages the eight phase utilities within a
seed. This prevents nodes that are inactive in low-load phases from entering
those phases' fairness calculation. The original whole-run aggregate remains a
secondary diagnostic and is reported even when it disagrees with the primary
metric.

## Evidence Protocol

The implementation is first checked on three development seeds. Before that
pilot starts, separate Adaptive and TMC matrices for a 32-seed confirmation are
committed. Those new seeds have not appeared in discovery or previous
confirmation data. Advancement requires a positive paired time-averaged phase
utility against fixed TMC in the repeated low/high scenario, no material
phase-averaged Jain-fairness loss (greater than 0.01), and every logged action
belonging to `{4,20}`.

## Testing

Unit tests cover candidate validation, deterministic tie-breaking, controller
forwarding, and unchanged default selection. A matrix smoke run checks raw
decision provenance. The independent pilot and, if advanced, the frozen
32-seed confirmation provide the empirical gate.
