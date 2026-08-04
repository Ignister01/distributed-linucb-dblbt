# Fixed-Arm Challenge Design

## Objective

Determine whether the independently confirmed LinUCB gain in the two
large-effect regimes comes from contextual adaptation or can be reproduced by
one fixed recovery profile from the same 24-action grid.

## Alternatives

1. Confirm all 24 fixed arms for 10 long-run seeds. This is exhaustive but
   requires 480 fixed-arm jobs before rerunning Adaptive.
2. Screen all 24 arms with three short independent seeds, then confirm the
   best arm per scenario against Adaptive on ten untouched long-run seeds.
   This retains independent confirmation with substantially less compute.
3. Test only the arms most frequently selected by LinUCB. This is fastest but
   could miss a better fixed arm and would not support a strong conclusion.

Use alternative 2.

## Frozen Comparison

- Scenarios: `combined-n06-p030` and
  `turnover-n06-p030-j10-l200`.
- Fixed candidates: all 24 legal recovery profiles through the existing
  `pretrain_arm` policy. No fixed candidate receives local context or online
  updates.
- Adaptive candidate: the unchanged frozen-model LinUCB controller.
- Fixed-arm pilot: 20,000 rounds and seeds `7103`, `7117`, `7121`.
- Confirmation: 100,000 rounds and seeds `8101`, `8111`, `8117`, `8123`,
  `8147`, `8161`, `8171`, `8179`, `8191`, `8209`.
- Exogenous randomness is paired by scenario and seed.

For each scenario, select the fixed arm with the highest three-seed mean
evaluation utility. Candidate selection never uses a confirmation seed.

## Decision Rule

Contextual adaptation is independently supported when Adaptive minus the
selected fixed arm has a paired 95% utility interval above zero, at least
8/10 positive seeds, and Jain-fairness difference of at least `-0.01`.
Report utility, collision probability, effective airtime, P95 delay, and Jain
fairness even if the rule fails.

If Adaptive does not beat the selected fixed arm, retain the large
Adaptive-versus-TMC result but characterize it as online profile-selection
gain rather than evidence that context-dependent switching is necessary.

## Outputs

- fixed-arm pilot and confirmation matrices;
- canonical summaries and selected-arm table;
- paired Adaptive-minus-best-fixed effects;
- an explicit interpretation in `FINDINGS.md`.

The existing paper is not modified in this experiment stage.
