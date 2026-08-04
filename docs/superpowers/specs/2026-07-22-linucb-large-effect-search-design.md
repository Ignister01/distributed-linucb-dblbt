# LinUCB Large-Effect Search Design

## Objective

Find a reproducible operating regime in which the unchanged distributed
LinUCB DB-LBT controller has a larger engineering effect than the already
confirmed `0.013--0.021` absolute utility gains over fixed TMC.

## Scientific Constraint

The controller, frozen initial model, 11 local features, 24 legal recovery
profiles, decision interval, reward, and simulator remain unchanged. The
search does not add spectrum, channel switching, global observations, or PHY
information. Only protocol-relevant environment variables change.

## Search Matrix

The pilot uses 20,000 rounds and search seeds `5101`, `5107`, and `5113`.
Every scenario runs fixed TMC and Adaptive DB-LBT under paired exogenous
randomness.

1. `load`: 3+3 through 6+6 Wi-Fi/NR-U nodes crossed with Poisson rates
   `0.015`, `0.025`, `0.035`, and `0.045` packets/ms/node (16 scenarios).
2. `turnover`: 3+3 through 6+6 nodes crossed with rates `0.020`, `0.025`,
   `0.030`, and `0.035` using `join=10`, `lifetime=200`, plus eight boundary
   variants at 4+4 and 5+5 nodes with `j05-l100` and `j20-l400` (24 scenarios).
3. `combined`: 3+3 through 6+6 nodes crossed with rates `0.020`, `0.025`,
   `0.030`, and `0.035`; every cell adds a 30 ms / 2 ms periodic busy process,
   sensing perturbation `0.4`, and `j10-l200` turnover (16 scenarios).

The matrix contains 56 scenarios and 336 jobs.

## Selection and Confirmation

Within each family, select the scenario with the largest paired mean utility
gain among cells where all three pilot seeds improve and Jain fairness changes
by at least `-0.01`. Select at most one scenario per family. Candidate ranking
uses scenario-level means, never individual seeds.

Confirmation uses 100,000 rounds and untouched seeds `6101`, `6113`, `6121`,
`6131`, `6133`, `6143`, `6151`, `6163`, `6173`, and `6197`. It runs Primary
DB-LBT, fixed TMC, and Adaptive DB-LBT.

## Decision Rule

A result is a strong primary effect when all conditions hold:

- paired 95% utility interval is above zero;
- absolute utility gain is at least `0.03`;
- relative utility gain is at least `5%`;
- at least 8/10 seeds improve;
- Jain fairness difference is at least `-0.01`.

A result below the primary target may still be reported as mechanistically
strong when it has at least a 30% relative collision reduction or a 10% P95
delay reduction, with a positive paired utility interval and no fairness loss.
The report must include unsuccessful families and must not promote an isolated
seed or an unconfirmed pilot point.

## Outputs

- isolated pilot and confirmation matrices;
- raw manifests and canonical summaries;
- paired effect tables and family selection;
- comparison with the previously confirmed maximum `+0.020555`;
- figures showing density/load response and confirmed mechanism;
- a compact Windows delivery directory, with raw runs retained in WSL.
