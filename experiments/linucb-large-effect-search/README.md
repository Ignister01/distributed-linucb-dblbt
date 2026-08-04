# LinUCB Large-Effect Search

This isolated experiment searches density, load, active-set turnover, and
structured channel occupancy for a larger Adaptive-versus-fixed-TMC effect.
It does not modify the LinUCB algorithm, model, action grid, reward, or paper.

## Frozen Contract

- Pilot seeds: `5101`, `5107`, `5113`
- Confirmation seeds: `6101`, `6113`, `6121`, `6131`, `6133`, `6143`,
  `6151`, `6163`, `6173`, `6197`
- Pilot: 56 scenarios, 20,000 rounds, 336 jobs
- Main comparison: Adaptive DB-LBT minus fixed TMC DB-LBT
- Strong target: absolute utility gain at least `0.03`, relative gain at least
  `5%`, paired 95% lower bound above zero, at least 8/10 positive seeds, and
  Jain-fairness difference at least `-0.01`

Run from `/root/codex-work/linucb-regime-discovery-20260722` so raw artifacts
remain on the WSL ext4 filesystem.

## Fixed-Arm Challenge

The two independently confirmed large-effect regimes are also screened
against all 24 fixed recovery profiles. `fixed-arm-pilot.yaml` uses seeds
`7103`, `7117`, and `7121` for selection. The selected fixed profile is then
compared with unchanged Adaptive DB-LBT using ten untouched `8101--8209`
confirmation seeds. This separates contextual-adaptation evidence from a
one-time profile-retuning effect.
