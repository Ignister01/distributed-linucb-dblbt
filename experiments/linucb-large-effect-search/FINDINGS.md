# LinUCB Large-Effect Search Findings

## Main Result

The unchanged distributed LinUCB controller produces a large, independently
confirmed improvement over the fixed TMC DB-LBT profile in two dense dynamic
regimes. Both results pass the frozen strong-effect rule: absolute utility
gain exceeds `0.03`, relative gain exceeds 5%, the paired 95% interval is
positive, all ten seeds improve, and Jain fairness does not decrease.

| Regime | Adaptive - TMC utility | Relative gain | Paired 95% CI | Positive seeds | Collision difference | P95-delay difference | Jain difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| Active-set turnover, 6+6 nodes, rate 0.030 | +0.213555 | +65.27% | [+0.212034, +0.214924] | 10/10 | -0.042960 | -523.304 ms | +0.004494 |
| Combined dynamics, 6+6 nodes, rate 0.030 | +0.209608 | +63.08% | [+0.208181, +0.211113] | 10/10 | -0.036245 | -468.600 ms | +0.001828 |
| Static-load control, 6+6 nodes, rate 0.015 | +0.019359 | +3.36% | [+0.019004, +0.019698] | 10/10 | -0.076943 | -0.213 ms | +0.000012 |

The two dynamic gains are approximately 10.4 and 10.2 times the previous
confirmed maximum of `+0.020555`. They are not isolated pilot peaks: the
three-seed pilot shows a contiguous high-gain band at 6+6 nodes.

| Poisson rate | Turnover gain | Combined-dynamics gain |
|---:|---:|---:|
| 0.020 | +0.154847 | +0.167449 |
| 0.025 | +0.203456 | +0.191911 |
| 0.030 | +0.214562 | +0.219058 |
| 0.035 | +0.065283 | +0.016793 |

## Regime Definition

Each technology has six nodes. A packet lasts 2 ms, and each node receives
Poisson traffic at `0.030 packets/ms`, giving 12 contenders and a nominal
aggregate offered airtime of 72% before contention overhead. In the turnover
case, nodes enter ten rounds apart and remain active for 200 rounds; the
310-round cycle repeatedly grows and shrinks the active set. The combined
case adds a 2 ms external busy interval every 30 ms and local sensing
perturbation with standard deviation `0.4`.

The large effect appears near a recoverable congestion boundary. At 5+5
nodes the same dynamic families give only about `+0.02--0.036`. At 6+6 nodes
and rate 0.035, the benefit falls sharply because both controllers have too
little spare capacity. Static 6+6 traffic at rate 0.045 is approximately
neutral. The result therefore applies to a dense, time-varying active set
with enough local structure to identify a recovery profile and enough spare
capacity for that profile to change queue stability.

## Mechanism

Fixed TMC uses `(kappa, beta, m, Binit) = (7, 3, 6, 15)`. In the two dynamic
confirmations, its mean P95 delay is 669.4 and 722.8 ms. LinUCB reduces these
values to 200.8 and 199.5 ms. It also lowers collisions and slightly improves
airtime and fairness.

The preregistered utility caps the delay penalty at 500 ms. Consequently,
moving the adaptive system below the cap contributes about `+0.199--0.200`
utility; collision reduction contributes another `+0.009--0.011`. Airtime
and fairness provide small positive contributions. The large gain is thus a
queue-stability and tail-latency effect, not a rounding artifact or a fairness
trade.

Across 20 Adaptive confirmation runs in the two dynamic regimes, 483,665
local decisions use 17 of 24 legal arms. Most decisions use `beta=2`, with
the largest shares on arms 4, 5, 16, and 17. The original TMC-equivalent arm
20 is selected only four times.

## Best-Fixed-Arm Challenge

To distinguish contextual adaptation from one-time parameter retuning, all
24 legal fixed profiles were screened on independent seeds `7103`, `7117`,
and `7121`. Both scenarios selected arm 4:

`(kappa, beta, m, Binit) = (5, 2, 10, 15)`.

Arm 4 and unchanged Adaptive DB-LBT were then rerun for 100,000 rounds on ten
untouched seeds `8101--8209` with paired exogenous randomness.

| Regime | Adaptive - best fixed | Relative difference | Paired 95% CI | Positive seeds | P95-delay difference | Jain difference |
|---|---:|---:|---:|---:|---:|---:|
| Combined dynamics | -0.001831 | -0.34% | [-0.002781, -0.000812] | 2/10 | +1.758 ms | -0.000691 |
| Active-set turnover | -0.000686 | -0.13% | [-0.002037, +0.000819] | 4/10 | +0.248 ms | -0.000573 |

LinUCB does not beat the independently selected best fixed profile in these
two regimes. The large Adaptive-versus-TMC effect remains valid, but the
strongest supported interpretation is profile-selection gain relative to the
published fixed TMC configuration. These experiments do not establish that
context-dependent switching is necessary once arm 4 is known offline.

A direct contextual-adaptation claim requires an environment that alternates
between locally distinguishable states whose best fixed profiles conflict.
One fixed arm must then be unable to serve both states, while each state must
persist long enough for local observations to be actionable. That is the
appropriate next experiment; merely adding faster randomness would violate
the learnability premise rather than strengthen the method.

## Experimental Contract

- Source revision used by the large-effect runs:
  `0a04d6df6f33200fb890da6beaa1681b74daf6a7`.
- Source revision used by the fixed-arm challenge:
  `81d0e943f45d076cb23e75d48c47384cc6f8bbb3`.
- Frozen model SHA-256:
  `70611e9712e3a8e4b0f35fc8e0e616f4fde9a20b17c844f1b6406da2833301e6`.
- Main pilot: 56 scenarios, two policies, three seeds, 20,000 rounds, 336 jobs.
- Main confirmation: three scenarios, three policies, ten untouched seeds,
  100,000 rounds, 90 jobs.
- Fixed-arm pilot: two scenarios, 24 arms, three independent seeds, 20,000
  rounds, 144 jobs.
- Fixed-arm challenge: two selected fixed-arm matrices plus one Adaptive
  matrix, ten untouched seeds, 100,000 rounds, 40 jobs.
- All policy comparisons use paired exogenous randomness by scenario and seed.
- Raw records remain in the WSL filesystem; compact summaries, figures, and
  matrices form the Windows delivery package.
- Claims apply to the validated event simulator. Packet-level ns-3 was not
  rerun for these newly discovered high-density regimes.

## Reproduction

Run the four matrices from the repository root:

```bash
.venv/bin/dblbt-fcn sweep \
  --matrix experiments/linucb-large-effect-search/pilot.yaml \
  --workers 24 \
  --output-dir experiments/linucb-large-effect-search/results/pilot-runs \
  --model models/linucb-initial.npz

.venv/bin/dblbt-fcn sweep \
  --matrix experiments/linucb-large-effect-search/confirmation.yaml \
  --workers 24 \
  --output-dir experiments/linucb-large-effect-search/results/confirmation-runs \
  --model models/linucb-initial.npz

.venv/bin/dblbt-fcn sweep \
  --matrix experiments/linucb-large-effect-search/fixed-arm-pilot.yaml \
  --workers 24 \
  --output-dir experiments/linucb-large-effect-search/results/fixed-arm-pilot-runs

.venv/bin/dblbt-fcn sweep \
  --matrix experiments/linucb-large-effect-search/adaptive-fixed-challenge.yaml \
  --workers 16 \
  --output-dir experiments/linucb-large-effect-search/results/adaptive-fixed-challenge-runs \
  --model models/linucb-initial.npz
```

The two `fixed-confirmation-*.yaml` matrices reproduce the best-fixed rows.
Use `dblbt-fcn summarize` for each manifest directory and
`dblbt-fcn regime-report` for the Adaptive-versus-TMC effects.
