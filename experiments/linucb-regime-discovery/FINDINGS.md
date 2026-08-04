# LinUCB Regime-Discovery Findings

## Result

The frozen distributed LinUCB controller produces a material, independently
confirmed improvement over fixed-TMC DB-LBT in five protocol-relevant regimes.
All five confirmation scenarios pass the preregistered rule: the paired 95%
lower bound is positive, mean utility gain exceeds `0.005`, all ten paired
seeds improve, and Jain fairness does not decrease by more than `0.01`.

| Confirmed regime | Adaptive - TMC utility | Paired 95% CI | Positive seeds | Collision difference | Airtime difference | P95-delay difference | Jain difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| Active-set turnover (`j10-l200`) | +0.020555 | [+0.019956, +0.021209] | 10/10 | -0.053402 | +0.004976 | -6.025 ms | +0.004588 |
| Combined dynamics (`combined-b`) | +0.019286 | [+0.018595, +0.019921] | 10/10 | -0.049250 | +0.005317 | -5.391 ms | +0.004820 |
| Light Poisson load (`p025`) | +0.018087 | [+0.017857, +0.018345] | 10/10 | -0.073037 | -0.000194 | +0.166 ms | +0.000009 |
| Sensing perturbation (`p035-s08`) | +0.013860 | [+0.013515, +0.014244] | 10/10 | -0.054518 | -0.000132 | -0.405 ms | +0.000013 |
| Periodic occupancy (`i100-d2000`) | +0.013036 | [+0.012747, +0.013385] | 10/10 | -0.051479 | +0.000041 | -0.216 ms | +0.000025 |

The gain is not a rounding artifact. Absolute improvements span `0.0130` to
`0.0206`, or `1.82%` to `3.48%` relative to fixed TMC. The Adaptive mean is
also higher than the Primary DB-LBT mean in every confirmed scenario; this is
a descriptive secondary comparison because candidate selection targeted TMC.

## What Improves

Collision avoidance is the common mechanism. In the light-load confirmation,
the collision-probability reduction of `0.073037` contributes approximately
`0.018259` utility through the fixed `-0.25 p_collision` term, while airtime,
delay, and fairness nearly cancel. Under active-set turnover and combined
dynamics, LinUCB also improves effective airtime, P95 delay, and fairness. The
controller is therefore most useful when local queue, CCA-interruption, delay,
and recent-outcome features reveal which legal DB-LBT recovery profile is
appropriate for the current contention state.

The channel variables remain exogenous. LinUCB does not receive a free extra
channel, global state, or cost-free channel switching. Periodic busy time and
sensing perturbation change the locally observed MAC context, after which the
controller selects one of the same 24 legal DB-LBT profiles available in every
comparison.

## Boundaries

The pilot sweep identifies where adaptation ceases to be useful.

### Offered load

| Poisson rate (packets/ms/node) | Adaptive - TMC utility |
|---:|---:|
| 0.015 | +0.016297 |
| 0.025 | +0.018814 |
| 0.035 | +0.013225 |
| 0.045 | +0.006409 |
| 0.055 | +0.002702 |
| 0.065 | -0.000129 |
| 0.075 | -0.000024 |
| 0.085 | -0.000165 |

The practical-gain boundary lies between `0.045` and `0.055`; the sign change
lies between `0.055` and `0.065`. At high load, persistent contention leaves
little recoverable idle structure and the fixed TMC profile is already near
the useful operating point.

### External occupancy and combined dynamics

At a 10 ms interference period, utility gain declines from `+0.010503` at 5%
busy duty to `+0.005109` at 20%, then to `-0.000176` at 50%. At a 30 ms period,
gain remains positive through a 16.7% duty cycle. Duty cycle alone is therefore
insufficient: the burst timescale determines whether local history contains a
learnable and actionable pattern.

All six active-set turnover cells are positive in the pilot. Combined 4+4-node
cells remain positive, whereas denser 8+8-node combined cases `f` and `g` are
negative and `h` is near zero. The useful regime requires enough temporal
structure to learn but enough remaining contention freedom for a profile
change to matter.

## Experimental Contract

- Simulator source revision: `ce94329139ae30fd7a23344b068fc1e4e94b4a3f`
- Frozen model SHA-256:
  `70611e9712e3a8e4b0f35fc8e0e616f4fde9a20b17c844f1b6406da2833301e6`
- Pilot: 36 scenarios, 2 policies, 3 search seeds, 20,000 rounds, 216 jobs.
- Confirmation: 5 selected scenarios, 3 policies, 10 untouched seeds,
  100,000 rounds, 150 jobs.
- Pilot and confirmation use paired exogenous randomness across policies.
- The confirmed claims apply to the event model. Packet-level ns-3 validation
  was not rerun in this regime-discovery experiment.

## Reproduction

Run from the repository root in the native WSL filesystem:

```bash
.venv/bin/dblbt-fcn sweep \
  --matrix experiments/linucb-regime-discovery/pilot.yaml \
  --workers 24 \
  --output-dir experiments/linucb-regime-discovery/results/pilot-runs \
  --model models/linucb-initial.npz

.venv/bin/dblbt-fcn summarize \
  --manifest-dir experiments/linucb-regime-discovery/results/pilot-runs/manifests \
  --output experiments/linucb-regime-discovery/results/pilot-summary.csv \
  --workers 24

.venv/bin/dblbt-fcn regime-report \
  --summary experiments/linucb-regime-discovery/results/pilot-summary.csv \
  --effects-output experiments/linucb-regime-discovery/results/pilot-effects.csv \
  --selection-output experiments/linucb-regime-discovery/results/selected-scenarios.txt

.venv/bin/dblbt-fcn regime-confirmation \
  --pilot-matrix experiments/linucb-regime-discovery/pilot.yaml \
  --selection experiments/linucb-regime-discovery/results/selected-scenarios.txt \
  --output experiments/linucb-regime-discovery/confirmation.yaml
```

Run the generated confirmation matrix in the same way, summarize it, and then
generate the figures with:

```bash
.venv/bin/python -m dblbt_fcn.regime_plotting \
  --pilot-effects experiments/linucb-regime-discovery/results/pilot-effects.csv \
  --confirmation-effects experiments/linucb-regime-discovery/results/confirmation-effects.csv \
  --output-dir experiments/linucb-regime-discovery/results/figures
```
