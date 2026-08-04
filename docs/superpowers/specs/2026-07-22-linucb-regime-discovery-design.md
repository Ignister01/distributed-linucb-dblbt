# LinUCB Regime-Discovery Experiment Design

## Objective

Identify reproducible operating regimes in which the existing distributed
LinUCB controller materially improves DB-LBT over fixed TMC, and determine
which protocol-level metric explains each gain. This work is experimental
only; it does not modify the paper.

## Constraints

- Retain distributed per-node learning and local MAC observations only.
- Retain the registered 24 legal recovery profiles and the existing reward.
- Compare policies under paired exogenous seeds.
- Treat traffic, channel occupancy, sensing perturbation, topology, and
  active-set turnover as environment factors, not privileged controller
  information.
- Do not add cost-free channel switching or extra spectrum. Such a comparison
  would change system capability rather than isolate LinUCB adaptation.
- Store all new configurations, raw results, summaries, and figures in a
  separate experiment directory.

## Approaches Considered

### 1. Structured regime search (selected)

Keep the algorithm frozen and scan offered load, periodic external occupancy,
occupancy duration, sensing perturbation, active-set turnover, and node
density. This directly answers where the current method is useful and keeps
comparisons with fixed TMC valid.

### 2. Lightweight LinUCB stabilization (conditional)

On search seeds only, test exploration coefficient, a penalty for changing
profiles, and a hold decision when the predicted improvement is too small.
Freeze any selected setting before confirmatory runs. This targets the current
heavy-load exploration loss without introducing another learning algorithm.

### 3. Joint channel/profile selection (deferred)

A multi-channel action would require a link-level channel model, scan
overhead, receiver coordination, switching latency, and new baselines. It is
not comparable with the existing single-channel results and is deferred until
the single-channel channel-condition study is complete.

## Experiment Stages

### Stage A: Pilot search

Use short event runs and three search seeds. Cover four factor families:

1. Poisson offered-load sweep around and beyond the existing 0.035 and 0.065
   packets/ms points.
2. Periodic external occupancy with multiple intervals and busy durations.
3. Active-set turnover with multiple join intervals and lifetimes.
4. Fractional combinations of load, occupancy, turnover, sensing
   perturbation, and node density.

Run adaptive DB-LBT and fixed TMC for every pilot cell. Rank scenario families
using paired utility difference, sign consistency, and component metrics. Do
not rank individual random seeds.

### Stage B: Independent confirmation

Select a small number of scenario families from Stage A, then rerun full
100,000-round jobs with ten seeds not used in search or pretraining. Include
adaptive DB-LBT, fixed TMC, and primary DB-LBT.

A positive result must satisfy all of the following:

- paired-bootstrap 95% interval for adaptive minus fixed-TMC utility is above
  zero;
- mean absolute utility gain is at least 0.005;
- at least eight of ten paired seeds improve;
- Jain fairness loss is no worse than 0.01;
- collision, delay, or effective-airtime components explain the gain.

Results below the practical threshold remain statistically descriptive and
are not labeled a material improvement.

### Stage C: Stabilization ablation

Run only if Stage A identifies regimes with potential gain or recoverable
exploration loss. Tune candidate LinUCB stabilization settings on search seeds,
freeze one setting, and compare it with the unchanged LinUCB on independent
seeds. The unchanged LinUCB remains the primary reference.

### Stage D: Packet-level cross-check

Port at most the strongest confirmed scenario family supported by the pinned
ns-3/5G-LENA setup. Use paired channel/topology seeds. Report directions for
Wi-Fi and NR-U separately; do not merge opposite directions into one claim.

## Outputs

- Pilot and confirmation matrix files.
- Immutable run manifests and raw per-seed records.
- A scenario-by-factor effect table.
- Paired confidence intervals and component decomposition.
- Plots for load, occupancy, and turnover response surfaces.
- A short experimental findings report, including negative results and the
  boundary at which adaptation stops helping.

## Stop Conditions

Stop a candidate family early when all search seeds are non-positive by more
than numerical tolerance. Do not extend an unsuccessful family solely to find
an isolated positive point. Advance only families with a plausible local-MAC
mechanism and enough variation for a controller to learn within the run.
