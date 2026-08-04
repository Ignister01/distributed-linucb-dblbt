# Comments0720 Paper and Experiment Revision Design

## Objective

Revise the six-page FCN manuscript and its reproducibility package in response
to `comments0720.pdf`. The paper's main claim becomes protocol-preserving
adaptive robustness under the A1--A3 mutually sensing event abstraction. A
performance-gain claim is made only where independent experiments show that
different regimes prefer incompatible fixed recovery profiles and the online
controller responds within the regime dwell time.

## Alternatives Considered

1. **Prose-only repair.** Reframe the contribution and relabel the ns-3 loss
   metric without new experiments. This is fast but does not establish why an
   adaptive controller is needed.
2. **Event-first evidence repair (selected).** Add fixed-arm conflict tests,
   multi-phase switching, adaptation-time measurement, and at least 30 event
   seeds. Split packet loss from simultaneous-access collision in ns-3 and use
   packet results only as a physical-model diagnostic.
3. **Full packet-level retraining.** Train all 24 arms directly in ns-3 and
   repeat the complete event study with 15--30 packet seeds. This would answer
   the cross-model issue most strongly, but it is not credible within the
   current short paper and runtime budget because event and packet contexts do
   not yet share identical sampling semantics.

## Claim Structure

The paper will make three claims, in this order:

1. Adaptive DB-LBT does not disturb the stable deterministic regime and does
   not change native Wi-Fi or NR-U procedures.
2. Per-scenario fixed-arm evaluation determines whether a common fixed tuple
   can serve all validated regimes. Adaptation is necessary only if confirmed
   scenario optima conflict and a best-global-fixed profile leaves a material
   worst-case gap.
3. In a known multi-phase run, online LinUCB is useful only if empirical
   `T_adapt` is below `T_dwell`. Otherwise the result is reported as a negative
   timescale boundary, not averaged into a positive claim.

`Distributed` denotes independent per-node deployment. The analysis supplies
only a conditional stationary-window LinUCB regret statement under slowly
changing neighbor policies; it makes no Nash, adversarial-regret, or strategic
game guarantee.

## Event Experiments

### Fixed-profile conflict discovery

Freeze the controller, action grid, reward, and event model. Screen all 24
legal profiles on a compact set of stationary and slow-turnover regimes using
three discovery seeds and 20,000 rounds. Rank arms per scenario, not per seed.
Advance a regime pair only when their best arms differ and each selected arm
has a reproducible margin over the other regime's arm.

Confirm the selected regimes with at least 30 untouched paired seeds and
100,000 rounds. Report per-scenario oracle, best-global-fixed, published TMC,
and adaptive LinUCB. If no fixed-profile conflict survives confirmation, the
paper must state that the experiment supports online retuning relative to TMC,
not context-dependent switching.

### Multi-phase and adaptation time

Construct a run with explicit change points and at least three repeated phase
transitions. Phase configurations come only from the independently selected
regime pair. Each phase lasts long enough to contain many 32-round decision
intervals.

For each change point, define `T_adapt` as the first post-change interval where
the rolling local reward recovers 90% of the gap from its immediate post-change
level to the final-quarter steady-state level and remains above that threshold
for three consecutive windows. Report median, P95, and censored transitions.
The learnability condition is checked directly as `P95(T_adapt) < T_dwell`.

### Statistical contract

- Discovery seeds and confirmation seeds are disjoint.
- Policy comparisons use paired exogenous randomness.
- Formal event comparisons use at least 30 seeds.
- Confidence intervals are paired bootstrap intervals over seed-level effects.
- Tail delay and worst-scenario utility accompany mean utility.
- Scenario selection, arm selection, and confirmation artifacts remain
  separate and hash-addressed.

## Packet-Level Diagnostic

The ns-3 schema and reducer will expose two distinct quantities:

- `packet_loss_ratio`: transmitted IP packets not received, including decoding
  and other end-to-end loss;
- `simultaneous_access_collision_rate`: transmissions overlapping another
  access, derived from the official simultaneous-transmission trace.

The paper will not compare event collision probability with packet loss ratio
as if they were the same metric. Packet runs will use at least 15 seeds, enable
log-normal shadowing, obtain independent channel realizations by seed, and use
a longer duration if a runtime benchmark makes this feasible. Because the
controller was initialized from event data, these runs are a diagnostic of the
A1--A3 boundary rather than evidence for packet-level generalization.

## Manuscript Revision

- Remove aggregate held-out gain as the headline result.
- Reframe the contribution as zero-protocol-cost adaptive robustness.
- Replace the training-seed fixed oracle with per-scenario oracle and
  best-global-fixed comparisons.
- State the A1--A3 indoor mutual-sensing scope in the abstract, introduction,
  evaluation, and conclusion.
- Replace the invalid cross-model collision vote with separately named packet
  metrics.
- Add empirical `T_adapt` and its comparison with `T_dwell`.
- Keep the main manuscript at exactly six pages; detailed matrices, seed lists,
  and extended figures remain in the GitHub artifact package.
- Retitle the paper if needed so `Distributed` is not presented as a theorem.

## Acceptance Criteria

1. Every numbered comment maps to a paper change or an explicitly reported
   scope limitation.
2. No adaptive-superiority sentence remains unless supported against the
   relevant per-scenario or best-global-fixed baseline.
3. Event results use at least 30 formal seeds and report paired intervals.
4. Packet loss and simultaneous-access collision are separate fields from the
   database through the paper table.
5. `T_adapt` is measured with a frozen definition and compared with dwell time.
6. The final PDF compiles without warnings that affect content, contains six
   pages, and passes page-image inspection for overlap and legibility.
