# Final Experiment Audit

## Audited Scope

This audit covers the completed event-level formal experiment, the official
ns-3.35/5G-LENA/NR-U compatibility gate, the 27-job packet-level validation
matrix, and the 18-job event-level cross-model matrix used for H5.

| Evidence set | Complete items | Audit basis |
|---|---:|---|
| Event pretraining | 792 jobs | immutable config, raw gzip, marker, manifest |
| Event formal | 940 jobs | raw-backed summary, figures, tables, read-only audit |
| ns-3 formal | 27 jobs | SQLite schema, local state, decisions, hashes, manifest |
| H5 event match | 18 jobs | raw-backed summary and frozen model provenance |

No historical archive value is used as reportable output. The archive may be
consulted only to understand the original papers and earlier implementation
choices.

## Final Hypotheses

The canonical final table is `results/tables/final-hypotheses.csv`.

| Hypothesis | Status | Main evidence |
|---|---|---|
| H1 stable-regime safety | pass | utility difference `-7.00e-7`, 95% interval crosses zero but remains above the -2% threshold |
| H2 non-stationary gain | fail | improvement `0.002593`, far below the registered 10% threshold `0.089742` |
| H3 fairness preservation | pass | Jain difference `-3.919e-5`, above the `-0.01` limit |
| H4 held-out generalization | pass | utility difference `0.001556`, 95% interval `[0.001470, 0.001651]` |
| H5 cross-model consistency | inconclusive | agreement tuple `(None, False, None)` |

H2 is retained as a failure. H5 is not upgraded to pass: the event model is
tied in static and non-ideal validation, while dynamic improves at event level
but degrades under the packet-level majority rule. The exact scenario rows are
in `results/tables/cross-model-scenarios.csv`.

## Integrity Chain

The model used by both simulation layers has SHA-256
`70611e9712e3a8e4b0f35fc8e0e616f4fde9a20b17c844f1b6406da2833301e6`.
The action grid has SHA-256
`558da7340dfa32d8cc484ba68a05951314936d7aff34a145cc34ea051c07707c`.

The ns-3 audit binds every database hash to one manifest and records
`audited=27` and `adaptive_decisions=2067`. The reduction metadata binds the
72 seed-level metric pairs and 24 scenario-level metric rows to those audited
databases. The cross-model audit then binds the event summary, ns-3 scenario
metrics, original event hypothesis table, final H5 table, and scenario table.

| Artifact | SHA-256 |
|---|---|
| `ns3/validation-results/audit.json` | `522b3fc38a1a2b6ef10740c7784a4f30263c95a7e0f96b102f287e1565acfcde` |
| `ns3/validation-results/reduction.json` | `9bfc94857088d989ae1b54aaa5a7391c3fea202fdecb994653faa2929bf247d7` |
| `results/tables/ns3-cross-validation-event.csv` | `d5fe82004c4bc11f92be995b26d8bdef424fdea9ba690bc745fc4bde27140a54` |
| `results/tables/cross-model-scenarios.csv` | `cbfe0a5955682fd05e3ecb6ddd787a743705cb9fce28062d7fe1395f9100e43c` |
| `results/tables/final-hypotheses.csv` | `95e8edff7f889c4340bfb2c0db819df283fd2c82afacb6a2418cd9fd5804f6f6` |
| `results/tables/cross-model-audit.json` | `531e55ae74acd8022e4eefdf6901400757318067515578b15ea2428af44ac749` |

## Metric Boundary

Event-level results use contention-round effective airtime, channel-access
delay, collision probability, and Jain fairness. Packet-level ns-3 results use
application throughput, mean E2E delay, MAC collision evidence, and channel
occupancy. Values from one layer are never pooled numerically with the other.
H5 compares only scenario directions.

Channel occupancy is reported but excluded from the H5 better/worse vote. More
occupancy can mean useful utilization, monopolization, or overlapping failed
transmissions, so assigning it a universal positive direction would be unsafe.

## Reproducibility And Locality

The adaptive controller owns a separate 11-feature history and 24-arm LinUCB
state for every AP or gNB. It observes only its own attempts, busy time,
interruptions, access delay, queue state, arrivals, retries, and effective data
time. Global contender count, other-node state, global throughput, and Jain
fairness are excluded from controller input and reward.

The formal simulations and audits are local computations. A temporary network
interruption did not alter any result because all databases and event records
were written locally with completion markers or atomic manifests. Post-event
checks confirmed normal DNS and HTTPS connectivity and no orphaned simulation
process.

## Fresh Final Verification

- Python tests: `1215 passed, 1 skipped, 0 failed` in 248.95 seconds.
- Complete smoke script: tests passed, three valid jobs were retained, the
  three-row report was regenerated, and the audit returned `audited=3`.
- Event formal audit: `audited=940` in 1251.2 seconds.
- ns-3 smoke: both TMC and adaptive evidence accepted from the pinned runtime.
- ns-3 formal audit and reduction: both exited zero; the audit retained 27
  databases and 2067 adaptive decisions.
- H5 regeneration: `h5_status=inconclusive`; all four outputs were
  byte-identical to the retained final evidence.
- Frozen artifact manifests: every entry in
  `results/tables/final-artifacts.sha256` and
  `results/tables/task14-artifacts.sha256` verified `OK`.

An independent report regeneration produced 24 files. Eight PNG figures and
the six stable non-overhead CSV/LaTeX tables were byte-identical. The eight PDF
containers had different generation metadata, but rendering both versions
through Poppler produced byte-identical PNGs. The two overhead tables changed
because decision latency is intentionally measured at runtime: the frozen
median/P95 was `306.6315/349.4983 us`, while the fresh run measured
`297.774/344.0847 us`. Model size, model hash, action-grid hash, warmup count,
and measurement count remained identical. The formal frozen directory itself
still matches its original SHA manifest completely.

## Limitations For The Paper

- The compatible packet stack is the older ns-3.35, 5G-LENA v1.2.y, and pinned
  NR-U code. Results must not be described as validation on current upstream
  NR-U.
- The ns-3 matrix has three paired seeds and two-second runs. It validates
  direction and implementation behavior; the 940-job event matrix remains the
  broader statistical experiment.
- Event-level Wi-Fi ACK duration is zero in the frozen abstraction. Packet-level
  ns-3 includes protocol timing and therefore must remain a separate metric
  family.
- The dynamic mechanisms differ: event-level contender joins/lifetimes versus
  a half-run ns-3 load change.
- The H5 direction aggregation completes a detail under-specified in the initial
  design. It is conservative and yields `inconclusive`; the paper should label
  this as cross-model evidence rather than a clean confirmatory success.
