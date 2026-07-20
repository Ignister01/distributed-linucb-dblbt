# Official ns-3 NR-U Build Gate

This directory freezes the official compatibility stack used by Task 15. The
gate is prepared here but must be run later from WSL2 Ubuntu 24.04; it is not a
Windows-native workflow. The default and required wall-clock limit for the
complete dependency, checkout, build, example, and validation sequence is
four-hour.

## Run The Gate

From the repository root in WSL2:

```bash
bash scripts/run_ns3_gate.sh
```

The runner checks and installs missing Ubuntu build dependencies, clones or
reuses the repositories below `ns3/worktree/ns-3-dev`, verifies their exact
detached commits, enables the official examples and tests, and builds the
official `nr` and `nr-u` modules with ns-3.35.

The official `ns-3.35` ref is an annotated tag. The lock therefore records and
verifies both tag object `020c5f533253c98ee805b715d3efbd559a0ac7b4` and its
dereferenced release commit `ac88b75eac1818c673cf2c939a96ac3005b1f051`.
The gate installs and records GCC 11 explicitly because this 2021 source stack
predates the transitive standard-header changes in Ubuntu 24.04's GCC 13. The
official sources remain unmodified.

It then runs the supplied example with these declared parameters:

```text
cttc-nr-wifi-interference --simTime=0.7 --seed=410 --runId=1 --enableNr=true --enableWifi=true --wifiStandard=11ax
```

The gate does not modify or copy the official example.

## Retained Evidence

Each invocation creates a separate
`ns3/worktree/gate-runs/<UTC timestamp>-<process id>/` directory. Attempts do
not overwrite one another. The directory retains the complete `gate.log`, the
last atomic `stage`, exact commit and compiler metadata, the official example
database when produced, and `gate-status.env`.

`gate-status.env` records the final status and exit code, whether the
four-hour limit expired, exact ns-3/NR/NR-U commits, compiler identity, log
SHA-256, and output database SHA-256 on success. The output database must
contain the official SINR, failed MAC, channel occupancy, simultaneous
transmission (collision), and E2E table families.

The compact tracked snapshot for the accepted gate is
`ns3/gate-evidence/20260718T104919Z-gate-status.env`; the full build log and
official database remain in the ignored run directory named by that snapshot.

## Pass And Failure Policy

Passing requires both official modules to build, the supplied example to exit
zero, and the SQLite schema checks to pass. A failure or timeout preserves the
last stage and log hash, records `h5_status=not_evaluated`, and stops before
Task 16. A passing gate records `h5_status=pending_ns3_validation`; Task 15 by
itself does not evaluate H5.

The event-level simulator must not be substituted for the official ns-3 NR-U
stack. A failed gate remains a failed gate and cannot be reported as
packet-level validation.
