#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
CLI="$ROOT/.venv/bin/dblbt-fcn"
RUN_ROOT="$ROOT/runs/smoke"
SUMMARY="$ROOT/results/tables/smoke.csv"
REPORT="$ROOT/results/figures/smoke"

"$PYTHON" -m pytest -q
"$CLI" sweep \
  --matrix "$ROOT/configs/matrices/smoke.yaml" \
  --workers 3 \
  --output-dir "$RUN_ROOT"
"$CLI" summarize \
  --manifest-dir "$RUN_ROOT/manifests" \
  --output "$SUMMARY"
"$CLI" plot \
  --summary "$SUMMARY" \
  --output-dir "$REPORT" \
  --manifest-dir "$RUN_ROOT/manifests"
"$CLI" audit \
  --manifest-dir "$RUN_ROOT/manifests" \
  --summary "$SUMMARY" \
  --output-dir "$REPORT"
