#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLI="$ROOT/.venv/bin/dblbt-fcn"
WORKERS="${DBLBT_WORKERS:-8}"
MIN_FREE_KB=$((40 * 1024 * 1024))
PRETRAIN_ROOT="$ROOT/runs/pretrain"
FORMAL_ROOT="$ROOT/runs/formal"
MODEL="$ROOT/models/linucb-initial.npz"
ORACLE="$ROOT/models/fixed-oracle-arm.json"
HASH_FILE="$ROOT/models/event-formal-inputs.sha256"
SUMMARY="$ROOT/results/tables/per-seed.csv"
REPORT="$ROOT/results/figures/formal"

if ! [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || ((WORKERS > 24)); then
  echo "error: DBLBT_WORKERS must be an integer in 1..24" >&2
  exit 1
fi

FREE_KB="$(df -Pk "$ROOT" | awk 'NR == 2 {print $4}')"
if ((FREE_KB < MIN_FREE_KB)); then
  echo "error: formal run requires at least 40 GB free on the project drive" >&2
  exit 1
fi

if ! git -C "$ROOT" diff --quiet || ! git -C "$ROOT" diff --cached --quiet; then
  echo "error: tracked source changes must be committed before a formal run" >&2
  exit 1
fi
UNTRACKED_SOURCE="$(
  git -C "$ROOT" ls-files --others --exclude-standard -- \
    pyproject.toml src configs scripts tests
)"
if [[ -n "$UNTRACKED_SOURCE" ]]; then
  echo "error: untracked source files must be committed before a formal run" >&2
  exit 1
fi

mkdir -p "$ROOT/models" "$ROOT/results/tables" "$ROOT/results/figures"

bash "$ROOT/scripts/run_smoke.sh"

"$CLI" pretrain \
  --matrix "$ROOT/configs/matrices/pretrain.yaml" \
  --workers "$WORKERS" \
  --output-dir "$PRETRAIN_ROOT" \
  --output "$MODEL" \
  --oracle-output "$ORACLE"

HASH_TEMP="$HASH_FILE.partial"
rm -f "$HASH_TEMP"
(
  cd "$ROOT"
  sha256sum models/linucb-initial.npz models/fixed-oracle-arm.json
) >"$HASH_TEMP"
mv -f "$HASH_TEMP" "$HASH_FILE"

"$CLI" sweep \
  --matrix "$ROOT/configs/matrices/reproduction.yaml" \
  --workers "$WORKERS" \
  --output-dir "$FORMAL_ROOT" \
  --model "$MODEL"
"$CLI" sweep \
  --matrix "$ROOT/configs/matrices/heldout.yaml" \
  --workers "$WORKERS" \
  --output-dir "$FORMAL_ROOT" \
  --model "$MODEL" \
  --oracle-arm-file "$ORACLE"
"$CLI" sweep \
  --matrix "$ROOT/configs/matrices/ablation.yaml" \
  --workers "$WORKERS" \
  --output-dir "$FORMAL_ROOT" \
  --model "$MODEL"

"$CLI" summarize \
  --manifest-dir "$FORMAL_ROOT/manifests" \
  --output "$SUMMARY"
"$CLI" plot \
  --summary "$SUMMARY" \
  --output-dir "$REPORT" \
  --manifest-dir "$FORMAL_ROOT/manifests" \
  --model "$MODEL" \
  --oracle-arm-file "$ORACLE"
"$CLI" audit \
  --manifest-dir "$FORMAL_ROOT/manifests" \
  --summary "$SUMMARY" \
  --output-dir "$REPORT" \
  --model "$MODEL" \
  --oracle-arm-file "$ORACLE"

echo "formal event batch complete"
cat "$HASH_FILE"
