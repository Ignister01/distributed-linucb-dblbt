#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPT_PATH="$(realpath -e -- "${BASH_SOURCE[0]}")"
ROOT="$(realpath -e -- "$SCRIPT_DIR/..")"
VALIDATION_LOCK="$ROOT/ns3/validation.env"
VALIDATION_CC="gcc-11"
VALIDATION_CXX="g++-11"
SOURCE_ROOT="${DBLBT_NS3_SOURCE_ROOT:-$ROOT/ns3/worktree/ns-3-dev}"
OUTPUT_ROOT="$ROOT/ns3/validation-results"
MODEL_PATH="$ROOT/ns3/models/linucb-initial.txt"
SCENARIO_PATH="$ROOT/ns3/scenarios/dblbt-nru-wifi-validation.cc"
PATCH_ROOT="$ROOT/ns3/patches"
VALIDATION_PYTHON="${DBLBT_VALIDATION_PYTHON:-$ROOT/.venv/bin/python}"

if [[ ! -f "$VALIDATION_LOCK" ]]; then
  echo "Missing ns-3 validation lock: $VALIDATION_LOCK" >&2
  return 1 2>/dev/null || exit 1
fi
# The tracked lock is restricted to scalar shell assignments.
source "$VALIDATION_LOCK"

CACHE_PARENT="${XDG_CACHE_HOME:-${HOME:?HOME must be set}/.cache}/dblbt-fcn"
RUNTIME_ROOT="${DBLBT_NS3_RUNTIME_ROOT:-$CACHE_PARENT/ns3-validation-$patch_bundle_sha256}"
RUNTIME_SOURCE="$RUNTIME_ROOT/source"
RUNTIME_BUILD="$RUNTIME_SOURCE/build"
RUNTIME_BINARY="$RUNTIME_BUILD/contrib/nr-u/examples/ns3.35-dblbt-nru-wifi-validation-optimized"

validation_compiler_identity() {
  "$VALIDATION_CXX" --version | sed -n '1p'
}

write_runtime_provenance() {
  printf 'patch_bundle_sha256=%s\n' "$patch_bundle_sha256"
  printf 'scenario_sha256=%s\n' "$scenario_sha256"
  printf 'compiler=%s\n' "$(validation_compiler_identity)"
  printf 'build_profile=%s\n' "$build_profile"
  printf 'ns3_commit=%s\n' "$ns3_commit"
  printf 'nr_commit=%s\n' "$nr_commit"
  printf 'nru_commit=%s\n' "$nru_commit"
}

runtime_provenance_matches() {
  if [[ "$#" -ne 1 || ! -f "$1" ]]; then
    return 1
  fi
  cmp -s -- "$1" <(write_runtime_provenance)
}

sha256_file() {
  sha256sum -- "$1" | cut -d' ' -f1
}

assert_descendant() {
  if [[ "$#" -ne 2 ]]; then
    echo "usage: assert_descendant PARENT CHILD" >&2
    return 2
  fi
  local parent child
  parent="$(realpath -m -- "$1")"
  child="$(realpath -m -- "$2")"
  if [[ "$child" != "$parent" && "$child" != "$parent/"* ]]; then
    echo "Path escapes declared root: $child is not below $parent" >&2
    return 1
  fi
}

validate_workers() {
  if [[ "$#" -ne 1 || ! "$1" =~ ^[1-8]$ ]]; then
    echo "DBLBT_NS3_WORKERS must be an integer in 1..8" >&2
    return 1
  fi
}

filesystem_type() {
  if [[ "$#" -ne 1 ]]; then
    echo "usage: filesystem_type PATH" >&2
    return 2
  fi
  local target
  target="$(realpath -m -- "$1")"
  while [[ ! -e "$target" && "$target" != "/" ]]; do
    target="$(dirname -- "$target")"
  done
  findmnt -T "$target" -n -o FSTYPE
}

require_ext4_path() {
  if [[ "$#" -ne 1 ]]; then
    echo "usage: require_ext4_path PATH" >&2
    return 2
  fi
  local target_type
  target_type="$(filesystem_type "$1")"
  if [[ "$target_type" != "ext4" ]]; then
    echo "ns-3 runtime requires ext4: $1 uses ${target_type:-unknown}" >&2
    return 1
  fi
}

require_runtime_filesystems() {
  require_ext4_path "$CACHE_PARENT"
  require_ext4_path "$RUNTIME_ROOT"
}

scratch_available_kib() {
  if [[ "$#" -ne 1 ]]; then
    echo "usage: scratch_available_kib PATH" >&2
    return 2
  fi
  df -Pk -- "$1" | awk 'NR == 2 { print $4; exit }'
}

select_scratch_root() {
  local workers="${DBLBT_NS3_WORKERS:-4}" required_kib available_kib candidate
  validate_workers "$workers"
  required_kib=$((workers * 64 * 1024))

  if [[ -n "${DBLBT_NS3_SCRATCH_ROOT:-}" ]]; then
    [[ "$DBLBT_NS3_SCRATCH_ROOT" == /* ]] || {
      echo "DBLBT_NS3_SCRATCH_ROOT must be absolute" >&2
      return 1
    }
    candidate="$(realpath -m -- "$DBLBT_NS3_SCRATCH_ROOT")"
    [[ "$candidate" != "/" ]] || {
      echo "DBLBT_NS3_SCRATCH_ROOT cannot be the filesystem root" >&2
      return 1
    }
    if ! mkdir -p -- "$candidate" || [[ ! -d "$candidate" || ! -w "$candidate" ]]; then
      echo "Scratch root is not writable: $candidate" >&2
      return 1
    fi
    available_kib="$(scratch_available_kib "$candidate")"
    if [[ ! "$available_kib" =~ ^[0-9]+$ ]] || (( available_kib < required_kib )); then
      echo "Insufficient scratch capacity: $candidate requires ${required_kib} KiB" >&2
      return 1
    fi
    printf '%s\n' "$candidate"
    return 0
  fi

  candidate="/dev/shm/dblbt-fcn/ns3-validation-$patch_bundle_sha256"
  if [[ -d /dev/shm && -w /dev/shm ]]; then
    available_kib="$(scratch_available_kib /dev/shm 2>/dev/null || printf '0\n')"
    if [[ "$available_kib" =~ ^[0-9]+$ ]] && (( available_kib >= required_kib )) &&
       mkdir -p -- "$candidate"
    then
      candidate="$(realpath -e -- "$candidate")"
      assert_descendant /dev/shm "$candidate"
      printf '%s\n' "$candidate"
      return 0
    fi
  fi

  candidate="$(realpath -m -- "$RUNTIME_ROOT/tmp")"
  mkdir -p -- "$candidate"
  available_kib="$(scratch_available_kib "$candidate")"
  if [[ ! "$available_kib" =~ ^[0-9]+$ ]] || (( available_kib < required_kib )); then
    echo "Insufficient scratch capacity: $candidate requires ${required_kib} KiB" >&2
    return 1
  fi
  assert_descendant "$RUNTIME_ROOT" "$candidate"
  printf '%s\n' "$candidate"
}

make_job_scratch() {
  if [[ "$#" -ne 2 || ! "$1" =~ ^[a-z0-9._-]+$ ]]; then
    echo "usage: make_job_scratch JOB_ID SCRATCH_ROOT" >&2
    return 2
  fi
  local scratch_root job_tmp
  scratch_root="$(realpath -e -- "$2")"
  job_tmp="$(mktemp -d "$scratch_root/$1.XXXXXX")"
  assert_descendant "$scratch_root" "$job_tmp"
  printf '%s\n' "$job_tmp"
}

wait_tracked_child() {
  if [[ "$#" -ne 1 ]]; then
    echo "usage: wait_tracked_child PID_ARRAY" >&2
    return 2
  fi
  local -n tracked_pids="$1"
  if (( ${#tracked_pids[@]} == 0 )); then
    echo "Cannot wait on an empty PID array" >&2
    return 2
  fi
  local completed_pid="" status=0 pid
  local -a remaining_pids=()
  if wait -n -p completed_pid "${tracked_pids[@]}" 2>/dev/null; then
    status=0
  else
    status=$?
  fi
  if [[ -z "${completed_pid:-}" ]]; then
    completed_pid="${tracked_pids[0]}"
    if wait "$completed_pid"; then
      status=0
    else
      status=$?
    fi
  fi
  for pid in "${tracked_pids[@]}"; do
    if [[ "$pid" != "$completed_pid" ]]; then
      remaining_pids+=("$pid")
    fi
  done
  tracked_pids=("${remaining_pids[@]}")
  return "$status"
}

verify_validation_sources() {
  if [[ "$#" -ne 2 ]]; then
    echo "usage: verify_validation_sources ROOT LOCK" >&2
    return 2
  fi
  local check_root lock_path source_root actual bundle
  check_root="$(realpath -e -- "$1")"
  lock_path="$(realpath -e -- "$2")"
  source_root="$(realpath -e -- "$SOURCE_ROOT")"
  (
    source "$lock_path"
    [[ "$(git -C "$source_root" rev-parse HEAD)" == "$ns3_commit" ]] || {
      echo "ns-3 commit mismatch" >&2; exit 1;
    }
    [[ "$(git -C "$source_root/contrib/nr" rev-parse HEAD)" == "$nr_commit" ]] || {
      echo "NR commit mismatch" >&2; exit 1;
    }
    [[ "$(git -C "$source_root/contrib/nr-u" rev-parse HEAD)" == "$nru_commit" ]] || {
      echo "NR-U commit mismatch" >&2; exit 1;
    }
    actual="$(sha256_file "$check_root/ns3/patches/0001-add-db-lbt-wifi-hook.patch")"
    [[ "$actual" == "$wifi_patch_sha256" ]] || { echo "Wi-Fi patch hash mismatch" >&2; exit 1; }
    actual="$(sha256_file "$check_root/ns3/patches/0002-fix-nr-dl-ul-symbol-overlap.patch")"
    [[ "$actual" == "$nr_patch_sha256" ]] || { echo "NR patch hash mismatch" >&2; exit 1; }
    actual="$(sha256_file "$check_root/ns3/patches/0003-add-db-lbt-nru-controller.patch")"
    [[ "$actual" == "$nru_patch_sha256" ]] || { echo "NR-U patch hash mismatch" >&2; exit 1; }
    bundle="$(cat \
      "$check_root/ns3/patches/0001-add-db-lbt-wifi-hook.patch" \
      "$check_root/ns3/patches/0002-fix-nr-dl-ul-symbol-overlap.patch" \
      "$check_root/ns3/patches/0003-add-db-lbt-nru-controller.patch" | sha256sum | cut -d' ' -f1)"
    [[ "$bundle" == "$patch_bundle_sha256" ]] || { echo "Patch bundle hash mismatch" >&2; exit 1; }
    actual="$(sha256_file "$check_root/ns3/scenarios/dblbt-nru-wifi-validation.cc")"
    [[ "$actual" == "$scenario_sha256" ]] || { echo "Scenario hash mismatch" >&2; exit 1; }
    actual="$(sha256_file "$check_root/ns3/models/linucb-initial.txt")"
    [[ "$actual" == "$model_export_sha256" ]] || { echo "Model export hash mismatch" >&2; exit 1; }
  )
}

validate_database_contract() {
  if [[ "$#" -ne 1 || ! -s "$1" ]]; then
    echo "Missing ns-3 validation database: ${1:-}" >&2
    return 1
  fi
  local database="$1" table family count
  for table in validation_metadata dblbt_nodes dblbt_attempts dblbt_decisions validation_metrics; do
    count="$(sqlite3 "$database" "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='$table';")"
    [[ "$count" == "1" ]] || { echo "Missing validation table: $table" >&2; return 1; }
  done
  for family in channel_occupancy_ simultaneous_tx_ mac_data_tx_failed_ sinr_results_ e2e_; do
    count="$(sqlite3 "$database" "SELECT count(*) FROM sqlite_master WHERE type='table' AND name GLOB '${family}*';")"
    [[ "$count" =~ ^[1-9][0-9]*$ ]] || { echo "Missing official table family: $family" >&2; return 1; }
  done
}

strict_validate_database() {
  if [[ "$#" -ne 2 ]]; then
    echo "usage: strict_validate_database DATABASE JOB_ID" >&2
    return 2
  fi
  PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$VALIDATION_PYTHON" - "$1" "$2" "$MODEL_PATH" \
    "$patch_bundle_sha256" "$scenario_sha256" "$build_profile" <<'PY'
import sys

from dblbt_fcn.ns3_validation import validate_ns3_job_database

database, job_id, model, patch_hash, scenario_hash, profile = sys.argv[1:]
validate_ns3_job_database(
    database,
    job_id=job_id,
    model_path=model,
    expected_patch_sha256=patch_hash,
    expected_scenario_sha256=scenario_hash,
    expected_build_profile=profile,
)
PY
}

remove_successful_job_scratch() {
  if [[ "$#" -ne 2 ]]; then
    echo "usage: remove_successful_job_scratch SCRATCH_ROOT JOB_DIRECTORY" >&2
    return 2
  fi
  local scratch_root job_directory job_name
  scratch_root="$(realpath -e -- "$1")"
  job_directory="$(realpath -e -- "$2")"
  job_name="$(basename -- "$job_directory")"
  assert_descendant "$scratch_root" "$job_directory"
  if [[ "$(dirname -- "$job_directory")" != "$scratch_root" ]] ||
     [[ ! "$job_name" =~ ^[a-z0-9._-]+\.[A-Za-z0-9]{6}$ ]]
  then
    echo "Refusing to remove non-job scratch path: $job_directory" >&2
    return 1
  fi
  rm -rf -- "$job_directory"
}

write_job_manifest() {
  if [[ "$#" -ne 2 ]]; then
    echo "usage: write_job_manifest OUTPUT_ROOT JOB_ID" >&2
    return 2
  fi
  local output job_id database manifest temporary database_hash database_bytes
  output="$(realpath -m -- "$1")"
  job_id="$2"
  [[ "$job_id" =~ ^[a-z0-9-]+(__seed-[0-9]+__[a-z]+)?$ ]] || {
    echo "Invalid ns-3 job id: $job_id" >&2; return 1;
  }
  database="$output/databases/$job_id.db"
  assert_descendant "$output" "$database"
  validate_database_contract "$database"
  strict_validate_database "$database" "$job_id"
  database_hash="$(sha256_file "$database")"
  database_bytes="$(stat -c %s -- "$database")"
  mkdir -p -- "$output/manifests"
  manifest="$output/manifests/$job_id.json"
  temporary="$output/manifests/.$job_id.json.tmp.$$"
  python3 - "$temporary" "$job_id" "$database_bytes" "$database_hash" \
    "$model_sha256" "$model_export_sha256" "$action_grid_hash" \
    "$patch_bundle_sha256" "$scenario_sha256" "$node_rate_bps" \
    "$traffic_mode" "$build_profile" <<'PY'
import json
import sys

(target, job_id, size, database_hash, model_hash, export_hash, grid_hash,
 patch_hash, scenario_hash, node_rate, traffic_mode, build_profile) = sys.argv[1:]
manifest = {
    "schema_version": 1,
    "status": "complete",
    "exit_code": 0,
    "job_id": job_id,
    "database": f"databases/{job_id}.db",
    "database_bytes": int(size),
    "database_sha256": database_hash,
    "model_sha256": model_hash,
    "model_export_sha256": export_hash,
    "action_grid_hash": grid_hash,
    "patch_sha256": patch_hash,
    "scenario_sha256": scenario_hash,
    "node_rate_bps": int(node_rate),
    "traffic_mode": traffic_mode,
    "srs_enabled": False,
    "build_profile": build_profile,
}
with open(target, "x", encoding="ascii", newline="\n") as stream:
    json.dump(manifest, stream, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
    stream.flush()
PY
  mv -fT -- "$temporary" "$manifest"
}

job_is_complete() {
  if [[ "$#" -ne 2 ]]; then
    echo "usage: job_is_complete OUTPUT_ROOT JOB_ID" >&2
    return 2
  fi
  local output job_id database manifest
  output="$(realpath -m -- "$1")"
  job_id="$2"
  database="$output/databases/$job_id.db"
  manifest="$output/manifests/$job_id.json"
  [[ -f "$manifest" ]] || return 1
  validate_database_contract "$database" >/dev/null 2>&1 || return 1
  python3 - "$manifest" "$database" "$job_id" "$model_sha256" \
    "$model_export_sha256" "$action_grid_hash" "$patch_bundle_sha256" \
    "$scenario_sha256" "$node_rate_bps" "$traffic_mode" "$build_profile" <<'PY'
import hashlib
import json
import os
import sys

(manifest_path, database, job_id, model_hash, export_hash, grid_hash,
 patch_hash, scenario_hash, node_rate, traffic_mode, build_profile) = sys.argv[1:]
try:
    with open(manifest_path, encoding="ascii") as stream:
        actual = json.load(stream)
    digest = hashlib.sha256()
    with open(database, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    expected = {
        "schema_version": 1,
        "status": "complete",
        "exit_code": 0,
        "job_id": job_id,
        "database": f"databases/{job_id}.db",
        "database_bytes": os.path.getsize(database),
        "database_sha256": digest.hexdigest(),
        "model_sha256": model_hash,
        "model_export_sha256": export_hash,
        "action_grid_hash": grid_hash,
        "patch_sha256": patch_hash,
        "scenario_sha256": scenario_hash,
        "node_rate_bps": int(node_rate),
        "traffic_mode": traffic_mode,
        "srs_enabled": False,
        "build_profile": build_profile,
    }
except (OSError, ValueError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if actual == expected else 1)
PY
  strict_validate_database "$database" "$job_id" >/dev/null 2>&1
}

write_smoke_marker() {
  if [[ "$#" -ne 2 || ( "$2" != "tmc" && "$2" != "adaptive" ) ]]; then
    echo "usage: write_smoke_marker SMOKE_ROOT tmc|adaptive" >&2
    return 2
  fi
  write_job_manifest "$1" "$2"
}

require_smoke_gate() {
  if [[ "$#" -ne 1 ]]; then
    echo "usage: require_smoke_gate SMOKE_ROOT" >&2
    return 2
  fi
  job_is_complete "$1" tmc && job_is_complete "$1" adaptive
}

ensure_python_environment() {
  if [[ ! -x "$VALIDATION_PYTHON" ]] ||
     ! "$VALIDATION_PYTHON" -c 'import dblbt_fcn, pytest' >/dev/null 2>&1
  then
    bash "$ROOT/scripts/bootstrap_linux.sh"
    VALIDATION_PYTHON="$ROOT/.venv/bin/python"
  fi
}

prepare_runtime() {
  require_runtime_filesystems
  ensure_python_environment
  verify_validation_sources "$ROOT" "$VALIDATION_LOCK"
  validate_workers "${DBLBT_NS3_WORKERS:-4}"
  command -v "$VALIDATION_CC" >/dev/null
  command -v "$VALIDATION_CXX" >/dev/null
  if [[ -x "$RUNTIME_BINARY" && -f "$RUNTIME_ROOT/runtime.env" ]] &&
     runtime_provenance_matches "$RUNTIME_ROOT/runtime.env"
  then
    echo "Reusing validated ns-3 runtime: $RUNTIME_ROOT"
    return 0
  fi
  if [[ -e "$RUNTIME_ROOT" ]]; then
    echo "Refusing unvalidated existing runtime: $RUNTIME_ROOT" >&2
    return 1
  fi
  mkdir -p -- "$CACHE_PARENT" "$OUTPUT_ROOT"
  local staging
  staging="$(mktemp -d "$CACHE_PARENT/.ns3-validation.XXXXXX")"
  mkdir -p -- "$staging/source/contrib/nr" "$staging/source/contrib/nr-u"
  git -C "$SOURCE_ROOT" archive "$ns3_commit" | tar -C "$staging/source" -xf -
  git -C "$SOURCE_ROOT/contrib/nr" archive "$nr_commit" | tar -C "$staging/source/contrib/nr" -xf -
  git -C "$SOURCE_ROOT/contrib/nr-u" archive "$nru_commit" | tar -C "$staging/source/contrib/nr-u" -xf -
  (cd -- "$staging/source" && git apply "$PATCH_ROOT/0001-add-db-lbt-wifi-hook.patch")
  (cd -- "$staging/source/contrib/nr" && git apply "$PATCH_ROOT/0002-fix-nr-dl-ul-symbol-overlap.patch")
  (cd -- "$staging/source/contrib/nr-u" && git apply "$PATCH_ROOT/0003-add-db-lbt-nru-controller.patch")
  [[ "$(sha256_file "$staging/source/contrib/nr-u/examples/dblbt-nru-wifi-validation.cc")" == "$scenario_sha256" ]]
  (
    cd -- "$staging/source"
    CC="$VALIDATION_CC" CXX="$VALIDATION_CXX" ./waf configure \
      --enable-examples --disable-python --build-profile=optimized
    ./waf build -j"${DBLBT_NS3_BUILD_JOBS:-8}"
  ) 2>&1 | tee "$OUTPUT_ROOT/build.log"
  [[ -x "$staging/source/build/contrib/nr-u/examples/ns3.35-dblbt-nru-wifi-validation-optimized" ]]
  write_runtime_provenance > "$staging/runtime.env"
  mv -fT -- "$staging" "$RUNTIME_ROOT"
}

run_binary() {
  LD_LIBRARY_PATH="$RUNTIME_BUILD/lib" "$RUNTIME_BINARY" "$@"
}

copy_database_atomic() {
  if [[ "$#" -ne 3 ]]; then
    echo "usage: copy_database_atomic SOURCE OUTPUT_ROOT JOB_ID" >&2
    return 2
  fi
  local source output job_id temporary destination
  source="$(realpath -e -- "$1")"
  output="$(realpath -m -- "$2")"
  job_id="$3"
  mkdir -p -- "$output/databases"
  destination="$output/databases/$job_id.db"
  temporary="$output/databases/.$job_id.db.tmp.$$"
  assert_descendant "$output" "$destination"
  cp -- "$source" "$temporary"
  validate_database_contract "$temporary"
  mv -fT -- "$temporary" "$destination"
}

validate_smoke_database() {
  if [[ "$#" -ne 2 ]]; then
    echo "usage: validate_smoke_database DATABASE POLICY" >&2
    return 2
  fi
  local database="$1" policy="$2" count
  validate_database_contract "$database"
  count="$(sqlite3 "$database" "SELECT count(*) FROM validation_metadata WHERE policy='$policy' AND alpha=11 AND context_dim=11 AND num_arms=24 AND node_rate_bps=$node_rate_bps AND traffic_mode='$traffic_mode' AND srs_enabled=0 AND model_sha256='$model_sha256' AND model_export_sha256='$model_export_sha256' AND patch_sha256='$patch_bundle_sha256' AND scenario_sha256='$scenario_sha256';")"
  [[ "$count" == "1" ]] || { echo "Smoke metadata contract failed: $policy" >&2; return 1; }
  count="$(sqlite3 "$database" "SELECT count(*)=2 AND count(DISTINCT state_id)=2 AND sum(technology='wifi')=1 AND sum(technology='nru')=1 FROM dblbt_nodes;")"
  [[ "$count" == "1" ]] || { echo "Smoke locality contract failed: $policy" >&2; return 1; }
  count="$(sqlite3 "$database" "SELECT count(DISTINCT node_id) FROM dblbt_attempts;")"
  [[ "$count" == "2" ]] || { echo "Smoke attempts missing: $policy" >&2; return 1; }
  if [[ "$policy" == "adaptive" ]]; then
    count="$(sqlite3 "$database" "SELECT count(DISTINCT node_id) FROM dblbt_decisions WHERE alpha=11 AND decision_round>=32 AND decision_round%32=0;")"
    [[ "$count" == "2" ]] || { echo "Adaptive smoke decisions missing" >&2; return 1; }
  fi
}

run_smoke_policy() {
  if [[ "$#" -ne 2 ]]; then
    echo "usage: run_smoke_policy SMOKE_ROOT POLICY" >&2
    return 2
  fi
  local smoke="$1" policy="$2" scratch_root job_tmp temp_db log_path
  if job_is_complete "$smoke" "$policy"; then
    echo "Skipping complete smoke: $policy"
    return 0
  fi
  mkdir -p -- "$smoke/logs" "$smoke/databases"
  scratch_root="$(select_scratch_root)"
  job_tmp="$(make_job_scratch "smoke-$policy" "$scratch_root")"
  temp_db="$job_tmp/$policy.db"
  log_path="$smoke/logs/$policy.log"
  run_binary \
    --policy="$policy" --scenario="smoke-$policy" --wifiAps=1 --nruGnbs=1 \
    --simTime=0.8 --appStart=0.2 --seed=410 --runId=1 --nodeRate=2Mbps \
    --modelPath="$MODEL_PATH" --modelSha256="$model_sha256" \
    --modelExportSha256="$model_export_sha256" --actionGridHash="$action_grid_hash" \
    --patchSha256="$patch_bundle_sha256" --scenarioSha256="$scenario_sha256" \
    --outputDb="$temp_db" 2>&1 | tee "$log_path"
  validate_smoke_database "$temp_db" "$policy"
  copy_database_atomic "$temp_db" "$smoke" "$policy"
  write_smoke_marker "$smoke" "$policy"
  remove_successful_job_scratch "$scratch_root" "$job_tmp"
}

run_smoke() {
  local smoke="$OUTPUT_ROOT/smoke"
  run_smoke_policy "$smoke" tmc
  run_smoke_policy "$smoke" adaptive
  require_smoke_gate "$smoke"
}

run_formal_job() {
  if [[ "$#" -ne 4 ]]; then
    echo "usage: run_formal_job SCENARIO SEED POLICY OUTPUT" >&2
    return 2
  fi
  local scenario="$1" seed="$2" policy="$3" output="$4"
  local job_id wifi_aps=4 nru_gnbs=4 interval=0 scratch_root job_tmp temp_db log_path
  job_id="${scenario}__seed-${seed}__${policy}"
  if job_is_complete "$output" "$job_id"; then
    echo "Skipping complete formal job: $job_id"
    return 0
  fi
  if [[ "$scenario" == "nonideal-6x6-300ms" ]]; then
    wifi_aps=6
    nru_gnbs=6
    interval=300
  fi
  mkdir -p -- "$output/logs" "$output/databases"
  scratch_root="$(select_scratch_root)"
  job_tmp="$(make_job_scratch "$job_id" "$scratch_root")"
  temp_db="$job_tmp/$job_id.db"
  log_path="$output/logs/${job_id}.log"
  run_binary \
    --formal=true --policy="$policy" --scenario="$scenario" \
    --wifiAps="$wifi_aps" --nruGnbs="$nru_gnbs" \
    --simTime=2.0 --appStart=0.2 --seed="$seed" --runId=1 --nodeRate=2Mbps \
    --interferenceIntervalMs="$interval" --interferenceDurationMs=2 \
    --modelPath="$MODEL_PATH" --modelSha256="$model_sha256" \
    --modelExportSha256="$model_export_sha256" --actionGridHash="$action_grid_hash" \
    --patchSha256="$patch_bundle_sha256" --scenarioSha256="$scenario_sha256" \
    --outputDb="$temp_db" 2>&1 | tee "$log_path"
  validate_database_contract "$temp_db"
  copy_database_atomic "$temp_db" "$output" "$job_id"
  write_job_manifest "$output" "$job_id"
  remove_successful_job_scratch "$scratch_root" "$job_tmp"
}

run_formal_matrix() {
  local output="$OUTPUT_ROOT/formal" workers="${DBLBT_NS3_WORKERS:-4}"
  local scenario seed policy failed=0
  local -a pids=()
  validate_workers "$workers"
  require_smoke_gate "$OUTPUT_ROOT/smoke"
  mkdir -p -- "$output"
  for scenario in static-4x4 dynamic-4x4 nonideal-6x6-300ms; do
    for seed in 410 523 631; do
      for policy in random tmc adaptive; do
        run_formal_job "$scenario" "$seed" "$policy" "$output" &
        pids+=("$!")
        if (( ${#pids[@]} >= workers )); then
          if ! wait_tracked_child pids; then failed=1; fi
        fi
      done
    done
  done
  while (( ${#pids[@]} > 0 )); do
    if ! wait_tracked_child pids; then failed=1; fi
  done
  [[ "$failed" == "0" ]]
}

run_audit() {
  ensure_python_environment
  local output="$OUTPUT_ROOT/formal" temporary="$OUTPUT_ROOT/.audit.json.tmp.$$"
  PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$VALIDATION_PYTHON" - "$output" "$MODEL_PATH" "$patch_bundle_sha256" \
    "$scenario_sha256" "$build_profile" "$temporary" <<'PY'
import json
import sys
from dblbt_fcn.ns3_validation import audit_ns3_validation

output, model, patch_hash, scenario_hash, profile, target = sys.argv[1:]
report = audit_ns3_validation(
    output,
    model_path=model,
    expected_patch_sha256=patch_hash,
    expected_scenario_sha256=scenario_hash,
    expected_build_profile=profile,
)
payload = {
    "schema_version": 1,
    "audited": report.audited,
    "adaptive_decisions": report.adaptive_decisions,
    "model_sha256": report.model_sha256,
    "model_export_sha256": report.model_export_sha256,
    "database_hashes": dict(report.database_hashes),
}
with open(target, "x", encoding="ascii", newline="\n") as stream:
    json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
PY
  mv -fT -- "$temporary" "$OUTPUT_ROOT/audit.json"
}

run_reduce() {
  ensure_python_environment
  local output="$OUTPUT_ROOT/formal"
  PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$VALIDATION_PYTHON" - "$output" "$MODEL_PATH" "$patch_bundle_sha256" \
    "$scenario_sha256" "$build_profile" "$OUTPUT_ROOT" <<'PY'
import sys

from dblbt_fcn.ns3_validation import (
    reduce_ns3_validation,
    write_ns3_reduction,
)

output, model, patch_hash, scenario_hash, profile, target = sys.argv[1:]
report = reduce_ns3_validation(
    output,
    model_path=model,
    expected_patch_sha256=patch_hash,
    expected_scenario_sha256=scenario_hash,
    expected_build_profile=profile,
)
write_ns3_reduction(report, target)
PY
}

main() {
  local stage="${1:-all}"
  if [[ "$#" -gt 1 ]]; then
    echo "usage: $SCRIPT_PATH [prepare|smoke|formal|audit|reduce|all]" >&2
    return 2
  fi
  assert_descendant "$ROOT" "$OUTPUT_ROOT"
  validate_workers "${DBLBT_NS3_WORKERS:-4}"
  case "$stage" in
    prepare) prepare_runtime ;;
    smoke) prepare_runtime; run_smoke ;;
    formal) prepare_runtime; run_formal_matrix ;;
    audit) run_audit ;;
    reduce) run_reduce ;;
    all)
      prepare_runtime
      run_smoke
      run_formal_matrix
      run_audit
      run_reduce
      ;;
    *) echo "Unknown ns-3 validation stage: $stage" >&2; return 2 ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
