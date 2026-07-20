#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPT_PATH="$(realpath -e -- "${BASH_SOURCE[0]}")"
ROOT="$(realpath -e -- "$SCRIPT_DIR/..")"
VERSIONS_FILE="$ROOT/ns3/VERSIONS"
GATE_TIMEOUT="4h"
GATE_JOBS="8"
GATE_CC="gcc-11"
GATE_CXX="g++-11"
OFFICIAL_EXAMPLE_ARGS="cttc-nr-wifi-interference --simTime=0.7 --seed=410 --runId=1 --enableNr=true --enableWifi=true --wifiStandard=11ax"
DEPENDENCY_PACKAGES=(
  build-essential
  ca-certificates
  g++-11
  git
  python3
  sqlite3
  libsqlite3-dev
  libc6-dev
  pkg-config
)

write_atomic() {
  if [[ "$#" -ne 2 ]]; then
    echo "usage: write_atomic TARGET VALUE" >&2
    return 2
  fi

  local target="$1" value="$2" temporary
  temporary="${target}.tmp.$$"
  printf '%s\n' "$value" > "$temporary"
  mv -fT -- "$temporary" "$target"
}

set_stage() {
  if [[ "$#" -ne 2 ]]; then
    echo "usage: set_stage RUN_DIR STAGE" >&2
    return 2
  fi
  write_atomic "$1/stage" "$2"
}

write_metadata() {
  if [[ "$#" -ne 3 || ! "$2" =~ ^[a-z0-9_]+$ ]]; then
    echo "usage: write_metadata RUN_DIR KEY VALUE" >&2
    return 2
  fi
  mkdir -p -- "$1/metadata"
  write_atomic "$1/metadata/$2" "$3"
}

read_metadata() {
  if [[ "$#" -ne 2 ]]; then
    echo "usage: read_metadata RUN_DIR KEY" >&2
    return 2
  fi
  if [[ -f "$1/metadata/$2" ]]; then
    cat -- "$1/metadata/$2"
  fi
}

manifest_entry() {
  if [[ "$#" -ne 2 || ! "$1" =~ ^[a-z0-9_]+$ ]]; then
    echo "usage: manifest_entry KEY VALUE" >&2
    return 2
  fi
  printf '%s=' "$1"
  printf '%q' "$2"
  printf '\n'
}

finalize_run() {
  if [[ "$#" -ne 2 || ! "$2" =~ ^[0-9]+$ ]]; then
    echo "usage: finalize_run RUN_DIR EXIT_CODE" >&2
    return 2
  fi

  local run_dir="$1" effective_exit_code="$2"
  local stage="unknown" status="failed" timed_out="false"
  local h5_status="not_evaluated" log_path log_sha256=""
  local output_database output_database_sha256="" finished_at_utc temporary
  local ns3_ref ns3_commit nr_release nr_commit nru_commit
  local declared_ns3_tag_object declared_ns3_commit
  local declared_nr_commit declared_nru_commit
  local compiler_path compiler_version started_at_utc required value

  if [[ -f "$run_dir/stage" ]]; then
    stage="$(<"$run_dir/stage")"
  fi
  log_path="$run_dir/gate.log"
  if [[ "$effective_exit_code" == "124" ]]; then
    timed_out="true"
  fi
  if [[ ! -f "$log_path" ]]; then
    printf '%s\n' "Supervisor did not retain gate.log." > "$log_path"
    if [[ "$effective_exit_code" == "0" ]]; then
      effective_exit_code=70
    fi
  fi

  ns3_ref="$(read_metadata "$run_dir" ns3_ref)"
  declared_ns3_tag_object="$(read_metadata "$run_dir" declared_ns3_tag_object)"
  declared_ns3_commit="$(read_metadata "$run_dir" declared_ns3_commit)"
  declared_nr_commit="$(read_metadata "$run_dir" declared_nr_commit)"
  declared_nru_commit="$(read_metadata "$run_dir" declared_nru_commit)"
  ns3_commit="$(read_metadata "$run_dir" ns3_commit)"
  nr_release="$(read_metadata "$run_dir" nr_release)"
  nr_commit="$(read_metadata "$run_dir" nr_commit)"
  nru_commit="$(read_metadata "$run_dir" nru_commit)"
  compiler_path="$(read_metadata "$run_dir" compiler_path)"
  compiler_version="$(read_metadata "$run_dir" compiler_version)"
  output_database="$(read_metadata "$run_dir" output_database)"
  started_at_utc="$(read_metadata "$run_dir" started_at_utc)"

  if [[ "$effective_exit_code" == "0" ]]; then
    if [[ "$stage" != "complete" ]]; then
      echo "Worker exited zero before the complete stage." >> "$log_path"
      effective_exit_code=70
    fi
    for required in \
      ns3_ref ns3_commit nr_release nr_commit nru_commit \
      compiler_path compiler_version output_database
    do
      value="${!required}"
      if [[ -z "$value" ]]; then
        echo "Worker exited zero without required metadata: $required" >> "$log_path"
        effective_exit_code=70
      fi
    done
    if [[ -z "$output_database" || ! -s "$output_database" ]]; then
      echo "Worker exited zero without a retained output database." >> "$log_path"
      effective_exit_code=70
    fi
  fi

  if [[ "$effective_exit_code" == "0" ]]; then
    status="passed"
    h5_status="pending_ns3_validation"
    output_database_sha256="$(sha256sum -- "$output_database")"
    output_database_sha256="${output_database_sha256%% *}"
  fi
  if [[ -f "$log_path" ]]; then
    log_sha256="$(sha256sum -- "$log_path")"
    log_sha256="${log_sha256%% *}"
  fi
  finished_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  temporary="$run_dir/gate-status.env.tmp.$$"
  {
    manifest_entry schema_version 1
    manifest_entry status "$status"
    manifest_entry stage "$stage"
    manifest_entry exit_code "$effective_exit_code"
    manifest_entry timed_out "$timed_out"
    manifest_entry h5_status "$h5_status"
    manifest_entry started_at_utc "$started_at_utc"
    manifest_entry finished_at_utc "$finished_at_utc"
    manifest_entry ns3_ref "$ns3_ref"
    manifest_entry declared_ns3_tag_object "$declared_ns3_tag_object"
    manifest_entry declared_ns3_commit "$declared_ns3_commit"
    manifest_entry declared_nr_commit "$declared_nr_commit"
    manifest_entry declared_nru_commit "$declared_nru_commit"
    manifest_entry ns3_commit "$ns3_commit"
    manifest_entry nr_release "$nr_release"
    manifest_entry nr_commit "$nr_commit"
    manifest_entry nru_commit "$nru_commit"
    manifest_entry compiler_path "$compiler_path"
    manifest_entry compiler_version "$compiler_version"
    manifest_entry log_path "$log_path"
    manifest_entry log_sha256 "$log_sha256"
    manifest_entry output_database "$output_database"
    manifest_entry output_database_sha256 "$output_database_sha256"
  } > "$temporary"
  mv -fT -- "$temporary" "$run_dir/gate-status.env"
  write_metadata "$run_dir" final_exit_code "$effective_exit_code"
}

run_supervisor() {
  if [[ "$#" -lt 3 ]]; then
    echo "usage: run_supervisor RUN_DIR TIMEOUT COMMAND [ARG ...]" >&2
    return 2
  fi

  local run_dir="$1" gate_timeout="$2" worker_exit final_exit
  local restore_errexit=0
  shift 2
  mkdir -p -- "$(dirname -- "$run_dir")"
  if ! mkdir -- "$run_dir"; then
    echo "Refusing to reuse gate run directory: $run_dir" >&2
    return 1
  fi
  mkdir -- "$run_dir/metadata"
  set_stage "$run_dir" initializing
  write_metadata "$run_dir" started_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  if [[ "$-" == *e* ]]; then
    restore_errexit=1
  fi
  set +e
  timeout --kill-after=30s "$gate_timeout" "$@" 2>&1 | tee "$run_dir/gate.log"
  worker_exit="${PIPESTATUS[0]}"
  if [[ "$restore_errexit" -eq 1 ]]; then
    set -e
  fi

  finalize_run "$run_dir" "$worker_exit"
  final_exit="$(read_metadata "$run_dir" final_exit_code)"
  return "$final_exit"
}

load_versions() {
  if [[ ! -f "$VERSIONS_FILE" ]]; then
    echo "Missing ns-3 version lock: $VERSIONS_FILE" >&2
    return 1
  fi

  # The repository-owned lock contains simple shell assignments.
  source "$VERSIONS_FILE"
  local name
  for name in \
    ns3_repo ns3_ref ns3_tag_object ns3_commit nr_repo nr_release nr_ref \
    nru_repo nru_ref official_example
  do
    if [[ -z "${!name:-}" ]]; then
      echo "Missing required version lock value: $name" >&2
      return 1
    fi
  done
  if [[ "$official_example" != "cttc-nr-wifi-interference" ]]; then
    echo "Unexpected official example in version lock: $official_example" >&2
    return 1
  fi
}

record_declared_versions() {
  if [[ "$#" -ne 1 ]]; then
    echo "usage: record_declared_versions RUN_DIR" >&2
    return 2
  fi

  local run_dir="$1"
  load_versions
  write_metadata "$run_dir" ns3_ref "$ns3_ref"
  write_metadata "$run_dir" nr_release "$nr_release"
  write_metadata "$run_dir" declared_ns3_tag_object "$ns3_tag_object"
  write_metadata "$run_dir" declared_ns3_commit "$ns3_commit"
  write_metadata "$run_dir" declared_nr_commit "$nr_ref"
  write_metadata "$run_dir" declared_nru_commit "$nru_ref"
}

ensure_dependencies() {
  local package
  local -a missing=()

  for package in "${DEPENDENCY_PACKAGES[@]}"; do
    if ! dpkg-query -W -f='${Status}\n' "$package" 2>/dev/null |
      grep -qx 'install ok installed'
    then
      missing+=("$package")
    fi
  done
  if [[ "${#missing[@]}" -eq 0 ]]; then
    echo "All ns-3 gate dependencies are installed."
    return 0
  fi

  echo "Installing missing ns-3 gate dependencies: ${missing[*]}"
  if [[ "$EUID" -eq 0 ]]; then
    DEBIAN_FRONTEND=noninteractive apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y -- "${missing[@]}"
  else
    if ! command -v sudo >/dev/null; then
      echo "sudo is required to install missing dependencies: ${missing[*]}" >&2
      return 1
    fi
    sudo env DEBIAN_FRONTEND=noninteractive apt-get update
    sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y -- "${missing[@]}"
  fi
}

assert_built_modules() {
  if [[ "$#" -ne 1 ]]; then
    echo "usage: assert_built_modules LIBRARY_DIR" >&2
    return 2
  fi

  local library_dir="$1" library base
  local nr_found=0 nru_found=0
  if [[ ! -d "$library_dir" ]]; then
    echo "Missing ns-3 build library directory: $library_dir" >&2
    return 1
  fi

  while IFS= read -r -d '' library; do
    base="$(basename -- "$library")"
    case "$base" in
      libns3*-nr-u-*.so|libns3*-nr-u-*.a)
        nru_found=1
        ;;
      libns3*-nr-*.so|libns3*-nr-*.a)
        nr_found=1
        ;;
    esac
  done < <(find "$library_dir" -maxdepth 1 -type f -print0)

  if [[ "$nr_found" -ne 1 ]]; then
    echo "Missing built ns-3 nr module in: $library_dir" >&2
    return 1
  fi
  if [[ "$nru_found" -ne 1 ]]; then
    echo "Missing built ns-3 nr-u module in: $library_dir" >&2
    return 1
  fi
}

gate_worker() {
  if [[ "$#" -ne 1 ]]; then
    echo "usage: gate_worker RUN_DIR" >&2
    return 2
  fi

  local run_dir="$1" source_root example_dir database
  local -a databases=()
  trap 'echo "ns-3 gate worker failed at stage $(<"$run_dir/stage")" >&2' ERR

  record_declared_versions "$run_dir"

  set_stage "$run_dir" dependencies
  ensure_dependencies
  write_metadata "$run_dir" compiler_path "$(command -v "$GATE_CXX")"
  write_metadata "$run_dir" compiler_version \
    "$("$GATE_CXX" --version | sed -n '1p')"

  source_root="$ROOT/ns3/worktree/ns-3-dev"
  mkdir -p -- "$ROOT/ns3/worktree"
  set_stage "$run_dir" checkout_ns3
  ensure_repo ns3 "$ns3_repo" "$source_root" "$ns3_ref" "$ns3_tag_object" "$ns3_commit"
  write_metadata "$run_dir" ns3_commit "$(git -C "$source_root" rev-parse HEAD)"
  set_stage "$run_dir" checkout_nr
  ensure_repo nr "$nr_repo" "$source_root/contrib/nr" "$nr_release" "$nr_ref" "$nr_ref"
  write_metadata "$run_dir" nr_commit \
    "$(git -C "$source_root/contrib/nr" rev-parse HEAD)"
  set_stage "$run_dir" checkout_nru
  ensure_repo nr-u "$nru_repo" "$source_root/contrib/nr-u" "$nru_ref" "$nru_ref" "$nru_ref"
  write_metadata "$run_dir" nru_commit \
    "$(git -C "$source_root/contrib/nr-u" rev-parse HEAD)"

  set_stage "$run_dir" configure
  (
    cd -- "$source_root"
    if [[ -d build ]]; then
      ./waf clean
    fi
    CC="$GATE_CC" CXX="$GATE_CXX" ./waf configure --enable-examples --enable-tests
  )

  set_stage "$run_dir" build
  (
    cd -- "$source_root"
    ./waf -j"$GATE_JOBS" build
  )
  assert_built_modules "$source_root/build/lib"

  example_dir="$run_dir/example"
  mkdir -- "$example_dir"
  set_stage "$run_dir" example
  (
    cd -- "$source_root"
    ./waf --cwd="$example_dir" --run "$OFFICIAL_EXAMPLE_ARGS"
  )

  mapfile -d '' databases < <(
    find "$example_dir" -maxdepth 1 -type f -name '*.db' -print0
  )
  if [[ "${#databases[@]}" -ne 1 ]]; then
    echo "Expected one official example database, found ${#databases[@]}." >&2
    return 1
  fi
  database="$(realpath -e -- "${databases[0]}")"
  write_metadata "$run_dir" output_database "$database"
  set_stage "$run_dir" validate_database
  validate_output_database "$database"
  set_stage "$run_dir" complete
  trap - ERR
}

main() {
  if [[ "${1:-}" == "--worker" ]]; then
    if [[ "$#" -ne 2 ]]; then
      echo "usage: $0 --worker RUN_DIR" >&2
      return 2
    fi
    gate_worker "$2"
    return
  fi
  if [[ "$#" -ne 0 ]]; then
    echo "usage: $0" >&2
    return 2
  fi

  local run_parent="$ROOT/ns3/worktree/gate-runs" run_dir exit_code
  run_dir="$run_parent/$(date -u +%Y%m%dT%H%M%SZ)-$$"
  echo "ns-3 gate attempt: $run_dir"
  set +e
  run_supervisor \
    "$run_dir" "$GATE_TIMEOUT" bash "$SCRIPT_PATH" --worker "$run_dir"
  exit_code="$?"
  set -e
  echo "ns-3 gate status: $run_dir/gate-status.env"
  return "$exit_code"
}

ensure_repo() {
  if [[ "$#" -ne 6 ]]; then
    echo "usage: ensure_repo LABEL ORIGIN DESTINATION REF REF_OBJECT COMMIT" >&2
    return 2
  fi

  local label="$1" origin="$2" destination="$3" ref="$4"
  local ref_object="$5" commit="$6"
  local actual_origin resolved_object resolved
  if [[ ! "$ref_object" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Invalid pinned ref object for $label: $ref_object" >&2
    return 1
  fi
  if [[ ! "$commit" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Invalid pinned commit for $label: $commit" >&2
    return 1
  fi

  if [[ ! -e "$destination" ]]; then
    git clone --no-checkout -- "$origin" "$destination"
  fi
  if ! git -C "$destination" rev-parse --git-dir >/dev/null 2>&1; then
    echo "Existing $label path is not a Git repository: $destination" >&2
    return 1
  fi

  actual_origin="$(git -C "$destination" remote get-url origin)"
  if [[ "$actual_origin" != "$origin" ]]; then
    echo "$label origin mismatch: expected $origin, found $actual_origin" >&2
    return 1
  fi
  if [[ "$ref" == "$commit" ]]; then
    if ! git -C "$destination" cat-file -e "${commit}^{commit}" 2>/dev/null; then
      git -C "$destination" fetch --force origin "$commit"
    fi
    resolved_object="$(git -C "$destination" rev-parse "$commit")"
    resolved="$(git -C "$destination" rev-parse "${commit}^{commit}")"
  else
    git -C "$destination" fetch --force origin "$ref"
    resolved_object="$(git -C "$destination" rev-parse FETCH_HEAD)"
    resolved="$(git -C "$destination" rev-parse 'FETCH_HEAD^{commit}')"
  fi
  if [[ "$resolved_object" != "$ref_object" ]]; then
    echo "$label ref $ref object resolved to $resolved_object instead of $ref_object" >&2
    return 1
  fi
  if [[ "$resolved" != "$commit" ]]; then
    echo "$label ref $ref resolved to $resolved instead of $commit" >&2
    return 1
  fi
  git -C "$destination" checkout --detach "$resolved"
  if [[ -n "$(git -C "$destination" status --porcelain --untracked-files=no)" ]]; then
    echo "Refusing tracked local changes in $label checkout: $destination" >&2
    return 1
  fi
  if [[ "$(git -C "$destination" rev-parse HEAD)" != "$commit" ]]; then
    echo "$label did not resolve to pinned commit $commit" >&2
    return 1
  fi
}

validate_output_database() {
  if [[ "$#" -ne 1 ]]; then
    echo "usage: validate_output_database DATABASE" >&2
    return 2
  fi

  local database="$1" family count
  if [[ ! -s "$database" ]]; then
    echo "Official example database is missing or empty: $database" >&2
    return 1
  fi

  for family in \
    sinr_results_ \
    mac_data_tx_failed_ \
    channel_occupancy_ \
    simultaneous_tx_ \
    e2e_
  do
    count="$(
      sqlite3 "$database" \
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name GLOB '${family}*';"
    )"
    if [[ ! "$count" =~ ^[1-9][0-9]*$ ]]; then
      echo "Missing official SQLite table family: $family" >&2
      return 1
    fi
  done
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
