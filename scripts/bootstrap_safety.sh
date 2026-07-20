#!/usr/bin/env bash

# This file is sourced by bootstrap_linux.sh and intentionally has no top-level effects.

dblbt_project_key() {
  if [[ "$#" -ne 1 ]]; then
    echo "usage: dblbt_project_key ROOT" >&2
    return 2
  fi

  local root project_name digest
  if ! root="$(realpath -e -- "$1")"; then
    echo "Cannot canonicalize project root: $1" >&2
    return 1
  fi
  project_name="$(basename -- "$root")"
  if ! digest="$(printf '%s' "$root" | sha256sum)"; then
    echo "Cannot hash project root: $root" >&2
    return 1
  fi
  digest="${digest%% *}"
  printf '%s-%s\n' "$project_name" "${digest:0:12}"
}

dblbt_validate_venv_path() {
  if [[ "$#" -ne 3 ]]; then
    echo "usage: dblbt_validate_venv_path ROOT HOME VENV" >&2
    return 2
  fi

  local root home venv
  if ! root="$(realpath -e -- "$1")"; then
    echo "Cannot canonicalize project root: $1" >&2
    return 1
  fi
  if ! home="$(realpath -m -- "$2")"; then
    echo "Cannot canonicalize home directory: $2" >&2
    return 1
  fi
  if ! venv="$(realpath -m -- "$3")"; then
    echo "Cannot canonicalize virtual environment path: $3" >&2
    return 1
  fi

  if [[ "$venv" == "/" || "$venv" == "$root" || "$venv" == "$home" ||
        "$root" == "${venv}/"* || "$home" == "${venv}/"* ]]; then
    echo "Refusing unsafe virtual environment path: $venv" >&2
    return 1
  fi
}

dblbt_assert_owned_or_empty() {
  if [[ "$#" -ne 2 ]]; then
    echo "usage: dblbt_assert_owned_or_empty ROOT VENV" >&2
    return 2
  fi

  local root venv first_entry marker
  if ! root="$(realpath -e -- "$1")"; then
    echo "Cannot canonicalize project root: $1" >&2
    return 1
  fi
  if ! venv="$(realpath -m -- "$2")"; then
    echo "Cannot canonicalize virtual environment path: $2" >&2
    return 1
  fi

  if [[ ! -e "$venv" && ! -L "$venv" ]]; then
    return 0
  fi
  if [[ ! -d "$venv" ]]; then
    echo "Refusing non-directory virtual environment target: $venv" >&2
    return 1
  fi
  if ! first_entry="$(find "$venv" -mindepth 1 -maxdepth 1 -print -quit)"; then
    echo "Cannot inspect virtual environment target: $venv" >&2
    return 1
  fi
  if [[ -z "$first_entry" ]]; then
    return 0
  fi

  marker="$venv/.dblbt-fcn-owner"
  if [[ ! -f "$marker" || -L "$marker" ]] ||
     ! printf '%s\n' "$root" | cmp -s - "$marker"; then
    echo "Refusing nonempty virtual environment not owned by $root: $venv" >&2
    return 1
  fi
}

dblbt_mark_owner() {
  if [[ "$#" -ne 2 ]]; then
    echo "usage: dblbt_mark_owner ROOT VENV" >&2
    return 2
  fi

  local root venv marker temporary_marker
  if ! root="$(realpath -e -- "$1")"; then
    echo "Cannot canonicalize project root: $1" >&2
    return 1
  fi
  if ! venv="$(realpath -e -- "$2")" || [[ ! -d "$venv" ]]; then
    echo "Cannot mark missing virtual environment directory: $2" >&2
    return 1
  fi

  marker="$venv/.dblbt-fcn-owner"
  if ! temporary_marker="$(mktemp --tmpdir="$venv" .dblbt-fcn-owner.tmp.XXXXXX)"; then
    echo "Cannot create ownership marker in: $venv" >&2
    return 1
  fi
  if ! printf '%s\n' "$root" > "$temporary_marker"; then
    rm -f -- "$temporary_marker"
    echo "Cannot write ownership marker in: $venv" >&2
    return 1
  fi
  if ! mv -fT -- "$temporary_marker" "$marker"; then
    rm -f -- "$temporary_marker"
    echo "Cannot install ownership marker in: $venv" >&2
    return 1
  fi
}

dblbt_assert_native_filesystem() {
  if [[ "$#" -ne 1 ]]; then
    echo "usage: dblbt_assert_native_filesystem VENV" >&2
    return 2
  fi

  local venv probe parent filesystem_type
  if ! venv="$(realpath -m -- "$1")"; then
    echo "Cannot canonicalize virtual environment path: $1" >&2
    return 1
  fi
  probe="$venv"
  while [[ ! -e "$probe" && ! -L "$probe" ]]; do
    parent="$(dirname -- "$probe")"
    if [[ "$parent" == "$probe" ]]; then
      echo "Cannot find an existing ancestor for: $venv" >&2
      return 1
    fi
    probe="$parent"
  done
  if ! filesystem_type="$(stat -f -c '%T' -- "$probe")"; then
    echo "Cannot inspect filesystem for: $probe" >&2
    return 1
  fi

  case "${filesystem_type,,}" in
    ext2/ext3|ext4|xfs|btrfs)
      return 0
      ;;
    *)
      echo "Virtual environment must be on a native Linux filesystem, not $filesystem_type: $venv" >&2
      return 1
      ;;
  esac
}

dblbt_assert_native_project_venv() {
  if [[ "$#" -ne 1 ]]; then
    echo "usage: dblbt_assert_native_project_venv PROJECT_VENV" >&2
    return 2
  fi

  if [[ -L "$1" ]]; then
    echo "Refusing project virtual environment symlink on native Linux: $1" >&2
    return 1
  fi
}
