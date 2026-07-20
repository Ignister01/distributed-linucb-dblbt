#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/bootstrap_safety.sh"

ROOT="$(realpath -e -- "$SCRIPT_DIR/..")"
PROJECT_VENV="$ROOT/.venv"
IS_WSL=0
if grep -qiE '(microsoft|wsl)' "/proc/sys/kernel/osrelease"; then
  IS_WSL=1
fi

if [[ "$IS_WSL" -eq 1 ]]; then
  PROJECT_KEY="$(dblbt_project_key "$ROOT")"
  VENV="$(realpath -m -- "${XDG_CACHE_HOME:-${HOME:?HOME must be set}/.cache}/dblbt-fcn/$PROJECT_KEY/venv")"
  dblbt_validate_venv_path "$ROOT" "${HOME:?HOME must be set}" "$VENV"
  dblbt_assert_native_filesystem "$VENV"
  dblbt_assert_owned_or_empty "$ROOT" "$VENV"

  if [[ -L "$PROJECT_VENV" ]]; then
    EXISTING_TARGET="$(realpath -m -- "$PROJECT_VENV")"
    if [[ "$EXISTING_TARGET" != "$VENV" ]]; then
      echo "Refusing to replace $PROJECT_VENV; it points to $EXISTING_TARGET instead of $VENV." >&2
      exit 1
    fi
  elif [[ -e "$PROJECT_VENV" ]]; then
    echo "Refusing to replace non-symlink project environment: $PROJECT_VENV" >&2
    exit 1
  fi
else
  dblbt_assert_native_project_venv "$PROJECT_VENV"
  VENV="$PROJECT_VENV"
  dblbt_validate_venv_path "$ROOT" "${HOME:?HOME must be set}" "$VENV"
  dblbt_assert_owned_or_empty "$ROOT" "$VENV"
fi

PYTHON="$VENV/bin/python"
LOCK="$ROOT/environment/requirements.lock"
LIVE_FREEZE="$(mktemp)"
trap 'rm -f -- "$LIVE_FREEZE"' EXIT

sudo apt-get update
sudo apt-get install -y build-essential cmake git ninja-build pkg-config \
  python3.12 python3.12-venv sqlite3 libsqlite3-dev libc6-dev

mkdir -p -- "$(dirname -- "$VENV")"
python3.12 -m venv --clear "$VENV"
dblbt_mark_owner "$ROOT" "$VENV"

"$PYTHON" -m pip install --requirement "$LOCK"
"$PYTHON" -m pip install --no-build-isolation --no-deps --editable "$ROOT[dev]"
"$PYTHON" -m pip check
"$PYTHON" -m pip freeze --all --exclude-editable | LC_ALL=C sort > "$LIVE_FREEZE"
if ! cmp -s -- "$LOCK" "$LIVE_FREEZE"; then
  echo "Installed environment does not match $LOCK:" >&2
  diff -u -- "$LOCK" "$LIVE_FREEZE" || true
  exit 1
fi
"$PYTHON" -c "import dblbt_fcn, dblbt_fcn.cli, hypothesis, matplotlib, numpy, pandas, pydantic, pytest, pytest_cov, scipy, typer, yaml; print('python gate ok')"
"$VENV/bin/dblbt-fcn" --help

if [[ "$IS_WSL" -eq 1 ]]; then
  if [[ -L "$PROJECT_VENV" ]]; then
    EXISTING_TARGET="$(realpath -m -- "$PROJECT_VENV")"
    if [[ "$EXISTING_TARGET" != "$VENV" ]]; then
      echo "Refusing to replace $PROJECT_VENV; it now points to $EXISTING_TARGET instead of $VENV." >&2
      exit 1
    fi
    rm -- "$PROJECT_VENV"
  elif [[ -e "$PROJECT_VENV" ]]; then
    echo "Refusing to replace non-symlink project environment: $PROJECT_VENV" >&2
    exit 1
  fi
  ln -s -- "$VENV" "$PROJECT_VENV"
fi
