#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != Linux ]]; then
    printf '%s\n' 'This launcher is for Linux.' >&2
    exit 1
fi

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
env_dir="${CELLMATCHR_LINUX_ENV:-$repo_dir/../CellMatchR-linux-env}"
if [[ ! -x "$env_dir/bin/python" ]]; then
    printf 'CellMatchR Linux Python was not found in %s\n' "$env_dir" >&2
    exit 1
fi
env_dir="$(cd -- "$env_dir" && pwd)"

export QT_API=pyqt5
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$env_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MICROSAM_CACHEDIR="${MICROSAM_CACHEDIR:-$env_dir/model-cache/micro-sam}"
export CELLPOSE_LOCAL_MODELS_PATH="${CELLPOSE_LOCAL_MODELS_PATH:-$env_dir/model-cache/cellpose}"

cd -- "$repo_dir"
exec "$env_dir/bin/python" "$repo_dir/scripts/run_cellmatchr.py" "$@"
