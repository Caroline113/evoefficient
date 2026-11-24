#!/usr/bin/env bash
set -euo pipefail
NAME="${1:?Usage: run_one.sh RUN_NAME MODE [runner flags...]}"
MODE="${2:?Usage: run_one.sh RUN_NAME MODE [runner flags...]}"
shift 2
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
PYTHON="${AEVOLVE_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
exec "${PYTHON}" "${SCRIPT_DIR}/run_emergent_efficiency_vwa.py" \
  --run-name "${NAME}" --mode "${MODE}" "$@"
