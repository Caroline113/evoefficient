#!/usr/bin/env bash
set -euo pipefail
NAME="${1:?Usage: run_matrix.sh RUN_NAME [runner flags...]}"
shift
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IFS=',' read -r -a MODES <<< "${VWA_MODES:-normal-evolve,cost-aware,predefined-control,predefined-cost-aware,category-seeded-cost-aware}"
for MODE in "${MODES[@]}"; do
  printf '\nRunning VisualWebArena mode: %s\n' "${MODE}"
  bash "${SCRIPT_DIR}/run_one.sh" "${NAME}" "${MODE}" "$@"
done
