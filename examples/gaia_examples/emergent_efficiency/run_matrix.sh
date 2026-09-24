#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:?Usage: $0 <RUN_NAME> [shared runner flags]}"
shift

MODES=(
  normal-evolve
  cost-aware
  predefined-control
  predefined-cost-aware
  category-seeded-cost-aware
)

for MODE in "${MODES[@]}"; do
  echo "============================================================"
  echo "  GAIA validation Level 3: ${RUN_NAME} / ${MODE}"
  echo "============================================================"
  bash examples/gaia_examples/emergent_efficiency/run_one.sh "$RUN_NAME" "$MODE" "$@"
done
