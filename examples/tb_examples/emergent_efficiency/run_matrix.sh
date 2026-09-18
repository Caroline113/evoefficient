#!/usr/bin/env bash
set -euo pipefail

# Run all Terminal-Bench emergent-efficiency conditions on the same task slice.
#
# Usage:
#   bash examples/tb_examples/emergent_efficiency/run_matrix.sh <RUN_NAME> [flags]
#
# Example:
#   bash examples/tb_examples/emergent_efficiency/run_matrix.sh tb20 \
#     --limit 20 --batch-size 5 --parallel 4

RUN_NAME="${1:?Usage: $0 <RUN_NAME> [shared runner flags]}"
shift

MODES=(
  # normal-evolve
  cost-aware
  predefined-control
  # predefined-cost-aware
  category-seeded-cost-aware
)

for MODE in "${MODES[@]}"; do
  echo "============================================================"
  echo "  Terminal-Bench emergent-efficiency: ${RUN_NAME} / ${MODE}"
  echo "============================================================"
  bash examples/tb_examples/emergent_efficiency/run_one.sh "$RUN_NAME" "$MODE" "$@"
done
