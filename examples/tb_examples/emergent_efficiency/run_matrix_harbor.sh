#!/usr/bin/env bash
set -euo pipefail

# Run all five Terminal-Bench emergent-efficiency conditions with Harbor.
#
# Shared runner args are passed to every mode. See README.md for task download,
# Harbor install, and agent adapter setup.
#
# Smoke matrix:
#   bash examples/tb_examples/emergent_efficiency/run_matrix_harbor.sh tb_smoke \
#     --challenges-dir agent_evolve/benchmarks/tb21/tasks \
#     --tasks count-dataset-tokens \
#     --limit 1 --batch-size 1 --parallel 1 --dry-run-harbor --dry-run-proposer
#
# Real Harbor matrix:
#   export TB21_TASKS_DIR=/path/to/terminal-bench-2-1/tasks
#   export HARBOR_REPO=/path/to/harbor
#   export TB_HARBOR_AGENT_IMPORT_PATH='agent_evolve.agents.terminal.harbor_adapter:AEvolveTerminalHarborAgent'
#   export TB_HARBOR_DELEGATE_AGENT_NAME=codex
  # bash examples/tb_examples/emergent_efficiency/run_matrix_harbor.sh tb21_matrix \
  #   --shuffle --shuffle-seed 42 --limit 20 --batch-size 4 --parallel 2

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_name> [shared runner args...]" >&2
  exit 64
fi

RUN_NAME="$1"
shift

MODES=(
  normal-evolve
  cost-aware
  # predefined-control
  # predefined-cost-aware
  # category-seeded-cost-aware
)

for MODE in "${MODES[@]}"; do
  echo "============================================================"
  echo "  Terminal-Bench Harbor emergent-efficiency: ${RUN_NAME} / ${MODE}"
  echo "============================================================"
  bash examples/tb_examples/emergent_efficiency/run_one_harbor.sh "${RUN_NAME}" "${MODE}" "$@"
done
