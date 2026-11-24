#!/usr/bin/env bash
set -euo pipefail

# Run all five OSWorld2 emergent-efficiency conditions on the same task slice.
#
# Shared runner args are passed to every mode. See README.md for dataset and
# Harbor setup. Typical real run:
#
#   export OSWORLD2_PATH=/path/to/OSWorld2
#   export OSWORLD2_TASK_FILE="$OSWORLD2_PATH/evaluation_examples/test_all.json"
#   export HARBOR_REPO=/path/to/harbor
#   export OSWORLD2_HARBOR_AGENT_IMPORT_PATH='agent_evolve.agents.osworld2.harbor_adapter:AEvolveOSWorld2HarborAgent'
#   export OSWORLD2_HARBOR_DELEGATE_AGENT_IMPORT_PATH='osworld.custom_agent:OSWorldAgent'
#   bash examples/osworld2_examples/emergent_efficiency/run_matrix.sh osw2_matrix \
#     --limit 20 --batch-size 2 --parallel 1
#
# Smoke matrix without Harbor:
#
#   bash examples/osworld2_examples/emergent_efficiency/run_matrix.sh osw2_smoke \
#     --task-file examples/osworld2_examples/emergent_efficiency/smoke_tasks.json \
#     --limit 2 --batch-size 1 --parallel 1 --dry-run-harbor --dry-run-proposer

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_name> [shared runner args...]" >&2
  exit 64
fi

RUN_NAME="$1"
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
  echo "  OSWorld2 Harbor emergent-efficiency: ${RUN_NAME} / ${MODE}"
  echo "============================================================"
  bash examples/osworld2_examples/emergent_efficiency/run_one.sh "${RUN_NAME}" "${MODE}" "$@"
done
