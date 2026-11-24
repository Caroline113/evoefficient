#!/usr/bin/env bash
set -euo pipefail

# Run one Terminal-Bench emergent-efficiency condition with Harbor evaluation
# and AdaptiveSkillEngine normal evolution.
#
# Usage:
#   bash examples/tb_examples/emergent_efficiency_adaptive/run_one_harbor.sh <RUN_NAME> <MODE> [flags]
#
# Modes:
#   static-control (no evolution)
#   normal-evolve
#   cost-aware
#   predefined-control
#   predefined-cost-aware
#   category-seeded-cost-aware

RUN_NAME="${1:?Usage: $0 <RUN_NAME> <MODE> [runner args...]}"
MODE="${2:?Usage: $0 <RUN_NAME> <MODE> [runner args...]}"
shift 2

export TB_HARBOR_AGENT_IMPORT_PATH="${TB_HARBOR_AGENT_IMPORT_PATH:-agent_evolve.agents.terminal.harbor_adapter:AEvolveTerminalHarborAgent}"
export TB_HARBOR_DELEGATE_AGENT_NAME="${TB_HARBOR_DELEGATE_AGENT_NAME:-codex}"

OUTPUT_DIR="logs/tb_emergent_efficiency_adaptive_harbor/${RUN_NAME}_${MODE}"

python examples/tb_examples/emergent_efficiency_adaptive/run_emergent_efficiency_tb_adaptive_harbor.py \
  --mode "${MODE}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"

