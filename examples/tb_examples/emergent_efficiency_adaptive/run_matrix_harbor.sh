#!/usr/bin/env bash
set -euo pipefail
RUN_NAME="${1:?Usage: $0 RUN_NAME [shared runner flags]}"
shift
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"
if [[ ! "$RUN_NAME" =~ ^[a-zA-Z0-9_.-]+$ ]]; then
  echo 'RUN_NAME must be a simple filename component' >&2
  exit 64
fi
PYTHON="${TB_ADAPTIVE_PYTHON:-python}"
if [[ -n "${TB_ADAPTIVE_SELECTION_FILE:-}" ]]; then
  SELECTION_FILE="$TB_ADAPTIVE_SELECTION_FILE"
  [[ -s "$SELECTION_FILE" ]] || { echo 'Frozen selection is missing or empty' >&2; exit 64; }
else
  mkdir -p logs/tb_emergent_efficiency_adaptive_harbor/_task_selections
  SELECTION_FILE="logs/tb_emergent_efficiency_adaptive_harbor/_task_selections/${RUN_NAME}_$(date -u +%Y%m%d_%H%M%S)_${BASHPID}.json"
  "$PYTHON" "$SCRIPT_DIR/run_emergent_efficiency_tb_adaptive_harbor.py" "$@" \
    --mode static-control --plan-only --selection-file "$SELECTION_FILE"
fi
IFS=',' read -r -a MODES <<< "${TB_ADAPTIVE_MODES:-static-control,normal-evolve,cost-aware}"
for MODE in "${MODES[@]}"; do
  printf '\nTerminal-Bench adaptive: %s / %s (selection: %s)\n' "$RUN_NAME" "$MODE" "$SELECTION_FILE"
  bash "$SCRIPT_DIR/run_one_harbor.sh" "$RUN_NAME" "$MODE" "$@" --selection-file "$SELECTION_FILE"
done
