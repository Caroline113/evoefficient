#!/usr/bin/env bash
set -euo pipefail

# Run one Terminal-Bench emergent-efficiency condition with Harbor evaluation.
#
# Environment variables used as defaults:
#   PYTHON_BIN                    Python executable, default: python
#   TB21_TASKS_DIR                Default --challenges-dir
#   HARBOR_REPO                   Harbor checkout used by --harbor-repo
#   TB_HARBOR_AGENT_IMPORT_PATH   Harbor agent adapter import path
#                                  Recommended:
#                                  agent_evolve.agents.terminal.harbor_adapter:AEvolveTerminalHarborAgent
#   TB_HARBOR_DELEGATE_AGENT_NAME Harbor built-in agent wrapped by the adapter,
#                                  default in runner: codex
#   TB_HARBOR_DELEGATE_MODEL_NAME Optional model name for the delegated agent
#   TB_HARBOR_DELEGATE_KWARGS     Optional JSON/YAML kwargs for the delegated agent
#   HARBOR_MODEL_NAME             Optional Harbor-facing model name
#   HARBOR_UV_CMD                 Harbor command prefix, default: uv run harbor run
#                                  If uv is not in PATH, use: .venv/bin/harbor run
#
# Smoke test without Harbor or Docker:
#   bash examples/tb_examples/emergent_efficiency/run_one_harbor.sh smoke normal-evolve \
#     --challenges-dir agent_evolve/benchmarks/tb21/tasks \
#     --tasks count-dataset-tokens \
#     --limit 1 --batch-size 1 --parallel 1 --dry-run-harbor --dry-run-proposer
#
# Real Harbor run:
#   export TB21_TASKS_DIR=/path/to/terminal-bench-2-1/tasks
#   export HARBOR_REPO=/path/to/harbor
#   export TB_HARBOR_AGENT_IMPORT_PATH='agent_evolve.agents.terminal.harbor_adapter:AEvolveTerminalHarborAgent'
#   export TB_HARBOR_DELEGATE_AGENT_NAME=codex
  # bash examples/tb_examples/emergent_efficiency/run_one_harbor.sh tb21 cost-aware \
  #   --limit 10 --batch-size 2 --parallel 1

export HARBOR_REPO=/home/azureuser/zj/harbor
export TB21_TASKS_DIR=/home/azureuser/zj/terminal-bench-2-1/tasks
export TB_HARBOR_AGENT_IMPORT_PATH='agent_evolve.agents.terminal.harbor_adapter:AEvolveTerminalHarborAgent'
export TB_HARBOR_DELEGATE_AGENT_NAME=codex
export HARBOR_UV_CMD='.venv/bin/harbor run'

if [[ $# -lt 2 ]]; then
  cat >&2 <<'USAGE'
Usage:
  run_one_harbor.sh <run_name> <mode> [runner args...]

Modes:
  normal-evolve
  cost-aware
  predefined-control
  predefined-cost-aware
  category-seeded-cost-aware

Required for real Harbor evaluation:
  HARBOR_REPO or --harbor-repo
  TB_HARBOR_AGENT_IMPORT_PATH or --harbor-agent-import-path

Recommended:
  TB21_TASKS_DIR or --challenges-dir

Use --dry-run-harbor for smoke tests that do not need Harbor or Docker.
USAGE
  exit 64
fi

RUN_NAME="$1"
MODE="$2"
shift 2

PYTHON_BIN="${PYTHON_BIN:-python}"

DEFAULT_ARGS=()
if [[ -n "${TB21_TASKS_DIR:-}" ]]; then
  DEFAULT_ARGS+=(--challenges-dir "${TB21_TASKS_DIR}")
fi
if [[ -n "${HARBOR_REPO:-}" ]]; then
  DEFAULT_ARGS+=(--harbor-repo "${HARBOR_REPO}")
fi
if [[ -n "${TB_HARBOR_AGENT_IMPORT_PATH:-}" ]]; then
  DEFAULT_ARGS+=(--harbor-agent-import-path "${TB_HARBOR_AGENT_IMPORT_PATH}")
fi
if [[ -n "${HARBOR_MODEL_NAME:-}" ]]; then
  DEFAULT_ARGS+=(--harbor-model-name "${HARBOR_MODEL_NAME}")
fi
if [[ -n "${TB_HARBOR_DELEGATE_AGENT_NAME:-}" ]]; then
  DEFAULT_ARGS+=(--harbor-delegate-agent-name "${TB_HARBOR_DELEGATE_AGENT_NAME}")
fi
if [[ -n "${TB_HARBOR_DELEGATE_MODEL_NAME:-}" ]]; then
  DEFAULT_ARGS+=(--harbor-delegate-model-name "${TB_HARBOR_DELEGATE_MODEL_NAME}")
fi
if [[ -n "${TB_HARBOR_DELEGATE_KWARGS:-}" ]]; then
  DEFAULT_ARGS+=(--harbor-delegate-kwargs "${TB_HARBOR_DELEGATE_KWARGS}")
fi
if [[ -n "${HARBOR_UV_CMD:-}" ]]; then
  DEFAULT_ARGS+=(--harbor-uv-cmd "${HARBOR_UV_CMD}")
fi

"${PYTHON_BIN}" examples/tb_examples/emergent_efficiency/run_emergent_efficiency_tb_harbor.py \
  --mode "${MODE}" \
  --output-dir "logs/tb_emergent_efficiency_harbor/${RUN_NAME}_${MODE}" \
  "${DEFAULT_ARGS[@]}" \
  "$@"
