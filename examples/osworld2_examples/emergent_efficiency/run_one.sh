#!/usr/bin/env bash
set -euo pipefail

# Run one OSWorld2 emergent-efficiency condition through Harbor.
#
# Environment variables used as defaults:
#   PYTHON_BIN                         Python executable, default: python
#   OSWORLD2_TASK_FILE                 Default --task-file
#   OSWORLD2_PATH                      Default --osworld2-path
#   HARBOR_REPO                        Harbor checkout used by --harbor-repo
#   OSWORLD2_HARBOR_AGENT_IMPORT_PATH  Harbor agent adapter import path
#                                      Default in runner:
#                                      agent_evolve.agents.osworld2.harbor_adapter:AEvolveOSWorld2HarborAgent
#   OSWORLD2_HARBOR_DELEGATE_AGENT_IMPORT_PATH
#                                      Harbor OSWorld agent wrapped by the adapter,
#                                      default in runner: osworld.custom_agent:OSWorldAgent
#   OSWORLD2_HARBOR_DELEGATE_MODEL_NAME
#                                      Optional model name for the delegated OSWorld agent
#   OSWORLD2_HARBOR_DELEGATE_KWARGS    Optional JSON/YAML kwargs for OSWorldAgent
#   HARBOR_MODEL_NAME                  Optional Harbor-facing model name
#   HARBOR_UV_CMD                      Harbor command prefix, default: uv run harbor run
#                                      If uv is not in PATH, use: .venv/bin/harbor run
#
# Smoke test without OSWorld2/Harbor:
#   bash examples/osworld2_examples/emergent_efficiency/run_one.sh smoke normal-evolve \
#     --task-file examples/osworld2_examples/emergent_efficiency/smoke_tasks.json \
#     --limit 2 --batch-size 1 --parallel 1 --dry-run-harbor --dry-run-proposer
#
# Real Harbor run:
#   export OSWORLD2_PATH=/path/to/OSWorld2
#   export OSWORLD2_TASK_FILE="$OSWORLD2_PATH/evaluation_examples/test_all.json"
#   export HARBOR_REPO=/path/to/harbor
#   export OSWORLD2_HARBOR_AGENT_IMPORT_PATH='agent_evolve.agents.osworld2.harbor_adapter:AEvolveOSWorld2HarborAgent'
#   export OSWORLD2_HARBOR_DELEGATE_AGENT_IMPORT_PATH='osworld.custom_agent:OSWorldAgent'
#   bash examples/osworld2_examples/emergent_efficiency/run_one.sh osw2 normal-evolve --limit 2 --batch-size 1 --parallel 1

if [[ $# -lt 2 ]]; then
  cat >&2 <<'USAGE'
Usage:
  run_one.sh <run_name> <mode> [runner args...]

Modes:
  normal-evolve
  cost-aware
  predefined-control
  predefined-cost-aware
  category-seeded-cost-aware

Required for real Harbor evaluation:
  OSWORLD2_TASK_FILE or --task-file
  OSWORLD2_PATH or --osworld2-path
  HARBOR_REPO or --harbor-repo
  OSWORLD2_HARBOR_AGENT_IMPORT_PATH or --harbor-agent-import-path

Use --dry-run-harbor for smoke tests that do not need OSWorld2 or Harbor.
USAGE
  exit 64
fi

RUN_NAME="$1"
MODE="$2"
shift 2

PYTHON_BIN="${PYTHON_BIN:-python}"

DEFAULT_ARGS=()
if [[ -n "${OSWORLD2_TASK_FILE:-}" ]]; then
  DEFAULT_ARGS+=(--task-file "${OSWORLD2_TASK_FILE}")
fi
if [[ -n "${OSWORLD2_PATH:-}" ]]; then
  DEFAULT_ARGS+=(--osworld2-path "${OSWORLD2_PATH}")
fi
if [[ -n "${HARBOR_REPO:-}" ]]; then
  DEFAULT_ARGS+=(--harbor-repo "${HARBOR_REPO}")
fi
if [[ -n "${OSWORLD2_HARBOR_AGENT_IMPORT_PATH:-}" ]]; then
  DEFAULT_ARGS+=(--harbor-agent-import-path "${OSWORLD2_HARBOR_AGENT_IMPORT_PATH}")
fi
if [[ -n "${OSWORLD2_HARBOR_DELEGATE_AGENT_IMPORT_PATH:-}" ]]; then
  DEFAULT_ARGS+=(--harbor-delegate-agent-import-path "${OSWORLD2_HARBOR_DELEGATE_AGENT_IMPORT_PATH}")
fi
if [[ -n "${OSWORLD2_HARBOR_DELEGATE_MODEL_NAME:-}" ]]; then
  DEFAULT_ARGS+=(--harbor-delegate-model-name "${OSWORLD2_HARBOR_DELEGATE_MODEL_NAME}")
fi
if [[ -n "${OSWORLD2_HARBOR_DELEGATE_KWARGS:-}" ]]; then
  DEFAULT_ARGS+=(--harbor-delegate-kwargs "${OSWORLD2_HARBOR_DELEGATE_KWARGS}")
fi
if [[ -n "${HARBOR_MODEL_NAME:-}" ]]; then
  DEFAULT_ARGS+=(--harbor-model-name "${HARBOR_MODEL_NAME}")
fi
if [[ -n "${HARBOR_UV_CMD:-}" ]]; then
  DEFAULT_ARGS+=(--harbor-uv-cmd "${HARBOR_UV_CMD}")
fi

"${PYTHON_BIN}" examples/osworld2_examples/emergent_efficiency/run_emergent_efficiency_osworld2.py \
  --mode "${MODE}" \
  --output-dir "logs/osworld2_emergent_efficiency/${RUN_NAME}_${MODE}" \
  "${DEFAULT_ARGS[@]}" \
  "$@"
