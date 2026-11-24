#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUNNER="${SCRIPT_DIR}/run_emergent_efficiency.py"

# The project .venv can be inconsistent on some machines; this conda env was
# verified to import swebench correctly. Override with PYTHON_BIN if needed.
PYTHON_BIN="${PYTHON_BIN:-/mnt/miniconda3/envs/aevolve/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

# Space-separated mode list. Override, for example:
#   MODES="normal-evolve cost-aware" bash run_multiple_modes.sh
# normal-evolve predefined-control category-seeded-cost-aware
MODES="${MODES:-cost-aware}"

# Shared experiment defaults. Override any of these as environment variables.
LIMIT="${LIMIT:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PARALLEL="${PARALLEL:-2}"
MAX_STEPS="${MAX_STEPS:-60}"
WINDOW_SIZE="${WINDOW_SIZE:-40}"
DATASET="${DATASET:-MariusHobbhahn/swe-bench-verified-mini}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/emergent_efficiency/multi_mode}"
MODEL_ID="${MODEL_ID:-deepseek-v4-flash}"
EVOLVER_MODEL="${EVOLVER_MODEL:-}"
REGION="${REGION:-us-west-2}"
MAX_PROPOSALS_PER_BATCH="${MAX_PROPOSALS_PER_BATCH:-2}"
CATEGORY_SEED="${CATEGORY_SEED:-basic-efficiency}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

for extra_arg in "$@"; do
  case "${extra_arg}" in
    --mode|--output-dir|--category-seed)
      echo "Do not pass ${extra_arg} to this wrapper. Use MODES, OUTPUT_ROOT, or CATEGORY_SEED instead." >&2
      exit 2
      ;;
  esac
done

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

echo "Python: ${PYTHON_BIN}"
echo "Runner: ${RUNNER}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Modes: ${MODES}"
echo

failures=()
for mode in ${MODES}; do
  mode_slug="${mode//-/_}"
  output_dir="${OUTPUT_ROOT}/${mode_slug}"

  args=(
    "${RUNNER}"
    --mode "${mode}"
    --dataset "${DATASET}"
    --limit "${LIMIT}"
    --batch-size "${BATCH_SIZE}"
    --parallel "${PARALLEL}"
    --max-steps "${MAX_STEPS}"
    --window-size "${WINDOW_SIZE}"
    --model-id "${MODEL_ID}"
    --region "${REGION}"
    --max-proposals-per-batch "${MAX_PROPOSALS_PER_BATCH}"
    --output-dir "${output_dir}"
  )

  if [[ -n "${EVOLVER_MODEL}" ]]; then
    args+=(--evolver-model "${EVOLVER_MODEL}")
  fi

  if [[ "${mode}" == "category-seeded-cost-aware" ]]; then
    args+=(--category-seed "${CATEGORY_SEED}")
  fi

  echo "============================================================"
  echo "Running mode: ${mode}"
  echo "Output prefix: ${output_dir}"
  echo "============================================================"

  if "${PYTHON_BIN}" "${args[@]}" "$@"; then
    :
  else
    status=$?
    failures+=("${mode}:${status}")
    echo "Mode failed: ${mode} (exit ${status})" >&2
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      exit "${status}"
    fi
  fi
  echo
done

if ((${#failures[@]} > 0)); then
  echo "Finished with failed mode(s): ${failures[*]}" >&2
  exit 1
fi

echo "All requested modes finished."
