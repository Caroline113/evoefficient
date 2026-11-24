#!/usr/bin/env bash
set -euo pipefail
TRAIN_DIR="${1:?Usage: $0 TRAIN_DIR TB41_TASKS_DIR RUN_PREFIX [extra flags]}"
EVAL_DIR="${2:?TB41 tasks directory is required}"
RUN_PREFIX="${3:?A new output prefix is required}"
shift 3
for MODE in static-control normal-evolve cost-aware; do
  python -m examples.tb_examples.emergent_efficiency_train_eval.run_experiment run \
    --train-dir "$TRAIN_DIR" --eval-dir "$EVAL_DIR" --mode "$MODE" \
    --output "${RUN_PREFIX}_${MODE}" "$@"
done
