#!/usr/bin/env bash
set -euo pipefail

# Run one Terminal-Bench emergent-efficiency condition.
#
# Usage:
#   bash examples/tb_examples/emergent_efficiency/run_one.sh <RUN_NAME> <MODE> [flags]
#
# Modes:
#   normal-evolve
#   cost-aware
#   predefined-control
#   predefined-cost-aware
#   category-seeded-cost-aware

RUN_NAME="${1:?Usage: $0 <RUN_NAME> <MODE> [--limit N] [--batch-size N] [--parallel N] [--tasks a,b] [--exclude a,b] [--dry-run-proposer]}"
MODE="${2:?Usage: $0 <RUN_NAME> <MODE> [flags]}"
shift 2

OUTPUT_DIR="logs/tb_emergent_efficiency/${RUN_NAME}_${MODE}"

python examples/tb_examples/emergent_efficiency/run_emergent_efficiency_tb.py \
  --mode "$MODE" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
