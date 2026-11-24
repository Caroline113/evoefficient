#!/usr/bin/env bash
set -euo pipefail
OUTPUT="${1:?Usage: $0 NEW_OUTPUT_DIR [round caps, e.g. 20 40 80 120]}"
shift
if [ "$#" -eq 0 ]; then set -- 20 40 80 120; fi
BASE=logs/tb_emergent_efficiency_adaptive_harbor
python -m examples.tb_examples.emergent_efficiency_train_eval.analyze_prefix_costs \
  --run "static=${BASE}/tb_new10_static-control_20261101_145526_022514" \
  --run "normal=${BASE}/tb_new10_normal-evolve_20261101_171205_178010" \
  --run "cost-aware=${BASE}/tb_new10_cost-aware_20261101_193933_417823" \
  --max-rounds "$@" --output "$OUTPUT"
