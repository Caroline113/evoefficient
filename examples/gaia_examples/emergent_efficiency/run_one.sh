#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:?Usage: $0 <RUN_NAME> <MODE> [runner flags]}"
MODE="${2:?Usage: $0 <RUN_NAME> <MODE> [runner flags]}"
shift 2

python examples/gaia_examples/emergent_efficiency/run_emergent_efficiency_gaia.py \
  --mode "$MODE" \
  --output-dir "logs/gaia_emergent_efficiency/${RUN_NAME}_${MODE}" \
  "$@"
