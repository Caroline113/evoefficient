#!/usr/bin/env bash
set -euo pipefail

# MODES=(static-control normal-evolve cost-aware)
MODES=(cost-aware)
for mode in "${MODES[@]}"; do
  test -f "$RUN_ROOT/$mode/trainval/artifact/artifact.json" || {
    echo "缺少冻结产物：$mode"
    break
  }
  python -m json.tool "$RUN_ROOT/$mode/trainval/artifact/artifact.json"
done

# 演化组的候选指标和选择历史在：
python -m json.tool "$RUN_ROOT/cost-aware/trainval/selection_history.json"

# 确定实验配置后，对最终冻结策略运行 test：
for mode in "${MODES[@]}"; do
  run_experiment evaluate \
    "${COMMON_ARGS[@]}" \
    --artifact "$RUN_ROOT/$mode/trainval/artifact" \
    --output "$RUN_ROOT/$mode/final_test" || break
done

# 查看结果和费用
for mode in "${MODES[@]}"; do
  echo "实验模式：$mode"
  python -m json.tool "$RUN_ROOT/$mode/final_test/test/summary.json"
  python -m json.tool "$RUN_ROOT/$mode/final_test/test/freeze_audit.json"
  python -m json.tool "$RUN_ROOT/$mode/final_test/pipeline_summary.json"
done