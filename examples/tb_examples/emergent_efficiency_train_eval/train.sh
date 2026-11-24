#!/usr/bin/env bash
set -euo pipefail

# 在该服务器上自动使用隔离的用户级 Docker 和容器安装代理。
if [[ -z "${EVOEFFICIENT_CONTAINER_PROXY:-}" ]]; then
  exec bash "$(dirname "${BASH_SOURCE[0]}")/with_container_proxy.sh" bash "$0" "$@"
fi

# 每次启动分配一个目录，同一次启动的各组共用。随机后缀避免同秒冲突。
if [[ -z "${RUN_ROOT:-}" ]]; then
  runs_dir=/home/zhangj777/zj/evolve/a-evolve-new/logs/tb41_three_way
  mkdir -p "$runs_dir"
  run_stamp=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)
  RUN_ROOT=$(mktemp -d "$runs_dir/seed42_${run_stamp}_XXXXXX")
fi
printf '本次实验输出目录：%s\n' "$RUN_ROOT"

# 三个模式共用同一份固定划分。
SPLIT=/home/zhangj777/zj/evolve/a-evolve-new/datasets/tb41_cpu_split_seed42.json

# Harbor 使用独立的 CPU 环境。
HARBOR_REPO=/home/zhangj777/zj/evolve/harbor
HARBOR_CMD="$HARBOR_REPO/.venv-cpu/bin/harbor run"

# 求解、普通进化、轨迹判断与效率提案使用相同模型配置。
MODEL_ID=deepseek:deepseek-flash

# 服务器已生成默认清单 datasets/tb41_cpu_split_seed42.json。以下命令会复用已有清单，只在不存在时创建：

if [ -f "$SPLIT" ]; then
  echo "复用已有数据划分：$SPLIT"
else
  python -m examples.tb_examples.emergent_efficiency_train_eval.prepare_dataset split \
    --dataset-dir /home/zhangj777/zj/evolve/a-evolve-new/datasets/terminal-bench-4.1.0-cpu/cpu-tasks \
    --output "$SPLIT" \
    --train-count 10 \
    --test-count 42 \
    --seed 42 \
    --method random
fi

python -m json.tool "$SPLIT"

COMMON_ARGS=(
  --split-manifest "$SPLIT"
  --harbor-repo "$HARBOR_REPO"
  --harbor-cmd "$HARBOR_CMD"
  --delegate-agent codex
  --parallel 2
  --selection-cost total_tokens
)

# python -m examples.tb_examples.emergent_efficiency_train_eval.run_experiment run \
#   "${COMMON_ARGS[@]}" \
#   --mode cost-aware \
#   --epochs 2 \
#   --candidates-per-epoch 2 \
#   --selection-tier medium \
#   --batch-size 4 \
#   --output "$RUN_ROOT/smoke" \
#   --dry-run

# python -m json.tool "$RUN_ROOT/smoke/test/freeze_audit.json"

# HARBOR_PY=/home/zhangj777/zj/evolve/harbor/.venv-cpu/bin/python
# "$HARBOR_PY" -m pip install socksio
# "$HARBOR_PY" -c 'import socksio; print("SOCKS dependency OK")'

# MODES=(static-control normal-evolve cost-aware)
MODES=(cost-aware)
for mode in "${MODES[@]}"; do
  python -m examples.tb_examples.emergent_efficiency_train_eval.run_experiment train \
    "${COMMON_ARGS[@]}" \
    --mode "$mode" \
    --seed-workspace seed_workspaces/terminal \
    --epochs 1 \
    --candidates-per-epoch 1 \
    --selection-tier medium \
    --batch-size 4 \
    --output "$RUN_ROOT/$mode/trainval"
done