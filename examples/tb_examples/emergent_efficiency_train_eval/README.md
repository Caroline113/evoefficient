# EvoEfficient 三阶段实验运行指南

本实验从 Terminal-Bench 4.1.0 CPU 数据集中固定划分训练集、验证集和测试集：

| 子集 | 任务数 | 用途 |
|---|---:|---|
| train（训练集） | 10 | 运行任务、收集轨迹，生成候选效率策略 |
| val（验证集） | 11 | 比较候选与父策略，选出下一轮使用的策略 |
| test（测试集） | 42 | 全部迭代结束后，评测冻结的最终策略 |

流程为：**固定数据划分 → train 生成候选 → val 筛选 → 重复迭代 → 冻结策略 → test 最终评测**。

本文只说明当前三阶段实验。以下命令全部在服务器的 **Bash 终端**执行；按顺序在同一个终端中运行，保留变量和函数定义。

## 第一步：进入项目并激活环境

~~~bash
cd /home/zhangj777/zj/evolve/a-evolve-new
source .venv/bin/activate

# 为本次手动运行分配输出根目录，后续步骤继续使用同一个值。
if [[ -z "${RUN_ROOT:-}" ]]; then
  mkdir -p logs/tb41_three_way
  RUN_ROOT=$(mktemp -d "logs/tb41_three_way/seed42_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_XXXXXX")
fi
printf '本次实验输出目录：%s\n' "$RUN_ROOT"

# 三个模式共用同一份固定划分。
SPLIT=datasets/tb41_cpu_split_seed42.json

# Harbor 使用独立的 CPU 环境。
HARBOR_REPO=/home/zhangj777/zj/evolve/harbor
HARBOR_CMD="$HARBOR_REPO/.venv-cpu/bin/harbor run"

# 求解、普通进化、轨迹判断与效率提案使用相同模型配置。
MODEL_ID=deepseek:deepseek-flash
~~~

项目的 Python 来自项目 .venv；实际任务由 Harbor 的 .venv-cpu 执行。后续命令显式指定 Harbor 路径，不依赖代码中保留的旧路径默认值。

直接运行 train.sh 时，脚本会自动分配并打印本次实验的输出目录。需要指定目录时，可以用 RUN_ROOT=目标路径 bash train.sh。后续测试使用训练时打印的目录。

## 第二步：检查环境

~~~bash
python -m examples.tb_examples.emergent_efficiency_train_eval.run_experiment preflight \
  --harbor-repo "$HARBOR_REPO"

"$HARBOR_REPO/.venv-cpu/bin/python" \
  examples/tb_examples/emergent_efficiency_train_eval/validate_setup.py

docker info
df -h /var/lib/docker

python -m pytest examples/tb_examples/emergent_efficiency_train_eval/tests -q
~~~

检查通过后继续：

- preflight 检查依赖与 Harbor 目录；Harbor 实际可用性以其独立环境的配置检查为准。
- validate_setup.py 检查任务配置和独立 verifier 支持。
- Docker 应能正常访问，并有足够空间构建任务镜像。
- pytest 不调用付费模型。
- 这些检查不代表模型接口、所有任务镜像下载或真实任务执行已经成功。

## 第三步：准备数据和固定划分

### 3.1 检查数据是否已经准备好

~~~bash
ls datasets/terminal-bench-4.1.0-cpu/cpu-tasks/dataset_receipt.json
~~~

文件存在时直接进行 3.2。只有尚未准备数据时，才执行：

~~~bash
python -m examples.tb_examples.emergent_efficiency_train_eval.prepare_dataset benchmark \
  --cpu-only \
  --output datasets/terminal-bench-4.1.0-cpu
~~~

准备器固定使用上游 v4.1.0，检查完整数据的 66 个任务，再按任务及 verifier 的加速器声明筛出 63 个 CPU 任务，并保存版本、提交和任务内容哈希。

下载目标目录必须不存在。如果之前下载失败但留下目录，请选择新的输出目录，并相应修改后续划分命令中的 --dataset-dir。

### 3.2 检查或创建划分清单

服务器已生成默认清单 datasets/tb41_cpu_split_seed42.json。以下命令会复用已有清单，只在不存在时创建：

~~~bash
if [ -f "$SPLIT" ]; then
  echo "复用已有数据划分：$SPLIT"
else
  python -m examples.tb_examples.emergent_efficiency_train_eval.prepare_dataset split \
    --dataset-dir datasets/terminal-bench-4.1.0-cpu/cpu-tasks \
    --output "$SPLIT" \
    --train-count 10 \
    --test-count 42 \
    --seed 42 \
    --method random
fi

python -m json.tool "$SPLIT"
~~~

这里使用固定种子的随机划分：10 个 train、42 个 test，其余 11 个作为 val。所有模式必须使用同一份清单。

划分清单不能覆盖；训练和测试时会校验数据内容、分区覆盖及重叠。确定划分后，不再修改清单或任务文件。seed 控制数据划分，不保证模型服务端生成结果完全确定。

## 第四步：检查计划并进行流程演练

先设置共用参数：

~~~bash
COMMON_ARGS=(
  --split-manifest "$SPLIT"
  --harbor-repo "$HARBOR_REPO"
  --harbor-cmd "$HARBOR_CMD"
  --delegate-agent codex
  --parallel 2
  --selection-cost total_tokens
)

python -m examples.tb_examples.emergent_efficiency_train_eval.run_experiment plan \
  "${COMMON_ARGS[@]}" \
  --mode cost-aware \
  --epochs 3 \
  --candidates-per-epoch 2 \
  --selection-tier medium \
  --batch-size 4 \
  --output "$RUN_ROOT/plan.json"
~~~

计划检查应显示 train=10、val=11、test=42，并保存任务与参数清单；不调用模型、不启动 Docker。

再运行一次完整流程演练：

~~~bash
python -m examples.tb_examples.emergent_efficiency_train_eval.run_experiment run \
  "${COMMON_ARGS[@]}" \
  --mode cost-aware \
  --epochs 2 \
  --candidates-per-epoch 2 \
  --selection-tier medium \
  --batch-size 4 \
  --output "$RUN_ROOT/smoke" \
  --dry-run

python -m json.tool "$RUN_ROOT/smoke/test/freeze_audit.json"
~~~

冻结审计应显示 passed=true、evaluation_evolution_calls=0。

--dry-run 只验证流程，训练、验证和测试均使用合成结果，不调用模型、不启动 Docker。它不会学习真实策略，成功率和费用不能作为实验成绩。正式运行必须去掉 --dry-run，并使用不同输出目录。

## 第五步：加载模型连接

正式运行沿用服务器项目 .env 中的模型配置。下面的 Bash 函数在每次启动时读取 .env，通过项目模型工厂解析连接，并将同一连接传给 Harbor 的求解器。

~~~bash
run_experiment() {
  python - "$MODEL_ID" "$@" <<'PY'
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from agent_evolve.llm.factory import create_openai_client_for_model
from examples.tb_examples.emergent_efficiency_train_eval.run_experiment import main

load_dotenv(Path.cwd() / ".env", override=False)
model_id = sys.argv[1]
client, solver_model, provider = create_openai_client_for_model(model_id)
os.environ["OPENAI_API_KEY"] = client.api_key
os.environ["OPENAI_BASE_URL"] = str(client.base_url)
client.close()

print(f"模型配置：{model_id}；求解模型：{solver_model}；连接类型：{provider}", flush=True)
main(sys.argv[2:] + [
    "--model", model_id,
    "--evolver-model", model_id,
    "--harbor-model", solver_model,
])
PY
}
~~~

此函数只在当前终端有效，不修改 .env 或全局认证配置，也不打印密钥。新开终端后，需要重新执行第一步、第四步的 COMMON_ARGS 定义及本步函数定义。

先用它检查一次实际模型配置和实验计划：

~~~bash
run_experiment plan \
  "${COMMON_ARGS[@]}" \
  --mode cost-aware \
  --epochs 3 \
  --candidates-per-epoch 2 \
  --selection-tier medium \
  --batch-size 4
~~~

这一步不发送模型推理请求，因此不能代替真实 API 连通性验证。环境中已经设置的变量优先于 .env；确认它们指向本次需要使用的连接。

## 第六步：正式训练并用 val 筛选

三个模式的区别：

| 模式 | 行为 |
|---|---|
| static-control | 保留初始策略，不进行训练和演化；运行初始验证并冻结 |
| normal-evolve | 在 train 上执行普通技能演化，通过 val 选择策略 |
| cost-aware | 在普通演化基础上生成成本感知效率策略，通过 val 选择策略 |

下面依次运行三组，采用相同划分、模型、初始工作区和筛选配置。会调用真实模型、启动任务环境并产生费用：

~~~bash
MODES=(static-control normal-evolve cost-aware)

for mode in "${MODES[@]}"; do
  run_experiment train \
    "${COMMON_ARGS[@]}" \
    --mode "$mode" \
    --seed-workspace seed_workspaces/terminal \
    --epochs 3 \
    --candidates-per-epoch 2 \
    --selection-tier medium \
    --batch-size 4 \
    --output "$RUN_ROOT/$mode/trainval" || break
done
~~~

如果只运行成本感知组，将 MODES 改为：

~~~bash
MODES=(cost-aware)
~~~

然后执行上面的循环。不要对已经完成的非空输出目录重新运行。

train 命令包含完整训练与验证，但不会运行 test。对两个演化模式，每轮：

1. 从上一轮选中的策略独立生成 2 个候选。
2. 每个候选在全部 10 个训练任务上按每批 4 个任务收集轨迹并更新；最后一个批次也会更新。
3. 在同一组 11 个验证任务上评估父策略及两个候选。
4. 用初始策略的验证成功率作为固定门槛，过滤不达标和被 Pareto 支配的候选。
5. 按 medium 权重选出下一轮策略；相同分数优先保留父策略。若所有候选均未达门槛，保留此前选中的策略。

3 轮完成后，最终策略写入各组 trainval/artifact/。验证轨迹不作为生成器的训练输入；这属于流程层面的隔离，并非操作系统级数据访问沙箱。

默认配置下，每个演化模式计划执行 60 次训练任务运行、110 次验证任务运行，最终测试另有 42 次；不含演化器自身的调用。增加轮数或候选数会增加费用。

## 第七步：核对训练产物，再进行最终 test 评测

先确认三组冻结产物都存在。若上一步只运行了一个模式，保持 MODES 为相同列表。

~~~bash
for mode in "${MODES[@]}"; do
  test -f "$RUN_ROOT/$mode/trainval/artifact/artifact.json" || {
    echo "缺少冻结产物：$mode"
    break
  }
  python -m json.tool "$RUN_ROOT/$mode/trainval/artifact/artifact.json"
done
~~~

演化组的候选指标和选择历史在：

~~~bash
python -m json.tool "$RUN_ROOT/cost-aware/trainval/selection_history.json"
~~~

确定实验配置后，对最终冻结策略运行 test：

~~~bash
for mode in "${MODES[@]}"; do
  run_experiment evaluate \
    "${COMMON_ARGS[@]}" \
    --artifact "$RUN_ROOT/$mode/trainval/artifact" \
    --output "$RUN_ROOT/$mode/final_test" || break
done
~~~

evaluate 从冻结产物读取实验模式，不需要 --mode、--epochs 或 --candidates-per-epoch。必须使用相同划分清单、模型、求解器和环境配置。

每组在全部 42 个测试任务上运行一次，每个任务使用独立策略副本。测试过程中不演化，输出也不回流用于策略筛选。不要根据 test 成绩重新选择候选、档位或超参数。

## 第八步：查看结果和费用

~~~bash
for mode in "${MODES[@]}"; do
  echo "实验模式：$mode"
  python -m json.tool "$RUN_ROOT/$mode/final_test/test/summary.json"
  python -m json.tool "$RUN_ROOT/$mode/final_test/test/freeze_audit.json"
  python -m json.tool "$RUN_ROOT/$mode/final_test/pipeline_summary.json"
done
~~~

重点查看：

| 文件或字段 | 含义 |
|---|---|
| test/summary.json 中的 tasks、successes、pass_rate | 测试任务数、成功数、成功率 |
| mean_solver_tokens | 每个测试任务平均求解 token |
| mean_solver_cost | 每个测试任务平均求解费用；缺少可靠价格时为 null |
| test/freeze_audit.json | 冻结策略是否未被修改，测试演化调用数是否为 0 |
| pipeline_summary.json | 训练、验证、最终测试的分项费用 |
| trainval/train/usage_ledger.json | 普通演化、轨迹判断及效率 proposer 的用量记录 |

本指南显式设置 --selection-cost total_tokens，即按 token 代理费用筛选，不等于按美元费用筛选。若所有验证结果都有可靠定价，可以在第四步改用 cost_units；三组必须保持一致。使用 cost_units 遇到未知价格时会停止筛选。

流水线汇总包含被淘汰候选的训练和验证开销；无法可靠定价的演化费用保留未知，总费用可能为 null，不能把它解释成零费用。多次运行中的 tasks 是运行次数，不是去重后的任务数。

目录结构：

~~~text
logs/tb41_three_way/seed42_YYYYMMDD_HHMMSS_六位后缀/
  plan.json
  smoke/                         # 合成流程演练
  static-control/
    trainval/
      artifact/                  # 初始策略冻结包
    final_test/
  normal-evolve/
    trainval/
    final_test/
  cost-aware/
    trainval/
      experiment_manifest.json
      candidates/                # 每个候选的轨迹、演化记录及冻结包
      train/                     # 所有候选的训练汇总和用量
      val/                       # 初始策略、父策略和候选的验证结果
      selection_history.json
      artifact/                  # 最终选中的冻结策略
    final_test/
      experiment_manifest.json
      test/
        results.json
        summary.json
        freeze_audit.json
        trials/                  # 逐任务结果及 Harbor 运行产物
      pipeline_summary.json
~~~

## 常见情况

| 情况 | 处理方式 |
|---|---|
| 输出目录非空 | 换一个 RUN_ROOT 或输出路径；没有自动覆盖或断点续训 |
| 划分清单已存在 | 复用原清单，不重复生成；新划分使用新文件名 |
| 数据哈希不一致 | 核对任务是否被修改；不要直接修改清单哈希绕过检查 |
| 模型认证失败 | 核对 .env、已导出的环境变量及第五步模型加载；不要把密钥写入日志 |
| Harbor、Docker 或任务执行异常 | 查看报错指出的任务目录与 inputs 下的 stdout.txt、stderr.txt，再检查 Harbor 原始结果 |
| 验证费用未知 | 完善定价，或明确使用 total_tokens 重新开展实验 |
| 希望缩小实验 | 固定划分协议不接受 train-limit / eval-limit；另建开发用数据划分，并保留最终测试集 |
| static-control 没有 selection_history.json | 正常；该组不生成候选、不进行迭代选择 |
| 需要重复实验 | 保持划分及配置，换输出目录；新目录名不代表设置了模型随机种子 |

旧的 run_cpu_demo.py 和 run_matrix.sh 服务于独立训练集／评测集协议，不用于本文三阶段流程。

## 可选：使用 DPP 选择训练核心集

主流程使用随机划分。如需采用论文的 DPP 核心集方法，需要先准备特征文件；当前代码不会自动采集初始轨迹、调用难度判断模型或生成嵌入。

准备步骤：

1. 保持 seed=42、test-count=42，先固定相同的测试集。
2. 仅在剩余 21 个任务上收集初始策略轨迹。
3. 为每个任务生成 0–10 的难度分数和抽象失败模式描述，再将描述转为同维度非零向量。
4. 将全部 21 个非测试任务的特征写入 datasets/tb41_cpu_dpp_features.json；不要包含 test 特征，也不要用候选验证成绩生成特征。

文件格式示例，实际必须覆盖全部非测试任务：

~~~json
{
  "任务ID": {
    "difficulty": 7.5,
    "embedding": [0.1, 0.4, 0.2]
  }
}
~~~

创建一份新的 DPP 清单：

~~~bash
python -m examples.tb_examples.emergent_efficiency_train_eval.prepare_dataset split \
  --dataset-dir datasets/terminal-bench-4.1.0-cpu/cpu-tasks \
  --output datasets/tb41_cpu_split_dpp_seed42.json \
  --train-count 10 \
  --test-count 42 \
  --seed 42 \
  --method dpp \
  --features datasets/tb41_cpu_dpp_features.json
~~~

特征维度及秩必须足以选择 10 个训练任务；上面的三维向量只展示格式。代码记录特征文件哈希，核验非测试任务覆盖，并在核矩阵秩不足时停止，不自动退回随机划分。

随后更换 SPLIT 和 RUN_ROOT，重新定义 COMMON_ARGS，再从第四步开始，三组都使用这份 DPP 清单。

## 方法边界

数据分工与验证筛选参考 [HarnessPro 第四节及附录](https://arxiv.org/html/2610.26770v2#S4.SS3)。当前使用项目原有的 EvoEfficient 候选生成方式，以及 Harbor 官方 verifier；未实现论文的 LLM 合成测试。每次实验保留一个指定档位，不同时演化三个档位。

本文的 10/11/42 是 TB4.1 CPU 数据集的实验配置，不是论文 TB2 划分的复刻。报告结果时应明确数据版本、随机或 DPP 划分方式，以及 token 或美元筛选口径。
