# DeepSWE v1.1 效率策略自进化实验

本目录将 TB 三阶段实验迁移到 DeepSWE v1.1。**train 生成策略、val 筛选策略，全部迭代后冻结，再运行 test。** 求解和评分均由 Harbor 执行。

普通进化参考同级 emergent_efficiency/run_emergent_efficiency.py，使用 GuidedSynthesisEngine 整理技能提案；成本感知组在此基础上复用原 SWE 的效率 proposer 和提案采纳逻辑。原 SWE 与 TB 实验文件保持不变。

以下命令全部直接在服务器 Bash 终端执行，按顺序使用同一个终端。

## 一、实验配置

| 项目 | 配置 |
|---|---|
| 数据集 | DeepSWE v1.1，113 个任务 |
| 固定源码提交 | 0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea |
| 默认数据划分 | 按仓库隔离，train 10、val 28、test 75 |
| 求解器 | Harbor 原生 Codex 的子类，在保留安装、网络白名单和轨迹能力的同时注入策略 |
| 初始策略 | seed_workspaces/swe |
| 普通进化 | 训练轨迹反思 → 结构化提案 → GuidedSynthesisEngine 整理 |
| 成本感知进化 | 普通进化 + 原 SWE 效率 proposer + 原 SWE 提案采纳 |
| 验证筛选 | 初始成功率门槛、Pareto 过滤、指定档位加权选择 |
| 最终评分 | 官方独立 verifier，评测 agent 已提交到 Git 的修改 |

按仓库划分时，train-count 和 test-count 是目标任务数；程序选择完整仓库组合尽量达到目标，不拆开一个仓库。此固定版本及默认种子得到 10/28/75。也可显式使用 --split-unit task 按任务随机划分，但该设置不保证跨仓库泛化。

[官方源码](https://github.com/datacurve-ai/deep-swe) · [v1.1 说明](https://deepswe.datacurve.ai/blog/deepswe-v1-1) · [Harbor 数据集](https://hub.harborframework.com/datasets/datacurve/deep-swe-1-1)

## 二、进入项目

~~~bash
cd /home/zhangj777/zj/evolve/a-evolve-new
source .venv/bin/activate

EXP=examples.swe_examples_new.deepswe_efficiency_train_eval
DATA=datasets/deepswe-v1.1/tasks
SPLIT=datasets/deepswe_v1.1_split_seed42.json
RUN_ROOT=logs/deepswe_three_way/seed42_r01
HARBOR_REPO=/home/zhangj777/zj/evolve/harbor
HARBOR_CMD="$HARBOR_REPO/.venv-cpu/bin/harbor run"
~~~

重新运行实验时更换 RUN_ROOT，避免覆盖已有记录。项目虚拟环境运行演化与统计，Harbor 的 .venv-cpu 运行任务。

## 三、准备数据

服务器上已准备默认数据和划分。先检查：

~~~bash
ls "$DATA/dataset_receipt.json"
ls "$SPLIT"
~~~

两者都存在时，直接进入第四节。

### 尚未下载数据时

~~~bash
python -m "$EXP.prepare_dataset" benchmark \
  --output datasets/deepswe-v1.1
~~~

准备器从官方仓库获取固定提交，不跟随主分支自动更新。目标目录必须不存在。数据集清单记录提交和全部任务哈希，并要求 113 个任务具有独立 verifier 和提交补丁采集配置。

如果已经有检出该固定提交的干净官方源码，可避免重新下载：

~~~bash
python -m "$EXP.prepare_dataset" benchmark \
  --source-dir datasets/deepswe-v1.1-source \
  --output datasets/deepswe-v1.1
~~~

上面两个准备命令只选择一个执行；已有数据不必再次准备。

### 尚未生成划分时

~~~bash
python -m "$EXP.prepare_dataset" split \
  --dataset-dir "$DATA" \
  --output "$SPLIT" \
  --split-unit repository \
  --train-count 10 \
  --test-count 75 \
  --seed 42

python -m json.tool "$SPLIT"
~~~

清单不能覆盖。三组实验应使用同一份清单；训练和测试前都会核验数据内容、任务覆盖、重叠及仓库隔离。划分种子不控制模型服务端的随机性。

## 四、检查 Harbor 环境

用 Harbor 的 Python 检查实际 113 个任务，而不是仅检查示例任务：

~~~bash
LITELLM_LOCAL_MODEL_COST_MAP=True "$HARBOR_REPO/.venv-cpu/bin/python" \
  -m "$EXP.run_experiment" preflight --dataset-dir "$DATA"

docker info
df -h /var/lib/docker

LITELLM_LOCAL_MODEL_COST_MAP=True python -m pytest \
  examples/swe_examples_new/deepswe_efficiency_train_eval/tests \
  examples/tb_examples/emergent_efficiency_train_eval/tests -q
~~~

配置检查应显示 tasks=113、separate_verifier=true、harbor_config_valid=true。它不启动任务容器，也不调用模型，不能替代镜像下载、网络与真实任务执行检查。

任务使用原始 task.toml 中的时间、资源和网络限制。本实验不将独立 verifier 改成与 agent 共用环境，也不修改官方测试。

## 五、检查计划并进行流程演练

~~~bash
COMMON_ARGS=(
  --split-manifest "$SPLIT"
  --harbor-repo "$HARBOR_REPO"
  --harbor-cmd "$HARBOR_CMD"
  --delegate-agent codex
  --model deepseek:deepseek-flash
  --evolver-model deepseek:deepseek-flash
  --harbor-model deepseek-flash
  --parallel 2
  --selection-cost total_tokens
)

python -m "$EXP.run_experiment" plan \
  "${COMMON_ARGS[@]}" \
  --mode cost-aware \
  --epochs 3 \
  --candidates-per-epoch 2 \
  --batch-size 4 \
  --selection-tier medium \
  --output "$RUN_ROOT/plan.json"

python -m "$EXP.run_experiment" run \
  "${COMMON_ARGS[@]}" \
  --mode cost-aware \
  --epochs 2 \
  --candidates-per-epoch 2 \
  --batch-size 4 \
  --selection-tier medium \
  --output "$RUN_ROOT/smoke" \
  --dry-run

python -m json.tool "$RUN_ROOT/smoke/test/freeze_audit.json"
~~~

plan 只检查数据与配置。--dry-run 使用合成运行结果，不调用模型、不启动 Docker；冻结审计应显示 passed=true、evaluation_evolution_calls=0。合成产物不会被当成正式训练策略，不能用来报告成功率或费用。

## 六、正式训练与验证

正式执行前，确认项目 .env 中已有模型连接配置。--load-model-env 会通过项目模型工厂加载连接，将对应密钥和地址传给 Harbor，并使用解析后的求解模型名称；不输出密钥、不修改 .env。环境中已设置的变量优先于 .env。

~~~bash
MODES=(static-control normal-evolve cost-aware)

for mode in "${MODES[@]}"; do
  python -m "$EXP.run_experiment" train \
    "${COMMON_ARGS[@]}" \
    --load-model-env \
    --mode "$mode" \
    --seed-workspace seed_workspaces/swe \
    --epochs 3 \
    --candidates-per-epoch 2 \
    --batch-size 4 \
    --selection-tier medium \
    --output "$RUN_ROOT/$mode/trainval" || break
done
~~~

这些命令会调用真实模型、启动任务容器并产生费用。如只运行成本感知组，先将 MODES 改为 (cost-aware)。

| 模式 | 执行行为 |
|---|---|
| static-control | 冻结初始 SWE 策略；不训练、不演化，仅测初始 val |
| normal-evolve | 训练任务完成后生成普通技能提案，用原 SWE 整理引擎更新策略 |
| cost-aware | 在相同普通进化之后额外提出并采纳效率策略 |

每轮从上一轮选中策略独立生成两个候选。每个候选遍历全部训练任务，每批之后进化，最后一批也进化。父策略和候选在相同 val 上比较，再保留一个指定档位的策略进入下一轮。

成功率门槛固定为初始策略的验证成功率；过滤低于门槛及被 Pareto 支配的候选。medium 使用性能／成本权重 (1,1)，lite 为 (1,2)，heavy 为 (2,1)。相同得分优先保留父策略；若没有候选达到固定门槛，保留此前选中的策略并记录回退。

train 命令不运行 test。默认三轮两候选的每个演化模式有 60 次训练任务运行和 280 次验证任务运行，反思、整理和效率提案的模型调用另行记录。

### Harbor 轨迹怎样进入 SWE 进化

原 SWE 求解器返回 _skill_proposal，但 Harbor 代理不提供这一字段。因此本实验从每条 train 轨迹生成结构化普通提案，再通过原 SWE 的 Observation 转换和 GuidedSynthesisEngine 整理。反思使用求解模型，整理使用 evolver-model。

这是为 Harbor 增加的桥接步骤，不能称作原 SweAgent 内部提案调用的逐字复现。val 和 test 不进行反思或演化。

成本感知组直接复用原 SWE 的 propose_emergent_efficiency 和 _apply_proposals，不加入 TB 特有的失败分布提示包装。原反馈中的 60 步参考值只用于诊断，不对 Harbor 设置外部硬停止器。

## 七、冻结后运行最终测试

确认训练产物存在，并查看成本感知组的候选选择记录：

~~~bash
for mode in "${MODES[@]}"; do
  test -f "$RUN_ROOT/$mode/trainval/artifact/artifact.json" || {
    echo "缺少冻结产物：$mode"
    break
  }
done

python -m json.tool "$RUN_ROOT/cost-aware/trainval/selection_history.json"
~~~

对已经确定的最终策略运行 test，保留与训练相同的模型及求解器配置：

~~~bash
for mode in "${MODES[@]}"; do
  python -m "$EXP.run_experiment" evaluate \
    "${COMMON_ARGS[@]}" \
    --load-model-env \
    --artifact "$RUN_ROOT/$mode/trainval/artifact" \
    --output "$RUN_ROOT/$mode/final_test" || break
done
~~~

evaluate 从冻结产物读取模式，不传 --mode、--epochs 或 --seed-workspace。每组在全部 75 个 test 任务上运行一次。每个测试任务使用独立策略副本，测试期间不发生策略更新。不要依据 test 结果重新选择候选或超参数。

DeepSWE v1.1 只评测已提交的 Git 修改。求解提示已明确要求在完成前提交代码，最终仍由任务自身的 collect hook 抽取补丁并在独立环境评分。

## 八、查看结果

~~~bash
for mode in "${MODES[@]}"; do
  echo "实验模式：$mode"
  python -m json.tool "$RUN_ROOT/$mode/final_test/test/summary.json"
  python -m json.tool "$RUN_ROOT/$mode/final_test/test/freeze_audit.json"
  python -m json.tool "$RUN_ROOT/$mode/final_test/pipeline_summary.json"
done
~~~

| 文件 | 内容 |
|---|---|
| trainval/experiment_manifest.json | 数据划分、任务身份与运行参数 |
| trainval/candidates/ | 候选训练轨迹、普通反思、效率提案和采纳记录 |
| trainval/train/usage_ledger.json | 训练反思、普通整理、效率 proposer 用量 |
| trainval/val/ | 每轮验证结果和全部验证汇总 |
| trainval/selection_history.json | 成功率、费用、前沿及选中策略 |
| trainval/artifact/ | 最终冻结策略和数据来源 |
| final_test/test/results.json | 逐题测试结果 |
| final_test/test/summary.json | 成功数、成功率、平均 token 和已知求解费用 |
| final_test/test/freeze_audit.json | 策略哈希不变及测试演化调用数为零 |
| final_test/pipeline_summary.json | 训练、验证、最终测试费用分项 |

筛选默认使用 total_tokens，这是 token 代理指标，不是美元费用。有可靠价格时可改用 cost_units；未知验证价格会阻止美元筛选。各组应使用相同口径。

被淘汰候选的开销也计入汇总。尚不能可靠定价的演化调用保留 null，因此完整流水线费用可能未知，不按零计算。

## 九、常见问题与边界

- **输出目录已存在**：更换 RUN_ROOT 或输出目录；不自动覆盖，也没有断点续训。
- **数据内容变动**：检查任务文件来源，不能改清单哈希绕过校验。
- **基础设施异常或缺失 reward**：停止并检查原始 Harbor 结果，不把运行失败当成策略低分。
- **训练轨迹或提案为空／格式无效**：停止并保留日志及已记录用量，避免普通进化静默退化为无操作。
- **没有测试结果**：train 只训练和验证；需单独执行 evaluate，或使用 run 一次性完成。
- **普通反思判断没有可复用经验**：允许 ACTION: NONE，此时原整理引擎可以不修改策略。
- **工具提案的执行方式**：工具源码与策略正文会注入 Harbor agent 提示，但不自动注册成 Codex 原生工具；不能把生成工具文件视为运行时已经调用。
- **数据隔离**：控制器只把 train 轨迹传给进化器；这不是操作系统级数据访问隔离。
- **排行榜可比性**：本实验只在 75 个留出任务上评测，且可能使用不同求解器，不等同于官方全量 113 题排行榜成绩。

## 十、代码入口

| 文件 | 用途 |
|---|---|
| prepare_dataset.py | 固定提交的数据准备与划分入口 |
| common.py | DeepSWE 身份、内容和 v1.1 verifier 校验 |
| splits.py | 按仓库或按任务生成固定三分清单 |
| run_experiment.py | 计划、训练、测试、完整运行及模型连接加载 |
| stages.py | 候选训练和冻结测试阶段 |
| three_way.py | 多轮候选生成、验证筛选和最终冻结控制 |
| backend.py | Harbor 执行、轨迹解析、SWE 进化桥接和用量记录 |
| harbor_agent.py | 继承 Harbor 原生 Codex，注入冻结策略并保留网络能力 |
| preflight.py | 使用真实任务验证 Harbor 配置兼容性 |
| tests/test_deepswe.py | 数据隔离、SWE 策略变更、账本和协议测试 |
