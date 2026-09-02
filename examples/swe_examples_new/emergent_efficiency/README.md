# 自然涌现效率策略实验 Runner

本目录是独立实验入口，和之前的 `evolve_sequential.py`、EvoEfficient MVP runner 分开。
目标是研究：不把预定义效率策略列表交给 evolver 时，成本信号是否能诱导 agent 自己产生可复用的效率行为。

## 文件

- `run_emergent_efficiency.py`：独立 runner。
- `README.md`：本文档。

runner 不修改 `examples/swe_examples_new/evolve_sequential.py`，也不从它 import 运行逻辑。
                                                                       


## 四个实验组

### A. normal-evolve baseline

正常 agent 自进化控制组：复制 seed workspace，清理 `efficiency/` 目录，并启用 solver
skill proposal + GuidedSynthesisEngine curation。runner 还会在复制后的 workspace 中
redaction 预定义策略名，避免 seed tools 源码里的策略标记影响泄漏审计。原始 seed
workspace 不会被修改。

```bash
python examples/swe_examples_new/emergent_efficiency/run_emergent_efficiency.py \
  --mode normal-evolve \
  --limit 2 \
  --batch-size 2 \
  --parallel 2 \
  --max-steps 60 \
  --output-dir logs/emergent_efficiency/normal_evolve
```

### B. cost-aware emergence

主实验组：复制 seed workspace，清理 `efficiency/` 目录；每个 batch 结束后，先执行
和 A 组相同的正常 agent 自进化，再额外把中性的成本和行为摘要交给 proposer，让它生成
普通 skill 或 prompt fragment。复制后的 workspace 会先经过预定义策略名 redaction；
因此 proposer 和后续 agent workspace 中都不应该出现策略 registry 的原始名称。

```bash
python examples/swe_examples_new/emergent_efficiency/run_emergent_efficiency.py \
  --mode cost-aware \
  --limit 4 \
  --batch-size 2 \
  --parallel 2 \
  --max-steps 60 \
  --output-dir logs/emergent_efficiency/cost_aware
```
```bash
python examples/swe_examples_new/emergent_efficiency/run_emergent_efficiency.py \
  --mode cost-aware \
  --limit 2 \
  --batch-size 2 \
  --parallel 2 \
  --max-steps 60 \
  --output-dir logs/emergent_efficiency/cost_aware_small
```

### C. predefined-control

工程对照组：执行与 A/B 相同的正常 agent 自进化，并显式启用已有 EvoEfficient 策略，
作为自然涌现组的上限/对照。

```bash
python examples/swe_examples_new/emergent_efficiency/run_emergent_efficiency.py \
  --mode predefined-control \
  --limit 10 \
  --batch-size 2 \
  --parallel 2 \
  --max-steps 60 \
  --output-dir logs/emergent_efficiency/predefined
```

可以用 `--predefined-strategies ...` 指定 control 组策略列表。这个参数对 `predefined-control` 和 `predefined-cost-aware` 生效。
默认 control 策略为 `observation_masking`、`plan_template_cache`、`adaptive_tool_trimming`、`stagnation_rescue_stop`、`skill_compilation`；默认不启用 `verbatim_compaction`。

### D. predefined-cost-aware

组合实验组：先显式启用已有 EvoEfficient 预定义效率策略；每个 batch 结束后，仍然执行正常
agent 自进化，并额外加入 cost-aware proposer，让 agent 在已有效率控制的基础上继续生成新的
效率 skill 或 prompt fragment。

这组用于回答：当 agent 已经有一套手工定义的效率策略时，成本反馈还能否诱导它继续自进化出
有增量价值的新效率策略。

与 B 组不同，D 组的 proposer 会知道当前已激活的预定义策略名称，但系统提示要求它只提出
已有策略未覆盖的增量行为，不能简单复述预定义策略。

```bash
python examples/swe_examples_new/emergent_efficiency/run_emergent_efficiency.py \
  --mode predefined-cost-aware \
  --limit 10 \
  --batch-size 2 \
  --parallel 2 \
  --max-steps 60 \
  --output-dir logs/emergent_efficiency/predefined_cost_aware
```

可以用 `--predefined-strategies ...` 指定这组启动时激活的预定义策略列表。

## 小规模 smoke

先用 dry-run proposer 验证 runner 的 batch、proposal 写入、summary 和图表输出，不调用 proposer LLM：

```bash
python examples/swe_examples_new/emergent_efficiency/run_emergent_efficiency.py \
  --mode cost-aware \
  --dry-run-proposer \
  --limit 4 \
  --batch-size 2 \
  --parallel 2 \
  --max-steps 20 \
  --output-dir logs/emergent_efficiency/smoke
```

注意：`--dry-run-proposer` 只跳过 B/D 组额外的 cost-aware proposer LLM；任务求解本身仍会调用
SWE agent 模型，batch 间的正常自进化 curation 也可能在存在 solver proposal 时调用 evolver 模型。

## 输出结构

每次运行会生成：

```text
logs/emergent_efficiency/<run_name>/
  workspace/
  results.json
  summary.json
  summary.csv
  patches/
  conversations/
  emergence/
    leakage_audit.json
    proposals.jsonl
    proposal_adoption.json
    batch_XX_proposal_report.json
    observations/
    workspace_snapshots/
  figures/
    strategy_avg_tokens.png
    strategy_avg_cost.png
    strategy_cost_success.png
    strategy_tool_calls.png
    strategy_efficiency_frontier.png
    strategy_hits.png
```

默认情况下，runner 会给最终输出目录追加时间戳，避免覆盖已有实验。例如：

```text
--output-dir logs/emergent_efficiency/cost_aware
```

实际输出类似：

```text
logs/emergent_efficiency/cost_aware_20260902_153012_123456/
```

如果确实要复用旧的固定路径行为，可以加 `--fixed-output-dir`；这时 `--overwrite` 才会删除已有目录。

关键文件：

- `emergence/leakage_audit.json`：检查 workspace 是否残留预定义策略名。`normal-evolve` 和 `cost-aware` 模式下如果发现泄漏，runner 会直接停止；predefined 相关模式允许预定义策略存在。
- `emergence/normal_evolution.jsonl`：A/B/C/D 四组正常自进化每个 batch 的 solver proposal 和 curation 结果摘要。
- `emergence/proposals.jsonl`：B/D 组 cost-aware proposer 的原始输出、输入成本摘要和解析后的 proposals。
- `emergence/proposal_adoption.json`：哪些 proposal 被写成 skill 或 prompt fragment。
- `emergence/workspace_snapshots/`：每个自进化 batch 后的 prompts/skills/memory 快照。
- `summary.csv`：按 batch 和 overall 汇总 success、token、cost、zero_token、estimated_usage 等指标。

## 指标解释

- `avg_tokens`：主 agent input + output token，不含 cache。
- `avg_cost`：包含 input、output、cache read/write，以及 route model cost。
- `cost_of_pass`：`total_cost / success`，越低越好；没有成功任务时为 `inf`。
- `zero_token_tasks`：完全没有 token 记录的任务数。该值大于 0 时，不应把低成本直接解释为策略收益。
- `estimated_usage_tasks`：provider 未返回 usage 时通过文本启发式估算的任务数。

## 如何判断自然涌现是否有效

先比较 A 与 B。二者都启用正常 agent 自进化；B 只额外增加 cost-aware emergent
efficiency proposal，因此更能体现自然涌现效率策略是否有效：

```text
B.success_rate >= A.success_rate - 0.03
B.avg_cost <= A.avg_cost * 0.95
B.zero_token_tasks == 0 或 token_record_quality != missing
```

再比较 B 与 C：

- B 接近 C：说明成本信号足以诱导有效策略。
- C 明显强于 B：说明人工预定义策略仍有工程优势。
- B 成本下降但成功率下降：说明 proposal 可能变成过早放弃，需要加入 success-first gate 或 rescue-first 约束。

## 后验分类

实验运行时不使用预定义策略名。实验结束后再人工或用独立 classifier 对 B 组 proposals 分类：

- context reduction
- observation/tool trimming
- verification-first workflow
- plan/template reuse
- model delegation
- early rescue/stop
- memory/skill compilation
- unknown/new

分类时应同时看：

- `proposals.jsonl` 里的 proposal 文本
- `proposal_adoption.json` 的落地目标
- `workspace_snapshots/` 中 prompt/skill 的实际变化
- 后续 batch 的 trajectory 行为变化

只有在 held-out 上保持成功率且降低成本的 proposal，才应升级为正式策略候选。

## 常用参数

- `--mode normal-evolve|cost-aware|predefined-control|predefined-cost-aware`
- `--limit`：任务数。
- `--batch-size`：每个 batch 任务数。四组都在 batch 间做正常自进化；cost-aware 和 predefined-cost-aware 额外写入 emergent proposal。
- `--parallel`：batch 内并发 worker 数。
- `--max-steps`：每个任务最大 tool call 数。
- `--model-id`：solver 模型。
- `--evolver-model`：proposal 生成模型；默认复用 `--model-id`。
- `--dry-run-proposer`：只用于 smoke，使用固定 dummy proposal。
- `--fixed-output-dir`：严格使用 `--output-dir` 指定的路径，不自动追加时间戳。
- `--overwrite`：仅建议配合 `--fixed-output-dir` 使用，用于覆盖已有输出目录。

## 建议实验规模

成本受限时：

```text
3 组 x 3 seeds x 10 tasks, batch_size=2
```

更稳定的主实验：

```text
3 组 x 3 seeds x 20 tasks, batch_size=2 或 4
```

如果 batch 太大，proposal 反馈频率低；如果 batch 太小，单批统计噪声大。建议先用 `batch_size=2` 做 smoke，再用 `batch_size=4` 做主实验。
