# 自然涌现效率策略实验设计

本文档用于设计一个研究问题：在不把现有效率策略列表直接提供给 agent/evolver 的前提下，
哪些效率策略会在自进化过程中自然出现、被复用，并在保持成功率的同时降低成本。

这里先描述实验方案和后续代码改造点，不直接增加实验运行代码。

## 1. 核心问题

当前 EvoEfficient 代码里的效率策略大多是预定义的：

- `agent_evolve/efficiency/strategies.py` 定义策略 id、参数和 prompt block。
- `agent_evolve/efficiency/selector.py` 根据 task profile 生成候选策略 config。
- `agent_evolve/efficiency/gate.py` 用 success-first gate 过滤候选策略。
- `examples/swe_examples_new/run_evoefficient_mvp.py` 会显式验证候选策略。

这套流程适合回答“预定义策略是否有效”，但不适合回答“策略是否自然涌现”。如果 evolver
已经看到了 `adaptive_tool_trimming`、`difficulty_model_routing`、`stagnation_rescue_stop`
这类名字或说明，那么后续出现相似行为只能算“选择/复用预设策略”，不能算严格意义上的自然涌现。

本实验要研究的是：

1. agent 在只看到任务轨迹、成本指标和成功反馈时，是否会自己提出减少 token/tool/cost 的可执行方法。
2. 这些方法是否能跨任务复用，而不是某个任务上的偶然短路。
3. 这些方法与人工预定义策略的相似度有多高，是否出现了预定义集合之外的新策略。

## 2. 操作性定义

一个策略可以被标记为“自然涌现”，需要同时满足：

- `novel_source`：策略描述、规则、工具改动或 skill 是由自进化过程产生的，而不是来自预定义策略 registry。
- `no_name_leakage`：在策略出现前，evolver 的 prompt、workspace、日志摘要中没有暴露预定义策略名称或完整说明。
- `behavioral_signature`：运行轨迹中能观察到稳定行为变化，例如更少无效搜索、更早进入验证、更短 observation、更少重复读文件等。
- `measurable_effect`：在独立验证集上，成功率不低于 baseline 容忍阈值，同时 `avg_cost`、`cost_of_pass` 或 tool calls 有改善。
- `persistence`：该策略进入后续 workspace、prompt、skill 或 tool 之后，在至少一个后续 batch 中仍然被使用。

不满足这些条件的情况应降级为：

- `one_off_tactic`：单个任务上的临时做法。
- `regurgitated_strategy`：与预定义策略高度相似，但实验存在名称/说明泄漏。
- `cost_cutting_failure`：成本下降主要来自提前失败、少探索或不提交有效 patch。
- `measurement_artifact`：token/cost 记录缺失或估算质量不足导致的假收益。

## 3. 实验组设计

建议做四组。这里不把 `no-evolve` 作为主比较组，因为它不能控制“正常自进化本身”
带来的变化。A 组使用无成本反馈的正常 agent 自进化；B 组在 A 的基础上额外加入
cost-aware emergence，让 evolver 看到中性的成本压力，但仍然看不到预定义策略名称。
D 组则在预定义效率策略已经启用的前提下继续加入成本反馈，用于测试手工效率控制和
自然涌现效率策略是否能叠加。

| 组别 | 目的 | 是否给效率指标 | 是否给预定义策略 | 是否允许自进化写入 workspace |
| --- | --- | --- | --- | --- |
| A. Normal-evolve baseline | 正常 agent 自进化控制组 | 否 | 否 | 是 |
| B. Cost-aware emergence | 在正常自进化上额外加入成本反馈 | 是，只给数字和失败案例 | 否 | 是 |
| C. Predefined control | 与当前 EvoEfficient 对照 | 是 | 是 | 是 |
| D. Predefined cost-aware | 预定义策略 + 成本反馈自进化 | 是，只给数字和失败案例 | 是 | 是 |

### A. Normal-evolve baseline

运行正常 agent 自进化：启用 solver skill proposal，并在 batch 间用
GuidedSynthesisEngine 做普通 skill curation。该组不提供成本目标、不提供预定义效率策略，
但允许 agent 像原系统一样通过解题经验改进 workspace。

作用：

- 控制“正常自进化本身”带来的成功率和成本变化。
- 给 B 组提供公平分母：B 相比 A 只多了中性的成本反馈和 emergent efficiency proposal。

### B. Cost-aware emergence

启用与 A 组相同的正常 agent 自进化，并额外提供每个任务的 compact 成本摘要，例如：

- success/fail
- total cost
- input/output/cache token
- tool calls
- max input per turn
- step limit 是否触发
- 哪些工具调用重复或无产出

但仍然不提供任何预定义策略名称、策略说明或候选 config。

作用：

- 测试“成本压力 + 正常自进化”是否会额外诱导自然效率策略。
- 这是本研究的主实验组。

### C. Predefined control

启用与 A/B 相同的正常 agent 自进化，并额外启用当前已有的 EvoEfficient 预定义策略。

作用：

- 作为上限或工程控制组。
- 对比自然涌现策略与人工策略的性能差距。

### D. Predefined cost-aware

启用与 C 组相同的预定义效率策略，同时在每个 batch 后继续加入 cost-aware proposer，
让 agent 在已有工程效率控制的基础上继续生成新的效率 skill 或 prompt fragment。

作用：

- 测试“预定义效率策略 + 成本反馈”是否能产生额外有效的新效率策略。
- 观察新生成策略是否只是重复预定义策略，还是补充了预定义策略未覆盖的行为。
- 与 C 组对比时，可以估计成本反馈在已有策略基础上的增量价值。

## 4. 防止策略泄漏

自然涌现实验最关键的是避免 leakage。后续写代码时应增加一个 `emergence` 模式，在该模式下：

- 不加载 `agent_evolve/efficiency/strategies.py` 的策略 prompt block。
- 不向 evolver 传递 `STRATEGY_ORDER`、`DEFAULT_STRATEGIES`、strategy library、active strategy 名称。
- 不在 workspace 中写入 `efficiency/active_strategies.yaml`。
- 不 materialize `efficiency.md`、`cost_profile.yaml`、`route_env.md` 等预定义策略文档。
- 不把 `strategy_hits`、`accepted_strategies`、`candidate_configs` 这类带策略名称的报告传给 evolver。
- 如果要给成本摘要，只使用中性字段名，例如 `expensive_repeated_actions`，不要写 `adaptive_tool_trimming` 这类标签。
- 新 runner 在 `normal-evolve` 和 `cost-aware` 模式下会删除复制后 workspace 的 `efficiency/`
  目录，并对 seed tools 源码中已有的预定义策略名做 redaction；原始 seed workspace 不会被修改。

可以保留 profiler 统计代码，但它的输出在传给 evolver 前需要过滤。

## 5. 自然策略如何被表达

不要要求 evolver 从预设策略列表中选择。应让它用自由文本或结构化草案表达自己发现的规则。

建议让 evolver 输出类似：

```yaml
emergent_efficiency_proposals:
  - name: short descriptive name written by evolver
    problem_pattern: when this helps
    action_rule: concrete behavior change
    safety_rule: when not to use it
    expected_savings: why it may reduce cost
    correctness_risk: possible failure mode
    evidence:
      - task_id: ...
        observed_waste: ...
        proposed_fix: ...
    implementation_target: prompt | skill | tool | workflow | stopping_rule
```

这里的 `name` 是 evolver 自己起的名字，不参与预定义策略匹配。后处理阶段再把它映射到人工 taxonomy。

## 6. 后验分类，而不是前置分类

实验运行时不使用预定义策略名。实验结束后再做 post-hoc analysis。

建议的后验分类流程：

1. 收集每代 workspace diff、evolver 输出、prompt/skill/tool 改动、任务轨迹。
2. 抽取 emergent proposal 文本和实际行为变化。
3. 由人工或独立 classifier 将其映射到 taxonomy：
   - context reduction
   - adaptive observation/tool trimming
   - verification-first workflow
   - plan/template reuse
   - model routing/delegation
   - early rescue/stop
   - memory/skill compilation
   - unknown/new
4. classifier 不能看到实验组标签和性能结果，避免结果导向打标。
5. 对每个策略给出 `similarity_to_predefined`：
   - `none`
   - `weak`
   - `strong`
   - `near_duplicate`

只有 post-hoc 被标记为 `none/weak` 且验证有效的策略，才有资格称为新涌现策略。

## 7. 评估指标

### 正确性指标

- `success_rate`
- `score_sum`
- `resolved_count`
- held-out success regression

### 成本指标

- `avg_tokens`
- `avg_cost`
- `cost_of_pass`
- `avg_tool_calls`
- `wall_time`
- `step_limit_failures`
- `zero_token_tasks`
- `estimated_usage_tasks`

需要特别注意：

- `avg_tokens` 不含 cache。
- `avg_cost` 包含 input、output、cache read/write、route model cost。

### 涌现指标

- `proposal_count`：产生多少条效率 proposal。
- `accepted_proposal_count`：多少条进入 workspace/prompt/skill/tool。
- `used_proposal_count`：多少条在后续轨迹中可观察到行为命中。
- `persistence_batches`：策略持续影响了几个后续 batch。
- `generalization_gain`：held-out 上的 cost/pass 改善。
- `novelty_score`：与预定义 taxonomy 的差异程度。

## 8. 判定标准

建议使用 success-first 标准。

一个 emergent strategy 可以算“有效”：

```text
heldout_success_rate >= baseline_success_rate - epsilon_success
heldout_avg_cost <= baseline_avg_cost * (1 - delta_cost)
```

推荐初始阈值：

- `epsilon_success = 0.03`
- `delta_cost = 0.05`
- 每个候选至少 `min_trials = 5`
- 至少跨 `2` 个 batch 或 `2` 个随机 seed 复现

对于 step-limit 相关策略，应额外要求：

- 不是简单减少探索导致少提交 patch。
- failure detail 中 step-limit failure 下降。
- 成功任务的 `cost_of_pass` 下降，而不是只在失败任务上省成本。

## 9. 推荐实验流程

### 阶段 1：泄漏审计

先跑一个不调用模型的检查脚本，确认 emergence 模式下 workspace 和 prompt 中不包含预定义策略名：

- `adaptive_tool_trimming`
- `difficulty_model_routing`
- `stagnation_rescue_stop`
- `context_reduction`
- `verbatim_compaction`
- 其它 `STRATEGY_ORDER` 中的 id

任何命中都应阻断实验。

### 阶段 2：小规模 smoke

后续应使用一个独立 runner，不复用、不修改 `evolve_sequential.py`。建议文件名：

```text
examples/swe_examples_new/emergent_efficiency/run_emergent_efficiency.py
```

新增 runner 后，先跑 cost-aware smoke：

```bash
python examples/swe_examples_new/emergent_efficiency/run_emergent_efficiency.py \
  --mode cost-aware \
  --limit 6 \
  --batch-size 2 \
  --parallel 2 \
  --max-steps 60 \
  --output-dir logs/emergence_costaware_smoke
```

runner 默认会在 `--output-dir` 指定的目录名后追加时间戳，例如实际输出可能是：

```text
logs/emergence_costaware_smoke_20260902_153012_123456/
```

这样同一命令重复运行时不会覆盖旧结果。如需旧的固定路径行为，可加 `--fixed-output-dir`；
只有这种情况下才建议配合 `--overwrite`。

### 阶段 3：主实验

每组至少跑多个 seed 或多个 task slice：

```text
A normal-evolve baseline:   3 seeds x 20 tasks
B cost-aware emergence:     3 seeds x 20 tasks
C predefined control:       3 seeds x 20 tasks
D predefined cost-aware:    3 seeds x 20 tasks
```

如果成本受限，先使用：

```text
3 seeds x 10 tasks, batch_size=2
```

### 阶段 4：后验策略分类

对 B/D 的输出做 post-hoc analysis：

- 从 evolution logs 中抽取 proposal。
- 从 workspace diff 中抽取 prompt/skill/tool 改动。
- 从 trajectories 中抽取行为证据。
- 生成 `emergent_strategy_report.json` 和 `REPORT.md`。

### 阶段 5：重放验证

把自然涌现出的策略固化为独立 config，在新 held-out task slice 上重放：

- 不再让 evolver继续改。
- 只启用该 emergent artifact。
- 和同 task slice 的 baseline 对比。

这个阶段用于区分“自然涌现的策略本身有效”与“当时那个演化 workspace 恰好有效”。

## 10. 后续代码改造建议

建议新增一个独立 runner，而不是直接复用当前预定义策略 runner，也不要继续修改
`evolve_sequential.py`：

```text
examples/swe_examples_new/emergent_efficiency/run_emergent_efficiency.py
```

主要参数：

- `--mode normal-evolve|cost-aware|predefined-control|predefined-cost-aware`
- `--hide-predefined-efficiency`
- `--cost-feedback none|summary|trajectory`
- `--proposal-format yaml|json`
- `--posthoc-taxonomy agent_evolve/efficiency/strategies.py`
- `--min-emergence-trials`
- `--heldout-limit`
- `--fixed-output-dir`：关闭默认时间戳目录，严格使用 `--output-dir`。

建议新增模块：

```text
agent_evolve/efficiency/emergence.py
```

职责：

- 过滤传给 evolver 的效率信息，防止策略名泄漏。
- 生成中性的 cost feedback。
- 解析 evolver 输出的 emergent proposals。
- 给每个 proposal 生成 id，例如 `emergent.gen02.p03`。
- 记录 proposal 的来源、证据和后续命中。

建议新增输出：

```text
logs/emergence_*/
  results.json
  summary.json
  summary.csv
  emergence/
    leakage_audit.json
    proposals.jsonl
    proposal_adoption.json
    posthoc_taxonomy.json
    emergent_strategy_report.json
    workspace_diffs/
  figures/
```

## 11. 最小可行版本

第一版不需要让 agent 自动写复杂工具，只需要：

1. 禁用预定义 efficiency controller。
2. 给 evolver 一个中性 cost feedback block。
3. 要求 evolver 输出 `emergent_efficiency_proposals`。
4. 允许 evolver 把 proposal 写成普通 skill 或 prompt patch。
5. 下一 batch 正常使用 evolved workspace。
6. 结束后做 post-hoc 分类和 held-out 验证。

这能回答最基本的问题：即使没有预定义策略列表，evolver 是否会自己总结出“少读重复文件”、
“先定位再编辑”、“失败前压缩证据”、“接近 step limit 时提交最小 patch”等效率行为。

## 12. 预期结果解释

可能出现几类结果：

| 结果 | 解释 |
| --- | --- |
| B 无策略 | 当前成本反馈不足，或自进化写入机制不足以形成可复用效率规则 |
| B 有策略但无收益 | 策略只是文字总结，没有改变后续行为，或只对 train batch 有效 |
| B 成本降但成功率降 | 策略变成过早放弃，需要 success-first gate 或 rescue-first 限制 |
| B 出现未知新策略且 held-out 有效 | 值得加入正式 registry，并与预定义策略做 ablation |
| C 明显强于 B | 预定义策略仍有工程优势，自然涌现需要更强反馈或更多 batch |
| B 接近或超过 C | 成本信号足以诱导有效策略，说明预定义 registry 不是唯一来源 |

最终论文/报告中应把“自然涌现”和“工程选择预设策略”分开汇报：

- Emergence：策略从无标签环境中出现。
- Selection：预定义策略被 gate 选择。
- Validation：策略在 held-out 上保持成功率并节省成本。
