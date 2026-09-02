# Token、Cost、zero_token_task 与可视化说明

本文档对应 `examples/swe_examples_new/evolve_sequential.py` 的顺序演化实验，以及
`agent_evolve/efficiency/profiler.py` 的指标口径。

## 指标来源

SWE agent 在每个任务结束时，会在 trajectory 最后追加一条 summary step。该 step
包含：

- `usage`：主 agent LLM 调用的累计 token/cost。
- `per_turn_usage`：逐轮 token 记录。
- `route_model_calls`：`difficulty_model_routing` 中显式调用 `route_model` 的 delegated model 记录。
- `usage_record_quality`：`ok`、`estimated` 或 `missing`。
- `usage_source`：usage 取自哪个来源，例如 `response_metrics`、`agent_metrics`、`agent_messages_metadata` 或启发式估算。

`evolve_sequential.py` 现在从 trajectory 尾部反向查找最后一条含 `usage` 的 step，
避免误读中间 step 或过期统计。

## Token 口径

`profiler.result_tokens()` 使用 `total_tokens`，语义是主 agent 的
`input_tokens + output_tokens`，不包含 cache token。

`summarize_results()` 中的主要 token 字段：

- `tokens`：所有任务主 agent 的 `total_tokens` 总和。
- `input_tokens` / `output_tokens`：主 agent 输入/输出 token 总和。
- `cache_read_input_tokens` / `cache_write_input_tokens`：cache read/write token 单独列出。
- `avg_tokens`：`tokens / tasks`。
- `avg_tokens_success`：成功任务上的平均主 agent token。

## Cost 口径

`profiler.result_cost()` 使用 `cost_units`。有价格表时按实际价格计算；没有价格表时回退为
token-equivalent cost。

`cost_units` 包含：

- input token cost
- output token cost
- cache read token cost
- cache write token cost
- `route_model` delegated calls 的 cost

`summarize_results()` 额外拆分：

- `main_model_cost`：主 hard-model agent loop 的 cost。
- `route_model_cost`：delegated `route_model` 调用的 cost。
- `easy_model_cost`：`route_model_calls` 中 `route_level="easy"` 的 cost。
- `hard_model_cost`：主 agent cost 加 hard route 记录 cost。
- `cost_of_pass`：`total_cost / success`，无成功任务时为 `inf`。

本次修复了一个鲁棒性问题：如果结果只保留顶层 `cost_units`、`main_cost_units`、
`route_model_cost_units` 和 `route_model_calls`，但不保留完整 `steps`，profiler
不会再把 route-model cost 重复加到 total cost。

## zero_token_task 处理

`zero_token_tasks` 表示 `result_tokens(result) == 0` 的任务数，也就是主 agent token
完全没有记录到。

正常情况下应尽量避免该情况。当前链路有三层兜底：

1. 优先读取 response metrics。
2. 如果 response metrics 缺失，读取 agent metrics 或 message metadata。
3. 如果 provider 完全不返回 usage，则根据 system prompt、user prompt、conversation
   和 tool trace 做保守估算，并标记：
   - `usage_record_quality="estimated"`
   - `usage_estimated=True`
   - `usage_source="text_heuristic_after_missing_provider_usage"`

因此，只有在 agent 在生成 trajectory 前就异常退出、或者没有任何可估算文本上下文时，
才应该出现 `zero_token_tasks > 0`。这类任务会标记为 `usage_record_quality="missing"`，
方便后续排查，而不会被伪装成真实低成本任务。

## 顺序实验输出

运行：

```bash
python examples/swe_examples_new/evolve_sequential.py
```

实验结束后会输出：

- `results.json`：逐任务完整结果。
- `summary.json`：overall 与按 evolution generation 分组的汇总。
- `summary.csv`：适合表格查看的汇总，包含 `zero_token_tasks`、`zero_cost_tasks`、
  `estimated_usage_tasks` 等质量字段。
- `figures/strategy_avg_tokens.png`
- `figures/strategy_avg_cost.png`
- `figures/strategy_cost_success.png`
- `figures/strategy_tool_calls.png`
- `figures/strategy_efficiency_frontier.png`
- `figures/strategy_hits.png`

可视化使用 matplotlib。脚本会把 `MPLCONFIGDIR` 设置到输出目录下，避免默认
`/root/.config/matplotlib` 不可写导致缓存警告。如果图表渲染失败，主实验结果和
CSV/JSON 仍会正常保存，并打印 warning。

## 运行前检查

轻量检查：

```bash
python3 -m py_compile \
  agent_evolve/efficiency/profiler.py \
  agent_evolve/efficiency/visualization.py \
  examples/swe_examples_new/evolve_sequential.py

python3 examples/swe_examples_new/evolve_sequential.py --help
```

完整实验还需要实际 SWE/Strands 运行依赖、模型访问配置、数据集访问和 seed workspace。
当前代码把 `SweAgent`、benchmark、evolver 的导入放在 argparse 之后，因此 `--help`
和静态检查不依赖这些运行时依赖。
