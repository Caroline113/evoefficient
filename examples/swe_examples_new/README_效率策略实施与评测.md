# 效率策略实施与评测说明

这个目录里的脚本，用来在 `SWE-bench Verified mini` 上测试效率策略。
重点有两件事：

1. 看单个策略省不省 token。
2. 看多个策略组合起来后，是否还能保持成功率。

## 一、改了什么

这次不是只做 SWE 专用的小修补，而是把效率层拆成了几块通用能力：

- 任务画像
- 策略提示词
- 工具输出裁剪
- 模型路由
- 计划模板缓存
- 评测和可视化

## 二、怎么跑单策略评测

先进入项目目录：

```powershell
cd E:\evoefficient\code\a-evolve-new
```

小批量试跑：

```powershell
python examples\swe_examples_new\run_efficiency_strategy_ablation.py `
  --limit 2 `
  --parallel 1 `
  --max-steps 60 `
  --model-id openai:qwen3.5-35b-a3b `
  --output-dir logs\evoefficient_new_strategy_smoke `
  --force
```

推荐的小批量：

```powershell
python examples\swe_examples_new\run_efficiency_strategy_ablation.py `
  --limit 8 `
  --parallel 2 `
  --max-steps 80 `
  --model-id openai:qwen3.5-35b-a3b `
  --output-dir logs\evoefficient_new_strategy_ablation `
  --force
```

默认会跑：

- baseline
- 每个新策略单独跑
- 预设组合
- 所有新策略一起跑

## 三、怎么换模型

如果要用 OpenAI 兼容接口，比如 `openai:qwen3.5-35b-a3b`，一般要同时准备：

- `AEVOLVE_OPENAI_API_KEY`
- `AEVOLVE_OPENAI_BASE_URL`

如果你想按任务难度换模型，可以加这些环境变量：

```text
AEVOLVE_ROUTE_FORCE_LEVEL=easy|hard
AEVOLVE_MODEL_EASY
AEVOLVE_MODEL_HARD
AEVOLVE_MODEL_PROVIDER_EASY
AEVOLVE_MODEL_PROVIDER_HARD
AEVOLVE_OPENAI_BASE_URL_EASY
AEVOLVE_OPENAI_BASE_URL_HARD
AEVOLVE_OPENROUTER_BASE_URL_EASY
AEVOLVE_OPENROUTER_BASE_URL_HARD
```

## 四、怎么跑 15.1 MVP

MVP 会先跑 baseline，再做策略选择，再做 gate 判断。

```powershell
python examples\swe_examples_new\run_evoefficient_mvp.py `
  --train-limit 4 `
  --reuse-limit 4 `
  --parallel 1 `
  --max-steps 80 `
  --model-id openai:qwen3.5-35b-a3b `
  --expected-future-hits 10 `
  --output-dir logs\evoefficient_mvp_new_strategies `
  --force
```

## 五、输出看哪里

每次运行后，主要看这几个文件：

- `REPORT.md`
- `summary.json`
- `summary.csv`
- `figures/strategy_avg_tokens.png`
- `figures/strategy_cost_success.png`
- `figures/strategy_tool_calls.png`
- `figures/strategy_efficiency_frontier.png`
- 每个组自己的 `results.json`

## 六、怎么看结果

- `avg_tokens` 越低越省。
- `success_rate` 不能掉太多。
- `cost_of_pass` 越低越好。
- `strategy_efficiency_frontier.png` 里越靠左上越好。

## 七、这套策略的通用思路

1. 先看任务画像。
2. 再决定要不要换模型、换提示词、裁剪工具输出。
3. 最后用实验结果判断是否真的省了成本。

这样就不会只看“前面花了多少”，而是能看“前面花的钱，后面有没有赚回来”。
