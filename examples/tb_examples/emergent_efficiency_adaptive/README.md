# Terminal-Bench Adaptive Emergent Efficiency

This directory is an additive variant of `examples/tb_examples/emergent_efficiency`.
It keeps Harbor evaluation, but changes the normal evolution baseline to the
`AdaptiveSkillEngine` flow used by `examples/tb_examples/batch_evolve_terminal.py`.

## What Is Different

- Task execution is still delegated to Harbor.
- The solver prompt/workspace path is still the Terminal-Bench workspace path.
- After each non-final batch, normal evolution uses `AdaptiveSkillEngine` by
  default, or `MasAdaptiveSkillEngine` with `--evolver mas_adaptive_skill`.
- Default adaptive flags for this variant are:
  `trajectory_only=True`, `skills_only=False`, `protect_skills=True`,
  `max_skills=10`.
- Cost-aware modes run after the adaptive normal evolution step, so they are
  layered on top of the AdaptiveSkillEngine baseline.

## Modes

- `static-control` (fixed initial agent, no evolution)
- `normal-evolve`
- `cost-aware`
- `predefined-control`
- `predefined-cost-aware`
- `category-seeded-cost-aware`

## Quick Smoke Test

Use one task and one batch to validate wiring without running evolution:

```bash
bash examples/tb_examples/emergent_efficiency_adaptive/run_one_harbor.sh smoke normal-evolve \
  --challenges-dir agent_evolve/benchmarks/tb21/tasks \
  --limit 1 \
  --batch-size 1 \
  --parallel 1 \
  --dry-run-harbor
```

Use two one-task batches to validate batching without paid evolution:

```bash
bash examples/tb_examples/emergent_efficiency_adaptive/run_one_harbor.sh smoke normal-evolve \
  --challenges-dir /home/azureuser/zj/terminal-bench-2-1/tasks \
  --limit 2 \
  --batch-size 1 \
  --parallel 1 \
  --dry-run-harbor
```

For real Harbor runs, set the same Harbor variables as the original runner:

```bash
export HARBOR_REPO=/path/to/harbor
export TB_HARBOR_AGENT_IMPORT_PATH='agent_evolve.agents.terminal.harbor_adapter:AEvolveTerminalHarborAgent'
export TB_HARBOR_DELEGATE_AGENT_NAME=codex
export HARBOR_UV_CMD='.venv/bin/harbor run'
```

Then run:

```bash
bash examples/tb_examples/emergent_efficiency_adaptive/run_one_harbor.sh tb21_adaptive normal-evolve \
  --challenges-dir /path/to/terminal-bench-2-1/tasks \
  --limit 10 \
  --batch-size 2 \
  --parallel 1
```

Outputs are written under:

```text
logs/tb_emergent_efficiency_adaptive_harbor/<run_name>_<mode>/
```

## 2026-11-01 更新：无进化基线、新任务和失败感知节省

### 无进化基线

新增 `--mode static-control`：每个任务使用相同的初始 seed workspace，不执行普通进化，也不生成效率提案；保留正常求解、官方评分、轨迹与用量记录。它不是 `normal-evolve` 的别名。已有模式的普通进化流程按此前实现保留；新增静态基线不改变其他模式的方法。

矩阵默认运行：`static-control → normal-evolve → cost-aware`。可设置
`TB_ADAPTIVE_MODES=static-control,normal-evolve,cost-aware,predefined-control` 调整组别。

### 默认选择未跑过的任务

- 默认读取 `logs/tb_emergent_efficiency_adaptive_harbor/*/results.json` 中的实际任务记录，成功、失败、超时尝试都排除。只出现在 manifest 中但还没产生结果的任务不算已跑。
- 标记 `dry_run_harbor=true` / synthetic 的运行不进入历史集合。不要把干跑成绩当作正式实验结果。
- **先排除历史任务，再应用 `--limit`**，避免前几个任务已跑就选不到后续新任务。
- `--history-root PATH` 可指定其他历史根目录或单个 results.json；重复指定可合并多个来源。指定后替代默认根目录，需合并默认目录时也显式传入它。
- `--allow-rerun` 明确允许历史任务重跑。即使手动指定 `--tasks`，默认仍不会绕过历史过滤。
- `--plan-only` 只输出计划，不复制工作区、不启动 Harbor、不调用模型。
- `--selection-file FILE` 首次保存任务顺序和历史审计，以后精确复用。冻结文件中的任务清单优先于 limit/shuffle；任务必须仍存在且满足类别/任务名过滤。

先预览下一批任务：

```bash
python examples/tb_examples/emergent_efficiency_adaptive/run_emergent_efficiency_tb_adaptive_harbor.py \
  --challenges-dir /home/azureuser/zj/terminal-bench-2-1/tasks \
  --plan-only --limit 10
```

**矩阵比较的公平性：**脚本先从历史快照挑选一批新任务，将清单保存在 `_task_selections/`，之后三组共用这批任务。后续组不会因为第一组刚执行过这些任务而被迫换题。这是有意的同题对照；独立启动的下一次实验则继续默认排除历史任务。并发启动多个独立矩阵没有全局任务预约锁，请避免同时规划相同历史快照。

如需复用一次已经冻结的矩阵：

```bash
export TB_ADAPTIVE_SELECTION_FILE=/绝对路径/已有任务清单.json
bash examples/tb_examples/emergent_efficiency_adaptive/run_matrix_harbor.sh resume_comparison \
  --challenges-dir /home/azureuser/zj/terminal-bench-2-1/tasks
```

### cost-aware 对失败任务的进化反馈

每批生成 `emergence/batch_XX_failure_tokens.json`，整组生成 `failure_token_distribution.json`：

- 按成功/失败分组，比较非缓存输入、缓存读取和输出 token 的数量与比例。
- 失败任务按总 token 排序，显示末段工具调用、原始用量质量和提前停止记录。
- 有 Codex session 原始记录时，补回真实工具轨迹，按累计用量更新序列分四段分析消耗；没有记录则标记未知。
- 四段按**模型用量更新次数**划分，不代表四段墙钟时间，也不能当成某条工具命令的精确账单。推理 token 已包含在输出中，缓存读取也不重复相加。
- 效率 proposer 会明确关注失败上的上下文膨胀、长输出、无效重试和后半段高消耗，并要求给出可观察的退出条件、最低探索和一次有意义的恢复尝试；不能根据任务名字或一次失败断言不可能完成。

普通 AdaptiveSkillEngine 的 `trajectory_only` 语义保持不变；官方失败标签与上述 token 对比提供给 cost-aware 的**额外效率进化阶段**，不会偷偷把官方分数注入普通 trajectory-only 进化。

### early stop 是额外效率策略，而非外部控制器

本次补充只在原 cost-aware 效率进化提示和原反馈后追加内容：**原有优化目标、已有方法、提案类型和应用流程全部保留**。失败任务 token 分布提供额外观察维度，不要求只优化失败任务，也不替换成功任务上的优化机会。

early stop 与其他效率策略并列：进化 agent 可根据轨迹提出可复用的识别特征、触发/不触发条件、有限恢复尝试和退出诊断，然后经过原有“生成提案 → 采用 → 应用到 workspace”流程，由求解 agent 自主判断是否采用。

示例特征包括关键资源确实不可用，或经过若干不同的合理恢复方案后仍反复处于相同失败状态、没有新信息。单次失败、任务名字或缓存 token 多不能单独证明无法完成；仍有明确可行修复时应继续。策略采用后若主动结束，应保留部分工作并诚实报告剩余阻碍，原官方 verifier 决定成功与否。

**不增加外部进程监控、强制中断、固定 token 截断或统一重试阈值，也不预先向每个求解提示硬塞停止规则。**原 Harbor 委托执行、既有超时和评分方式保持。上一轮新增的 `--early-stop-*` / `--no-early-stop` 运行时控制参数已撤回。

失败分析读取原始 session 仅用于附加统计，不覆写原始 steps、conversation、调用计数及普通进化输入。普通自适应进化和其他效率提案仍走原有实现。

### 正式实验示例

沿用前文 HARBOR_REPO、HARBOR_UV_CMD 和模型服务配置，在服务器项目根目录运行：

```bash
# 三组同题对照，自动选择此前未跑过的10个可加载任务
bash examples/tb_examples/emergent_efficiency_adaptive/run_matrix_harbor.sh tb_new10 \
  --challenges-dir /home/azureuser/zj/terminal-bench-2-1/tasks \
  --limit 10 --batch-size 2 --parallel 2

# 只跑无进化基线
bash examples/tb_examples/emergent_efficiency_adaptive/run_one_harbor.sh baseline10 static-control \
  --challenges-dir /home/azureuser/zj/terminal-bench-2-1/tasks --limit 10

# 只跑cost-aware：在原效率进化上增加失败token关注与可选early-stop策略
bash examples/tb_examples/emergent_efficiency_adaptive/run_one_harbor.sh cost_extra_focus cost-aware \
  --challenges-dir /home/azureuser/zj/terminal-bench-2-1/tasks --limit 10
```

任务目录应使用实际完整的 Terminal-Bench 2.1 challenges。当前项目的通用 loader 会提示并跳过缺少 instruction/test 或无法识别镜像配置的任务；这与历史过滤不同，应先查看 `--plan-only` 确认可运行任务清单。

### 无付费自检

```bash
# --dry-run-harbor 现在同时跳过普通进化，并强制额外proposer使用占位提案，不调用付费模型
bash examples/tb_examples/emergent_efficiency_adaptive/run_matrix_harbor.sh smoke_new_controls \
  --dry-run-harbor --allow-rerun --limit 2 --batch-size 1 --parallel 1

python -m pytest examples/tb_examples/emergent_efficiency_adaptive/tests tests/test_harbor_usage.py -q
```

`--dry-run-proposer` 单独使用时只替换额外效率proposer；真实普通进化仍可能调用模型。只有 `--dry-run-harbor` 的上述完整自检保证整条流程不调用付费模型。


### 修正说明

按用户澄清，本轮已撤回外部强制停止实现；以本 README 的“额外效率策略”描述为准。先前的 `VALIDATION_20261101.md` 属于已撤回实现的历史验证，不代表当前机制。当前验证见 `VALIDATION_STRATEGY_ONLY.md`。
