# OSWorld2 Emergent Efficiency with Harbor

本目录用于在 OSWorld 2.x 上运行 emergent-efficiency 实验。A-Evolve 负责
分批、workspace 进化、normal-evolve skill curation、cost-aware proposal
和结果汇总；真正的桌面环境执行与评分交给 Harbor。

支持的实验模式：

- `normal-evolve`
- `cost-aware`
- `predefined-control`
- `predefined-cost-aware`
- `category-seeded-cost-aware`

## 运行流程

每一批任务按下面的顺序执行：

1. 初始化 workspace，并把 OSWorld seed skills 复制到 `workspace/skills/seed/`。
2. 根据当前 `skills/seed`、`skills/topic`、`skills/general`、`skills/evolved` 生成给 Harbor agent 的 system prompt。
3. 对每个 OSWorld2 task 写入 Harbor 输入文件：
   - `harbor_inputs/<task>/task.json`
   - `harbor_inputs/<task>/system_prompt.md`
4. 调用 Harbor 运行任务并读取 Harbor 产出的 `result.json` 与 `trajectory.json`。
5. 将 Harbor 结果标准化为 A-Evolve result schema，写入 `results.json`。
6. 若不是最后一个 batch，先运行 OSWorld 风格的 normal evolution：
   - 收集 solver/Harbor 轨迹中的 skill proposal。
   - 按 topic 写入 `workspace/skills/topic/<topic>/<skill>/SKILL.md`。
   - 对多任务失败模式写入 `workspace/skills/general/<skill>/SKILL.md`。
   - 更新 `workspace/skills/SKILL_TREE.md`。
7. 对 `cost-aware`、`predefined-cost-aware`、`category-seeded-cost-aware`，在上述框架基础上继续运行 cost-aware proposal 与 adoption。
8. 写出 summary、behavior stats、figures、workspace snapshot。

## 需要提前准备什么

### Python 环境

在本仓库根目录使用项目环境运行：

```bash
.venv/bin/python -m py_compile examples/osworld2_examples/emergent_efficiency/run_emergent_efficiency_osworld2.py
.venv/bin/ruff check examples/osworld2_examples/emergent_efficiency
```

如果在本机没有 `.venv`，先按项目根目录的安装说明创建环境。GPU 服务器上通常使用：

```bash
cd /home/azureuser/zj/a-evolve-new7/a-evolve-new
.venv/bin/python --version
```

### OSWorld2 数据集

真实评测需要提前准备 OSWorld2 数据集或任务仓库。runner 不会自动下载 OSWorld2。

推荐放置结构如下：

```text
/path/to/OSWorld2/
  evaluation_examples/
    test_all.json
    <domain>/<task_id>.json
```

然后设置：

```bash
export OSWORLD2_PATH=/path/to/OSWorld2
export OSWORLD2_TASK_FILE="$OSWORLD2_PATH/evaluation_examples/test_all.json"
```

也可以不设置环境变量，直接在命令里传：

```bash
--task-file /path/to/OSWorld2/evaluation_examples/test_all.json
--osworld2-path /path/to/OSWorld2
```

如果只想验证 runner 流程，不需要 OSWorld2 数据集，可以使用本目录自带的
`smoke_tasks.json` 并加 `--dry-run-harbor`。

### Harbor

真实评测需要一个 Harbor checkout。runner 默认在 Harbor 仓库里执行：

```bash
uv run harbor run -c <generated_config.yaml>
```

准备步骤：

```bash
git clone <your-harbor-repo-url> /path/to/harbor
cd /path/to/harbor
uv sync
uv run harbor --help
uv run harbor run --help
```

Harbor 的 OSWorld agent 需要 Harbor task directory（目录中包含 `task.toml`、
`instruction.md`、`environment/`、`tests/` 等）。如果使用 Harbor 自带 OSWorld
adapter，可以先在 Harbor 仓库中导出数据集：

```bash
cd /path/to/harbor/adapters/osworld
uv run osworld --output-dir ../../datasets/osworld-verified --overwrite
```

然后把 `OSWORLD2_PATH` 或 `--osworld2-path` 指向导出的
`/path/to/harbor/datasets/osworld-verified`。如果你的 OSWorld2.0 数据集已经有 Harbor
task directory 结构，也可以直接指向该目录。

如果没有设置 `OSWORLD2_TASK_FILE` / `--task-file`，runner 会自动扫描
`OSWORLD2_PATH` 下所有包含 `task.toml` 的子目录作为任务列表。因此真实跑 Harbor
dataset 时不需要每次手写任务 index。常用选择方式：

```bash
# 跑整个 Harbor dataset
unset OSWORLD2_TASK_FILE
export OSWORLD2_PATH=/path/to/harbor/datasets/osworld-verified

# 只跑前 N 个
--limit 20

# 只跑指定 domain。Harbor OSWorld task 通常形如 chrome__<uuid>
--domain chrome

# 只跑指定任务，逗号分隔
--tasks chrome__task_id_1,libreoffice_calc__task_id_2

# 打乱后抽样
--shuffle --shuffle-seed 0 --limit 20
```

如果你有自己的 OSWorld2 task index，也仍然可以显式设置 `OSWORLD2_TASK_FILE`；
这种情况下 runner 优先使用 index，而不是自动扫描目录。

然后设置：

```bash
export HARBOR_REPO=/path/to/harbor
```

### OSWorld2 Harbor agent adapter

本实验已经提供了一个小的 custom adapter：

```bash
export OSWORLD2_HARBOR_AGENT_IMPORT_PATH='agent_evolve.agents.osworld2.harbor_adapter:AEvolveOSWorld2HarborAgent'
export OSWORLD2_HARBOR_DELEGATE_AGENT_IMPORT_PATH='osworld.custom_agent:OSWorldAgent'
```

外层 `AEvolveOSWorld2HarborAgent` 负责读取 A-Evolve workspace/system prompt/skills，
并把这些内容注入 Harbor OSWorld prompt template；真正的桌面执行、Docker/QEMU 环境和
评测仍交给 Harbor 自带的 `osworld.custom_agent:OSWorldAgent`。

runner 会把以下 kwargs 写进 Harbor config：

- `workspace_dir`
- `system_prompt_path`
- `skills_dir`
- `task_json_path`
- `task_config_path`
- `task_id`
- `domain`
- `max_steps`
- `aws_region_name`
- `delegate_agent_import_path`
- `delegate_model_name`

如果要覆盖 Harbor OSWorld agent 的参数，可以用 JSON/YAML 字符串：

```bash
export OSWORLD2_HARBOR_DELEGATE_KWARGS='{"client_password":"password","initial_observation_delay":60,"final_evaluation_delay":20}'
```

两种配置方式任选一种：

1. 直接传 agent import path：

```bash
export OSWORLD2_HARBOR_AGENT_IMPORT_PATH='agent_evolve.agents.osworld2.harbor_adapter:AEvolveOSWorld2HarborAgent'
```

2. 使用 Harbor config template：

```bash
--harbor-config-template /path/to/osworld2_harbor_template.yaml
```

如果 Harbor CLI 不是 `uv run harbor run -c <config>`，用：

```bash
--harbor-command-template 'uv run harbor run --launch osworld2 -c {config_path}'
```

如果服务器的非交互 shell 没有 `uv`，可以改用 Harbor venv 里的命令：

```bash
export HARBOR_UV_CMD='.venv/bin/harbor run'
```

常用 placeholders：

- `{config_path}`
- `{task_id}`
- `{task_json_path}`
- `{system_prompt_path}`
- `{workspace_dir}`
- `{skills_dir}`
- `{jobs_dir}`
- `{job_name}`
- `{model_id}`
- `{harbor_model_name}`
- `{osworld2_path}`

## 快速 smoke test

这个命令不会启动真实桌面或 Harbor，只验证 batching、workspace、skill tree、
summary 输出是否正常：

```bash
.venv/bin/python examples/osworld2_examples/emergent_efficiency/run_emergent_efficiency_osworld2.py \
  --mode normal-evolve \
  --task-file examples/osworld2_examples/emergent_efficiency/smoke_tasks.json \
  --limit 2 \
  --batch-size 1 \
  --parallel 1 \
  --dry-run-harbor \
  --dry-run-proposer \
  --fixed-output-dir \
  --overwrite \
  --output-dir logs/osworld2_emergent_efficiency/smoke
```

也可以使用脚本：

```bash
bash examples/osworld2_examples/emergent_efficiency/run_one.sh osworld2_smoke normal-evolve \
  --task-file examples/osworld2_examples/emergent_efficiency/smoke_tasks.json \
  --limit 2 \
  --batch-size 1 \
  --parallel 1 \
  --dry-run-harbor \
  --dry-run-proposer
```

## 真实 Harbor 运行

单个模式：

```bash
export OSWORLD2_PATH=/path/to/OSWorld2
export OSWORLD2_TASK_FILE="$OSWORLD2_PATH/evaluation_examples/test_all.json"
export HARBOR_REPO=/path/to/harbor
export OSWORLD2_HARBOR_AGENT_IMPORT_PATH='agent_evolve.agents.osworld2.harbor_adapter:AEvolveOSWorld2HarborAgent'
export OSWORLD2_HARBOR_DELEGATE_AGENT_IMPORT_PATH='osworld.custom_agent:OSWorldAgent'
export HARBOR_UV_CMD='.venv/bin/harbor run'

bash examples/osworld2_examples/emergent_efficiency/run_one.sh osworld2_real cost-aware \
  --limit 10 \
  --batch-size 2 \
  --parallel 1 \
  --model-id us.anthropic.claude-opus-4-6-v1 \
  --evolver-model us.anthropic.claude-sonnet-4-5-20250929-v1:0
```

完整五模式矩阵：

```bash
bash examples/osworld2_examples/emergent_efficiency/run_matrix.sh osworld2_matrix \
  --limit 20 \
  --batch-size 2 \
  --parallel 1
```

只跑某些 domain 或 task：

```bash
bash examples/osworld2_examples/emergent_efficiency/run_one.sh osworld2_calc normal-evolve \
  --domain libreoffice_calc \
  --tasks task_id_1,task_id_2 \
  --limit 4 \
  --batch-size 2
```

## 常用参数

- `--task-file`: OSWorld2 task index 文件。
- `--osworld2-path`: OSWorld2 数据集根目录。
- `--domain`: 逗号分隔的 domain filter。
- `--tasks`: 逗号分隔的 task id filter。
- `--limit`: 最多运行多少个 task。
- `--batch-size`: 每批多少个 task。
- `--parallel`: 并行 Harbor job 数。GUI 桌面任务建议先用 `1`。
- `--dry-run-harbor`: 不调用 Harbor，生成合成结果。
- `--dry-run-proposer`: 不调用 evolver/curator 模型。
- `--max-skills-per-topic`: 每个 topic 下最多保留多少 skill。
- `--max-general-skills`: general skill 上限。
- `--evolve-all`: 允许从 passed task 中也提炼 skill。
- `--harbor-jobs-dir`: 指定 Harbor job 输出目录。
- `--harbor-timeout-sec`: 单个 Harbor 命令超时。

## 输出目录

默认输出在：

```text
logs/osworld2_emergent_efficiency/<run_name>_<mode>_<timestamp>/
```

主要文件：

- `experiment_manifest.json`: 本次运行配置。
- `results.json`: 标准化后的 per-task 结果。
- `summary.json` / `summary.csv`: batch 和 overall 指标。
- `behavior_stats.json`: GUI 行为统计。
- `workspace/skills/seed`: OSWorld seed skills。
- `workspace/skills/topic`: normal-evolve 按 topic 产出的 skills。
- `workspace/skills/general`: normal-evolve 产出的跨任务 general skills。
- `workspace/skills/SKILL_TREE.md`: skill 索引。
- `harbor_inputs/`: 每个 task 的 Harbor 输入。
- `harbor_jobs/`: Harbor job 输出，除非指定 `--harbor-jobs-dir`。
- `raw_harbor_results/`: runner 解析前的 Harbor 原始结果。
- `emergence/normal_evolution.jsonl`: OSWorld-style normal evolution 记录。
- `emergence/proposals.jsonl`: cost-aware proposal 记录。
- `emergence/proposal_adoption.json`: cost-aware adoption 记录。
- `figures/`: 图表输出。

## 服务器运行提示

如果本地工作区是挂载到 GPU 服务器的 `E:\codex-work`，可以直接在服务器项目目录运行：

```bash
cd /home/azureuser/zj/a-evolve-new7/a-evolve-new
.venv/bin/python examples/osworld2_examples/emergent_efficiency/run_emergent_efficiency_osworld2.py ...
```

或者从本地通过 SSH 包装脚本执行。注意当前脚本里的 SSH alias 必须能解析；如果
`gpu` alias 不存在，就需要用实际 host、端口和 key 连接服务器。
