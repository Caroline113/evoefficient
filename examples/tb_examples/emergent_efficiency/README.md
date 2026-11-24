# Terminal-Bench Emergent Efficiency

本目录包含 Terminal-Bench 2.1 的 emergent-efficiency 实验。当前有两条评测路径：

1. Docker 原版路径：
   - runner: `run_emergent_efficiency_tb.py`
   - scripts: `run_one.sh`, `run_matrix.sh`
   - 行为：加载 Terminal-Bench task 后，由本仓库的 Terminal agent 直接 `pull_image`、
     启动 `TB2Container`，再在容器内运行 tests。
2. Harbor 路径：
   - runner: `run_emergent_efficiency_tb_harbor.py`
   - scripts: `run_one_harbor.sh`, `run_matrix_harbor.sh`
   - 行为：A-Evolve 仍负责 batching、workspace evolution、normal-evolve 和
     cost-aware proposal；评测交给 Harbor。这个 runner 不直接 pull Docker image，
     不直接调用 `TB2Container`。

支持的实验模式：

- `normal-evolve`
- `cost-aware`
- `predefined-control`
- `predefined-cost-aware`
- `category-seeded-cost-aware`

## 原 Docker 路径是否依赖下载好的镜像

是。原 runner 会走下面的链路：

- `agent_evolve/agents/terminal/dataset.py` 从 `task.toml` 或 `compose.yaml` 读取 `docker_image`。
- `examples/tb_examples/batch_evolve_terminal.py` 调用 `pull_image(task.docker_image)`。
- `agent_evolve/agents/terminal/docker_env.py` 先 `docker image inspect`，不存在时 `docker pull`。
- `TB2Container.start()` 使用该镜像启动容器。
- `run_tests_with_retry()` 在容器中执行 task 的 tests。

如果要完全避免 runner 直接管理镜像和容器，请使用 Harbor 路径。

## 运行流程

两条路径共享 A-Evolve 外层流程：

1. 加载 Terminal-Bench 2.1 tasks。
2. 复制 seed workspace 到输出目录下的 `workspace/`。
3. 按 batch 运行任务。
4. 写入 `results.json`、`summary.json`、`summary.csv`、`behavior_stats.json`。
5. 若不是最后一个 batch，执行 normal evolution，把 solver proposal 写入 workspace。
6. 对 `cost-aware`、`predefined-cost-aware`、`category-seeded-cost-aware`，继续执行 cost-aware proposal adoption。
7. 保存 workspace snapshot 和 proposal records。

Harbor 路径的 batch evaluation 会额外做：

1. 为每个 task 写入：
   - `harbor_inputs/<task>/task.json`
   - `harbor_inputs/<task>/system_prompt.md`
2. 生成临时 Harbor config。
3. 在 `HARBOR_REPO` 中执行 Harbor 命令，默认：
   - `uv run harbor run -c <generated_config.yaml>`
4. 解析 Harbor 产出的 `result.json` 与 `trajectory.json`。
5. 在标准化结果里标记：
   - `metadata.execution_backend = "harbor"`
   - `metadata.docker_image_not_pulled_by_runner = true`

## 需要提前准备什么

### Python 环境

在本仓库根目录使用项目环境：

```bash
.venv/bin/python -m py_compile examples/tb_examples/emergent_efficiency/run_emergent_efficiency_tb.py
.venv/bin/python -m py_compile examples/tb_examples/emergent_efficiency/run_emergent_efficiency_tb_harbor.py
.venv/bin/ruff check examples/tb_examples/emergent_efficiency
```

### Terminal-Bench 2.1 tasks

默认路径是：

```text
agent_evolve/benchmarks/tb21/tasks
```

当 `--challenges-dir` 为空时，loader 会尝试从固定的 Terminal-Bench 2.1
仓库自动下载 tasks 到默认路径。服务器不能联网或希望固定版本时，建议提前下载：

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/harbor-framework/terminal-bench-2-1.git \
  /path/to/terminal-bench-2-1
cd /path/to/terminal-bench-2-1
git sparse-checkout set tasks
```

然后运行时传：

```bash
--challenges-dir /path/to/terminal-bench-2-1/tasks
```

注意：当前 loader 仍会读取 `environment.docker_image`、`instruction.md`、
`tests/test.sh` 等字段，即使 Harbor 路径不直接 pull 镜像。如果某些 task 缺这些字段，
会被 loader 跳过并打印 warning。

### Harbor

Harbor 路径需要提前准备 Harbor checkout：

```bash
git clone <your-harbor-repo-url> /path/to/harbor
cd /path/to/harbor
uv sync
uv run harbor --help
uv run harbor run --help
```

设置：

```bash
export HARBOR_REPO=/path/to/harbor
```

默认 Harbor 命令是 `uv run harbor run -c <config>`。如果服务器的非交互 shell
里没有 `uv`，但 Harbor repo 已经 `uv sync` 过，可以改用 Harbor venv 里的 CLI：

```bash
export HARBOR_UV_CMD=".venv/bin/harbor run"
```

### Terminal-Bench Harbor agent adapter

本仓库已经提供一个小的 custom adapter：

```bash
agent_evolve.agents.terminal.harbor_adapter:AEvolveTerminalHarborAgent
```

它的作用是：

1. 读取本 runner 生成的 `system_prompt_path`、`workspace_dir`、`task_json_path`。
2. 把 A-Evolve 当前 workspace 的 prompt、skills、fragments 拼进 Terminal-Bench task instruction。
3. 委托给 Harbor 内置 agent 真正执行任务。默认委托 agent 是 `codex`，可用
   `TB_HARBOR_DELEGATE_AGENT_NAME` 或 `--harbor-delegate-agent-name` 改成
   `claude-code`、`openhands`、`terminus-2` 等 Harbor 已安装 agent。

推荐配置：

```bash
export TB_HARBOR_AGENT_IMPORT_PATH='agent_evolve.agents.terminal.harbor_adapter:AEvolveTerminalHarborAgent'
export TB_HARBOR_DELEGATE_AGENT_NAME=codex
```

runner 会在调用 Harbor 时把当前 A-Evolve repo 加进 `PYTHONPATH`，所以 Harbor
可以 import 到上面的 adapter。

runner 会给 adapter 传：

- `workspace_dir`
- `system_prompt_path`
- `skills_dir`
- `task_json_path`
- `task_name`
- `max_steps`
- `aws_region_name`
- `delegate_agent_name`
- `delegate_model_name`

同时 Harbor dataset kwargs 中包含：

- `task_name`
- `task_dir`
- `task_json_path`
- `docker_image`
- `test_sh_path`
- `tests_dir`

如果想委托给另一个 custom Harbor agent，而不是 Harbor 内置 agent，可以设置：

```bash
export TB_HARBOR_DELEGATE_AGENT_IMPORT_PATH='your_package.your_module:YourAgent'
```

如果你完全不想使用本 adapter，也可以使用 Harbor config template：

```bash
--harbor-config-template /path/to/tb21_harbor_template.yaml
```

如果 Harbor CLI 不是默认形式，用：

```bash
--harbor-command-template 'uv run harbor run --suite tb21 -c {config_path}'
```

常用 placeholders：

- `{config_path}`
- `{task_name}`
- `{task_json_path}`
- `{system_prompt_path}`
- `{workspace_dir}`
- `{jobs_dir}`
- `{job_name}`
- `{model_id}`
- `{harbor_model_name}`
- `{task_dir}`

## Smoke test

Harbor dry-run 不需要启动 Harbor，也不会 pull Docker image：

```bash
bash examples/tb_examples/emergent_efficiency/run_one_harbor.sh tb_harbor_smoke normal-evolve \
  --challenges-dir agent_evolve/benchmarks/tb21/tasks \
  --tasks count-dataset-tokens \
  --limit 1 \
  --batch-size 1 \
  --parallel 1 \
  --dry-run-harbor \
  --dry-run-proposer
```

如果默认 tasks 目录为空，并且服务器能联网，也可以不传 `--challenges-dir`，让 loader
自动下载 Terminal-Bench 2.1 tasks。

## 真实 Harbor 运行

单个模式：

```bash
export HARBOR_REPO=/path/to/harbor
export TB_HARBOR_AGENT_IMPORT_PATH='agent_evolve.agents.terminal.harbor_adapter:AEvolveTerminalHarborAgent'
export TB_HARBOR_DELEGATE_AGENT_NAME=codex
export HARBOR_UV_CMD=".venv/bin/harbor run"
export TB21_TASKS_DIR=/path/to/terminal-bench-2-1/tasks

bash examples/tb_examples/emergent_efficiency/run_one_harbor.sh tb21_harbor cost-aware \
  --limit 10 \
  --batch-size 2 \
  --parallel 1 \
  --model-id deepseek:deepseek-v4-flash
```

完整五模式矩阵：

```bash
bash examples/tb_examples/emergent_efficiency/run_matrix_harbor.sh tb21_harbor_matrix \
  --limit 20 \
  --batch-size 2 \
  --parallel 1
```

只跑指定 task：

```bash
bash examples/tb_examples/emergent_efficiency/run_one_harbor.sh tb21_one normal-evolve \
  --tasks count-dataset-tokens \
  --batch-size 1 \
  --parallel 1
```

## 原 Docker 路径运行

如果你仍要使用原始 Docker 评测路径：

```bash
export DEEPSEEK_API_KEY=...
bash examples/tb_examples/emergent_efficiency/run_one.sh tb21_docker cost-aware \
  --challenges-dir /path/to/terminal-bench-2-1/tasks \
  --limit 10 \
  --batch-size 2 \
  --parallel 2
```

这条路径会直接使用 Docker 镜像并在本机/服务器 Docker 中跑 tests。

## 常用参数

- `--challenges-dir`: Terminal-Bench tasks 目录。
- `--tasks`: 逗号分隔的 task name。
- `--exclude`: 排除 task name。
- `--category`: 按 metadata category 过滤。
- `--difficulty`: 按 difficulty 过滤。
- `--limit`: 最多运行多少个 task。
- `--batch-size`: 每批多少个 task。
- `--parallel`: 并行任务数。真实 Harbor 评测建议先用 `1`。
- `--dry-run-harbor`: Harbor 路径专用，不调用 Harbor。
- `--dry-run-proposer`: 不调用 evolver/curator 模型。
- `--harbor-repo`: Harbor checkout。
- `--harbor-agent-import-path`: Harbor agent adapter。
- `--harbor-delegate-agent-name`: A-Evolve adapter 委托的 Harbor 内置 agent，默认 `codex`。
- `--harbor-delegate-agent-import-path`: A-Evolve adapter 委托的自定义 Harbor agent。
- `--harbor-delegate-model-name`: 委托 agent 使用的 model name，默认继承 Harbor/model 配置。
- `--harbor-config-template`: Harbor config 模板。
- `--harbor-command-template`: 自定义 Harbor CLI。
- `--harbor-jobs-dir`: Harbor job 输出目录。
- `--harbor-timeout-sec`: 单个 Harbor 命令超时。

## 输出目录

Docker 原版默认：

```text
logs/tb_emergent_efficiency/<run_name>_<mode>_<timestamp>/
```

Harbor 版本默认：

```text
logs/tb_emergent_efficiency_harbor/<run_name>_<mode>_<timestamp>/
```

主要文件：

- `experiment_manifest.json`: 本次运行配置。
- `results.json`: 标准化 per-task 结果。
- `summary.json` / `summary.csv`: batch 和 overall 指标。
- `behavior_stats.json`: 命令、测试、重复行为统计。
- `conversations/`: per-task conversation。
- `system_prompts/`: per-task system prompt。
- `harbor_inputs/`: Harbor 路径的 task/system prompt 输入。
- `harbor_jobs/`: Harbor job 输出，除非指定 `--harbor-jobs-dir`。
- `raw_harbor_results/`: Harbor 路径的原始结果记录。
- `emergence/proposals.jsonl`: cost-aware proposal 记录。
- `emergence/proposal_adoption.json`: proposal adoption 记录。
- `emergence/workspace_snapshots/`: 每个 evolution batch 后的 workspace snapshot。

## 服务器运行提示

在 GPU 服务器上通常这样运行：

```bash
cd /home/azureuser/zj/a-evolve-new7/a-evolve-new
.venv/bin/python examples/tb_examples/emergent_efficiency/run_emergent_efficiency_tb_harbor.py ...
```

如果从本地调用服务器脚本，确保 SSH alias 或 host 配置可用。当前工作区通过 SSHFS
挂载时，文件修改会同步到服务器项目目录。
