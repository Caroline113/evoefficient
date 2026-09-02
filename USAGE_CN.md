# A-Evolve 中文使用说明

这份说明面向“刚拿到仓库，不知道从哪里运行”的情况。仓库路径按你当前机器上的位置写：

```bash
cd /mnt/data/zj/evolve/a-evolve
```

本文优先使用仓库里已经提供的 `examples/*` 脚本，因为它们把模型、数据集、输出目录、断点续跑和演化工作区都处理好了。`README.md` 里的三行 `ae.Evolver(...)` 更适合二次开发或接入自定义 agent，不建议作为第一次跑通仓库的入口。

## 1. 这个仓库在做什么

A-Evolve 是一个“让 agent 自动进化”的框架。你给它一个 agent 工作区，里面有 prompt、skills、tools、memory 等文件；它让 agent 在 benchmark 上做题，收集失败轨迹，再调用 evolver 修改这些文件，下一轮继续评测。

核心循环是：

```text
solve tasks -> collect trajectories/feedback -> evolve workspace files -> reload agent -> repeat
```

仓库里已经带了几类可运行示例：

| 示例 | 目录 | 主要依赖 | 适合第一次跑吗 |
|---|---|---|---|
| MCP-Atlas | `examples/mcp_examples` | Docker、AWS Bedrock 或其他 LLM provider、可选 MCP API keys | 推荐 |
| SWE-bench | `examples/swe_examples` | Docker、HuggingFace dataset、AWS Bedrock | 推荐，但更慢 |
| Terminal-Bench 2.0 | `examples/tb_examples` | Docker、`uv`、AWS Bedrock | 推荐，小样本可跑 |
| SkillBench | `examples/skillbench_examples` | Docker、SkillBench 任务数据、AWS Bedrock | 有数据后再跑 |
| OSWorld | `examples/osworld_examples` | OSWorld、AWS EC2、桌面环境 | 不建议第一次跑 |
| ARC-AGI-3 | `examples/arc_examples` | Python 3.12、`arc-agi`、AWS Bedrock | 当前分支相关，可单独跑 |

## 2. 环境准备

### 2.1 Python

主项目要求 Python 3.11+。建议用 `uv` 创建环境：

```bash
cd /mnt/data/zj/evolve/a-evolve
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[all,dev]"
```

如果没有 `uv`，可以用标准 venv：

```bash
cd /mnt/data/zj/evolve/a-evolve
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[all,dev]"
```

按 benchmark 只装一部分依赖也可以：

```bash
pip install -e ".[mcp,dev]"
pip install -e ".[swe,dev]"
pip install -e ".[skillbench,dev]"
pip install -e ".[osworld,dev]"
```

`examples/tb_examples/*.sh` 和部分 wrapper 会直接调用 `uv run python`，所以如果你要跑 Terminal-Bench，强烈建议安装 `uv`。

### 2.2 验证安装

从仓库根目录执行：

```bash
python - <<'PY'
import agent_evolve as ae
print(ae.__version__)
print(ae.Evolver)
print(ae.BaseAgent)
PY
```

能打印版本和类名就说明包导入正常。

可以再跑几个轻量测试：

```bash
pytest tests/test_unified_engine.py -q
pytest tests/gepa -q
```

### 2.3 Docker

MCP-Atlas、SWE-bench、Terminal-Bench、SkillBench 都会用到 Docker：

```bash
docker info
```

MCP-Atlas 建议先拉镜像：

```bash
docker pull ghcr.io/scaleapi/mcp-atlas:latest
```

如果遇到 `permission denied while trying to connect to the Docker daemon`，说明当前用户没有 Docker 权限。需要把用户加入 docker 组，或用有 Docker 权限的 shell 运行。

### 2.4 模型和凭证

仓库里的示例大多默认使用 AWS Bedrock Claude 模型。至少需要配置 AWS 凭证和 region：

```bash
export AWS_DEFAULT_REGION=us-west-2
# 二选一：使用 profile
export AWS_PROFILE=your_profile

# 或直接导出 key
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

常见默认模型变量：

```bash
export MODEL_ID=us.anthropic.claude-opus-4-6-v1
export EVOLVER_MODEL_ID=us.anthropic.claude-sonnet-4-6
export REGION=us-west-2
```

如果不想走 AWS Bedrock，可以统一切到 OpenRouter。推荐在运行前导出：

```bash
export OPENROUTER_API_KEY=sk-or-...
export AEVOLVE_MODEL_PROVIDER=openrouter
export OPENROUTER_MODEL=anthropic/claude-sonnet-4
export OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# 可选，用于 OpenRouter 控制台归因
export OPENROUTER_SITE_URL=http://localhost
export OPENROUTER_APP_NAME=a-evolve
```

也可以只给某个参数加前缀：

```bash
--model-id openrouter:anthropic/claude-sonnet-4
--solver-model openrouter:anthropic/claude-sonnet-4
--judge-model openrouter:anthropic/claude-sonnet-4
--evolver-model-id openrouter:anthropic/claude-sonnet-4
```

模型名使用 OpenRouter 的模型 slug，例如 `anthropic/claude-sonnet-4`、`openai/gpt-5.2`、`google/gemini-2.5-pro`。如果你保留脚本默认的 Bedrock 模型 ID，又设置了 `AEVOLVE_MODEL_PROVIDER=openrouter`，请同时设置 `OPENROUTER_MODEL`，否则 OpenRouter 会收到一个 AWS 模型 ID。

如果使用 agicto 或其他 OpenAI-compatible 平台，例如 `https://api.agicto.cn/v1`：

```bash
export OPENAI_API_KEY=你的_api_key
export OPENAI_BASE_URL=https://api.agicto.cn/v1
export EVOLVER_OPENAI_API_KEY="$OPENAI_API_KEY"
export EVOLVER_OPENAI_BASE_URL="$OPENAI_BASE_URL"

export AEVOLVE_MODEL_PROVIDER=openai
export AEVOLVE_OPENAI_MODEL=平台支持的模型名

# MCP-Atlas judge / LiteLLM
export EVAL_USE_LITELLM=true
export EVAL_LLM_API_KEY="$OPENAI_API_KEY"
export EVAL_LLM_BASE_URL="$OPENAI_BASE_URL"
```

单个参数也可以写成 `openai:模型名`，例如 `--solver-model openai:gpt-4o --evolver-model openai:gpt-4o`。MCP-Atlas 的 `--judge-model openai:gpt-4o` 会自动转换为 LiteLLM 使用的 `openai/gpt-4o`。

如果报 `AccessDeniedException`、`ValidationException`、`Could not load credentials`，优先检查：

- AWS 凭证是否生效。
- `AWS_DEFAULT_REGION` 或脚本里的 `REGION` 是否和开通 Bedrock 模型的区域一致。
- Bedrock 控制台里对应 Claude 模型是否已启用。
- 模型 ID 是否是你账号当前可用的 ID。

### 2.5 MCP-Atlas API keys

MCP-Atlas 有些任务需要 Brave、Exa、GitHub 等第三方 API key。没有这些 key 也能跑一部分不需要 key 的任务，但任务会被过滤。

推荐在仓库根目录建 `.env`：

```bash
BRAVE_API_KEY=...
EXA_API_KEY=...
GITHUB_TOKEN=...
```

MCP wrapper 默认读 `.env`，也可以用：

```bash
export ENV_FILE=.env
```

更多 key 映射见 `docs/mcp-atlas-demo.md` 和 `agent_evolve/agents/mcp/server_keys.yaml`。

## 3. 第一次跑通推荐路线

不要一上来跑全量 500 题。先把任务数和并发都降到最低，确认环境、模型和 Docker 都通。

推荐顺序：

1. 跑导入验证和测试。
2. 跑 MCP-Atlas baseline，`LIMIT=2`。
3. 跑 MCP-Atlas evolution，`LIMIT=4`、`BATCH_SIZE=2`。
4. 再按需要跑 SWE-bench 或 Terminal-Bench。

## 4. MCP-Atlas 运行

### 4.1 Baseline，无演化

```bash
cd /mnt/data/zj/evolve/a-evolve
source .venv/bin/activate

LIMIT=2 \
BATCH_SIZE=2 \
WORKERS=1 \
OUTPUT_DIR=logs/mcp_baseline_smoke \
bash examples/mcp_examples/run_adaptive_evolve_baseline.sh
```

默认参数来自脚本：

- `SOLVER_MODEL=us.anthropic.claude-opus-4-6-v1`
- `JUDGE_MODEL=us.anthropic.claude-sonnet-4-6`
- `REGION=us-west-2`
- `DOCKER_IMAGE=ghcr.io/scaleapi/mcp-atlas:latest`
- `ENV_FILE=.env`

常用覆盖方式：

```bash
SOLVER_MODEL=your_solver_model \
JUDGE_MODEL=your_judge_model \
REGION=us-west-2 \
LIMIT=5 \
BATCH_SIZE=5 \
OUTPUT_DIR=logs/mcp_baseline_test \
bash examples/mcp_examples/run_adaptive_evolve_baseline.sh
```

输出位置：

```text
logs/mcp_baseline_smoke/
  baseline.log
  summary.csv
  output_<task_id>.txt
  conversation_<task_id>.json
```

### 4.2 In-situ evolution，同一任务流边跑边演化

```bash
cd /mnt/data/zj/evolve/a-evolve
source .venv/bin/activate

LIMIT=4 \
BATCH_SIZE=2 \
PARALLEL=1 \
CYCLES=2 \
OUTPUT_DIR=logs/mcp_evolve_smoke \
bash examples/mcp_examples/run_adaptive_evolve_in-situ_unified.sh
```

常用变量：

| 变量 | 含义 |
|---|---|
| `LIMIT` | 总任务数上限 |
| `BATCH_SIZE` | 每个演化 batch 的任务数 |
| `CYCLES` | 演化轮数 |
| `PARALLEL` | batch 内并发 |
| `SOLVER_MODEL` | 做题模型 |
| `EVOLVER_MODEL` 或 `EVOLVER_MODEL_ID` | 修改 workspace 的模型 |
| `JUDGE_MODEL` | 评测或裁判模型 |
| `OUTPUT_DIR` | 结果目录 |

输出位置：

```text
logs/mcp_evolve_smoke/
  evolve.log
  results.jsonl
  results.metrics.json
  workspace/
```

### 4.3 Split evolution，先训练演化再测试

```bash
EVOLVE_LIMIT=4 \
EVAL_LIMIT=4 \
BATCH_SIZE=2 \
TRAIN_PARALLEL=1 \
TEST_PARALLEL=1 \
LIMIT=8 \
OUTPUT_DIR=logs/mcp_split_smoke \
bash examples/mcp_examples/run_adaptive_evolve_split_unified.sh
```

输出包括：

```text
results.train.jsonl
results.test.jsonl
results.jsonl
results.metrics.json
evolve.log
```

## 5. SWE-bench 运行

SWE-bench 会读取 HuggingFace dataset，并可能拉取/运行 SWE-bench Docker 镜像。第一次建议使用 mini 数据集或很小的 `LIMIT`。

### 5.1 Baseline，无演化

快速只生成 patch、不跑官方评测：

```bash
cd /mnt/data/zj/evolve/a-evolve
source .venv/bin/activate

DATASET=MariusHobbhahn/swe-bench-verified-mini \
LIMIT=2 \
WORKERS=1 \
RUN_EVAL=false \
OUTPUT_DIR=logs/swe_baseline_smoke \
bash examples/swe_examples/run_solve_all.sh
```

如果要跑官方 evaluation，把 `RUN_EVAL=true`，但会更慢并依赖 Docker 镜像。

常用变量：

| 变量 | 默认 | 含义 |
|---|---|---|
| `DATASET` | `princeton-nlp/SWE-bench_Verified` | 数据集 |
| `LIMIT` | `500` | 任务数 |
| `WORKERS` | `20` | 并发 |
| `MAX_TURNS` | `140` | 每题最大 agent 回合数 |
| `RUN_EVAL` | `true` | 是否跑 SWE-bench 官方评测 |
| `OUTPUT_DIR` | `logs/swe_solve_all_<run_id>` | 输出目录 |

### 5.2 In-situ evolution

```bash
DATASET=MariusHobbhahn/swe-bench-verified-mini \
LIMIT=4 \
BATCH_SIZE=2 \
PARALLEL=1 \
CYCLES=2 \
SOLVER_PROPOSES=true \
VERIFICATION_FOCUS=true \
EFFICIENCY_PROMPT=true \
OUTPUT_DIR=logs/swe_evolve_smoke \
bash examples/swe_examples/run_swe_evolve_in-situ_unified.sh
```

输出位置：

```text
logs/swe_evolve_smoke/
  evolve.log
  results.jsonl
  results.metrics.json
  workspace/
```

### 5.3 Split evolution

```bash
DATASET=MariusHobbhahn/swe-bench-verified-mini \
EVOLVE_LIMIT=4 \
EVAL_LIMIT=4 \
BATCH_SIZE=2 \
TRAIN_PARALLEL=1 \
TEST_PARALLEL=1 \
LIMIT=8 \
OUTPUT_DIR=logs/swe_split_smoke \
bash examples/swe_examples/run_swe_evolve_split_unified.sh
```

更详细的 SWE 参数说明见 `examples/swe_examples/README.md`。

## 6. Terminal-Bench 2.0 运行

Terminal-Bench 脚本会通过 `uv run python` 启动，所以需要 `uv`。脚本内部会设置/使用 tool consent 相关环境；批量运行时建议显式设置：

```bash
export BYPASS_TOOL_CONSENT=true
```

### 6.1 Baseline

```bash
cd /mnt/data/zj/evolve/a-evolve

BYPASS_TOOL_CONSENT=true \
bash examples/tb_examples/run_baseline.sh tb_smoke --limit 1 --workers 1
```

输出：

```text
logs/baseline_tb_smoke/
  batch.log
  results.jsonl
  errors.jsonl
```

### 6.2 Evolution

```bash
BYPASS_TOOL_CONSENT=true \
bash examples/tb_examples/run_evolution.sh tb_evolve_smoke \
  --evolve-limit 1 \
  --eval-limit 1 \
  --batch-size 1 \
  --workers 1
```

输出：

```text
logs/evolve_tb_evolve_smoke/
  batch.log
  results.jsonl
  errors.jsonl

evolution_workdir/evolve_tb_evolve_smoke/
  manifest.yaml
  prompts/
  skills/
  memory/
```

常用参数：

| 参数 | 含义 |
|---|---|
| `--limit N` | baseline 跑 N 个任务 |
| `--workers N` | 并发数 |
| `--evolve-limit N` | phase 1 用 N 个任务演化 |
| `--eval-limit N` | phase 2 用 N 个额外任务测试 |
| `--batch-size N` | 每 N 个任务演化一次 |
| `--max-skills N` | skill 总数上限 |
| `--exclude a,b,c` | 排除指定任务 |

更详细说明见 `examples/tb_examples/README.md`。

## 7. SkillBench 运行

SkillBench 可以自动 bootstrap 公共 `benchflow-ai/skillsbench` 仓库，也可以使用你本地已有的任务目录。先安装：

```bash
pip install -e ".[skillbench,dev]"
```

单题测试：

```bash
python examples/skillbench_examples/skillbench_solve_one.py \
  --mode native \
  --use-skills true \
  --region us-west-2
```

小规模 evolution：

```bash
python examples/skillbench_examples/skillbench_evolve_in_situ_cycle.py \
  --batch-size 2 \
  --max-cycles 3 \
  --use-skills false \
  --limit 2 \
  --mode native \
  --region us-west-2 \
  -v
```

如果你有本地 SkillBench clone：

```bash
export SKILLBENCH_REPO_DIR=/path/to/skillsbench
```

或者直接指定任务目录：

```bash
export SKILLBENCH_TASKS_DIR=/path/to/tasks
export SKILLBENCH_TASKS_NO_SKILLS_DIR=/path/to/tasks-no-skills
```

更多说明见 `docs/skillbench-setup.md`。

## 8. OSWorld 运行

OSWorld 不是最适合第一次跑的 benchmark，因为它需要 AWS EC2 桌面环境、VPC、安全组、OSWorld 数据和缓存。

安装：

```bash
pip install -e ".[osworld,dev]"
```

需要先准备 OSWorld：

```bash
git clone https://github.com/xlang-ai/OSWorld
cd OSWorld
pip install -e .
```

环境变量示例：

```bash
export OSWORLD_PATH=/path/to/OSWorld
export AWS_REGION=us-east-1
export AWS_SUBNET_ID=subnet-xxxxxxxx
export AWS_SECURITY_GROUP_ID=sg-xxxxxxxx
```

小规模 baseline：

```bash
python examples/osworld_examples/evolve_osworld.py \
  --task-file "$OSWORLD_PATH/evaluation_examples/test_all.json" \
  --provider aws \
  --solver-model 1 \
  --no-evolve \
  --no-seed-skills \
  --limit 2 \
  --max-steps 30 \
  --batch-size 1 \
  --workers 1 \
  --output-dir outputs/osworld_smoke
```

完整 evolution 参考 `examples/osworld_examples/README.md`。

## 9. ARC-AGI-3 运行

当前分支包含 ARC-AGI-3 相关代码。ARC 示例要求 Python 3.12+，和主项目的 Python 3.11 环境可以分开建：

```bash
cd /mnt/data/zj/evolve/a-evolve
uv venv --python 3.12 .venv-arc
source .venv-arc/bin/activate
uv pip install -e . arc-agi strands-agents strands-agents-bedrock boto3 numpy flask
```

跑 LS20 可视化示例：

```bash
python examples/arc_examples/play_ls20.py
```

脚本会在本地生成日志，并可打开：

```text
http://localhost:7889
```

跑两个 game 的 quick eval：

```bash
python examples/arc_examples/eval_2games.py
```

跑 evolution：

```bash
python examples/arc_examples/evolve_arc.py \
  --cycles 2 \
  --batch-size 1 \
  --operation-mode offline \
  --output-dir evolution_workdir/arc_smoke
```

如果使用 competition/online 模式，需要 ARC API key：

```bash
export ARC_API_KEY=...
```

## 10. 输出目录怎么看

常见输出目录：

| 目录 | 用途 |
|---|---|
| `logs/...` | 运行日志、结果 JSONL、metrics |
| `results/...` | 部分老脚本使用的结果目录 |
| `evolution_workdir/...` | 演化后的 agent 工作区 |
| `outputs/...` | OSWorld、SkillBench 等输出 |
| `/tmp/...` | baseline 临时工作区 |

常见文件：

| 文件 | 含义 |
|---|---|
| `evolve.log` 或 `batch.log` | 主日志 |
| `results.jsonl` | 每题结果 |
| `results.metrics.json` | 汇总指标 |
| `errors.jsonl` | 失败任务 |
| `summary.csv` | MCP baseline 汇总 |
| `conversation_<task_id>.json` | agent 轨迹 |
| `output_<task_id>.txt` | agent 最终输出 |

演化后的 workspace 通常长这样：

```text
workspace/
  manifest.yaml
  prompts/
  skills/
  tools/
  memory/
```

可以查看演化带来的变化：

```bash
cd evolution_workdir/evolve_tb_evolve_smoke
git status
git log --oneline --decorate --all --max-count=20
find skills -maxdepth 2 -type f | sort
```

## 11. 常用配置速查

### 11.1 任务规模

第一次运行建议：

```bash
LIMIT=1
WORKERS=1
BATCH_SIZE=1
PARALLEL=1
```

确认能跑通后再扩大：

```bash
LIMIT=20
WORKERS=4
BATCH_SIZE=5
PARALLEL=4
```

### 11.2 模型

多数脚本可用环境变量覆盖：

```bash
MODEL_ID=...
SOLVER_MODEL=...
EVOLVER_MODEL=...
EVOLVER_MODEL_ID=...
JUDGE_MODEL=...
REGION=...
MAX_TOKENS=...
```

如果你不确定账号支持哪个模型，先用一个你已经验证能在 Bedrock 调通的模型 ID。

### 11.3 数据集

SWE-bench：

```bash
DATASET=MariusHobbhahn/swe-bench-verified-mini
DATASET=princeton-nlp/SWE-bench_Verified
```

MCP-Atlas：

```bash
DATASET=ScaleAI/MCP-Atlas
```

### 11.4 输出和断点续跑

大部分脚本会跳过已有结果。想续跑时，使用相同的 `OUTPUT_DIR`、`LOG_DIR` 或 `RUN_NAME` 重新执行同一命令。

如果想从干净状态重跑，换一个新的 `RUN_ID` 或 `OUTPUT_DIR`，不要直接覆盖旧目录。

## 12. Bring Your Own Agent

如果你想把自己的 agent 接入 A-Evolve，需要提供一个工作区和一个 `BaseAgent`。

工作区结构：

```text
my_agent/
  manifest.yaml
  prompts/
    system.md
  skills/
  tools/
  memory/
```

`manifest.yaml` 示例：

```yaml
name: my-agent
version: 0.1.0
contract_version: "1.0"

agent:
  type: reference
  entrypoint: my_package.my_agent.MyAgent

evolvable_layers:
  - prompts
  - skills
  - memory

reload_strategy: hot
```

Agent 示例：

```python
from agent_evolve.protocol.base_agent import BaseAgent
from agent_evolve.types import Task, Trajectory


class MyAgent(BaseAgent):
    def solve(self, task: Task) -> Trajectory:
        prompt = self.system_prompt
        skills = self.skills

        # 在这里调用你自己的 LLM、工具或执行器
        output = run_my_solver(prompt=prompt, skills=skills, task=task)

        return Trajectory(
            task_id=task.id,
            output=output,
            steps=[],
        )
```

然后可以用 `ae.Evolver` 接自定义 benchmark 或内置 benchmark adapter。注意：本仓库当前更常用、也更可复现的运行方式仍然是 `examples/*` 脚本；顶层 API 适合开发集成，不是所有 benchmark 名称都等价于完整实验脚本。

## 13. 常见报错和处理

### 13.1 `ModuleNotFoundError: No module named 'agent_evolve'`

没有激活虚拟环境，或没有 editable install：

```bash
cd /mnt/data/zj/evolve/a-evolve
source .venv/bin/activate
pip install -e ".[all,dev]"
```

### 13.2 `uv: command not found`

部分脚本直接用 `uv run python`。安装 `uv`，或改用会尊重当前 venv 的 wrapper。最简单是安装：

```bash
pip install uv
```

### 13.3 Docker 相关错误

检查 Docker daemon：

```bash
docker info
```

MCP-Atlas 预拉镜像：

```bash
docker pull ghcr.io/scaleapi/mcp-atlas:latest
```

### 13.4 Bedrock 鉴权或模型不可用

检查：

```bash
aws sts get-caller-identity
echo "$AWS_DEFAULT_REGION"
```

然后确认 Bedrock 控制台里对应模型在该 region 可用。

### 13.5 MCP-Atlas 任务数量很少或全被过滤

说明缺少 MCP server API keys。先建 `.env`，至少配置：

```bash
BRAVE_API_KEY=...
EXA_API_KEY=...
GITHUB_TOKEN=...
```

没有 key 时只能跑不需要 key 的任务。

### 13.6 HuggingFace dataset 下载失败

SWE-bench 和 MCP-Atlas 可能需要从 HuggingFace 拉数据。检查网络和缓存。如果机器不能联网，需要提前把数据集缓存到本机，或把 `DATASET` 指向可访问的数据源。

### 13.7 Terminal-Bench 卡在工具确认

设置：

```bash
export BYPASS_TOOL_CONSENT=true
```

### 13.8 跑得太慢或费用太高

先把规模降到：

```bash
LIMIT=1
WORKERS=1
BATCH_SIZE=1
PARALLEL=1
```

确认端到端没问题后再扩大。SWE-bench、Terminal-Bench 和 OSWorld 都可能产生较高 token 或云资源成本。

## 14. 推荐工作流

第一次建议按下面顺序执行：

```bash
cd /mnt/data/zj/evolve/a-evolve
source .venv/bin/activate

python - <<'PY'
import agent_evolve as ae
print(ae.__version__)
PY

pytest tests/test_unified_engine.py -q

docker pull ghcr.io/scaleapi/mcp-atlas:latest

LIMIT=2 BATCH_SIZE=2 WORKERS=1 OUTPUT_DIR=logs/mcp_baseline_smoke \
bash examples/mcp_examples/run_adaptive_evolve_baseline.sh

LIMIT=4 BATCH_SIZE=2 PARALLEL=1 CYCLES=2 OUTPUT_DIR=logs/mcp_evolve_smoke \
bash examples/mcp_examples/run_adaptive_evolve_in-situ_unified.sh
```

如果这条链路能跑通，说明 Python 包、Docker、模型凭证和基本演化流程都正常。之后再跑 SWE-bench 或 Terminal-Bench。
