# SWE-bench Scripts for Windows

本文件是 `README.md` 的 Windows 命令行版本。下面命令默认在 `cmd.exe` 中运行。

进入项目目录：

```bat
cd /d E:\evoefficient\code\a-evolve
```

如果你使用 PowerShell，也可以直接运行同样的单行命令；多行命令需要把 `^` 换成反引号 `` ` ``。

## 0. 运行前检查

确保已经准备好：

- 已安装 `uv`
- 已安装并启动 Docker Desktop
- 已配置可用的模型访问凭据，例如 Bedrock / AWS 凭据
- 当前目录是 `E:\evoefficient\code\a-evolve`

快速检查：

```bat
uv --version
docker --version
```

## 1. Baseline：`solve_all.py`

无进化 baseline：直接用基础 SWE agent 解题，不做 skill evolution。

### 1.1 SWE-bench Verified 全量

```bat
uv run python examples/swe_examples/solve_all.py ^
  --dataset princeton-nlp/SWE-bench_Verified ^
  --model-id us.anthropic.claude-opus-4-6-v1 ^
  --workers 16 ^
  --max-turns 140 ^
  --output-dir logs/baseline ^
  --limit 500
```

### 1.2 Mini 快速测试

```bat
uv run python examples/swe_examples/solve_all.py `
  --dataset MariusHobbhahn/swe-bench-verified-mini `
  --workers 5 `
  --max-turns 140 `
  --output-dir logs/baseline-mini `
  --limit 50
```

### 1.3 更小的 smoke test

```bat
uv run python examples/swe_examples/solve_all.py `
  --dataset MariusHobbhahn/swe-bench-verified-mini `
  --model-id openrouter:anthropic/claude-sonnet-4 `
  --workers 1 `
  --max-turns 80 `
  --output-dir logs/baseline-smoke `
  --limit 2
```

## 2. Evolution：`evolve_sequential.py`

按 batch 解题，并在 batch 之间进行 skill evolution。

### 2.1 v32g 全量配置

```bat
uv run python examples/swe_examples/evolve_sequential.py ^
  --dataset princeton-nlp/SWE-bench_Verified ^
  --batch-size 20 ^
  --parallel 20 ^
  --max-steps 140 ^
  --window-size 70 ^
  --efficiency-prompt ^
  --solver-proposes ^
  --verification-focus ^
  --feedback none ^
  --model-id us.anthropic.claude-opus-4-6-v1 ^
  --seed-workspace seed_workspaces/swe ^
  --output-dir logs/v32g-full ^
  --limit 500
```

### 2.2 Mini 快速 evolution

```bat
uv run python examples/swe_examples/evolve_sequential.py ^
  --dataset MariusHobbhahn/swe-bench-verified-mini ^
  --batch-size 5 ^
  --parallel 5 ^
  --max-steps 140 ^
  --window-size 40 ^
  --efficiency-prompt ^
  --solver-proposes ^
  --verification-focus ^
  --feedback none ^
  --model-id us.anthropic.claude-opus-4-6-v1 ^
  --seed-workspace seed_workspaces/swe ^
  --output-dir logs/test-mini ^
  --limit 50
```

### 2.3 无进化 workspace baseline

```bat
uv run python examples/swe_examples/evolve_sequential.py ^
  --dataset princeton-nlp/SWE-bench_Verified ^
  --batch-size 10 ^
  --parallel 10 ^
  --max-steps 140 ^
  --window-size 70 ^
  --efficiency-prompt ^
  --no-evolve ^
  --model-id us.anthropic.claude-opus-4-6-v1 ^
  --seed-workspace seed_workspaces/swe ^
  --output-dir logs/baseline-ws ^
  --limit 500
```

## 3. 常用参数

| 参数 | 说明 |
|---|---|
| `--dataset` | HuggingFace dataset 名称 |
| `--model-id` | Bedrock 模型 ID |
| `--workers` | baseline 并行任务数 |
| `--parallel` | evolution 每个 batch 内的并行任务数 |
| `--max-turns` | baseline 最大工具调用轮数 |
| `--max-steps` | evolution 最大工具调用步数 |
| `--limit` | 最大任务数 |
| `--no-eval` | baseline 跳过评测，只生成 patch |
| `--efficiency-prompt` | 启用效率提示 |
| `--solver-proposes` | solver 在任务后提出 skill |
| `--verification-focus` | 只提出验证相关 skill |
| `--feedback none` | evolver 不看 pass/fail 分数 |
| `--no-evolve` | 禁用进化 |

## 4. 输出目录

```text
logs/<experiment>/
  patches/
  conversations/
  workspace/
  results.json
```

查看结果：

```bat
type logs\baseline-mini\results.json
type logs\test-mini\results.json
```

