# EvoEfficient SWE Pilot for Windows

本文件是 `README_PILOT.md` 的 Windows 命令行版本。下面命令默认在 `cmd.exe` 中运行。

进入项目目录：

```bat
cd /d E:\evoefficient\code\a-evolve
```

## 0. 运行前检查

确保：

- 已安装 `uv`
- 已安装并启动 Docker Desktop
- 已配置模型 API / Bedrock 凭据
- 当前目录是 `E:\evoefficient\code\a-evolve`

```bat
uv --version
docker --version
```

如果没有安装 `uv`，可以用普通 Python 虚拟环境替代：

```bat
cd /d E:\evoefficient\code\a-evolve
python -m venv .venv-win
.\.venv-win\Scripts\activate.bat
python -m pip install -U pip
python -m pip install -e ".[swe]"
```

之后把所有命令里的：

```bat
uv run python
```

替换成：

```bat
python
```

如果没有安装 Docker，需要注意：SWE-bench 的解题和评测都依赖 Docker 容器。没有 Docker 时，下面的 pilot 命令不能产生真实实验结果。可选做法是：

1. 安装 Docker Desktop 后再运行。
2. 换到有 Docker 的远程/Linux 机器运行。
3. 只分析已有的 `baseline/results.json` 和 `evolution/results.json`，不跑新任务。

## 1. 推荐小批量验证命令

```bat
uv run python examples/swe_examples/run_evoefficient_pilot.py ^
  --dataset MariusHobbhahn/swe-bench-verified-mini ^
  --limit 8 ^
  --pilot-limit 8 ^
  --batch-size 2 ^
  --parallel 2 ^
  --output-dir logs/evoefficient_pilot
```

不用 `uv` 时，对应命令是：

```bat
python examples/swe_examples/run_evoefficient_pilot.py ^
  --dataset MariusHobbhahn/swe-bench-verified-mini ^
  --limit 8 ^
  --pilot-limit 8 ^
  --batch-size 2 ^
  --parallel 2 ^
  --output-dir logs/evoefficient_pilot
```

这个命令会依次运行：

1. baseline：`solve_all.py`
2. EvoEfficient pilot：`evolve_sequential.py`
3. 自动汇总对比：`pilot_report.json`

## 2. 更便宜的 smoke test

如果你只是想确认代码链路能跑通，先跑 2 条任务：

```bat
uv run python examples/swe_examples/run_evoefficient_pilot.py ^
  --dataset MariusHobbhahn/swe-bench-verified-mini ^
  --limit 2 ^
  --pilot-limit 2 ^
  --batch-size 1 ^
  --parallel 1 ^
  --max-steps 60 ^
  --output-dir logs/evoefficient_pilot_smoke
```

## 3. 只跑 EvoEfficient pilot，不重复跑 baseline

如果 `logs\evoefficient_pilot\baseline\results.json` 已经存在，可以跳过 baseline：

```bat
uv run python examples/swe_examples/run_evoefficient_pilot.py ^
  --dataset MariusHobbhahn/swe-bench-verified-mini ^
  --limit 8 ^
  --pilot-limit 8 ^
  --batch-size 2 ^
  --parallel 2 ^
  --output-dir logs/evoefficient_pilot ^
  --no-baseline
```

## 4. 输出文件

```text
logs/evoefficient_pilot/
  baseline/results.json
  evolution/results.json
  pilot_report.json
```

查看汇总结果：

```bat
type logs\evoefficient_pilot\pilot_report.json
```

查看 baseline 和 pilot 原始结果：

```bat
type logs\evoefficient_pilot\baseline\results.json
type logs\evoefficient_pilot\evolution\results.json
```

## 5. 如何判断结果

重点看 `pilot_report.json` 里的这些字段：

- `success_rate`
- `tokens`
- `steps`
- `tool_calls`
- `wall_time`

推荐判断：

| 现象 | 解释 |
|---|---|
| token / steps / tool calls 下降，成功率基本不变 | 方向正确，可以扩大任务量 |
| 成功率上升但成本也上升 | 进化偏“加法策略”，需要更强效率约束 |
| 成功率明显下降 | 策略过激，需要收紧 Gate 或降低 `max_steps` 之外的变异 |
| 没有明显成本变化 | 小样本太小或策略未命中，可扩大到 8-20 条 |

## 6. PowerShell 写法

如果你在 PowerShell 中运行，多行命令请把 `^` 改成反引号 `` ` ``：

```powershell
uv run python examples/swe_examples/run_evoefficient_pilot.py `
  --dataset MariusHobbhahn/swe-bench-verified-mini `
  --limit 8 `
  --pilot-limit 8 `
  --batch-size 2 `
  --parallel 2 `
  --output-dir logs/evoefficient_pilot
```
