# EvoEfficient SWE Pilot

This repo now includes a small-batch pilot runner for EvoEfficient:

```bash
uv run python examples/swe_examples_new/run_evoefficient_pilot.py \
  --dataset MariusHobbhahn/swe-bench-verified-mini \
  --limit 2 \
  --model-id openai:gpt-5-mini \
  --pilot-limit 2 \
  --batch-size 2 \
  --parallel 2 \
  --output-dir logs/evoefficient_pilot2
```

## If you do not have `uv`

Use a normal Python virtual environment instead. On Windows PowerShell:

```powershell
cd E:\evoefficient\code\a-evolve
python -m venv .venv-win
.\.venv-win\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[swe]"
```

Then replace:

```bash
uv run python examples/swe_examples_new/run_evoefficient_pilot.py ...
```

with:

```powershell
python examples/swe_examples/run_evoefficient_pilot.py `
  --dataset MariusHobbhahn/swe-bench-verified-mini `
  --limit 8 `
  --pilot-limit 8 `
  --batch-size 2 `
  --parallel 2 `
  --output-dir logs/evoefficient_pilot
```

On Windows `cmd.exe`, use `^` for line continuation:

```bat
python examples/swe_examples/run_evoefficient_pilot.py ^
  --dataset MariusHobbhahn/swe-bench-verified-mini ^
  --limit 8 ^
  --pilot-limit 8 ^
  --batch-size 2 ^
  --parallel 2 ^
  --output-dir logs/evoefficient_pilot
```

## If you do not have Docker

SWE-bench execution needs Docker because each task is solved and evaluated inside a benchmark container. Without Docker, the SWE pilot cannot produce real solve/eval results.

You have three options:

1. Install Docker Desktop, then run the commands above.
2. Run on a remote/Linux machine that has Docker.
3. Only analyze existing result files, if you already have `baseline/results.json` and `evolution/results.json`; this does not run new tasks.

What it does:

1. Runs a baseline on a small SWE-bench mini subset.
2. Runs an EvoEfficient pilot on the same-style subset.
3. Writes a comparison report to `pilot_report.json`.

Outputs:

```text
logs/evoefficient_pilot/
  ├── baseline/results.json
  ├── evolution/results.json
  └── pilot_report.json
```

The pilot report includes:

- success rate
- token usage
- step count
- tool call count
- wall time

Recommended interpretation:

- If token/step/tool-call counts drop and success rate stays roughly stable, the efficiency idea is working.
- If success rises but costs also rise, the strategy is too additive.
- If success collapses, tighten gate thresholds and reduce mutation scope.
