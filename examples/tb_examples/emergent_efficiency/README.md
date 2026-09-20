# Terminal-Bench Emergent Efficiency

This runner ports the SWE emergent-efficiency experiment matrix to Terminal-Bench 2.1.
When `--challenges-dir` is omitted, the official Terminal-Bench 2.1 task
repository is downloaded automatically on first use.
It supports all five experiment groups:

- `normal-evolve`
- `cost-aware`
- `predefined-control`
- `predefined-cost-aware`
- `category-seeded-cost-aware`

Smoke run:

```bash
export DEEPSEEK_API_KEY=...
python examples/tb_examples/emergent_efficiency/run_emergent_efficiency_tb.py \
  --mode normal-evolve \
  --limit 4 \
  --batch-size 2 \
  --parallel 2 \
  --output-dir logs/tb_emergent_efficiency/normal-evolve
```

The default solver model is `deepseek:deepseek-v4-flash`, which uses the
official DeepSeek OpenAI-compatible API at `https://api.deepseek.com`. The
evolver model defaults to the same value unless `--evolver-model` is provided.
Use `--model-id` only when you want to override that default.

Run one named condition:

```bash
export DEEPSEEK_API_KEY=...
bash examples/tb_examples/emergent_efficiency/run_one.sh tb_smoke cost-aware \
  --limit 4 \
  --batch-size 2 \
  --parallel 2 \
  --dry-run-proposer
```

Run the full five-condition matrix on the same task slice:

```bash
export DEEPSEEK_API_KEY=...
bash examples/tb_examples/emergent_efficiency/run_matrix.sh tb_matrix \
  --limit 10 \
  --batch-size 2 \
  --parallel 2
```

Full cost-aware run:

```bash
export DEEPSEEK_API_KEY=...
uv run python examples/tb_examples/emergent_efficiency/run_emergent_efficiency_tb.py \
  --mode cost-aware \
  --limit 20 \
  --batch-size 5 \
  --parallel 4 \
  --output-dir logs/tb_emergent_efficiency/cost_aware
```

Predefined strategy control:

```bash
export DEEPSEEK_API_KEY=...
uv run python examples/tb_examples/emergent_efficiency/run_emergent_efficiency_tb.py \
  --mode predefined-control \
  --limit 20 \
  --batch-size 5 \
  --parallel 4
```

Category-seeded cost-aware run:

```bash
export DEEPSEEK_API_KEY=...
uv run python examples/tb_examples/emergent_efficiency/run_emergent_efficiency_tb.py \
  --mode category-seeded-cost-aware \
  --category-seed basic-efficiency \
  --limit 20 \
  --batch-size 5 \
  --parallel 4
```

Useful task filters:

```bash
--challenges-dir /path/to/terminal-bench-2-1/tasks
--tasks task_a,task_b
--exclude task_c
--category scientific-computing
--difficulty medium
--shuffle --shuffle-seed 13
```

Outputs are written under `logs/tb_emergent_efficiency/<mode>_<timestamp>/` by default:

- `results.json`: normalized per-task results
- `summary.json` and `summary.csv`: batch and overall metrics
- `behavior_stats.json`: repeated command/test and prompt-fragment adherence signals
- `emergence/proposals.jsonl`: cost-aware proposer inputs and outputs
- `emergence/proposal_adoption.json`: applied/skipped proposal records
- `emergence/workspace_snapshots/`: workspace state after each evolution batch
- `conversations/` and `system_prompts/`: per-task artifacts used by analysis
