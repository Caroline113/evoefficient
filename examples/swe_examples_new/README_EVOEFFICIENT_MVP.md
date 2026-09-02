# EvoEfficient SWE-bench Verified Mini Experiments

This README explains how to run two small-batch experiments:

1. **Single-strategy ablation**: test each efficiency strategy separately and visualize the efficiency gain.
2. **Meta-selector MVP loop**: run `profile -> candidate configs -> independent validation -> gate -> Pareto -> task-level reuse` on a small SWE-bench Verified batch.

The scripts are designed for quick validation before running larger experiments.

## Prerequisites

Run commands from the repository root:

```powershell
cd E:\evoefficient\code\a-evolve-new copy
```

You need:

- Docker running locally.
- Access to the SWE-bench Docker images.
- Python dependencies installed from this repo.
- Model/provider credentials configured in `.env` or your shell.
- HuggingFace dataset access for `MariusHobbhahn/swe-bench-verified-mini`.

Install SWE dependencies if your current Python environment does not already
include `swebench`, `datasets`, and `strands-agents`:

```powershell
python -m pip install -e ".[swe]"
```

Or, with `uv`:

```powershell
uv pip install -e ".[swe]"
```

The default model is:

```text
us.anthropic.claude-opus-4-6-v1
```

Change it with `--model-id` if your environment uses another model.

## What Was Added

Core code:

```text
agent_evolve/efficiency/
  strategies.py       # strategy registry and workspace activation
  profiler.py         # task/cost profile summaries
  gate.py             # success-first Efficiency Gate
  selector.py         # meta selector and task-level config matching
  pareto.py           # success-first cost-of-pass selection
  library.py          # structured strategy library and online stats
  visualization.py    # CSV and PNG result plots
```

SWE integration:

```text
agent_evolve/agents/swe/agent.py
seed_workspaces/swe/tools/bash.py
```

Experiment runners:

```text
examples/swe_examples_new/run_efficiency_strategy_ablation.py
examples/swe_examples_new/run_evoefficient_mvp.py
```

## Efficiency Strategies

The registry currently supports these strategy IDs. The MVP runner may select a
subset depending on `--candidate-mode`; run the ablation script when you want an
exhaustive per-strategy comparison.

| Strategy | Module | Effect |
|---|---|---|
| `simple_memory` | Context | Compact working memory with task facts, hypothesis, changed artifacts, evidence, and checks |
| `observation_masking` | Context | Mask stale large observations after enough tool calls |
| `verbatim_compaction` | Context | Delete low-value lines from long outputs while preserving retained lines verbatim |
| `cognitive_router` | Reasoning | Use low/high cognitive depth only where it changes the decision |
| `plan_template_cache` | Planning | Reuse compact plan templates by task intent |
| `adaptive_tool_trimming` | Tools | Trim shell/editor/Python outputs with task-aware rules |
| `difficulty_model_routing` | Routing | Keep the main task on the hard model and delegate only bounded simple substeps to the easy model |
| `context_reduction` | Context | Smaller conversation window and targeted file reading instructions |
| `reasoning_budget` | Reasoning | Hypothesis-first, concise reasoning with escalation only when needed |
| `plan_template` | Planning | Compact locate -> hypothesize -> patch -> verify workflow |
| `tool_lazy_loading` | Tools | Smaller SWE tool surface to reduce tool schema overhead |
| `observation_filter` | Observation | Filters long shell/test output before it enters model context |
| `model_routing_prompt` | Routing | Prompts difficulty-aware use of reasoning effort |
| `skill_compilation` | Compilation | Adds a compact reusable SWE debug-loop skill |

Strategies are activated per workspace through:

```text
<workspace>/efficiency/active_strategies.yaml
```

## 1. Run Single-Strategy Ablation

This runs:

- `baseline`
- each individual strategy
- `all_strategies`

Use a tiny batch first:

```powershell
python examples\swe_examples_new\run_efficiency_strategy_ablation.py `
  --limit 2 `
  --parallel 1 `
  --max-steps 60 `
  --output-dir logs\evoefficient_strategy_ablation_smoke
```

Recommended small batch:

```powershell
python examples\swe_examples_new\run_efficiency_strategy_ablation.py `
  --limit 8 `
  --parallel 2 `
  --max-steps 80 `
  --output-dir logs\evoefficient_strategy_ablation
```

Run only selected strategies:

```powershell
python examples\swe_examples_new\run_efficiency_strategy_ablation.py `
  --limit 6 `
  --strategies observation_filter reasoning_budget tool_lazy_loading `
  --output-dir logs\evoefficient_strategy_ablation_lowrisk
```

### Ablation Outputs

```text
logs/evoefficient_strategy_ablation/
  REPORT.md
  summary.json
  summary.csv
  figures/
    strategy_avg_tokens.png
    strategy_cost_success.png
    strategy_tool_calls.png
  baseline/results.json
  observation_filter/results.json
  ...
```

Read `REPORT.md` first. The clearest figures are:

- `figures/strategy_avg_tokens.png`: average token cost per task.
- `figures/strategy_cost_success.png`: token cost and success rate together.
- `figures/strategy_tool_calls.png`: average tool calls per task.

## 2. Run Meta-Selector MVP Experiment

The MVP script now implements the documented EvoEfficient loop:

```text
baseline profile
  -> meta task/bottleneck profile
  -> multiple structured candidate configs
  -> independent candidate validation
  -> success-first Efficiency Gate for each config
  -> Pareto frontier and cost-of-pass selection
  -> structured strategy library
  -> paired held-out baseline/reuse comparison with per-task config matching
```

The Gate is overhead-aware. It does not only ask whether the candidate run is
cheaper. It also accounts for the one-time cost of profiling, selecting, and
validating the strategy:

```text
net_saving_per_task =
  baseline_avg_cost
- candidate_avg_cost
- discovery_overhead / expected_future_hits
```

A strategy is accepted only when this amortized net saving is positive and the
required break-even hits are within the expected future reuse count.

Tiny smoke run:

```powershell
python examples/swe_examples_new/run_evoefficient_mvp.py \
  --train-limit 2 \
  --reuse-limit 2 \
  --candidate-mode safe \
  --max-candidate-configs 5 \
  --parallel 1 \
  --max-steps 60 \
  --output-dir logs/evoefficient_mvp_smoke
```

Recommended small-batch MVP:

```powershell
python examples/swe_examples_new/run_evoefficient_mvp.py \
  --train-limit 6 \
  --reuse-limit 6 \
  --candidate-mode safe \
  --max-candidate-configs 5 \
  --parallel 2 \
  --max-steps 60 \
  --expected-future-hits 5 \
  --output-dir logs/evoefficient_mvp3
```

Loosen the Gate for early debugging:

```powershell
python examples\swe_examples_new\run_evoefficient_mvp.py `
  --train-limit 4 `
  --reuse-limit 4 `
  --candidate-mode explicit `
  --candidate-strategies difficulty_model_routing plan_template_cache `
  --delta-cost 0.05 `
  --epsilon-success 0.05 `
  --min-trials 2 `
  --expected-future-hits 8 `
  --output-dir logs\evoefficient_mvp_debug
```

If profiling or strategy selection uses an extra LLM call, include that cost:

```powershell
python examples\swe_examples_new\run_evoefficient_mvp.py `
  --train-limit 6 `
  --reuse-limit 6 `
  --candidate-mode safe `
  --profile-overhead-tokens 12000 `
  --selector-overhead-tokens 8000 `
  --validation-overhead-tokens 0 `
  --expected-future-hits 12 `
  --max-break-even-hits 12 `
  --output-dir logs\evoefficient_mvp_overhead
```

### MVP Outputs

```text
logs/evoefficient_mvp/
  REPORT.md
  mvp_report.json
  summary.json
  summary.csv
  efficiency/
    strategy_library.yaml
  figures/
    strategy_avg_tokens.png
    strategy_cost_success.png
    strategy_tool_calls.png
    strategy_hits.png
  01_train_baseline_profile/results.json
  02_train_candidate_validate/results.json
  03_heldout_baseline/results.json
  04_heldout_reuse_accepted/results.json
```

`mvp_report.json` records:

- Baseline cost profile.
- Detected bottlenecks.
- Candidate mode.
- Selected candidate strategies.
- Gate decision.
- Discovery overhead.
- Amortized overhead per expected future hit.
- Net saving per future task.
- Break-even hits.
- Accepted strategy list.
- Summary metrics for train baseline, train validation, held-out baseline, and held-out reuse.
- Held-out paired comparison verdict.

## Interpreting Results

Main metrics:

| Metric | Meaning |
|---|---|
| `success_rate` | Fraction of tasks resolved |
| `avg_tokens` | Average input + output tokens per task |
| `cost_of_pass` | `avg_tokens / success_rate`; `inf` when success rate is zero |
| `avg_tool_calls` | Average tool calls per task |
| `wall_time` | Total wall time in seconds |

A good MVP result looks like:

```text
candidate success_rate >= baseline success_rate - epsilon
candidate avg_tokens <= baseline avg_tokens * (1 - delta_cost)
net_saving_per_task > 0 after discovery overhead is amortized
break_even_hits <= expected_future_hits
reuse batch keeps similar success with lower average tokens/tool calls
```

If the Gate rejects the candidate, the reuse stage falls back to baseline strategies. This is expected behavior: the MVP is success-first.

If the train Gate accepts the candidate but the paired held-out reuse run
regresses success or fails to improve cost, the final strategy library marks
the candidate entries as `quarantined` and leaves `active_strategies` empty.
Use `--disable-heldout-gate` only when you intentionally want to inspect the
old train-gate-only behavior.

### Candidate Modes

| Mode | Meaning |
|---|---|
| `meta` | Use the meta selector on the train task/cost profile. This may generate only relevant configs. |
| `bottleneck` | Backward-compatible alias for `meta`. |
| `safe` | Validate low-risk single-strategy configs plus a conservative combo. |
| `all` | Validate each registered strategy independently. |
| `explicit` | Use exactly the strategy IDs passed to `--candidate-strategies`. |

`--max-candidate-configs` caps how many independent configs are validated. Use
a small value for smoke tests and a larger value when you want clearer
synergy/conflict evidence.

## 3. Run Sequential Self-Evolution With Efficiency Learning

`evolve_sequential.py` keeps the original GuidedSynthesisEngine for prompt,
skill, and memory evolution, then adds an efficiency controller between
batches:

```text
batch solve -> prompt/skill evolution -> efficiency config validation -> strategy library -> next-batch task matching
```

Tiny smoke:

```powershell
python examples\swe_examples_new\evolve_sequential.py `
  --dataset MariusHobbhahn/swe-bench-verified-mini `
  --batch-size 2 `
  --limit 6 `
  --parallel 1 `
  --max-steps 60 `
  --efficiency-controller meta `
  --efficiency-max-candidates 2 `
  --output-dir logs\evoefficient_sequential_meta_smoke
```

Disable only the efficiency controller while keeping the older evolution path:

```powershell
python examples\swe_examples_new\evolve_sequential.py `
  --batch-size 2 `
  --limit 6 `
  --efficiency-controller off `
  --output-dir logs\evoefficient_sequential_no_efficiency
```

Detailed implementation notes are in:

```text
docs/EVOEFFICIENT_META_SELECTOR_IMPLEMENTATION_CN.md
```

## Common Issues

### Docker image pull is slow or fails

The first run may spend most of its time pulling SWE-bench images. Re-run the same command after images are cached.

### HuggingFace dataset cannot load

Check network access and dataset credentials. You can switch datasets with:

```powershell
--dataset MariusHobbhahn/swe-bench-verified-mini
```

### No cost improvement

Try:

- Increase `--limit`.
- Use lower-risk strategies only.
- Lower `--max-steps` to expose step/tool-call efficiency.
- Inspect `results.json` for tasks with empty patches or early failures.

### Matplotlib cannot render

The scripts use non-interactive `savefig`, so no GUI is required. If plotting still fails, check that `matplotlib` is installed from `pyproject.toml`.

## Suggested Experiment Order

1. Run ablation smoke with `--limit 2`.
2. Run MVP smoke with `--train-limit 2 --reuse-limit 2`.
3. Run ablation with `--limit 8`.
4. Run MVP with `--train-limit 6 --reuse-limit 6`.
5. Compare:

```text
logs/evoefficient_strategy_ablation/REPORT.md
logs/evoefficient_mvp/REPORT.md
```

This gives both a clear per-strategy contribution view and a first proof of the 15.1 MVP self-evolution loop.
