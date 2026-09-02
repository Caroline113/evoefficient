# EvoEfficient SWE Verified Small Experiment

This README describes the smallest useful experiment to verify the full
efficiency-learning loop:

1. Train on one small batch and let the agent learn efficiency strategies.
2. Evaluate the next held-out batch twice: baseline vs learned strategies.
3. Compare success, token cost, tool calls, and strategy hits.
4. Inspect plots to see which strategies actually helped.

## 1. Prerequisites

```powershell
cd E:\evoefficient\code\a-evolve-new copy
python -m pip install -e ".[swe]"
```

You also need:

- Docker running locally
- SWE-bench image access
- HuggingFace access for `MariusHobbhahn/swe-bench-verified-mini`
- Model credentials in your environment

## 2. Smallest Smoke Run

```powershell
python examples\swe_examples_new\run_evoefficient_mvp.py `
  --train-limit 2 `
  --reuse-limit 2 `
  --candidate-mode meta `
  --parallel 1 `
  --max-steps 60 `
  --output-dir logs\evoefficient_mvp_smoke `
  --force
```

This validates the pipeline shape:

- train baseline
- train candidate
- held-out baseline
- held-out reuse

## 3. Recommended Small Experiment

```powershell
python examples\swe_examples_new\run_evoefficient_mvp.py `
  --train-limit 6 `
  --reuse-limit 6 `
  --candidate-mode safe `
  --parallel 2 `
  --max-steps 80 `
  --expected-future-hits 10 `
  --output-dir logs\evoefficient_mvp_small `
  --force
```

## 4. What You Should See

Output directory:

```text
logs/evoefficient_mvp_small/
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
    strategy_efficiency_frontier.png
    strategy_hits.png
  01_train_baseline_profile/results.json
  02_train_candidate_validate/results.json
  03_heldout_baseline/results.json
  04_heldout_reuse_accepted/results.json
```

## 5. How to Read the Results

Open `REPORT.md` first.

Important signals:

- `token_record_quality` tells you whether token accounting is trustworthy.
- `heldout_comparison.verdict` tells you whether learned strategies preserved
  success and improved cost on the next batch.
- `strategy_hits.png` shows which strategies were actually used.
- `03_heldout_baseline` and `04_heldout_reuse_*` use the same held-out issue slice.

Good outcome:

- train candidate passes the gate
- held-out success stays within tolerance
- held-out avg tokens or tool calls go down
- the plotted strategy hits are non-zero for the intended strategies

## 6. Candidate Strategy Modes

The MVP runner does not have to validate every registered strategy on every
run. Use `--candidate-mode` to make the choice explicit:

| Mode | Meaning | When to use |
|---|---|---|
| `meta` | Select configs from the train task/cost profile. | Fast smoke tests and profile-driven learning. |
| `bottleneck` | Backward-compatible alias for `meta`. | Keeps older commands runnable. |
| `safe` | Use a conservative low-risk set: context compaction, memory, plan cache, and step-level model routing. | Recommended first small experiment. |
| `all` | Validate every strategy in `STRATEGY_ORDER`. | Coverage/debug runs when you want to check that the full registry is wired. |
| `explicit` | Use only `--candidate-strategies ...`. | Targeted tests for one strategy or a manually chosen combination. |

Examples:

```powershell
python examples\swe_examples_new\run_evoefficient_mvp.py `
  --train-limit 2 `
  --reuse-limit 2 `
  --candidate-mode all `
  --output-dir logs\evoefficient_mvp_all_smoke `
  --force
```

```powershell
python examples\swe_examples_new\run_evoefficient_mvp.py `
  --train-limit 2 `
  --reuse-limit 2 `
  --candidate-mode explicit `
  --candidate-strategies difficulty_model_routing plan_template_cache `
  --output-dir logs\evoefficient_mvp_route_plan_smoke `
  --force
```

The exhaustive per-strategy view still comes from
`run_efficiency_strategy_ablation.py`. The MVP runner learns and reuses a
candidate combination; the ablation runner is the better tool for attributing
which individual strategies helped.

## 7. Held-Out Safety Gate

The train Efficiency Gate can accept a candidate that looks cheaper on the
training issues. The runner now also checks the paired held-out reuse result.
If held-out success regresses or cost does not improve, the final
`efficiency/strategy_library.yaml` leaves those strategies inactive and marks
them as `quarantined`.

Use `--disable-heldout-gate` only when you intentionally want the old behavior
for debugging.

## 8. If You Want to Debug Faster

Use very small batches and lower step limits:

```powershell
python examples\swe_examples_new\run_evoefficient_mvp.py `
  --train-limit 2 `
  --reuse-limit 2 `
  --candidate-mode explicit `
  --candidate-strategies difficulty_model_routing `
  --parallel 1 `
  --max-steps 40 `
  --epsilon-success 0.05 `
  --delta-cost 0.05 `
  --output-dir logs\evoefficient_mvp_debug `
  --force
```
