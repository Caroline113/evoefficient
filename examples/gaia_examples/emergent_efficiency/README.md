# GAIA Validation Level 3 Emergent Efficiency

This isolated runner applies the same five-condition experiment design used by
the SWE and Terminal-Bench runners to the 26 GAIA 2023 validation Level 3
tasks. It does not change either existing benchmark implementation.

## Prerequisites

1. Accept the gated dataset terms at
   <https://huggingface.co/datasets/gaia-benchmark/GAIA>.
2. Authenticate with `hf auth login` or export `HF_TOKEN`.
3. Export the solver/evolver key (`DEEPSEEK_API_KEY` for the default model).
4. Optional but recommended: export `TAVILY_API_KEY` for reliable web search.

Install dependencies with:

```bash
uv sync --extra gaia
```

Alternatively, in an existing environment:

```bash
pip install strands-agents datasets huggingface-hub openai 'httpx[socks]' \
  pandas openpyxl pypdf pillow
```

## Environment configuration

```bash
# 1. GAIA dataset access (required unless already authenticated with hf CLI)
export HF_TOKEN=hf_...

# 2. Default solver and evolver provider
export DEEPSEEK_API_KEY=...

# 3. Reliable live search for Level 3 (strongly recommended)
export TAVILY_API_KEY=tvly-...
```

Before running, visit the official GAIA dataset page while logged in and
accept its gated-access terms. `HF_TOKEN` must belong to that same account.
Without `TAVILY_API_KEY`, the agent falls back to DuckDuckGo HTML search,
which is less stable and may be rate-limited.

If the dataset has already been downloaded, avoid any Hub access by passing
the snapshot root explicitly:

```bash
--dataset-dir /mnt/cache/modelscope/datasets/AI-ModelScope--GAIA/snapshots/master
```

That directory must contain the official `README.md` dataset configuration
and `2023/validation/metadata.level3.parquet`, together with its attachments.

## Smoke test

```bash
python examples/gaia_examples/emergent_efficiency/run_emergent_efficiency_gaia.py \
  --dataset-dir /mnt/cache/modelscope/datasets/AI-ModelScope--GAIA/snapshots/master \
  --mode normal-evolve \
  --batch-size 4 \
  --parallel 2
```

## Full five-condition matrix

```bash
bash examples/gaia_examples/emergent_efficiency/run_matrix.sh gaia_l3 \
  --dataset-dir /mnt/cache/modelscope/datasets/AI-ModelScope--GAIA/snapshots/master \
  --batch-size 4 \
  --parallel 2
```

Omit `--limit` to run all 26 Level 3 validation tasks. The official-style
deterministic scorer is used for numeric, list, and string answers.

## Metrics and outputs

The output schema matches the SWE/TB experiment family:

- success count/rate and per-task exact-match result;
- input, output, cache-read, cache-write, and total non-cache tokens;
- USD cost, cost per successful task, tool calls, turns, and elapsed time;
- per-batch summaries, behavior signals, proposal/adoption records, workspace
  snapshots, conversations, and evolved workspace state.

GAIA-only tools are web search, page fetch, attachment inspection, Python
calculation, skill loading, and exact final-answer submission. Attachments and
ground-truth answers remain local to the evaluator; answers are never placed in
the agent prompt.

The generated logs contain validation ground-truth answers in `results.json`
for local analysis. Because GAIA is gated, do not publish or commit those logs
to a public repository.

## Pricing

The default price profile is intentionally aligned with the SWE/TB experiment
workspace and lives at:

```text
agent_evolve/agents/gaia/seed_workspace/efficiency/cost_profile.yaml
```

Update the per-million-token prices there before a run if the selected model's
pricing differs. Input, output, cache-read, and cache-write tokens are recorded
separately and all four components contribute to `total_cost`.
