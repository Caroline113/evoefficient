#!/usr/bin/env python3
"""Run single-strategy EvoEfficient ablations on SWE-bench Verified mini.

Each group receives an isolated workspace so the same task batch can be run
with baseline behavior, one efficiency strategy at a time, and an all-strategy
configuration. Results are written as JSON, CSV, Markdown, and PNG plots.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agent_evolve.efficiency.profiler import build_cost_profile, summarize_results
from agent_evolve.efficiency.strategies import (
    DEFAULT_STRATEGIES,
    STRATEGY_ORDER,
    apply_strategy_set_to_workspace,
)
from agent_evolve.efficiency.visualization import (
    write_plots,
    write_strategy_library_plot,
    write_summary_table,
)


logger = logging.getLogger("strategy_ablation")


NEW_STRATEGIES = [
    "simple_memory",
    "observation_masking",
    "verbatim_compaction",
    "cognitive_router",
    "plan_template_cache",
    "adaptive_tool_trimming",
    "difficulty_model_routing",
]

PRESET_COMBINATIONS: dict[str, list[str]] = {
    "combo_m1_context": ["simple_memory", "observation_masking", "verbatim_compaction"],
    "combo_m2_m4_reason_plan": ["cognitive_router", "plan_template_cache"],
    "combo_m5_m7_tool_route": ["adaptive_tool_trimming", "difficulty_model_routing"],
    "combo_context_tool": ["observation_masking", "verbatim_compaction", "adaptive_tool_trimming"],
    "combo_new_all": NEW_STRATEGIES,
}


def _task_dicts(dataset: str, limit: int) -> list[dict]:
    from agent_evolve.benchmarks.swe_verified_mini.benchmark import SweVerifiedMiniBenchmark

    bm = SweVerifiedMiniBenchmark(dataset_name=dataset, shuffle=False)
    tasks = bm.get_tasks(split="test", limit=limit)
    return [{"id": t.id, "input": t.input, "metadata": t.metadata} for t in tasks]


def _run_group(
    *,
    group: str,
    strategies: list[str],
    tasks: list[dict],
    args: argparse.Namespace,
) -> dict:
    group_dir = Path(args.output_dir) / group
    workspace_dir = group_dir / "workspace"
    results_path = group_dir / "results.json"
    if results_path.exists() and not args.force:
        results = json.loads(results_path.read_text(encoding="utf-8"))
        summary = summarize_results(results)
        return {"group": group, "strategies": strategies, "summary": summary, "results_path": str(results_path)}

    group_dir.mkdir(parents=True, exist_ok=True)
    apply_strategy_set_to_workspace(
        args.seed_workspace,
        workspace_dir,
        strategies,
        metadata={"experiment": "single_strategy_ablation", "group": group},
    )

    logger.info("Running group=%s strategies=%s tasks=%d", group, strategies, len(tasks))
    from evolve_sequential import solve_one_task

    started = time.time()
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(
                solve_one_task,
                td,
                str(workspace_dir),
                args.model_id,
                args.region,
                args.max_tokens,
                args.max_steps,
                args.window_size,
                False,
                False,
                args.dataset,
            ): td["id"]
            for td in tasks
        }
        for fut in as_completed(futures):
            iid = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                result = {
                    "instance_id": iid,
                    "success": False,
                    "score": 0.0,
                    "error": str(exc),
                    "elapsed": 0.0,
                    "patch_len": 0,
                }
            result["group"] = group
            result["strategies"] = strategies
            results.append(result)
            logger.info(
                "[%s %d/%d] %s %s tokens=%s calls=%s",
                group,
                len(results),
                len(tasks),
                "PASS" if result.get("success") else "FAIL",
                iid,
                int(result.get("cumulative_input_tokens", 0) or 0)
                + int(result.get("cumulative_output_tokens", 0) or 0),
                result.get("num_tool_calls", 0),
            )

    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = summarize_results(results)
    report = {
        "group": group,
        "strategies": strategies,
        "summary": summary,
        "profile": build_cost_profile(results, group=group),
        "elapsed_wall_time": round(time.time() - started, 1),
        "results_path": str(results_path),
    }
    (group_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _write_markdown_report(rows: list[dict], output_dir: Path) -> None:
    lines = [
        "# EvoEfficient SWE Strategy Ablation Results",
        "",
        "| Group | Strategies | Tasks | Success | Success Rate | Avg Tokens | Avg Tokens(success) | Avg Cost | Main Cost | Route Cost | Cost-of-Pass | Avg Tool Calls |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    baseline = next((r for r in rows if r["group"] == "baseline"), None)
    baseline_avg = float(baseline["avg_tokens"]) if baseline else 0.0
    for row in rows:
        strategies = row.get("strategies", [])
        saving = ""
        if baseline_avg and row["group"] != "baseline":
            saving = f" ({(baseline_avg - float(row['avg_tokens'])) / baseline_avg:.1%})"
        lines.append(
            f"| {row['group']} | {', '.join(strategies) or '-'} | {row['tasks']} | "
            f"{row['success']} | {float(row['success_rate']):.1%} | "
            f"{row['avg_tokens']}{saving} | {row.get('avg_tokens_success', 0)} | "
            f"{row.get('avg_cost', row['avg_tokens'])} | "
            f"{row.get('main_model_cost', 0)} | {row.get('route_model_cost', 0)} | "
            f"{row['cost_of_pass']} | "
            f"{row['avg_tool_calls']} |"
        )
    lines.extend([
        "",
        "## Plots",
        "",
        "![Average tokens](figures/strategy_avg_tokens.png)",
        "",
        "![Average cost](figures/strategy_avg_cost.png)",
        "",
        "![Cost vs success](figures/strategy_cost_success.png)",
        "",
        "![Tool calls](figures/strategy_tool_calls.png)",
        "",
        "![Efficiency frontier](figures/strategy_efficiency_frontier.png)",
        "",
        "## Notes",
        "",
        "- `avg_tokens` is input plus output tokens recorded by the solver.",
        "- `avg_cost` includes input, output, cache-read, and cache-write cost units.",
        "- `main_model_cost` is the hard-model agent loop cost; `route_model_cost` is delegated model cost.",
        "- `cost_of_pass` is total cost units divided by successful tasks; lower is better.",
        "- `strategy_hits.png` shows whether runtime strategies actually fired.",
        "- Runtime errors should be checked in each group `results.json` before drawing final conclusions.",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SWE-bench mini efficiency strategy ablations")
    parser.add_argument("--dataset", default="MariusHobbhahn/swe-bench-verified-mini")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--model-id", default="us.anthropic.claude-opus-4-6-v1")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--window-size", type=int, default=40)
    parser.add_argument("--seed-workspace", default=str(REPO_ROOT / "seed_workspaces" / "swe"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "logs" / "evoefficient_strategy_ablation"))
    parser.add_argument("--strategies", nargs="*", default=NEW_STRATEGIES,
                        help="Strategy ids to test. Defaults to all registered strategy ids.")
    parser.add_argument("--include-all", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-legacy", action="store_true",
                        help="Also test legacy strategies that existed before this implementation pass.")
    parser.add_argument("--include-preset-combos", action=argparse.BooleanOptionalAction, default=True,
                        help="Run predefined useful strategy combinations.")
    parser.add_argument("--combo", action="append", default=[],
                        help="Custom combo in the form name:s1,s2,s3. Can be used multiple times.")
    parser.add_argument("--force", action="store_true", help="Re-run groups even if results already exist.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    tasks = _task_dicts(args.dataset, args.limit)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_strategies = list(dict.fromkeys(args.strategies))
    if args.include_legacy:
        selected_strategies = list(dict.fromkeys([*selected_strategies, *STRATEGY_ORDER]))

    groups: list[tuple[str, list[str]]] = [("baseline", [])]
    for strategy in selected_strategies:
        if strategy not in DEFAULT_STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}")
        groups.append((strategy, [strategy]))

    if args.include_preset_combos:
        for name, combo_strategies in PRESET_COMBINATIONS.items():
            active = [s for s in combo_strategies if s in selected_strategies and s in DEFAULT_STRATEGIES]
            if len(active) >= 2:
                groups.append((name, active))

    for spec in args.combo:
        if ":" not in spec:
            raise ValueError(f"Invalid --combo {spec!r}. Expected name:s1,s2,s3")
        name, raw = spec.split(":", 1)
        combo_strategies = [s.strip() for s in raw.split(",") if s.strip()]
        unknown = [s for s in combo_strategies if s not in DEFAULT_STRATEGIES]
        if unknown:
            raise ValueError(f"Unknown strategies in --combo {name}: {unknown}")
        groups.append((name.strip(), combo_strategies))

    if args.include_all:
        groups.append(("all_selected_strategies", [s for s in selected_strategies if s in DEFAULT_STRATEGIES]))

    deduped_groups: list[tuple[str, list[str]]] = []
    seen = set()
    for name, strategies in groups:
        key = (name, tuple(strategies))
        if key in seen:
            continue
        seen.add(key)
        deduped_groups.append((name, strategies))

    reports = [_run_group(group=name, strategies=strategies, tasks=tasks, args=args) for name, strategies in deduped_groups]

    rows = []
    for report in reports:
        row = {"group": report["group"], "strategies": report["strategies"]}
        row.update(report["summary"])
        rows.append(row)

    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_table(rows, output_dir / "summary.csv")
    write_plots(rows, output_dir / "figures")
    write_strategy_library_plot(rows, output_dir / "figures")
    _write_markdown_report(rows, output_dir)

    print(f"\nAblation complete: {output_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
