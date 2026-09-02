#!/usr/bin/env python3
"""Run the EvoEfficient 15.1 MVP loop on SWE-bench Verified mini.

Flow:
  1. Run a train baseline batch and build a cost profile.
  2. Select low-risk candidate strategies from the profile.
  3. Validate the candidate on the same train batch.
  4. Gate the candidate against baseline.
  5. Run paired held-out baseline and reuse batches on the next tasks.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agent_evolve.efficiency.gate import GateConfig, gate_candidate
from agent_evolve.efficiency.library import (
    apply_library_to_workspace,
    build_library_from_gate,
    write_strategy_library,
    record_config_outcome,
)
from agent_evolve.efficiency.pareto import choose_best_candidate, pareto_frontier
from agent_evolve.efficiency.profiler import (
    build_cost_profile,
    result_strategy_hits,
    summarize_results,
)
from agent_evolve.efficiency.selector import (
    candidate_configs_for_profile,
    config_to_runtime_config,
)
from agent_evolve.efficiency.strategies import (
    DEFAULT_STRATEGIES,
    STRATEGY_ORDER,
    apply_strategy_set_to_workspace,
    materialize_strategy_artifacts,
    write_active_config,
)
from agent_evolve.efficiency.visualization import (
    write_plots,
    write_strategy_library_plot,
    write_summary_table,
)


logger = logging.getLogger("evoefficient_mvp")


SAFE_CANDIDATE_STRATEGIES = [
    "observation_masking",
    "verbatim_compaction",
    "simple_memory",
    "plan_template_cache",
    "difficulty_model_routing",
]


def _slug(value: str) -> str:
    return "".join(
        char.lower() if char.isalnum() else "_"
        for char in str(value)
    ).strip("_")[:80] or "config"


def _unique_strategies(configs: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for config in configs:
        for strategy in config.get("strategies", []) or []:
            if strategy in DEFAULT_STRATEGIES and strategy not in values:
                values.append(strategy)
    return values


def _annotate_candidate_config(
    config: dict[str, Any],
    *,
    report: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Attach train evidence to a config before it enters the library."""
    enriched = dict(config)
    enriched["gate"] = dict(gate)
    summary = report.get("summary", {})
    enriched["stats"] = {
        "train_summary": summary,
        "success_rate": summary.get("success_rate", 0.0),
        "avg_tokens": summary.get("avg_tokens", 0.0),
        "avg_tokens_success": summary.get("avg_tokens_success", 0.0),
        "avg_cost": summary.get("avg_cost", summary.get("avg_tokens", 0.0)),
        "avg_cost_success": summary.get("avg_cost_success", 0.0),
        "cost_of_pass": summary.get("cost_of_pass", "inf"),
        "cost_delta_pct": gate.get("cost_delta_pct", 0.0),
        "trials": summary.get("tasks", 0),
        "successes": summary.get("success", 0),
    }
    return enriched


def _load_tasks(dataset: str, limit: int) -> list[dict]:
    from agent_evolve.benchmarks.swe_verified_mini.benchmark import SweVerifiedMiniBenchmark

    bm = SweVerifiedMiniBenchmark(dataset_name=dataset, shuffle=False)
    tasks = bm.get_tasks(split="test", limit=limit)
    return [{"id": t.id, "input": t.input, "metadata": t.metadata} for t in tasks]


def _dedupe_known_strategies(strategies: list[str]) -> list[str]:
    """Keep candidate order stable while dropping unknown strategy ids."""
    return list(dict.fromkeys(s for s in strategies if s in DEFAULT_STRATEGIES))


def _select_strategies(
    profile: dict,
    *,
    mode: str = "bottleneck",
    explicit_strategies: list[str] | None = None,
) -> list[str]:
    """Select candidate strategies for the MVP validation stage."""
    if mode == "explicit":
        return _dedupe_known_strategies(explicit_strategies or [])
    if mode == "all":
        return list(STRATEGY_ORDER)
    if mode == "safe":
        return _dedupe_known_strategies(SAFE_CANDIDATE_STRATEGIES)

    bottlenecks = set(profile.get("bottlenecks", []))
    selected: list[str] = []
    if "context-heavy" in bottlenecks or "long-observation" in bottlenecks:
        selected.append("observation_masking")
        selected.append("verbatim_compaction")
        selected.append("simple_memory")
    if "reasoning-heavy" in bottlenecks or "balanced" in bottlenecks:
        selected.append("cognitive_router")
    if "tool-heavy" in bottlenecks or "balanced" in bottlenecks:
        selected.append("adaptive_tool_trimming")
    if "balanced" in bottlenecks:
        selected.append("plan_template_cache")
        selected.append("difficulty_model_routing")
    if not selected:
        selected = ["verbatim_compaction", "cognitive_router", "adaptive_tool_trimming"]
    return _dedupe_known_strategies(selected)


def _run_batch(
    *,
    group: str,
    strategies: list[str],
    tasks: list[dict],
    args: argparse.Namespace,
    runtime_config: dict[str, Any] | None = None,
    library: dict[str, Any] | None = None,
) -> dict:
    group_dir = Path(args.output_dir) / group
    workspace_dir = group_dir / "workspace"
    results_path = group_dir / "results.json"
    if results_path.exists() and not args.force:
        results = json.loads(results_path.read_text(encoding="utf-8"))
        config_id = next(
            (
                str(result.get("efficiency_config_id", ""))
                for result in results
                if result.get("efficiency_config_id")
            ),
            "",
        )
        return {
            "group": group,
            "strategies": sorted(
                {
                    strategy
                    for result in results
                    for strategy in result.get(
                        "active_efficiency_strategies",
                        result.get("strategies", []),
                    )
                }
            ) or strategies,
            "config_id": config_id,
            "results": results,
            "summary": summarize_results(results),
        }

    group_dir.mkdir(parents=True, exist_ok=True)
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    shutil.copytree(args.seed_workspace, workspace_dir)
    if library is not None:
        write_strategy_library(workspace_dir, library)
        # A structured library is intentionally selected per task by SweAgent.
        # Keep the workspace baseline when no task profile is available here.
        write_active_config(
            workspace_dir,
            [],
            metadata={"experiment": "mvp", "group": group, "library_selection": "per_task"},
        )
    else:
        write_active_config(
            workspace_dir,
            strategies,
            metadata={
                "experiment": "mvp",
                "group": group,
                "efficiency_config_id": (
                    runtime_config.get("config_id", "") if runtime_config else ""
                ),
            },
            parameters=(runtime_config or {}).get("parameters", {}),
        )
        materialize_strategy_artifacts(workspace_dir, strategies)

    logger.info("Running %s with strategies=%s tasks=%d", group, strategies, len(tasks))
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
                runtime_config=runtime_config,
            ): td["id"]
            for td in tasks
        }
        for fut in as_completed(futures):
            iid = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                result = {"instance_id": iid, "success": False, "score": 0.0, "error": str(exc), "elapsed": 0.0}
            result["group"] = group
            result["strategies"] = list(
                result.get("active_efficiency_strategies") or strategies
            )
            if runtime_config and not result.get("efficiency_config_id"):
                result["efficiency_config_id"] = runtime_config.get("config_id", "")
                result["efficiency_selection_reason"] = "explicit_runtime_config"
            result["total_tokens"] = (
                int(result.get("cumulative_input_tokens", 0) or 0)
                + int(result.get("cumulative_output_tokens", 0) or 0)
            )
            result["strategy_hits"] = result_strategy_hits(result)
            results.append(result)
            logger.info("[%s %d/%d] %s %s", group, len(results), len(tasks),
                        "PASS" if result.get("success") else "FAIL", iid)

    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = summarize_results(results)
    report = {
        "group": group,
        "strategies": sorted(
            {
                strategy
                for result in results
                for strategy in result.get(
                    "active_efficiency_strategies",
                    result.get("strategies", []),
                )
            }
        ) or strategies,
        "config_id": next(
            (
                str(result.get("efficiency_config_id", ""))
                for result in results
                if result.get("efficiency_config_id")
            ),
            "",
        ),
        "results": results,
        "summary": summary,
        "elapsed_wall_time": round(time.time() - started, 1),
    }
    (group_dir / "summary.json").write_text(
        json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def _run_batch_from_library(
    *,
    group: str,
    library: dict,
    tasks: list[dict],
    args: argparse.Namespace,
) -> dict:
    return _run_batch(
        group=group,
        strategies=[],
        tasks=tasks,
        args=args,
        library=library,
    )


def _write_report(output_dir: Path, payload: dict, rows: list[dict]) -> None:
    lines = [
        "# EvoEfficient 15.1 MVP SWE Results",
        "",
        "This run separates learning from reuse:",
        "",
        "- train: baseline profile -> candidate validation -> Efficiency Gate",
        "- held-out: baseline and learned strategies run on the same next issue slice",
        "",
        "## Selector",
        "",
        f"- Candidate mode: {payload['candidate_mode']}",
        f"- Baseline bottlenecks: {', '.join(payload['baseline_profile']['bottlenecks'])}",
        f"- Candidate configs: {len(payload.get('candidate_configs', []))}",
        f"- Accepted configs before held-out: {len(payload.get('accepted_configs', []))}",
        f"- Pareto frontier: {', '.join(payload.get('pareto_frontier', [])) or '-'}",
        f"- Selected config: {payload.get('selected_config_id') or '-'}",
        f"- Final active config: {payload.get('final_active_config_id') or '-'}",
        "",
        "## Held-Out Reuse",
        "",
        f"- Held-out baseline success: {payload['heldout_comparison']['baseline_success_rate']:.1%}",
        f"- Held-out reuse success: {payload['heldout_comparison']['reuse_success_rate']:.1%}",
        f"- Held-out success delta: {payload['heldout_comparison']['success_delta']:.1%}",
        f"- Held-out avg token delta: {payload['heldout_comparison']['avg_token_delta_pct']:.1%}",
        f"- Held-out avg cost delta: {payload['heldout_comparison'].get('avg_cost_delta_pct', 0.0):.1%}",
        f"- Held-out avg tool-call delta: {payload['heldout_comparison']['avg_tool_call_delta_pct']:.1%}",
        f"- Held-out verdict: {payload['heldout_comparison']['verdict']}",
        "",
        "## Metrics",
        "",
        "| Group | Strategies | Tasks | Success Rate | Avg Tokens | Avg Tokens(success) | Avg Cost | Main Cost | Route Cost | Cost-of-Pass | Avg Tool Calls | Token Quality |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {', '.join(row.get('strategies', [])) or '-'} | {row['tasks']} | "
            f"{float(row['success_rate']):.1%} | {row['avg_tokens']} | "
            f"{row.get('avg_tokens_success', 0)} | {row.get('avg_cost', row['avg_tokens'])} | "
            f"{row.get('main_model_cost', 0)} | {row.get('route_model_cost', 0)} | "
            f"{row['cost_of_pass']} | "
            f"{row['avg_tool_calls']} | {row.get('token_record_quality', 'unknown')} |"
        )
    lines.extend([
        "",
        "## Strategy Hits",
        "",
        "| Group | Hits |",
        "|---|---|",
    ])
    for row in rows:
        hits = row.get("strategy_hits", {}) or {}
        if hits:
            hit_text = ", ".join(
                f"{name}: {data.get('hits', 0)}"
                for name, data in sorted(hits.items())
                if isinstance(data, dict)
            )
        else:
            hit_text = "-"
        lines.append(f"| {row['group']} | {hit_text or '-'} |")

    lines.extend([
        "",
        "## Artifacts",
        "",
        "- `mvp_report.json`",
        "- `efficiency/strategy_library.yaml`",
        "- `summary.csv`",
        "- `figures/strategy_avg_tokens.png`",
        "- `figures/strategy_avg_cost.png`",
        "- `figures/strategy_cost_success.png`",
        "- `figures/strategy_tool_calls.png`",
        "- `figures/strategy_efficiency_frontier.png`",
        "- `figures/strategy_hits.png`",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compare_heldout(
    baseline_summary: dict,
    reuse_summary: dict,
    *,
    epsilon_success: float,
) -> dict[str, float | str | bool]:
    b_success = float(baseline_summary.get("success_rate", 0.0) or 0.0)
    r_success = float(reuse_summary.get("success_rate", 0.0) or 0.0)
    b_tokens = float(baseline_summary.get("avg_tokens", 0.0) or 0.0)
    r_tokens = float(reuse_summary.get("avg_tokens", 0.0) or 0.0)
    b_cost = float(baseline_summary.get("avg_cost", b_tokens) or 0.0)
    r_cost = float(reuse_summary.get("avg_cost", r_tokens) or 0.0)
    b_calls = float(baseline_summary.get("avg_tool_calls", 0.0) or 0.0)
    r_calls = float(reuse_summary.get("avg_tool_calls", 0.0) or 0.0)
    success_ok = r_success >= b_success - epsilon_success
    cost_improved = r_cost < b_cost if b_cost > 0 else False
    tool_improved = r_calls < b_calls if b_calls > 0 else False
    verdict = (
        "success_preserved_and_cost_reduced"
        if success_ok and (cost_improved or tool_improved)
        else "success_preserved_no_cost_gain"
        if success_ok
        else "success_regressed"
    )
    return {
        "baseline_success_rate": b_success,
        "reuse_success_rate": r_success,
        "success_delta": round(r_success - b_success, 6),
        "baseline_avg_tokens": b_tokens,
        "reuse_avg_tokens": r_tokens,
        "avg_token_delta_pct": round((r_tokens - b_tokens) / b_tokens, 6) if b_tokens else 0.0,
        "baseline_avg_cost": b_cost,
        "reuse_avg_cost": r_cost,
        "avg_cost_delta_pct": round((r_cost - b_cost) / b_cost, 6) if b_cost else 0.0,
        "baseline_avg_tool_calls": b_calls,
        "reuse_avg_tool_calls": r_calls,
        "avg_tool_call_delta_pct": round((r_calls - b_calls) / b_calls, 6) if b_calls else 0.0,
        "success_ok": success_ok,
        "cost_improved": token_improved or tool_improved,
        "verdict": verdict,
    }


def _accepted_after_heldout(
    accepted_strategies: list[str],
    heldout_comparison: dict[str, Any],
    *,
    enforce_heldout_gate: bool,
) -> list[str]:
    if not enforce_heldout_gate:
        return accepted_strategies
    if heldout_comparison.get("success_ok") and heldout_comparison.get("cost_improved"):
        return accepted_strategies
    return []


def _task_profile_from_cost_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Build the selector-facing task profile for the train slice."""
    task_profile = dict(profile.get("task_profile", {}) or {})
    task_profile.setdefault("task_type", "coding")
    task_profile.setdefault("domain", "swebench_verified")
    bottlenecks = set(profile.get("bottlenecks", []) or [])
    task_profile["bottlenecks"] = sorted(bottlenecks)
    task_profile.setdefault("difficulty", "hard")
    task_profile["context_pressure"] = (
        "hard" if {"context-heavy", "long-observation"} & bottlenecks else "medium"
    )
    task_profile["tool_pressure"] = "hard" if "tool-heavy" in bottlenecks else "medium"
    task_profile["verification_pressure"] = "hard"
    return task_profile


def _config_eval_record(
    config: dict[str, Any],
    report: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "config_id": str(config.get("id", "")),
        "config": config,
        "summary": report.get("summary", {}),
        "gate": gate,
        "strategies": list(config.get("strategies", []) or []),
        "group": report.get("group", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run EvoEfficient 15.1 MVP on SWE-bench mini")
    parser.add_argument("--dataset", default="MariusHobbhahn/swe-bench-verified-mini")
    parser.add_argument("--train-limit", type=int, default=4)
    parser.add_argument("--reuse-limit", type=int, default=4,
                        help="Number of held-out tasks for paired baseline/reuse evaluation.")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--model-id", default="us.anthropic.claude-opus-4-6-v1")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--window-size", type=int, default=40)
    parser.add_argument("--seed-workspace", default=str(REPO_ROOT / "seed_workspaces" / "swe"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "logs" / "evoefficient_mvp"))
    parser.add_argument(
        "--candidate-mode",
        choices=["meta", "bottleneck", "safe", "all", "explicit"],
        default="meta",
        help=(
            "How to choose candidate strategies. meta uses the baseline profile; "
            "bottleneck is a backward-compatible alias; "
            "safe uses a conservative low-risk set; all validates every registered strategy; "
            "explicit uses --candidate-strategies."
        ),
    )
    parser.add_argument(
        "--candidate-strategies",
        nargs="*",
        default=None,
        help="Strategy ids used when --candidate-mode explicit is selected.",
    )
    parser.add_argument(
        "--max-candidate-configs",
        type=int,
        default=8,
        help="Maximum independent efficiency configs to validate in the train slice.",
    )
    parser.add_argument(
        "--disable-heldout-gate",
        action="store_true",
        help="Keep train-gate accepted strategies active even when held-out reuse regresses.",
    )
    parser.add_argument("--epsilon-success", type=float, default=0.03)
    parser.add_argument("--delta-cost", type=float, default=0.10)
    parser.add_argument("--min-trials", type=int, default=2)
    parser.add_argument("--profile-overhead-tokens", type=float, default=0.0,
                        help="Estimated one-time token cost for profiling the batch.")
    parser.add_argument("--selector-overhead-tokens", type=float, default=0.0,
                        help="Estimated one-time token cost for choosing candidate strategies.")
    parser.add_argument("--validation-overhead-tokens", type=float, default=0.0,
                        help="Extra validation overhead beyond normal candidate execution.")
    parser.add_argument("--expected-future-hits", type=int, default=10,
                        help="Expected future tasks that will reuse an accepted strategy.")
    parser.add_argument("--max-break-even-hits", type=int, default=None,
                        help="Reject strategies that require more hits than this to recover discovery overhead.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    total_limit = args.train_limit + args.reuse_limit
    all_tasks = _load_tasks(args.dataset, total_limit)
    train_tasks = all_tasks[: args.train_limit]
    reuse_tasks = all_tasks[args.train_limit: args.train_limit + args.reuse_limit] or train_tasks

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = _run_batch(
        group="01_train_baseline_profile",
        strategies=[],
        tasks=train_tasks,
        args=args,
    )
    baseline_profile = build_cost_profile(baseline["results"], group="baseline")
    selector_task_profile = _task_profile_from_cost_profile(baseline_profile)
    selector_mode = (
        "meta"
        if args.candidate_mode in {"meta", "bottleneck"}
        else args.candidate_mode
    )
    candidate_configs = candidate_configs_for_profile(
        selector_task_profile,
        baseline_profile,
        mode=selector_mode,
        explicit_strategies=args.candidate_strategies,
        max_configs=args.max_candidate_configs,
    )
    if not candidate_configs:
        logger.warning("No candidate efficiency configs were generated; continuing with baseline reuse.")

    gate_config = GateConfig(
        epsilon_success=args.epsilon_success,
        delta_cost=args.delta_cost,
        min_trials=args.min_trials,
        discovery_overhead_tokens=(
            args.profile_overhead_tokens
            + args.selector_overhead_tokens
            + args.validation_overhead_tokens
        ),
        expected_future_hits=args.expected_future_hits,
        max_break_even_hits=args.max_break_even_hits,
    )

    candidate_reports: list[dict[str, Any]] = []
    candidate_evals: list[dict[str, Any]] = []
    for index, raw_config in enumerate(candidate_configs, start=1):
        config = dict(raw_config)
        config_id = str(config.get("id", f"candidate-{index}"))
        runtime_config = config_to_runtime_config(config)
        candidate = _run_batch(
            group=f"02_train_{index:02d}_{_slug(config_id)}",
            strategies=list(config.get("strategies", []) or []),
            tasks=train_tasks,
            args=args,
            runtime_config=runtime_config,
        )
        candidate_profile = build_cost_profile(
            candidate["results"],
            group=config_id,
        )
        gate = gate_candidate(
            baseline_profile,
            candidate_profile,
            config=gate_config,
        )
        enriched = _annotate_candidate_config(
            config,
            report=candidate,
            gate=gate,
        )
        candidate_reports.append(candidate)
        candidate_evals.append(_config_eval_record(enriched, candidate, gate))
        candidate_configs[index - 1] = enriched
        logger.info(
            "Candidate %s: success=%.3f avg_tokens=%s gate=%s",
            config_id,
            float(candidate["summary"].get("success_rate", 0.0) or 0.0),
            candidate["summary"].get("avg_tokens", 0),
            gate.get("reason", "unknown"),
        )

    accepted_evals = [
        item for item in candidate_evals
        if item.get("gate", {}).get("accepted")
    ]
    frontier_evals = pareto_frontier(
        accepted_evals,
        epsilon_success=args.epsilon_success,
    ) if accepted_evals else []
    selected_eval = choose_best_candidate(
        frontier_evals,
        baseline=baseline_profile,
        epsilon_success=args.epsilon_success,
    ) if frontier_evals else None
    selected_config_id = (
        str(selected_eval.get("config_id", ""))
        if selected_eval else ""
    )
    selected_gate = (
        selected_eval.get("gate", {})
        if selected_eval
        else {
            "accepted": False,
            "reason": "no_candidate_config_passed_gate",
        }
    )
    selected_config = (
        selected_eval.get("config", {})
        if selected_eval
        else {}
    )
    accepted_strategies = (
        list(selected_config.get("strategies", []) or [])
        if selected_config_id else []
    )
    pareto_ids = [
        str(item.get("config_id", ""))
        for item in frontier_evals
        if item.get("config_id")
    ]

    heldout_baseline = _run_batch(
        group="03_heldout_baseline",
        strategies=[],
        tasks=reuse_tasks,
        args=args,
    )
    strategy_library = build_library_from_gate(
        candidate_strategies=_unique_strategies(candidate_configs),
        accepted_strategies=accepted_strategies,
        gate=selected_gate,
        baseline_summary=baseline["summary"],
        candidate_summary=(
            selected_eval.get("summary", {})
            if selected_eval else baseline["summary"]
        ),
        selector_profile=selector_task_profile,
        candidate_configs=candidate_configs,
        selected_config_id=selected_config_id,
        pareto_frontier=pareto_ids,
    )
    write_strategy_library(output_dir, strategy_library)
    heldout_reuse = _run_batch_from_library(
        group=(
            "04_heldout_reuse_selected"
            if selected_config_id
            else "04_heldout_reuse_baseline_fallback"
        ),
        library=strategy_library,
        tasks=reuse_tasks,
        args=args,
    )
    for result in heldout_reuse["results"]:
        record_config_outcome(
            strategy_library,
            config_id=result.get("efficiency_config_id"),
            result=result,
        )
    heldout_comparison = _compare_heldout(
        heldout_baseline["summary"],
        heldout_reuse["summary"],
        epsilon_success=args.epsilon_success,
    )
    final_active_config_id = (
        selected_config_id
        if (
            selected_config_id
            and (
                args.disable_heldout_gate
                or (
                    heldout_comparison.get("success_ok")
                    and heldout_comparison.get("cost_improved")
                )
            )
        )
        else ""
    )
    final_accepted_strategies = (
        accepted_strategies if final_active_config_id else []
    )
    status_reason = (
        f"heldout_verdict={heldout_comparison['verdict']}"
        if selected_config_id and final_active_config_id != selected_config_id
        else ""
    )
    # Preserve online held-out statistics in the config records passed to the
    # final library snapshot.
    updated_stats = {
        str(config.get("id", "")): config.get("stats", {})
        for config in strategy_library.get("configs", [])
        if isinstance(config, dict)
    }
    for config in candidate_configs:
        config_id = str(config.get("id", ""))
        if config_id in updated_stats:
            config["stats"] = updated_stats[config_id]
    strategy_library = build_library_from_gate(
        candidate_strategies=_unique_strategies(candidate_configs),
        accepted_strategies=final_accepted_strategies,
        gate=selected_gate,
        baseline_summary=baseline["summary"],
        candidate_summary=(
            selected_eval.get("summary", {})
            if selected_eval else baseline["summary"]
        ),
        reuse_summary=heldout_reuse["summary"],
        selector_profile=selector_task_profile,
        status_reason=status_reason,
        candidate_configs=candidate_configs,
        selected_config_id=final_active_config_id or selected_config_id,
        pareto_frontier=pareto_ids,
    )
    write_strategy_library(output_dir, strategy_library)

    rows = []
    for report in (
        [baseline]
        + candidate_reports
        + [heldout_baseline, heldout_reuse]
    ):
        row = {"group": report["group"], "strategies": report["strategies"]}
        row.update(report["summary"])
        rows.append(row)

    payload = {
        "baseline_profile": baseline_profile,
        "candidate_mode": args.candidate_mode,
        "candidate_configs": candidate_configs,
        "accepted_configs": [
            item.get("config_id") for item in accepted_evals
        ],
        "pareto_frontier": pareto_ids,
        "selected_config_id": selected_config_id,
        "final_active_config_id": final_active_config_id,
        "gate": selected_gate,
        "train_gate_accepted_strategies": accepted_strategies,
        "accepted_strategies": final_accepted_strategies,
        "heldout_comparison": heldout_comparison,
        "heldout_gate_enforced": not args.disable_heldout_gate,
        "strategy_library": strategy_library,
        "rows": rows,
    }
    (output_dir / "mvp_report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_table(rows, output_dir / "summary.csv")
    write_plots(rows, output_dir / "figures")
    write_strategy_library_plot(rows, output_dir / "figures")
    _write_report(output_dir, payload, rows)

    print(f"\nMVP complete: {output_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
