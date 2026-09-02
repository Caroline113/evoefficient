#!/usr/bin/env python3
"""In-situ batched evolution: solve tasks in batches, evolve between batches.

The agent solves all 50 swe_verified_mini tasks in batches of N (default 5).
Each batch runs in parallel with a shared workspace snapshot. After each batch,
the evolver analyzes results and generates interventions that carry forward
to the next batch.

Two feedback modes:
  - none:    evolver sees trajectories but NOT whether the solver succeeded
  - minimal: evolver sees trajectories AND whether the solver succeeded (score)

Usage:
    python evolve_sequential.py --batch-size 5 --parallel 5 --feedback none --output-dir logs/seq_none
    python evolve_sequential.py --batch-size 5 --parallel 5 --feedback minimal --output-dir logs/seq_minimal
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent_evolve.config import EvolveConfig
from agent_evolve.engine.observer import Observer
from agent_evolve.types import Feedback, Observation, Trajectory
from agent_evolve.efficiency.gate import GateConfig, gate_candidate
from agent_evolve.efficiency.library import (
    load_strategy_library,
    record_config_outcome,
    write_strategy_library,
)
from agent_evolve.efficiency.pareto import choose_best_candidate, pareto_frontier
from agent_evolve.efficiency.profiler import build_cost_profile, normalize_usage, summarize_results
from agent_evolve.efficiency.selector import (
    candidate_configs_for_profile,
    config_to_runtime_config,
)
from agent_evolve.efficiency.strategies import (
    materialize_strategy_artifacts,
    write_active_config,
)
from agent_evolve.efficiency.visualization import (
    write_plots,
    write_strategy_library_plot,
    write_summary_table,
)

logger = logging.getLogger("evolve_seq")


def _slug(value: str) -> str:
    return "".join(
        char.lower() if char.isalnum() else "_"
        for char in str(value)
    ).strip("_")[:80] or "config"


def _usage_quality(usage: dict[str, Any]) -> str:
    quality = str(usage.get("usage_record_quality") or "").strip()
    if quality:
        return quality
    if any(
        int(usage.get(name, 0) or 0) > 0
        for name in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cache_read_input_tokens",
            "cache_write_input_tokens",
        )
    ):
        return "ok"
    return "missing"


def _usage_from_result_fields(result: dict[str, Any]) -> dict[str, Any]:
    return normalize_usage(
        {
            "input_tokens": result.get("cumulative_input_tokens", 0),
            "output_tokens": result.get("cumulative_output_tokens", 0),
            "total_tokens": result.get("total_tokens", 0),
            "cache_read_input_tokens": result.get("cumulative_cache_read_input_tokens", 0),
            "cache_write_input_tokens": result.get("cumulative_cache_write_input_tokens", 0),
            "cost_units": result.get("cost_units", 0),
            "main_cost_units": result.get("main_cost_units", 0),
            "route_model_cost_units": result.get("route_model_cost_units", 0),
            "model_id": result.get("main_model_id", result.get("model_id", "")),
            "provider": result.get("main_model_provider", result.get("provider", "")),
            "route_level": result.get("main_route_level", result.get("route_level", "")),
            "input_price_per_1m": result.get("input_price_per_1m", 0),
            "output_price_per_1m": result.get("output_price_per_1m", 0),
            "cache_read_price_per_1m": result.get("cache_read_price_per_1m", 0),
            "cache_write_price_per_1m": result.get("cache_write_price_per_1m", 0),
            "cost_currency": result.get("cost_currency", ""),
            "cost_price_source": result.get("cost_price_source", ""),
            "cost_priced": result.get("cost_priced", False),
            "usage_record_quality": result.get("usage_record_quality", ""),
            "usage_source": result.get("usage_source", ""),
            "usage_estimated": result.get("usage_estimated", False),
        }
    )


def _attach_usage_fields(result: dict[str, Any]) -> dict[str, Any]:
    usage = _usage_from_result_fields(result)
    result["cumulative_input_tokens"] = int(usage.get("input_tokens", 0) or 0)
    result["cumulative_output_tokens"] = int(usage.get("output_tokens", 0) or 0)
    result["total_tokens"] = int(usage.get("total_tokens", 0) or 0)
    result["cumulative_cache_read_input_tokens"] = int(
        usage.get("cache_read_input_tokens", 0) or 0
    )
    result["cumulative_cache_write_input_tokens"] = int(
        usage.get("cache_write_input_tokens", 0) or 0
    )
    result["cost_units"] = float(usage.get("cost_units", 0.0) or 0.0)
    result["main_cost_units"] = float(usage.get("main_cost_units", result["cost_units"]) or 0.0)
    result["route_model_cost_units"] = float(usage.get("route_model_cost_units", 0.0) or 0.0)
    result["main_model_id"] = str(usage.get("model_id", ""))
    result["main_model_provider"] = str(usage.get("provider", ""))
    result["main_route_level"] = str(usage.get("route_level", ""))
    result["input_price_per_1m"] = float(usage.get("input_price_per_1m", 0.0) or 0.0)
    result["output_price_per_1m"] = float(usage.get("output_price_per_1m", 0.0) or 0.0)
    result["cache_read_price_per_1m"] = float(usage.get("cache_read_price_per_1m", 0.0) or 0.0)
    result["cache_write_price_per_1m"] = float(usage.get("cache_write_price_per_1m", 0.0) or 0.0)
    result["cost_currency"] = str(usage.get("cost_currency", ""))
    result["cost_price_source"] = str(usage.get("cost_price_source", ""))
    result["cost_priced"] = bool(usage.get("cost_priced", False))
    result["usage_record_quality"] = _usage_quality(usage)
    result["usage_source"] = str(usage.get("usage_source", ""))
    result["usage_estimated"] = bool(usage.get("usage_estimated", False))
    return result


def _unique_strings(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if isinstance(value, list):
            nested = _unique_strings(value)
            for item in nested:
                if item not in seen:
                    seen.add(item)
                    out.append(item)
            continue
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _summary_row(group: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    strategies = _unique_strings([
        result.get("active_efficiency_strategies", [])
        for result in results
    ])
    row: dict[str, Any] = {"group": group, "strategies": strategies}
    row.update(summarize_results(results))
    return row


def _sequential_summary_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    generations = sorted({
        int(result.get("evolve_generation", 0) or 0)
        for result in results
    })
    for generation in generations:
        gen_results = [
            result for result in results
            if int(result.get("evolve_generation", 0) or 0) == generation
        ]
        rows.append(_summary_row(f"gen{generation}", gen_results))
    rows.append(_summary_row("overall", results))
    return rows


def solve_one_task(
    task_dict: dict,
    workspace_dir: str,
    model_id: str,
    region: str,
    max_tokens: int,
    max_steps: int = 0,
    window_size: int = 40,
    verification_focus: bool = False,
    efficiency_prompt: bool = False,
    dataset: str = "MariusHobbhahn/swe-bench-verified-mini",
    enable_skill_proposal: bool = False,
    runtime_config: dict | None = None,
) -> dict:
    """Solve a single task in its own process. Workspace is read-only during batch."""
    import logging
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from agent_evolve.agents.swe.agent import SweAgent
    from agent_evolve.benchmarks.swe_verified_mini.benchmark import SweVerifiedMiniBenchmark
    from agent_evolve.types import Task

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [{task_dict['id']}] %(message)s",
    )
    for n in ("botocore", "urllib3", "httpcore", "httpx",
              "strands.models", "strands.tools", "strands.telemetry"):
        logging.getLogger(n).setLevel(logging.WARNING)
    log = logging.getLogger("worker")

    task = Task(id=task_dict["id"], input=task_dict["input"], metadata=task_dict["metadata"])
    instance_id = task.id

    # Create agent from shared workspace (read-only during parallel solve)
    agent = SweAgent(
        workspace_dir=workspace_dir,
        model_id=model_id,
        region=region,
        max_tokens=max_tokens,
        max_steps=max_steps,
        window_size=window_size,
        verification_focus=verification_focus,
        efficiency_prompt=efficiency_prompt,
        enable_skill_proposal=enable_skill_proposal,
        runtime_config=runtime_config,
    )

    bm = SweVerifiedMiniBenchmark(dataset_name=dataset, shuffle=False)

    t0 = time.time()
    trajectory = None
    for attempt in range(3):
        try:
            trajectory = agent.solve(task)
            break
        except Exception as e:
            err_str = str(e)
            transient = any(k in err_str for k in (
                "internalServerException", "ThrottlingException",
                "timed out", "Read timed out", "ServiceUnavailableException",
            ))
            if transient and attempt < 2:
                wait = 30 * (attempt + 1)
                log.warning("solve attempt %d failed (transient), retrying in %ds: %s",
                            attempt + 1, wait, e)
                time.sleep(wait)
                continue
            log.error("solve failed after %d attempt(s): %s", attempt + 1, e)
            return _attach_usage_fields({
                "instance_id": instance_id,
                "success": False,
                "score": 0.0,
                "error": err_str,
                "elapsed": round(time.time() - t0, 1),
                "patch_len": 0,
                "patch": "",
                "usage_record_quality": "missing",
                "usage_source": "solve_exception_before_trajectory",
            })

    if trajectory is None:
        return _attach_usage_fields({
            "instance_id": instance_id,
            "success": False,
            "score": 0.0,
            "error": "no trajectory",
            "elapsed": round(time.time() - t0, 1),
            "patch_len": 0,
            "patch": "",
            "usage_record_quality": "missing",
            "usage_source": "no_trajectory",
        })

    elapsed = time.time() - t0
    feedback = bm.evaluate(task, trajectory)
    log.info("%s score=%.1f elapsed=%.1fs patch=%dch",
             "PASS" if feedback.success else "FAIL",
             feedback.score, elapsed, len(trajectory.output))

    # Extract per-turn stats from trajectory
    num_tool_calls = len([s for s in trajectory.steps if isinstance(s, dict) and "tool" in s])
    usage_step = next(
        (
            s for s in reversed(trajectory.steps)
            if isinstance(s, dict) and "usage" in s
        ),
        {},
    )
    usage = normalize_usage(usage_step.get("usage", {}) if isinstance(usage_step, dict) else {})
    per_turn = usage_step.get("per_turn_usage", []) if isinstance(usage_step, dict) else []
    max_input = usage_step.get("max_input_tokens_per_turn", 0) if isinstance(usage_step, dict) else 0
    route_model_calls = usage_step.get("route_model_calls", []) if isinstance(usage_step, dict) else []
    active_strategies = usage_step.get("active_efficiency_strategies", []) if isinstance(usage_step, dict) else []

    log.info("stats: %d tool calls, %d turns, max_input=%d, cumulative_input=%d",
             num_tool_calls, len(per_turn), max_input, usage.get("input_tokens", 0))

    # Save full conversation to file
    conversation = getattr(trajectory, "_conversation", None)
    conversation_data = None
    if conversation:
        try:
            import json as _json
            safe_id = instance_id.replace("/", "_")
            conv_dir = Path(workspace_dir).parent / "conversations"
            conv_dir.mkdir(parents=True, exist_ok=True)
            conv_path = conv_dir / f"{safe_id}.json"
            conv_path.write_text(_json.dumps(conversation, indent=2, default=str, ensure_ascii=False))
            log.info("Saved conversation to %s (%d messages)", conv_path, len(conversation))
        except Exception as conv_err:
            log.warning("Failed to save conversation: %s", conv_err)

    return {
        "instance_id": instance_id,
        "success": feedback.success,
        "score": feedback.score,
        "elapsed": round(elapsed, 1),
        "patch_len": len(trajectory.output),
        "patch": trajectory.output,
        "skill_proposal": getattr(trajectory, "_skill_proposal", ""),
        "feedback_detail": feedback.detail,
        "num_tool_calls": num_tool_calls,
        "num_turns": len(per_turn),
        "cumulative_input_tokens": usage.get("input_tokens", 0),
        "cumulative_output_tokens": usage.get("output_tokens", 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "cumulative_cache_read_input_tokens": int(
            usage.get("cache_read_input_tokens", 0) or 0
        ),
        "cumulative_cache_write_input_tokens": int(
            usage.get("cache_write_input_tokens", 0) or 0
        ),
        "cost_units": float(usage.get("cost_units", 0.0) or 0.0),
        "main_cost_units": float(usage.get("main_cost_units", usage.get("cost_units", 0.0)) or 0.0),
        "route_model_cost_units": float(usage.get("route_model_cost_units", 0.0) or 0.0),
        "main_model_id": str(usage.get("model_id", "")),
        "main_model_provider": str(usage.get("provider", "")),
        "main_route_level": str(usage.get("route_level", "")),
        "input_price_per_1m": float(usage.get("input_price_per_1m", 0.0) or 0.0),
        "output_price_per_1m": float(usage.get("output_price_per_1m", 0.0) or 0.0),
        "cache_read_price_per_1m": float(usage.get("cache_read_price_per_1m", 0.0) or 0.0),
        "cache_write_price_per_1m": float(usage.get("cache_write_price_per_1m", 0.0) or 0.0),
        "cost_currency": str(usage.get("cost_currency", "")),
        "cost_price_source": str(usage.get("cost_price_source", "")),
        "cost_priced": bool(usage.get("cost_priced", False)),
        "usage_record_quality": _usage_quality(usage),
        "usage_source": str(usage.get("usage_source", "")),
        "usage_estimated": bool(usage.get("usage_estimated", False)),
        "max_input_tokens_per_turn": max_input,
        "active_efficiency_strategies": active_strategies,
        "efficiency_config_id": (
            usage_step.get("efficiency_config_id", "") if isinstance(usage_step, dict) else ""
        ),
        "efficiency_selection_reason": (
            usage_step.get("efficiency_selection_reason", "")
            if isinstance(usage_step, dict) else ""
        ),
        "efficiency_config": (
            usage_step.get("efficiency_config", {})
            if isinstance(usage_step, dict) else {}
        ),
        "route_model_calls": route_model_calls,
        "steps": trajectory.steps,
    }


def _seq_task_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Convert a batch cost profile into selector-facing task metadata."""
    task_profile = dict(profile.get("task_profile", {}) or {})
    bottlenecks = set(profile.get("bottlenecks", []) or [])
    task_profile.setdefault("task_type", "coding")
    task_profile.setdefault("domain", "swebench_verified")
    task_profile["bottlenecks"] = sorted(bottlenecks)
    task_profile.setdefault("difficulty", "hard")
    task_profile["context_pressure"] = (
        "hard" if {"context-heavy", "long-observation"} & bottlenecks else "medium"
    )
    task_profile["tool_pressure"] = "hard" if "tool-heavy" in bottlenecks else "medium"
    task_profile["verification_pressure"] = "hard"
    return task_profile


def _validate_efficiency_config(
    *,
    config: dict[str, Any],
    task_dicts: list[dict[str, Any]],
    source_workspace_dir: Path,
    validation_workspace_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run one candidate config on an isolated snapshot of the batch workspace."""
    if validation_workspace_dir.exists():
        shutil.rmtree(validation_workspace_dir)
    validation_workspace_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_workspace_dir, validation_workspace_dir)

    runtime_config = config_to_runtime_config(config)
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(
                solve_one_task,
                task_dict,
                str(validation_workspace_dir),
                args.model_id,
                args.region,
                args.max_tokens,
                args.max_steps,
                args.window_size,
                args.verification_focus,
                getattr(args, "efficiency_prompt", False),
                args.dataset,
                getattr(args, "solver_proposes", False),
                runtime_config=runtime_config,
            ): task_dict["id"]
            for task_dict in task_dicts
        }
        for future in as_completed(futures):
            instance_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = _attach_usage_fields({
                    "instance_id": instance_id,
                    "success": False,
                    "score": 0.0,
                    "error": str(exc),
                    "elapsed": 0.0,
                    "patch_len": 0,
                    "patch": "",
                    "usage_record_quality": "missing",
                    "usage_source": "validation_future_exception",
                })
            result["efficiency_validation_config_id"] = config.get("id", "")
            result["active_efficiency_strategies"] = list(
                config.get("strategies", []) or []
            )
            result["strategies"] = list(config.get("strategies", []) or [])
            _attach_usage_fields(result)
            results.append(result)

    return {
        "config_id": str(config.get("id", "")),
        "strategies": list(config.get("strategies", []) or []),
        "results": results,
        "summary": summarize_results(results),
        "profile": build_cost_profile(results, group=str(config.get("id", ""))),
    }


def _update_sequential_efficiency_library(
    *,
    strategy_library: dict[str, Any],
    batch_results: list[dict[str, Any]],
    batch_tasks: list[Any],
    batch_num: int,
    work_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Learn and deploy efficiency configs between two sequential batches."""
    baseline_profile = build_cost_profile(batch_results, group=f"batch_{batch_num}_baseline")
    task_profile = _seq_task_profile(baseline_profile)

    # Update the online statistics of configs actually used by this batch.
    for result in batch_results:
        record_config_outcome(
            strategy_library,
            config_id=result.get("efficiency_config_id"),
            result=result,
        )

    candidates = candidate_configs_for_profile(
        task_profile,
        baseline_profile,
        mode=args.efficiency_candidate_mode,
        max_configs=args.efficiency_max_candidates,
    )
    candidate_records: list[dict[str, Any]] = []
    gate_config = GateConfig(
        epsilon_success=args.efficiency_epsilon_success,
        delta_cost=args.efficiency_delta_cost,
        min_trials=args.efficiency_min_trials,
    )
    task_dicts = [
        {"id": task.id, "input": task.input, "metadata": task.metadata}
        for task in batch_tasks
    ]

    for config in candidates:
        config_id = str(config.get("id", ""))
        validation_workspace_dir = (
            output_dir
            / "efficiency"
            / "validation_workspaces"
            / f"batch_{batch_num:02d}_{_slug(config_id)}"
        )
        validation = _validate_efficiency_config(
            config=config,
            task_dicts=task_dicts,
            source_workspace_dir=work_dir,
            validation_workspace_dir=validation_workspace_dir,
            args=args,
        )
        gate = gate_candidate(
            baseline_profile,
            validation["profile"],
            config=gate_config,
        )
        config["gate"] = gate
        config["stats"] = {
            "batch": batch_num,
            "train_summary": validation["summary"],
            "success_rate": validation["summary"].get("success_rate", 0.0),
            "avg_tokens": validation["summary"].get("avg_tokens", 0.0),
            "avg_tokens_success": validation["summary"].get("avg_tokens_success", 0.0),
            "avg_cost": validation["summary"].get("avg_cost", 0.0),
            "avg_cost_success": validation["summary"].get("avg_cost_success", 0.0),
            "cost_of_pass": validation["summary"].get("cost_of_pass", "inf"),
            "cost_delta_pct": gate.get("cost_delta_pct", 0.0),
            "trials": validation["summary"].get("tasks", 0),
            "successes": validation["summary"].get("success", 0),
        }
        candidate_records.append(
            {
                "config_id": config_id,
                "config": config,
                "summary": validation["summary"],
                "gate": gate,
            }
        )

    accepted = [
        record
        for record in candidate_records
        if record["gate"].get("accepted")
    ]
    frontier = pareto_frontier(
        accepted,
        epsilon_success=args.efficiency_epsilon_success,
    ) if accepted else []
    selected = choose_best_candidate(
        frontier,
        baseline=baseline_profile,
        epsilon_success=args.efficiency_epsilon_success,
    ) if frontier else None
    selected_id = str(selected.get("config_id", "")) if selected else ""

    existing_configs = {
        str(config.get("id", "")): dict(config)
        for config in strategy_library.get("configs", []) or []
        if isinstance(config, dict) and config.get("id")
    }
    candidate_ids = {str(config.get("id", "")) for config in candidates}
    for config in candidates:
        existing_configs[str(config["id"])] = dict(config)

    # Keep a previous active config when the new batch has no accepted
    # successor. A candidate only replaces it after success-first validation.
    previous_active = [
        str(config_id)
        for config_id in strategy_library.get("active_configs", []) or []
        if config_id in existing_configs
    ]
    active_ids = [selected_id] if selected_id else previous_active
    for config_id, config in existing_configs.items():
        if config_id in active_ids:
            config["status"] = "active"
        elif config.get("gate", {}).get("accepted"):
            config["status"] = "validated"
        elif config_id in candidate_ids:
            config["status"] = "rejected"
        else:
            config["status"] = str(config.get("status", "rejected"))

    strategy_library["configs"] = list(existing_configs.values())
    strategy_library["active_configs"] = active_ids
    strategy_library["active_strategies"] = []
    for config_id in active_ids:
        for strategy in existing_configs[config_id].get("strategies", []) or []:
            if strategy not in strategy_library["active_strategies"]:
                strategy_library["active_strategies"].append(strategy)
    strategy_library.setdefault("meta", {})
    strategy_library["meta"].update(
        {
            "selection_layer": "meta_strategy_selector",
            "last_batch": batch_num,
            "last_bottlenecks": baseline_profile.get("bottlenecks", []),
            "last_selected_config_id": selected_id,
            "last_pareto_frontier": [
                str(record.get("config_id", "")) for record in frontier
            ],
        }
    )

    write_strategy_library(work_dir, strategy_library)
    write_strategy_library(output_dir, strategy_library)
    active_strategies = list(strategy_library.get("active_strategies", []) or [])
    active_metadata = {
        "experiment": "sequential_meta",
        "batch": batch_num,
        "selected_config_id": selected_id,
        "efficiency_config_id": selected_id,
        "active_configs": active_ids,
        "selection_layer": "meta_strategy_selector",
        "efficiency_config": existing_configs.get(selected_id, {}) if selected_id else {},
    }
    write_active_config(work_dir, active_strategies, metadata=active_metadata)
    write_active_config(output_dir, active_strategies, metadata=active_metadata)
    materialize_strategy_artifacts(work_dir, active_strategies)
    materialize_strategy_artifacts(output_dir, active_strategies)
    controller_report = {
        "batch": batch_num,
        "baseline_profile": baseline_profile,
        "candidate_configs": candidates,
        "accepted_configs": [record["config_id"] for record in accepted],
        "pareto_frontier": [record["config_id"] for record in frontier],
        "selected_config_id": selected_id,
        "active_configs": active_ids,
    }
    controller_path = output_dir / "efficiency" / f"batch_{batch_num:02d}_controller.json"
    controller_path.parent.mkdir(parents=True, exist_ok=True)
    controller_path.write_text(
        json.dumps(controller_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return strategy_library, controller_report


def main():
    p = argparse.ArgumentParser(description="Batched in-situ evolution")
    p.add_argument("--batch-size", type=int, default=5,
                   help="Tasks per batch (default: 5). Evolver runs after each batch.")
    p.add_argument("--parallel", type=int, default=5,
                   help="Parallel workers within each batch (default: 5)")
    p.add_argument("--feedback", type=str, default="minimal",
                   choices=["none", "minimal"],
                   help="Feedback to evolver: none=no scores, minimal=scores included")
    p.add_argument("--no-evolve", action="store_true",
                   help="Baseline mode: solve all tasks without any evolution")
    p.add_argument("--solver-proposes", action="store_true",
                   help="V11: solver proposes skills after each task, applied between batches")
    p.add_argument("--verification-focus", action="store_true",
                   help="V21: solver only proposes verification skills, evolver only curates verification")
    p.add_argument("--efficiency-prompt", action="store_true",
                   help="V22: add hypothesis-first efficiency constraints to system prompt")
    p.add_argument("--model-id", type=str, default="us.anthropic.claude-opus-4-6-v1")
    p.add_argument("--region", type=str, default="us-west-2")
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--max-steps", type=int, default=0,
                   help="Max tool calls per task (0=unlimited, 80 recommended)")
    p.add_argument("--window-size", type=int, default=40,
                   help="Sliding window size for conversation memory (default: 40)")
    p.add_argument("--seed-workspace", type=str, default="seed_workspaces/swe")
    p.add_argument("--output-dir", type=str, default="logs/seq_evolve")
    p.add_argument("--dataset", type=str, default="MariusHobbhahn/swe-bench-verified-mini",
                   help="HuggingFace dataset name (default: swe-bench-verified-mini)")
    p.add_argument("--limit", type=int, default=50,
                   help="Max tasks to solve (default: 50)")
    p.add_argument(
        "--efficiency-controller",
        choices=["off", "meta"],
        default="meta",
        help="Use the task-profiled efficiency library between batches.",
    )
    p.add_argument(
        "--efficiency-candidate-mode",
        choices=["meta", "safe", "all"],
        default="all",
        help="Candidate family used by the between-batch efficiency controller.",
    )
    p.add_argument(
        "--efficiency-max-candidates",
        type=int,
        default=3,
        help="Maximum candidate configs proposed after each batch.",
    )
    p.add_argument(
        "--efficiency-min-trials",
        type=int,
        default=2,
        help="Minimum observations before a sequential candidate can become active.",
    )
    p.add_argument(
        "--efficiency-epsilon-success",
        type=float,
        default=0.03,
        help="Allowed success-rate drop for the sequential efficiency gate.",
    )
    p.add_argument(
        "--efficiency-delta-cost",
        type=float,
        default=0.05,
        help="Required relative token improvement for the sequential gate.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    # SWE/Strands dependencies are optional for repository tooling. Import
    # them only after argparse so ``--help`` remains usable in a lean env.
    from agent_evolve.agents.swe.agent import SweAgent
    from agent_evolve.algorithms.guided_synth.engine import GuidedSynthesisEngine
    from agent_evolve.benchmarks.swe_verified_mini.benchmark import SweVerifiedMiniBenchmark

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    for n in ("botocore", "urllib3", "httpcore", "httpx",
              "strands.models", "strands.tools", "strands.telemetry"):
        logging.getLogger(n).setLevel(logging.WARNING)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all tasks
    bm = SweVerifiedMiniBenchmark(dataset_name=args.dataset, shuffle=False)
    tasks = bm.get_tasks(split="test", limit=args.limit)
    logger.info("Loaded %d tasks", len(tasks))

    # Initialize shared workspace
    work_dir = output_dir / "workspace"
    seed_dir = Path(args.seed_workspace)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    shutil.copytree(seed_dir, work_dir)
    strategy_library = load_strategy_library(work_dir)
    strategy_library.setdefault("configs", [])
    strategy_library.setdefault("active_configs", [])
    strategy_library.setdefault("active_strategies", [])
    write_strategy_library(work_dir, strategy_library)
    logger.info("Copied seed workspace %s → %s", seed_dir, work_dir)

    # Setup evolution
    evolution_dir = work_dir / "evolution"
    evolution_dir.mkdir(parents=True, exist_ok=True)
    observer = Observer(evolution_dir)

    config = EvolveConfig(evolver_model=args.model_id, extra={"region": args.region})
    evolver = GuidedSynthesisEngine(config, write_memory=False,
                                     verification_focus=getattr(args, 'verification_focus', False))
    n_batches = (len(tasks) + args.batch_size - 1) // args.batch_size
    evolver.MAX_INTERVENTIONS = n_batches  # one intervention per batch

    # Track results
    all_results = []
    evolve_count = 0

    # Split tasks into batches
    batches = []
    for i in range(0, len(tasks), args.batch_size):
        batches.append(tasks[i:i + args.batch_size])

    print(f"Solving {len(tasks)} tasks in {len(batches)} batches of {args.batch_size}")
    print(f"Parallel workers: {args.parallel}")
    print(f"Feedback mode: {args.feedback}")
    print(f"Output: {output_dir.resolve()}")
    print()

    for batch_idx, batch_tasks in enumerate(batches):
        batch_num = batch_idx + 1
        task_start = batch_idx * args.batch_size + 1
        task_end = task_start + len(batch_tasks) - 1
        print(f"=== Batch {batch_num}/{len(batches)} (tasks {task_start}-{task_end}) ===")

        # Prepare task dicts for parallel execution
        task_dicts = [{"id": t.id, "input": t.input, "metadata": t.metadata} for t in batch_tasks]

        # Solve batch in parallel
        batch_results = []
        with ProcessPoolExecutor(max_workers=args.parallel) as pool:
            futures = {
                pool.submit(
                    solve_one_task, td, str(work_dir),
                    args.model_id, args.region, args.max_tokens,
                    args.max_steps, args.window_size,
                    args.verification_focus,
                    getattr(args, 'efficiency_prompt', False),
                    args.dataset,
                    getattr(args, 'solver_proposes', False),
                ): td["id"]
                for td in task_dicts
            }
            for fut in as_completed(futures):
                iid = futures[fut]
                try:
                    r = fut.result()
                    batch_results.append(r)
                    status = "PASS" if r["success"] else "FAIL"
                    passed_total = sum(1 for x in all_results if x["success"]) + sum(1 for x in batch_results if x["success"])
                    done_total = len(all_results) + len(batch_results)
                    print(f"  {status} {iid} ({r['elapsed']:.0f}s) | running: {passed_total}/{done_total}")
                except Exception as e:
                    print(f"  ERROR {iid}: {e}")
                    batch_results.append(_attach_usage_fields({
                        "instance_id": iid,
                        "success": False,
                        "score": 0.0,
                        "error": str(e),
                        "elapsed": 0,
                        "patch_len": 0,
                        "patch": "",
                        "usage_record_quality": "missing",
                        "usage_source": "batch_future_exception",
                    }))

        # Record results
        for r in batch_results:
            usage = _usage_from_result_fields(r)
            record = {
                "instance_id": r["instance_id"],
                "batch": batch_num,
                "success": r["success"],
                "score": r["score"],
                "elapsed": r.get("elapsed", 0),
                "patch_len": r.get("patch_len", 0),
                "error": r.get("error"),
                "evolve_generation": evolve_count,
                "num_tool_calls": r.get("num_tool_calls", 0),
                "num_turns": r.get("num_turns", 0),
                "cumulative_input_tokens": int(usage.get("input_tokens", 0) or 0),
                "cumulative_output_tokens": int(usage.get("output_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
                "cumulative_cache_read_input_tokens": int(
                    usage.get("cache_read_input_tokens", 0) or 0
                ),
                "cumulative_cache_write_input_tokens": int(
                    usage.get("cache_write_input_tokens", 0) or 0
                ),
                "cost_units": float(usage.get("cost_units", 0.0) or 0.0),
                "main_cost_units": float(
                    usage.get("main_cost_units", usage.get("cost_units", 0.0)) or 0.0
                ),
                "route_model_cost_units": float(
                    usage.get("route_model_cost_units", 0.0) or 0.0
                ),
                "main_model_id": str(usage.get("model_id", "")),
                "main_model_provider": str(usage.get("provider", "")),
                "main_route_level": str(usage.get("route_level", "")),
                "input_price_per_1m": float(usage.get("input_price_per_1m", 0.0) or 0.0),
                "output_price_per_1m": float(usage.get("output_price_per_1m", 0.0) or 0.0),
                "cache_read_price_per_1m": float(
                    usage.get("cache_read_price_per_1m", 0.0) or 0.0
                ),
                "cache_write_price_per_1m": float(
                    usage.get("cache_write_price_per_1m", 0.0) or 0.0
                ),
                "cost_currency": str(usage.get("cost_currency", "")),
                "cost_price_source": str(usage.get("cost_price_source", "")),
                "cost_priced": bool(usage.get("cost_priced", False)),
                "usage_record_quality": _usage_quality(usage),
                "usage_source": str(usage.get("usage_source", "")),
                "usage_estimated": bool(usage.get("usage_estimated", False)),
                "max_input_tokens_per_turn": r.get("max_input_tokens_per_turn", 0),
                "active_efficiency_strategies": r.get("active_efficiency_strategies", []),
                "efficiency_config_id": r.get("efficiency_config_id", ""),
                "efficiency_selection_reason": r.get("efficiency_selection_reason", ""),
                "efficiency_config": r.get("efficiency_config", {}),
                "route_model_calls": r.get("route_model_calls", []),
                "steps": r.get("steps", []),
            }
            all_results.append(record)

        # Save patches
        for r in batch_results:
            if r.get("patch"):
                patch_path = output_dir / "patches" / f"{r['instance_id'].replace('/', '_')}.diff"
                patch_path.parent.mkdir(parents=True, exist_ok=True)
                patch_path.write_text(r["patch"])

        # Evolve after this batch
        observations = []
        for r in batch_results:
            if r.get("patch") is not None:
                task_obj = next((t for t in batch_tasks if t.id == r["instance_id"]), None)
                if not task_obj:
                    continue
                traj = Trajectory(task_id=r["instance_id"], output=r.get("patch", ""), steps=r.get("steps", []))
                if args.feedback == "none":
                    fb = Feedback(success=False, score=0.0, detail="", raw={})
                else:
                    fb = Feedback(
                        success=r["success"],
                        score=r["score"],
                        detail=r.get("feedback_detail", ""),
                        raw={},
                    )
                observations.append(Observation(task=task_obj, trajectory=traj, feedback=fb))

        if (
            args.efficiency_controller != "off"
            and batch_results
        ):
            if batch_num < len(batches):
                logger.info(
                    "=== EFFICIENCY CONTROLLER (batch %d -> %d) ===",
                    batch_num,
                    batch_num + 1,
                )
                try:
                    strategy_library, controller_report = _update_sequential_efficiency_library(
                        strategy_library=strategy_library,
                        batch_results=batch_results,
                        batch_tasks=batch_tasks,
                        batch_num=batch_num,
                        work_dir=work_dir,
                        output_dir=output_dir,
                        args=args,
                    )
                    print(
                        "  [efficiency] selected="
                        f"{controller_report.get('selected_config_id') or 'baseline'} "
                        f"active={controller_report.get('active_configs') or []}"
                    )
                except Exception as exc:
                    logger.exception(
                        "Efficiency controller failed after batch %d: %s",
                        batch_num,
                        exc,
                    )
            else:
                for result in batch_results:
                    record_config_outcome(
                        strategy_library,
                        config_id=result.get("efficiency_config_id"),
                        result=result,
                    )
                write_strategy_library(work_dir, strategy_library)
                write_strategy_library(output_dir, strategy_library)

        if observations and not args.no_evolve:
            evolve_count += 1
            logger.info("=== EVOLVING (generation %d, %d observations) ===",
                        evolve_count, len(observations))

            # Use a fresh agent to export workspace state
            agent = SweAgent(workspace_dir=work_dir, model_id=args.model_id,
                             region=args.region, max_tokens=args.max_tokens)
            agent.export_to_fs()
            observer.collect(observations)

            t_evo = time.time()
            try:
                evo_result = evolver.evolve(
                    workspace=agent.workspace,
                    observation_logs=observations,
                    evo_number=evolve_count,
                )
                logger.info("Evolution %d complete in %.1fs", evolve_count, time.time() - t_evo)
            except Exception as e:
                logger.error("Evolution %d failed: %s", evolve_count, e)

            # Log workspace state
            agent_check = SweAgent(workspace_dir=work_dir, model_id=args.model_id,
                                   region=args.region, max_tokens=args.max_tokens)
            prompt_len = len(agent_check._build_system_prompt())
            n_skills = len(agent_check.skills)
            print(f"  [evolved gen{evolve_count}] prompt={prompt_len} chars, skills={n_skills}")

        # V11: Solver proposes skills → evolver curates (accept/reject/merge)
        if args.solver_proposes:
            # Attach proposals to trajectory objects for evolver
            for r in batch_results:
                proposal = r.get("skill_proposal", "")
                if r.get("patch") is not None:
                    task_obj = next((t for t in batch_tasks if t.id == r["instance_id"]), None)
                    if not task_obj:
                        continue
                    # Find or create the observation's trajectory
                    for obs in observations:
                        if obs.task.id == r["instance_id"]:
                            obs.trajectory._skill_proposal = proposal
                            break

            if observations:
                evolve_count += 1
                logger.info("=== CURATING PROPOSALS (generation %d) ===", evolve_count)
                agent = SweAgent(workspace_dir=work_dir, model_id=args.model_id,
                                 region=args.region, max_tokens=args.max_tokens)
                agent.export_to_fs()
                observer.collect(observations)

                t_evo = time.time()
                try:
                    evo_result = evolver.evolve(
                        workspace=agent.workspace,
                        observation_logs=observations,
                        evo_number=evolve_count,
                    )
                    logger.info("Curation %d complete in %.1fs: %s",
                                evolve_count, time.time() - t_evo, evo_result)
                except Exception as e:
                    logger.error("Curation %d failed: %s", evolve_count, e)

                agent_check = SweAgent(workspace_dir=work_dir, model_id=args.model_id,
                                       region=args.region, max_tokens=args.max_tokens)
                prompt_len = len(agent_check._build_system_prompt())
                n_skills = len(agent_check.skills)
                print(f"  [curated gen{evolve_count}] prompt={prompt_len} chars, skills={n_skills}")

        batch_passed = sum(1 for r in batch_results if r["success"])
        print(f"  Batch {batch_num}: {batch_passed}/{len(batch_results)}")
        print()

    # Final summary
    passed = sum(1 for r in all_results if r["success"])
    total = len(all_results)

    print(f"{'='*60}")
    final_rate = 100 * passed / total if total else 0.0
    print(f"FINAL: {passed}/{total} resolved ({final_rate:.1f}%)")
    print(f"Feedback mode: {args.feedback}")
    print(f"Batch size: {args.batch_size}, Parallel: {args.parallel}")
    print(f"Evolutions run: {evolve_count}")
    print(f"{'='*60}")

    for r in all_results:
        mark = "✓" if r["success"] else "✗"
        gen = f"gen{r['evolve_generation']}"
        calls = r.get("num_tool_calls", 0)
        turns = r.get("num_turns", 0)
        cum_in = r.get("cumulative_input_tokens", 0)
        cost = float(r.get("cost_units", 0.0) or 0.0)
        usage_quality = r.get("usage_record_quality", "")
        max_in = r.get("max_input_tokens_per_turn", 0)
        print(f"  {mark} {r['instance_id']} ({r['elapsed']:.0f}s) [{gen}] "
              f"calls={calls} turns={turns} cum_in={cum_in:,d} "
              f"cost={cost:,.0f} usage={usage_quality} max_in={max_in:,d}")

    # Performance by generation
    print(f"\nPerformance by evolution generation:")
    for gen in range(evolve_count + 1):
        gen_results = [r for r in all_results if r["evolve_generation"] == gen]
        gen_passed = sum(1 for r in gen_results if r["success"])
        if gen_results:
            print(f"  gen{gen}: {gen_passed}/{len(gen_results)} ({100*gen_passed/len(gen_results):.0f}%)")

    # Save results
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {results_path}")

    summary_rows = _sequential_summary_rows(all_results)
    summary_payload = {
        "overall": summary_rows[-1] if summary_rows else _summary_row("overall", []),
        "by_generation": summary_rows[:-1],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False))
    write_summary_table(summary_rows, output_dir / "summary.csv")
    try:
        plot_paths = write_plots(summary_rows, output_dir / "figures")
        hits_path = write_strategy_library_plot(summary_rows, output_dir / "figures")
        print(f"Summary saved to {summary_path}")
        print(f"Figures saved to {output_dir / 'figures'} ({len(plot_paths) + 1} files)")
    except Exception as e:
        logger.warning("Failed to write visualization figures: %s", e)
        print(f"Summary saved to {summary_path}; figure rendering failed: {e}")

    # Print metrics dashboard
    try:
        from metrics import compute_metrics, print_dashboard
        print(f"\n{'='*60}")
        print("METRICS DASHBOARD")
        print(f"{'='*60}")
        print_dashboard([output_dir])
    except Exception as e:
        logger.warning("Failed to print metrics dashboard: %s", e)


if __name__ == "__main__":
    main()
