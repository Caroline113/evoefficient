#!/usr/bin/env python3
"""Run cost-aware emergent efficiency experiments.

This runner is intentionally separate from evolve_sequential.py.  The normal
mode runs ordinary agent self-evolution. The cost-aware mode runs that same
self-evolution and then adds neutral cost/trajectory summaries so the evolver
can write reusable efficiency interventions. It does not expose predefined
EvoEfficient strategy names or candidate configs to the evolver. The
category-seeded mode adds only broad M1-M7 efficiency categories. The
predefined-cost-aware mode intentionally combines predefined runtime efficiency
strategies with the same cost-aware proposer.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from agent_evolve.config import EvolveConfig
from agent_evolve.contract.workspace import AgentWorkspace
from agent_evolve.efficiency.profiler import normalize_usage, result_strategy_hits, summarize_results
from agent_evolve.efficiency.strategies import (
    DEFAULT_STRATEGIES,
    load_active_config,
    materialize_strategy_artifacts,
    write_active_config,
)
from agent_evolve.efficiency.visualization import (
    write_proposal_category_plots,
    write_plots,
    write_strategy_library_plot,
    write_summary_table,
)
from agent_evolve.llm.base import LLMMessage
from agent_evolve.types import Feedback, Observation, Task, Trajectory

logger = logging.getLogger("emergent_efficiency")

PREDEFINED_STRATEGY_NAMES = sorted(DEFAULT_STRATEGIES)
DEFAULT_OUTPUT_ROOT = "logs/emergent_efficiency"

CATEGORY_SEED_TAXONOMY: list[dict[str, str]] = [
    {
        "id": "M1",
        "name": "Context",
        "cost_dimension": "input tokens",
        "observable_signals": "prompt/context length, max input tokens per turn, cache read/write",
        "scope": "Context selection, compression, and retention of evidence that can change the patch decision.",
    },
    {
        "id": "M2",
        "name": "Reasoning",
        "cost_dimension": "reasoning tokens",
        "observable_signals": "reasoning budget when available; otherwise turns, reflection length, repeated analysis",
        "scope": "Shorter reasoning loops, hypothesis convergence, and avoiding repeated analysis.",
    },
    {
        "id": "M3",
        "name": "Output",
        "cost_dimension": "output tokens",
        "observable_signals": "output tokens, long observations, long command output",
        "scope": "Concise outputs, structured observation summaries, and less repeated printing.",
    },
    {
        "id": "M4",
        "name": "Planning",
        "cost_dimension": "planning tokens / steps",
        "observable_signals": "planning turns, plan updates, total step count",
        "scope": "Plan templates, stage gates, and replanning after failed hypotheses.",
    },
    {
        "id": "M5",
        "name": "Tools",
        "cost_dimension": "tool calls",
        "observable_signals": "tool call count, repeated commands, same-test reruns without edits",
        "scope": "Tool choice, batched probes, and repeated-command suppression.",
    },
    {
        "id": "M6",
        "name": "Compilation",
        "cost_dimension": "steps / LLM calls",
        "observable_signals": "read_skill calls, repeated skill/prompt loading, LLM calls",
        "scope": "Skill or memory compilation, compact task playbooks, and reusable knowledge summaries.",
    },
    {
        "id": "M7",
        "name": "Routing",
        "cost_dimension": "cost per call",
        "observable_signals": "main/easy/hard model cost, route model cost",
        "scope": "Difficulty judgment, cheap-model prescreening, and reserving expensive calls for key decisions.",
    },
]

CATEGORY_LABELS = {
    f"{item['id']} {item['name']}": item
    for item in CATEGORY_SEED_TAXONOMY
}
CATEGORY_ALIASES = {
    item["id"].lower(): f"{item['id']} {item['name']}"
    for item in CATEGORY_SEED_TAXONOMY
}
CATEGORY_ALIASES.update({
    item["name"].lower(): f"{item['id']} {item['name']}"
    for item in CATEGORY_SEED_TAXONOMY
})


EMERGENCE_SYSTEM_PROMPT = """\
You are an experiment assistant for a SWE-bench solving agent.

Goal: discover cost-saving behavior changes from evidence, without using any
named predefined strategy library.

You may propose ordinary reusable skills, short prompt fragments, compact
memory notes, or deterministic Strands tools.  Each proposal must preserve
correctness first.  Avoid shortcuts that merely stop early, skip investigation,
or reduce validation.

Output JSON only:
{
  "proposals": [
    {
      "name": "short_generic_name",
      "kind": "skill" | "prompt_fragment" | "tool" | "memory",
      "description": "one sentence with trigger and do-not-trigger conditions",
      "trigger": "when to use it",
      "do_not_trigger": "when not to use it",
      "content": "procedure under 350 words, or complete Strands @tool Python source for kind=tool",
      "expected_savings": "why this should reduce cost",
      "correctness_risk": "what could go wrong",
      "confidence": "HIGH" | "MEDIUM" | "LOW",
      "evidence": ["task_id: observed waste -> proposed fix"]
    }
  ]
}

Return {"proposals": []} when there is no cross-task efficiency pattern.
"""


PREDEFINED_COST_AWARE_SYSTEM_PROMPT = """\
You are an experiment assistant for a SWE-bench solving agent.

Goal: discover incremental cost-saving behavior changes from evidence while
predefined runtime efficiency strategies are already active.

You may refer to active predefined strategies only to explain what is already
covered. Propose ordinary reusable skills, short prompt fragments, compact
memory notes, or deterministic Strands tools only when they add behavior not
already covered by the active strategy set. Each proposal must preserve
correctness first. Avoid shortcuts that merely stop early, skip investigation,
or reduce validation.

Output JSON only:
{
  "proposals": [
    {
      "name": "short_generic_name",
      "kind": "skill" | "prompt_fragment" | "tool" | "memory",
      "description": "one sentence with trigger and do-not-trigger conditions",
      "trigger": "when to use it",
      "do_not_trigger": "when not to use it",
      "content": "procedure under 350 words, or complete Strands @tool Python source for kind=tool",
      "expected_savings": "why this should reduce cost beyond active strategies",
      "correctness_risk": "what could go wrong",
      "confidence": "HIGH" | "MEDIUM" | "LOW",
      "evidence": ["task_id: observed waste -> proposed incremental fix"]
    }
  ]
}

Return {"proposals": []} when there is no cross-task incremental pattern.
"""


CATEGORY_SEEDED_SYSTEM_PROMPT = """\
You are an experiment assistant for a SWE-bench solving agent.

Goal: discover concrete cost-saving behavior changes from evidence. You are
given only broad efficiency categories, not predefined strategy names or
implementations. Use the categories as weak labels for cost dimensions; do not
treat them as ready-made strategies.

Each proposal must preserve correctness first. Avoid shortcuts that merely stop
early, skip investigation, or reduce validation. Propose a tool only when it is
a deterministic, safe helper whose complete Strands @tool Python source can be
written in the content field.

Output JSON only:
{
  "proposals": [
    {
      "name": "short_generic_name",
      "kind": "skill" | "prompt_fragment" | "tool" | "memory",
      "categories": [""],
      "description": "one sentence with trigger and do-not-trigger conditions",
      "trigger": "when to use it",
      "do_not_trigger": "when not to use it",
      "content": "procedure under 350 words, or complete Strands @tool Python source for kind=tool",
      "expected_savings": "which category metric should improve and why",
      "correctness_risk": "what could go wrong",
      "confidence": "HIGH" | "MEDIUM" | "LOW",
      "evidence": ["task_id: observed waste -> proposed fix"]
    }
  ]
}

Return {"proposals": []} when there is no cross-task efficiency pattern.
"""


def _slug(value: str, *, default: str = "proposal", limit: int = 80) -> str:
    text = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value).strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return (text or default)[:limit]


def _normalize_kind(value: str) -> str:
    kind = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "prompt": "prompt_fragment",
        "fragment": "prompt_fragment",
        "prompt_fragment": "prompt_fragment",
        "skill": "skill",
        "tool": "tool",
        "memory": "memory",
        "memory_entry": "memory",
    }
    return aliases.get(kind, kind)


def _tool_function_name(value: str, *, default: str = "emergent_tool") -> str:
    name = _slug(value, default=default)
    if not re.match(r"^[A-Za-z_]", name):
        name = f"{default}_{name}"
    return name[:80]


def _strip_code_fence(content: str) -> str:
    text = str(content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
    return text


def _tool_validation_error(content: str) -> str:
    text = str(content or "").strip()
    if len(text) < 120:
        return "tool content too short"
    if "def " not in text:
        return "tool content has no function definition"
    if "@tool" not in text:
        return "tool content is not a Strands @tool"
    return ""


def _category_taxonomy_payload() -> list[dict[str, str]]:
    return [
        {
            "category": f"{item['id']} {item['name']}",
            "cost_dimension": item["cost_dimension"],
            "observable_signals": item["observable_signals"],
            "scope": item["scope"],
        }
        for item in CATEGORY_SEED_TAXONOMY
    ]


def _normalize_categories(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        text = str(raw or "").strip()
        if not text:
            continue
        compact = re.sub(r"\s+", " ", text)
        key = compact.lower()
        match = re.match(r"^(m[1-7])(?:\s+(.+))?$", key)
        if match:
            normalized = CATEGORY_ALIASES.get(match.group(1), compact)
        else:
            normalized = CATEGORY_ALIASES.get(key, compact)
        if normalized in CATEGORY_LABELS and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out


def _proposal_categories(proposal: dict[str, Any]) -> list[str]:
    return _normalize_categories(proposal.get("categories") or proposal.get("category"))


def _json_default(value: Any) -> str:
    return str(value)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    """Resolve the final output directory.

    By default, create a timestamped run directory so repeated experiments do
    not overwrite one another.  ``--fixed-output-dir`` restores the old exact
    path behavior for intentional reruns.
    """
    raw_output_dir = getattr(args, "output_dir", None) or DEFAULT_OUTPUT_ROOT
    output_dir = Path(raw_output_dir)
    if getattr(args, "fixed_output_dir", False):
        return output_dir

    stamp = _timestamp()
    mode_slug = str(args.mode).replace("-", "_")
    if raw_output_dir == DEFAULT_OUTPUT_ROOT:
        return output_dir / f"{mode_slug}_{stamp}"
    return output_dir.with_name(f"{output_dir.name}_{stamp}")


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _summary_usage_step(steps: list[Any]) -> dict[str, Any]:
    for step in reversed(steps or []):
        if isinstance(step, dict) and "usage" in step:
            return step
    return {}


def _usage_quality(usage: dict[str, Any]) -> str:
    quality = str(usage.get("usage_record_quality") or "").strip()
    if quality:
        return quality
    if any(
        _safe_int(usage.get(name)) > 0
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


def _normalize_result_usage(result: dict[str, Any]) -> dict[str, Any]:
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
    usage = _normalize_result_usage(result)
    result["cumulative_input_tokens"] = _safe_int(usage.get("input_tokens"))
    result["cumulative_output_tokens"] = _safe_int(usage.get("output_tokens"))
    result["total_tokens"] = _safe_int(usage.get("total_tokens"))
    result["cumulative_cache_read_input_tokens"] = _safe_int(
        usage.get("cache_read_input_tokens")
    )
    result["cumulative_cache_write_input_tokens"] = _safe_int(
        usage.get("cache_write_input_tokens")
    )
    result["cost_units"] = _safe_float(usage.get("cost_units"))
    result["main_cost_units"] = _safe_float(
        usage.get("main_cost_units", result["cost_units"])
    )
    result["route_model_cost_units"] = _safe_float(usage.get("route_model_cost_units"))
    result["main_model_id"] = str(usage.get("model_id", ""))
    result["main_model_provider"] = str(usage.get("provider", ""))
    result["main_route_level"] = str(usage.get("route_level", ""))
    result["input_price_per_1m"] = _safe_float(usage.get("input_price_per_1m"))
    result["output_price_per_1m"] = _safe_float(usage.get("output_price_per_1m"))
    result["cache_read_price_per_1m"] = _safe_float(
        usage.get("cache_read_price_per_1m")
    )
    result["cache_write_price_per_1m"] = _safe_float(
        usage.get("cache_write_price_per_1m")
    )
    result["cost_currency"] = str(usage.get("cost_currency", ""))
    result["cost_price_source"] = str(usage.get("cost_price_source", ""))
    result["cost_priced"] = bool(usage.get("cost_priced", False))
    result["usage_record_quality"] = _usage_quality(usage)
    result["usage_source"] = str(usage.get("usage_source", ""))
    result["usage_estimated"] = bool(usage.get("usage_estimated", False))
    return result


def solve_one_task(
    task_dict: dict[str, Any],
    workspace_dir: str,
    artifacts_dir: str,
    model_id: str,
    region: str,
    max_tokens: int,
    max_steps: int,
    window_size: int,
    enable_skill_proposal: bool,
    runtime_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Solve one SWE task in a worker process."""
    from agent_evolve.agents.swe.agent import SweAgent
    from agent_evolve.benchmarks.swe_verified_mini.benchmark import SweVerifiedMiniBenchmark

    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s [{task_dict['id']}] %(message)s")
    for name in ("botocore", "urllib3", "httpcore", "httpx", "strands.models", "strands.tools"):
        logging.getLogger(name).setLevel(logging.WARNING)

    task = Task(id=task_dict["id"], input=task_dict["input"], metadata=task_dict["metadata"])
    instance_id = task.id
    started = time.time()
    trajectory = None

    agent = SweAgent(
        workspace_dir=workspace_dir,
        model_id=model_id,
        region=region,
        max_tokens=max_tokens,
        max_steps=max_steps,
        window_size=window_size,
        efficiency_prompt=False,
        enable_skill_proposal=enable_skill_proposal,
        runtime_config=runtime_config,
    )
    benchmark = SweVerifiedMiniBenchmark(dataset_name=task_dict["dataset"], shuffle=False)

    try:
        trajectory = agent.solve(task)
        feedback = benchmark.evaluate(task, trajectory)
    except Exception as exc:  # noqa: BLE001
        return _attach_usage_fields(
            {
                "instance_id": instance_id,
                "success": False,
                "score": 0.0,
                "error": str(exc),
                "elapsed": round(time.time() - started, 1),
                "patch_len": 0,
                "patch": "",
                "usage_record_quality": "missing",
                "usage_source": "solve_or_evaluate_exception",
            }
        )

    steps = trajectory.steps or []
    usage_step = _summary_usage_step(steps)
    usage = normalize_usage(usage_step.get("usage", {}) if isinstance(usage_step, dict) else {})
    per_turn = usage_step.get("per_turn_usage", []) if isinstance(usage_step, dict) else []
    route_model_calls = usage_step.get("route_model_calls", []) if isinstance(usage_step, dict) else []
    max_input = usage_step.get("max_input_tokens_per_turn", 0) if isinstance(usage_step, dict) else 0
    active_strategies = (
        usage_step.get("active_efficiency_strategies", [])
        if isinstance(usage_step, dict) else []
    )
    num_tool_calls = len([s for s in steps if isinstance(s, dict) and "tool" in s])
    strategy_hits = result_strategy_hits({"steps": steps, "strategies": active_strategies})
    skill_usage_counts = {
        key.removeprefix("read_skill:"): value
        for key, value in strategy_hits.items()
        if key.startswith("read_skill:")
    }
    emergent_skill_usage_counts = {
        key.removeprefix("emergent_skill:"): value
        for key, value in strategy_hits.items()
        if key.startswith("emergent_skill:")
    }

    conversation_data = None
    conversation = getattr(trajectory, "_conversation", None)
    if conversation:
        try:
            conv_dir = Path(artifacts_dir) / "conversations"
            conv_dir.mkdir(parents=True, exist_ok=True)
            conv_path = conv_dir / f"{instance_id.replace('/', '_')}.json"
            conv_path.write_text(
                json.dumps(conversation, indent=2, default=_json_default, ensure_ascii=False),
                encoding="utf-8",
            )
            conversation_data = str(conv_path)
        except Exception:
            conversation_data = None

    system_prompt_data = None
    system_prompt = getattr(trajectory, "_system_prompt", "")
    if system_prompt:
        try:
            prompt_dir = Path(artifacts_dir) / "system_prompts"
            prompt_dir.mkdir(parents=True, exist_ok=True)
            prompt_path = prompt_dir / f"{instance_id.replace('/', '_')}.md"
            prompt_path.write_text(str(system_prompt), encoding="utf-8")
            system_prompt_data = str(prompt_path)
        except Exception:
            system_prompt_data = None

    return _attach_usage_fields(
        {
            "instance_id": instance_id,
            "success": bool(feedback.success),
            "score": float(feedback.score),
            "feedback_detail": feedback.detail,
            "elapsed": round(time.time() - started, 1),
            "patch_len": len(trajectory.output or ""),
            "patch": trajectory.output or "",
            "skill_proposal": getattr(trajectory, "_skill_proposal", ""),
            "num_tool_calls": num_tool_calls,
            "num_turns": len(per_turn),
            "cumulative_input_tokens": usage.get("input_tokens", 0),
            "cumulative_output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "cumulative_cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
            "cumulative_cache_write_input_tokens": usage.get("cache_write_input_tokens", 0),
            "cost_units": usage.get("cost_units", 0.0),
            "main_cost_units": usage.get("main_cost_units", usage.get("cost_units", 0.0)),
            "route_model_cost_units": usage.get("route_model_cost_units", 0.0),
            "main_model_id": usage.get("model_id", ""),
            "main_model_provider": usage.get("provider", ""),
            "main_route_level": usage.get("route_level", ""),
            "input_price_per_1m": usage.get("input_price_per_1m", 0.0),
            "output_price_per_1m": usage.get("output_price_per_1m", 0.0),
            "cache_read_price_per_1m": usage.get("cache_read_price_per_1m", 0.0),
            "cache_write_price_per_1m": usage.get("cache_write_price_per_1m", 0.0),
            "cost_currency": usage.get("cost_currency", ""),
            "cost_price_source": usage.get("cost_price_source", ""),
            "cost_priced": bool(usage.get("cost_priced", False)),
            "usage_record_quality": _usage_quality(usage),
            "usage_source": usage.get("usage_source", ""),
            "usage_estimated": bool(usage.get("usage_estimated", False)),
            "max_input_tokens_per_turn": max_input,
            "active_efficiency_strategies": active_strategies,
            "strategies": active_strategies,
            "skill_usage_counts": skill_usage_counts,
            "emergent_skill_usage_counts": emergent_skill_usage_counts,
            "efficiency_config_id": (
                usage_step.get("efficiency_config_id", "") if isinstance(usage_step, dict) else ""
            ),
            "route_model_calls": route_model_calls,
            "conversation_path": conversation_data,
            "system_prompt_path": system_prompt_data,
            "steps": steps,
        }
    )


def _copy_seed_workspace(seed_workspace: Path, workspace_dir: Path, *, hide_efficiency: bool) -> None:
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    shutil.copytree(seed_workspace, workspace_dir)
    if hide_efficiency:
        cost_profile_text = ""
        cost_profile = workspace_dir / "efficiency" / "cost_profile.yaml"
        if cost_profile.exists():
            cost_profile_text = cost_profile.read_text(encoding="utf-8")
        shutil.rmtree(workspace_dir / "efficiency", ignore_errors=True)
        if cost_profile_text.strip():
            cost_profile.parent.mkdir(parents=True, exist_ok=True)
            cost_profile.write_text(cost_profile_text, encoding="utf-8")
        _redact_predefined_names(workspace_dir)
    # A stale active config would leak predefined strategy names.
    active = workspace_dir / "efficiency" / "active_strategies.yaml"
    if hide_efficiency and active.exists():
        active.unlink()


def _redact_predefined_names(workspace_dir: Path) -> None:
    """Remove predefined strategy names from the copied hidden-mode workspace."""
    suffixes = {".md", ".py", ".yaml", ".yml", ".txt", ".json"}
    for path in workspace_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = text
        for index, name in enumerate(PREDEFINED_STRATEGY_NAMES, start=1):
            updated = updated.replace(name, f"redacted_efficiency_label_{index}")
        if updated != text:
            path.write_text(updated, encoding="utf-8")


def _configure_predefined_control(
    workspace_dir: Path,
    strategies: list[str],
    *,
    experiment: str = "emergent_efficiency_predefined_control",
) -> dict[str, Any]:
    clean = [name for name in strategies if name in DEFAULT_STRATEGIES]
    write_active_config(
        workspace_dir,
        clean,
        metadata={
            "experiment": experiment,
            "source": "explicit_control",
        },
    )
    materialize_strategy_artifacts(workspace_dir, clean)
    return load_active_config(workspace_dir)


def _strategy_leakage_scan(root: Path) -> dict[str, Any]:
    hits: list[dict[str, str]] = []
    ignored_parts = {
        ".git",
        "__pycache__",
        "conversations",
        "patches",
        "results.json",
        "summary.json",
        "summary.csv",
    }
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in ignored_parts for part in rel.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for name in PREDEFINED_STRATEGY_NAMES:
            if name in text:
                hits.append({"file": str(rel), "strategy": name})
    return {
        "predefined_strategy_names": PREDEFINED_STRATEGY_NAMES,
        "hits": hits,
        "passed": not hits,
    }


def _waste_signals(result: dict[str, Any], *, max_steps: int) -> dict[str, Any]:
    steps = [s for s in result.get("steps", []) if isinstance(s, dict)]
    tool_steps = [s for s in steps if "tool" in s]
    repeated: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    edited_at = []
    for index, step in enumerate(tool_steps):
        action = str(step.get("action") or step.get("tool") or "")
        action_counts[action] = action_counts.get(action, 0) + 1
        key = f"{action}:{str(step.get('input_summary', ''))[:120]}"
        repeated[key] = repeated.get(key, 0) + 1
        if _is_edit_step(step):
            edited_at.append(index + 1)
    repeated_top = [
        {"signature": key[:180], "count": count}
        for key, count in sorted(repeated.items(), key=lambda item: item[1], reverse=True)
        if count >= 2
    ][:5]
    last_edit_step = max(edited_at) if edited_at else 0
    behavior_stats = _behavior_hit_stats(result)
    return {
        "task_id": result.get("instance_id", ""),
        "success": bool(result.get("success")),
        "score": result.get("score", 0.0),
        "cost_units": round(_safe_float(result.get("cost_units")), 2),
        "total_tokens": _safe_int(result.get("total_tokens")),
        "input_tokens": _safe_int(result.get("cumulative_input_tokens")),
        "output_tokens": _safe_int(result.get("cumulative_output_tokens")),
        "cache_read_input_tokens": _safe_int(result.get("cumulative_cache_read_input_tokens")),
        "cache_write_input_tokens": _safe_int(result.get("cumulative_cache_write_input_tokens")),
        "main_model_cost_units": round(_safe_float(result.get("main_cost_units")), 4),
        "route_model_cost_units": round(_safe_float(result.get("route_model_cost_units")), 4),
        "tool_calls": _safe_int(result.get("num_tool_calls")),
        "turns": _safe_int(result.get("num_turns")),
        "max_input_tokens_per_turn": _safe_int(result.get("max_input_tokens_per_turn")),
        "near_step_limit": bool(max_steps and _safe_int(result.get("num_tool_calls")) >= int(max_steps * 0.9)),
        "no_edit_before_end": bool(tool_steps and last_edit_step == 0),
        "last_edit_step": last_edit_step,
        "action_counts": action_counts,
        "repeated_actions": repeated_top,
        "behavior_stats": {
            "prompt_fragment_visible": behavior_stats.get("prompt_fragment_visible"),
            "repeated_exact_inputs": behavior_stats.get("repeated_exact_inputs"),
            "repeated_test_inputs": behavior_stats.get("repeated_test_inputs"),
            "same_state_repeated_tests": behavior_stats.get("same_state_repeated_tests"),
            "post_edit_repeated_tests": behavior_stats.get("post_edit_repeated_tests"),
            "prompt_fragment_violation_events": behavior_stats.get("prompt_fragment_violation_events"),
        },
        "error": str(result.get("error") or "")[:240],
    }


def _step_action(step: dict[str, Any]) -> str:
    return str(step.get("action") or step.get("tool") or "").strip().lower()


def _step_signature(step: dict[str, Any]) -> str:
    action = _step_action(step)
    summary = str(step.get("input_summary") or step.get("input") or "")[:180].strip()
    return f"{action}:{summary}"


def _is_edit_step(step: dict[str, Any]) -> bool:
    action = _step_action(step)
    tool = str(step.get("tool") or "").strip().lower()
    if action in {"edit_file", "write_file", "edit", "write"}:
        return True
    return tool == "text_editor" and action not in {"read_file", "view", "search", "submit"}


def _is_test_step(step: dict[str, Any]) -> bool:
    action = _step_action(step)
    haystack = " ".join(
        str(step.get(key) or "")
        for key in ("input_summary", "input", "command", "args")
    ).lower()
    return action in {"run_test", "test"} or any(
        marker in haystack
        for marker in ("pytest", "tox ", " unittest", "manage.py test", "npm test")
    )


def _prompt_fragment_visible(result: dict[str, Any]) -> bool:
    prompt_text = str(result.get("system_prompt") or "")
    prompt_path = str(result.get("system_prompt_path") or "")
    if prompt_path:
        path = Path(prompt_path)
        if path.exists():
            try:
                prompt_text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                pass
    return "emergent_efficiency_proposal:" in prompt_text or "<!-- evolve:" in prompt_text


def _behavior_hit_stats(result: dict[str, Any]) -> dict[str, Any]:
    """Track whether behavioral prompt fragments appear to be obeyed or missed."""
    tool_steps = [
        step for step in result.get("steps", [])
        if isinstance(step, dict) and "tool" in step
    ]
    seen: dict[str, int] = {}
    test_seen_at_edit_version: dict[str, int] = {}
    edit_version = 0
    repeated_exact_inputs = 0
    repeated_test_inputs = 0
    same_state_repeated_tests = 0
    post_edit_repeated_tests = 0
    duplicate_examples: list[dict[str, Any]] = []

    for index, step in enumerate(tool_steps, start=1):
        if _is_edit_step(step):
            edit_version += 1
        signature = _step_signature(step)
        previous = seen.get(signature, 0)
        if previous:
            repeated_exact_inputs += 1
            if len(duplicate_examples) < 5:
                duplicate_examples.append({
                    "step": index,
                    "signature": signature[:220],
                    "previous_count": previous,
                })
        seen[signature] = previous + 1

        if not _is_test_step(step):
            continue
        if signature in test_seen_at_edit_version:
            repeated_test_inputs += 1
            if test_seen_at_edit_version[signature] == edit_version:
                same_state_repeated_tests += 1
            else:
                post_edit_repeated_tests += 1
        test_seen_at_edit_version[signature] = edit_version

    fragment_visible = _prompt_fragment_visible(result)
    return {
        "task_id": result.get("instance_id", ""),
        "prompt_fragment_visible": fragment_visible,
        "repeated_exact_inputs": repeated_exact_inputs,
        "repeated_test_inputs": repeated_test_inputs,
        "same_state_repeated_tests": same_state_repeated_tests,
        "post_edit_repeated_tests": post_edit_repeated_tests,
        "prompt_fragment_violation_events": same_state_repeated_tests if fragment_visible else 0,
        "duplicate_examples": duplicate_examples,
    }


def _summarize_behavior_stats(results: list[dict[str, Any]]) -> dict[str, Any]:
    stats = [r.get("behavior_stats") or _behavior_hit_stats(r) for r in results]
    return {
        "prompt_fragment_visible_tasks": sum(1 for item in stats if item.get("prompt_fragment_visible")),
        "repeated_exact_inputs": sum(_safe_int(item.get("repeated_exact_inputs")) for item in stats),
        "repeated_test_inputs": sum(_safe_int(item.get("repeated_test_inputs")) for item in stats),
        "same_state_repeated_tests": sum(_safe_int(item.get("same_state_repeated_tests")) for item in stats),
        "post_edit_repeated_tests": sum(_safe_int(item.get("post_edit_repeated_tests")) for item in stats),
        "prompt_fragment_violation_events": sum(
            _safe_int(item.get("prompt_fragment_violation_events")) for item in stats
        ),
    }


def _build_cost_feedback(
    batch_results: list[dict[str, Any]],
    *,
    batch_num: int,
    max_steps: int,
    allow_predefined: bool = False,
    include_category_seed: bool = False,
) -> str:
    summary = summarize_results(batch_results)
    signals = [_waste_signals(result, max_steps=max_steps) for result in batch_results]
    active_predefined = sorted(
        {
            strategy
            for result in batch_results
            for strategy in (
                result.get("active_efficiency_strategies")
                or result.get("strategies")
                or []
            )
            if strategy in DEFAULT_STRATEGIES
        }
    )
    payload = {
        "batch": batch_num,
        "summary": {
            "tasks": summary.get("tasks"),
            "success": summary.get("success"),
            "success_rate": summary.get("success_rate"),
            "avg_tokens": summary.get("avg_tokens"),
            "avg_tokens_success": summary.get("avg_tokens_success"),
            "avg_cost": summary.get("avg_cost"),
            "avg_cost_success": summary.get("avg_cost_success"),
            "cost_of_pass": summary.get("cost_of_pass"),
            "avg_tool_calls": summary.get("avg_tool_calls"),
            "cache_read_input_tokens": summary.get("cache_read_input_tokens"),
            "cache_write_input_tokens": summary.get("cache_write_input_tokens"),
            "main_model_cost": summary.get("main_model_cost"),
            "route_model_cost": summary.get("route_model_cost"),
            "easy_model_cost": summary.get("easy_model_cost"),
            "hard_model_cost": summary.get("hard_model_cost"),
            "zero_token_tasks": summary.get("zero_token_tasks"),
            "estimated_usage_tasks": summary.get("estimated_usage_tasks"),
            "token_record_quality": summary.get("token_record_quality"),
        },
        "task_cost_and_behavior_signals": signals,
        "instructions": (
            "Infer incremental reusable cost-saving behaviors from these signals. "
            "Active predefined strategies are already available; do not simply restate them."
            if allow_predefined
            else (
                "Infer reusable cost-saving behaviors only from these neutral signals. "
                "Do not refer to any predefined strategy registry, candidate list, or strategy id."
            )
        ),
    }
    if allow_predefined:
        payload["active_predefined_strategies"] = active_predefined
    if include_category_seed:
        payload["efficiency_category_taxonomy"] = _category_taxonomy_payload()
        payload["instructions"] = (
            "Infer concrete reusable behaviors from the task signals. Use the M1-M7 "
            "categories only as broad cost-dimension labels. Do not refer to any "
            "predefined strategy registry, candidate list, strategy id, prompt block, "
            "or gate implementation. Each proposal must include a categories list."
        )
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped.strip(), flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(stripped[start : end + 1])
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _recover_proposals_from_partial_json(text: str) -> list[dict[str, Any]]:
    """Recover complete proposal objects from a truncated proposer response."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    decoder = json.JSONDecoder()
    proposals: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        if isinstance(value.get("proposals"), list):
            for item in value["proposals"]:
                if isinstance(item, dict):
                    key = json.dumps(item, sort_keys=True, ensure_ascii=False)
                    if key not in seen:
                        seen.add(key)
                        proposals.append(item)
            continue
        required = {"name", "kind", "content"}
        if required.issubset(value):
            key = json.dumps(value, sort_keys=True, ensure_ascii=False)
            if key not in seen:
                seen.add(key)
                proposals.append(value)
    return proposals


def _load_proposer_llm(args: argparse.Namespace):
    from agent_evolve.algorithms.adaptive_skill.tools import create_default_llm

    config = EvolveConfig(
        evolver_model=args.evolver_model or args.model_id,
        evolver_max_tokens=args.evolver_max_tokens,
        evolver_temperature=args.evolver_temperature,
        extra={"region": args.region},
    )
    return create_default_llm(config)


def _complete_proposer_with_retries(
    llm: Any,
    messages: list[LLMMessage],
    *,
    max_tokens: int,
    temperature: float,
    retries: int,
    initial_delay: float,
    max_delay: float,
) -> tuple[str, dict[str, Any]]:
    """Call the proposer LLM with bounded retries and return raw text + metadata."""
    retry_count = max(0, int(retries))
    max_attempts = retry_count + 1
    delay = max(0.0, float(initial_delay))
    delay_cap = max(delay, float(max_delay))
    errors: list[dict[str, Any]] = []

    for attempt in range(1, max_attempts + 1):
        try:
            response = llm.complete(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return str(response.content or ""), {
                "status": "ok",
                "attempts": attempt,
                "max_attempts": max_attempts,
                "errors": errors,
            }
        except Exception as exc:
            error = {
                "attempt": attempt,
                "type": exc.__class__.__name__,
                "message": str(exc),
            }
            errors.append(error)
            if attempt >= max_attempts:
                logger.error(
                    "Proposer LLM call failed after %d attempt(s): %s: %s",
                    attempt,
                    error["type"],
                    error["message"],
                )
                break
            sleep_for = min(delay_cap, delay * (2 ** (attempt - 1))) if delay else 0.0
            logger.warning(
                "Proposer LLM call failed on attempt %d/%d; retrying in %.1fs: %s: %s",
                attempt,
                max_attempts,
                sleep_for,
                error["type"],
                error["message"],
            )
            if sleep_for:
                time.sleep(sleep_for)

    return "", {
        "status": "failed",
        "attempts": max_attempts,
        "max_attempts": max_attempts,
        "errors": errors,
    }


def propose_emergent_efficiency(
    *,
    batch_results: list[dict[str, Any]],
    batch_num: int,
    max_steps: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    allow_predefined = args.mode == "predefined-cost-aware"
    include_category_seed = (
        args.mode == "category-seeded-cost-aware"
        and getattr(args, "category_seed", "basic-efficiency") == "basic-efficiency"
    )
    feedback = _build_cost_feedback(
        batch_results,
        batch_num=batch_num,
        max_steps=max_steps,
        allow_predefined=allow_predefined,
        include_category_seed=include_category_seed,
    )
    if args.dry_run_proposer:
        raw = json.dumps(
            {
                "proposals": [
                    {
                        "name": f"dry_run_cost_review_gen_{batch_num}",
                        "kind": "prompt_fragment",
                        "description": "TRIGGER when task cost is high; DO NOT TRIGGER when evidence is insufficient.",
                        "trigger": "high repeated tool use or near step limit",
                        "do_not_trigger": "when a targeted edit and verification are already complete",
                        "content": (
                            "Before each new exploratory command, state the specific uncertainty it resolves. "
                            "Prefer the smallest command that can change the patch decision, then edit and verify."
                        ),
                        "expected_savings": "reduces repeated broad exploration",
                        "correctness_risk": "may skip useful context if applied too early",
                        "confidence": "MEDIUM",
                        "categories": ["M5 Tools"] if include_category_seed else [],
                        "evidence": [f"batch {batch_num}: dry-run placeholder"],
                    }
                ]
            },
            ensure_ascii=False,
        )
        proposer_call = {
            "status": "dry_run",
            "attempts": 0,
            "max_attempts": 0,
            "errors": [],
        }
    else:
        llm = _load_proposer_llm(args)
        raw, proposer_call = _complete_proposer_with_retries(
            llm,
            [
                LLMMessage(
                    role="system",
                    content=(
                        PREDEFINED_COST_AWARE_SYSTEM_PROMPT
                        if allow_predefined
                        else (
                            CATEGORY_SEEDED_SYSTEM_PROMPT
                            if include_category_seed
                            else EMERGENCE_SYSTEM_PROMPT
                        )
                    ),
                ),
                LLMMessage(role="user", content=feedback),
            ],
            max_tokens=args.evolver_max_tokens,
            temperature=args.evolver_temperature,
            retries=args.evolver_call_retries,
            initial_delay=args.evolver_retry_initial_delay,
            max_delay=args.evolver_retry_max_delay,
        )

    parsed = _extract_json_object(raw)
    parse_status = "parsed" if parsed else ("empty_response" if not raw.strip() else "parse_failed")
    if proposer_call.get("status") == "failed":
        parse_status = "llm_call_failed"
    recovered_count = 0
    if not parsed and raw.strip():
        recovered = _recover_proposals_from_partial_json(raw)
        if recovered:
            parsed = {"proposals": recovered}
            recovered_count = len(recovered)
            parse_status = "partial_recovered"
    proposals = parsed.get("proposals", [])
    if not isinstance(proposals, list):
        proposals = []
        parse_status = "invalid_proposals_field"
    return {
        "batch": batch_num,
        "mode": args.mode,
        "category_seeded": include_category_seed,
        "category_taxonomy": _category_taxonomy_payload() if include_category_seed else [],
        "raw_response": raw,
        "feedback": feedback,
        "parse_status": parse_status,
        "proposer_call": proposer_call,
        "recovered_proposals": recovered_count,
        "proposals": [p for p in proposals if isinstance(p, dict)],
    }


def _proposal_contains_predefined_name(proposal: dict[str, Any]) -> list[str]:
    text = json.dumps(proposal, ensure_ascii=False)
    return [name for name in PREDEFINED_STRATEGY_NAMES if name in text]


def _apply_proposals(
    workspace_dir: Path,
    proposal_report: dict[str, Any],
    *,
    output_dir: Path,
    max_apply: int,
) -> list[dict[str, Any]]:
    workspace = AgentWorkspace(workspace_dir)
    applied: list[dict[str, Any]] = []
    proposals = proposal_report.get("proposals", []) or []
    category_seeded = bool(proposal_report.get("category_seeded"))
    for index, proposal in enumerate(proposals[: max(0, max_apply)], start=1):
        confidence = str(proposal.get("confidence") or "MEDIUM").upper()
        categories = _proposal_categories(proposal)
        category_status = "ok"
        if category_seeded and not categories:
            category_status = "missing_categories"
        if confidence == "LOW":
            applied.append({
                "index": index,
                "status": "skipped_low_confidence",
                "categories": categories,
                "category_status": category_status,
                "proposal": proposal,
            })
            continue
        name = _slug(str(proposal.get("name") or f"proposal_{index}"), default=f"proposal_{index}")
        kind = _normalize_kind(str(proposal.get("kind") or "skill"))
        content = _strip_code_fence(str(proposal.get("content") or ""))
        if not content:
            applied.append({
                "index": index,
                "status": "skipped_empty_content",
                "categories": categories,
                "category_status": category_status,
                "proposal": proposal,
            })
            continue
        leaked_names = _proposal_contains_predefined_name(proposal)
        description = str(proposal.get("description") or proposal.get("trigger") or content[:120]).strip()
        proposal_id = f"emergent_gen{proposal_report.get('batch', 0):02d}_{index:02d}_{name}"
        if kind == "prompt_fragment":
            filename = f"{proposal_id}.md"
            workspace.write_fragment(
                filename,
                (
                    f"<!-- emergent_efficiency_proposal: {proposal_id} -->\n"
                    f"<!-- categories: {', '.join(categories) if categories else 'none'} -->\n"
                    f"{content.strip()}\n"
                ),
            )
            target = f"prompts/fragments/{filename}"
        elif kind == "skill":
            skill_name = proposal_id
            workspace.write_skill(
                skill_name,
                (
                    "---\n"
                    f"name: {skill_name}\n"
                    f"description: {description[:180]}\n"
                    "---\n\n"
                    "# Emergent Efficiency Skill\n\n"
                    f"Categories: {', '.join(categories) if categories else 'none'}\n\n"
                    f"Trigger: {proposal.get('trigger', '')}\n\n"
                    f"Do not trigger: {proposal.get('do_not_trigger', '')}\n\n"
                    f"{content.strip()}\n\n"
                    f"Expected savings: {proposal.get('expected_savings', '')}\n\n"
                    f"Correctness risk: {proposal.get('correctness_risk', '')}\n"
                ),
            )
            target = f"skills/{skill_name}/SKILL.md"
        elif kind == "tool":
            validation_error = _tool_validation_error(content)
            if validation_error:
                applied.append({
                    "index": index,
                    "status": f"skipped_invalid_tool:{validation_error}",
                    "kind": kind,
                    "categories": categories,
                    "category_status": category_status,
                    "proposal": proposal,
                })
                continue
            function_match = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", content)
            tool_name = function_match.group(1) if function_match else _tool_function_name(proposal_id)
            registry = workspace.read_tool_registry()
            if any(entry.get("name") == tool_name for entry in registry):
                applied.append({
                    "index": index,
                    "status": "skipped_duplicate_tool",
                    "proposal_id": proposal_id,
                    "kind": kind,
                    "categories": categories,
                    "category_status": category_status,
                    "target": f"tools/{tool_name}.py",
                    "proposal": proposal,
                })
                continue
            workspace.write_tool(tool_name, content)
            registry.append({
                "name": tool_name,
                "file": f"{tool_name}.py",
                "description": description[:180],
            })
            workspace.write_tool_registry(registry)
            target = f"tools/{tool_name}.py"
        elif kind == "memory":
            workspace.add_memory(
                {
                    "type": "emergent_efficiency_strategy",
                    "name": proposal_id,
                    "description": description[:180],
                    "categories": categories,
                    "trigger": proposal.get("trigger", ""),
                    "do_not_trigger": proposal.get("do_not_trigger", ""),
                    "content": content.strip(),
                    "expected_savings": proposal.get("expected_savings", ""),
                    "correctness_risk": proposal.get("correctness_risk", ""),
                    "confidence": confidence,
                    "evidence": proposal.get("evidence", []),
                },
                category="strategy",
            )
            target = "memory/strategy.jsonl"
        else:
            applied.append({
                "index": index,
                "status": f"skipped_unknown_kind:{kind}",
                "kind": kind,
                "categories": categories,
                "category_status": category_status,
                "proposal": proposal,
            })
            continue
        record = {
            "index": index,
            "status": "applied",
            "proposal_id": proposal_id,
            "kind": kind,
            "categories": categories,
            "category_status": category_status,
            "target": target,
            "confidence": confidence,
            "predefined_name_hits": leaked_names,
            "proposal": proposal,
        }
        applied.append(record)

    adoption_path = output_dir / "emergence" / "proposal_adoption.json"
    adoption_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if adoption_path.exists():
        try:
            existing = json.loads(adoption_path.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    existing.extend(applied)
    adoption_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    return applied


def _write_proposals(output_dir: Path, report: dict[str, Any]) -> None:
    emergence_dir = output_dir / "emergence"
    emergence_dir.mkdir(parents=True, exist_ok=True)
    if report.get("category_seeded"):
        (emergence_dir / "category_taxonomy.json").write_text(
            json.dumps(_category_taxonomy_payload(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    jsonl = emergence_dir / "proposals.jsonl"
    with jsonl.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, default=_json_default) + "\n")
    latest = emergence_dir / f"batch_{int(report.get('batch', 0)):02d}_proposal_report.json"
    latest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_proposal_adoptions(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "emergence" / "proposal_adoption.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _summarize_proposal_categories(adoptions: list[dict[str, Any]]) -> dict[str, Any]:
    category_counts: dict[str, int] = {}
    applied_category_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {}
    applied_kind_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    missing_category_records = 0
    for record in adoptions:
        status = str(record.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        proposal = record.get("proposal", {}) if isinstance(record.get("proposal"), dict) else {}
        categories = _normalize_categories(record.get("categories") or proposal.get("categories"))
        if record.get("category_status") == "missing_categories":
            missing_category_records += 1
        kind = _normalize_kind(str(record.get("kind") or proposal.get("kind") or "unknown"))
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        if status == "applied":
            applied_kind_counts[kind] = applied_kind_counts.get(kind, 0) + 1
        for category in categories:
            category_counts[category] = category_counts.get(category, 0) + 1
            if status == "applied":
                applied_category_counts[category] = applied_category_counts.get(category, 0) + 1
    return {
        "total_records": len(adoptions),
        "status_counts": status_counts,
        "kind_counts": kind_counts,
        "applied_kind_counts": applied_kind_counts,
        "category_counts": category_counts,
        "applied_category_counts": applied_category_counts,
        "missing_category_records": missing_category_records,
    }


def _write_proposal_category_outputs(output_dir: Path) -> None:
    adoptions = _read_proposal_adoptions(output_dir)
    if not adoptions:
        return
    emergence_dir = output_dir / "emergence"
    emergence_dir.mkdir(parents=True, exist_ok=True)
    summary = _summarize_proposal_categories(adoptions)
    (emergence_dir / "proposal_category_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    try:
        write_proposal_category_plots(adoptions, output_dir / "figures")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to render proposal category figures: %s", exc)


def _record_workspace_snapshot(workspace_dir: Path, output_dir: Path, batch_num: int) -> None:
    snapshot_dir = output_dir / "emergence" / "workspace_snapshots" / f"after_batch_{batch_num:02d}"
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
    for rel_name in ("prompts", "skills", "tools", "memory"):
        src = workspace_dir / rel_name
        if src.exists():
            shutil.copytree(src, snapshot_dir / rel_name)


def _results_to_observations(results: list[dict[str, Any]], tasks_by_id: dict[str, Task]) -> list[Observation]:
    observations = []
    for result in results:
        task = tasks_by_id.get(str(result.get("instance_id", "")))
        if task is None:
            continue
        feedback = Feedback(
            success=bool(result.get("success")),
            score=float(result.get("score", 0.0) or 0.0),
            detail=str(result.get("feedback_detail") or ""),
            raw={},
        )
        trajectory = Trajectory(
            task_id=task.id,
            output=str(result.get("patch") or ""),
            steps=result.get("steps", []) if isinstance(result.get("steps"), list) else [],
        )
        trajectory._skill_proposal = str(result.get("skill_proposal") or "")
        observations.append(Observation(task=task, trajectory=trajectory, feedback=feedback))
    return observations


def _write_workspace_evolution_records(
    *,
    workspace_dir: Path,
    batch_results: list[dict[str, Any]],
    observations: list[Observation],
    batch_num: int,
) -> None:
    """Persist batch trajectories/observations into the workspace evolution tree."""
    evolution_dir = workspace_dir / "evolution"
    trajectories_dir = evolution_dir / "trajectories"
    observations_dir = evolution_dir / "observations"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    observations_dir.mkdir(parents=True, exist_ok=True)

    batch_slug = f"batch_{batch_num:02d}"
    result_by_id = {str(r.get("instance_id", "")): r for r in batch_results}
    trajectory_index: list[dict[str, Any]] = []
    for task_id, result in sorted(result_by_id.items()):
        safe_task_id = task_id.replace("/", "__")
        path = trajectories_dir / f"{batch_slug}__{safe_task_id}.json"
        trajectory_payload = dict(result)
        conversation_path = str(result.get("conversation_path") or "")
        if conversation_path:
            conv_path = Path(conversation_path)
            if not conv_path.is_absolute():
                conv_path = Path.cwd() / conv_path
            if conv_path.exists():
                try:
                    trajectory_payload["conversation"] = json.loads(conv_path.read_text(encoding="utf-8"))
                except Exception:
                    trajectory_payload["conversation"] = conv_path.read_text(
                        encoding="utf-8",
                        errors="replace",
                    )
        system_prompt_path = str(result.get("system_prompt_path") or "")
        if system_prompt_path:
            prompt_path = Path(system_prompt_path)
            if not prompt_path.is_absolute():
                prompt_path = Path.cwd() / prompt_path
            if prompt_path.exists():
                trajectory_payload["system_prompt"] = prompt_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
        path.write_text(
            json.dumps(trajectory_payload, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        trajectory_index.append({
            "task_id": task_id,
            "path": str(path.relative_to(evolution_dir)),
            "success": bool(result.get("success")),
            "score": result.get("score", 0.0),
            "steps": len(result.get("steps", []) if isinstance(result.get("steps"), list) else []),
            "conversation_path": result.get("conversation_path", ""),
            "system_prompt_path": result.get("system_prompt_path", ""),
        })

    observation_payload = []
    for obs in observations:
        observation_payload.append({
            "task": {
                "id": obs.task.id,
                "input": obs.task.input,
                "metadata": obs.task.metadata,
            },
            "feedback": {
                "success": obs.feedback.success,
                "score": obs.feedback.score,
                "detail": obs.feedback.detail,
                "raw": obs.feedback.raw,
            },
            "trajectory": {
                "task_id": obs.trajectory.task_id,
                "output": obs.trajectory.output,
                "steps": obs.trajectory.steps,
                "skill_proposal": getattr(obs.trajectory, "_skill_proposal", ""),
            },
        })
    (observations_dir / f"{batch_slug}.json").write_text(
        json.dumps(observation_payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    with (evolution_dir / "trajectory_index.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "batch": batch_num,
            "trajectories": trajectory_index,
        }, ensure_ascii=False) + "\n")


def _run_normal_evolution(
    *,
    workspace_dir: Path,
    output_dir: Path,
    observations: list[Observation],
    batch_num: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run the ordinary solver-proposal curation path used as A/B control."""
    from agent_evolve.algorithms.guided_synth.engine import GuidedSynthesisEngine

    config = EvolveConfig(
        evolver_model=args.evolver_model or args.model_id,
        evolver_max_tokens=args.evolver_max_tokens,
        evolver_temperature=args.evolver_temperature,
        extra={"region": args.region},
    )
    engine = GuidedSynthesisEngine(config, write_memory=False)
    started = time.time()
    result = engine.evolve(
        workspace=AgentWorkspace(workspace_dir),
        observation_logs=observations,
        evo_number=batch_num,
    )
    record = {
        "batch": batch_num,
        "mode": args.mode,
        "elapsed": round(time.time() - started, 1),
        "result": result,
        "solver_proposals": sum(
            1 for obs in observations
            if str(getattr(obs.trajectory, "_skill_proposal", "") or "").strip()
        ),
        "solver_proposal_previews": [
            {
                "task_id": obs.task.id,
                "proposal": str(getattr(obs.trajectory, "_skill_proposal", "") or "")[:1200],
            }
            for obs in observations
            if str(getattr(obs.trajectory, "_skill_proposal", "") or "").strip()
        ],
    }
    path = output_dir / "emergence" / "normal_evolution.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
    return record


def _solve_batch(
    *,
    batch_tasks: list[Task],
    batch_num: int,
    workspace_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    runtime_config: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    task_dicts = [
        {
            "id": task.id,
            "input": task.input,
            "metadata": task.metadata,
            "dataset": args.dataset,
        }
        for task in batch_tasks
    ]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(
                solve_one_task,
                task_dict,
                str(workspace_dir),
                str(output_dir),
                args.model_id,
                args.region,
                args.max_tokens,
                args.max_steps,
                args.window_size,
                True,
                runtime_config,
            ): task_dict["id"]
            for task_dict in task_dicts
        }
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = _attach_usage_fields(
                    {
                        "instance_id": task_id,
                        "success": False,
                        "score": 0.0,
                        "error": str(exc),
                        "elapsed": 0.0,
                        "patch_len": 0,
                        "patch": "",
                        "usage_record_quality": "missing",
                        "usage_source": "batch_future_exception",
                    }
                )
            result["batch"] = batch_num
            result["mode"] = args.mode
            result["behavior_stats"] = _behavior_hit_stats(result)
            results.append(result)
            status = "PASS" if result.get("success") else "FAIL"
            print(
                f"  {status} {task_id} "
                f"cost={_safe_float(result.get('cost_units')):,.0f} "
                f"tokens={_safe_int(result.get('total_tokens')):,d} "
                f"calls={_safe_int(result.get('num_tool_calls'))}"
            )
    return results


def _make_summary_rows(results: list[dict[str, Any]], *, mode: str) -> list[dict[str, Any]]:
    def strategies_for(items: list[dict[str, Any]]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for result in items:
            for key in ("strategies", "active_efficiency_strategies"):
                values = result.get(key, []) or []
                if isinstance(values, str):
                    values = [values]
                if not isinstance(values, list):
                    continue
                for value in values:
                    name = str(value or "").strip()
                    if name and name not in seen:
                        seen.add(name)
                        out.append(name)
        return out

    rows: list[dict[str, Any]] = []
    for batch in sorted({_safe_int(r.get("batch")) for r in results}):
        batch_results = [r for r in results if _safe_int(r.get("batch")) == batch]
        row = {"group": f"{mode}_batch_{batch:02d}", "strategies": strategies_for(batch_results)}
        row.update(summarize_results(batch_results))
        row.update(_summarize_behavior_stats(batch_results))
        rows.append(row)
    overall = {"group": f"{mode}_overall", "strategies": strategies_for(results)}
    overall.update(summarize_results(results))
    overall.update(_summarize_behavior_stats(results))
    rows.append(overall)
    return rows


def _write_run_outputs(output_dir: Path, results: list[dict[str, Any]], *, mode: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    rows = _make_summary_rows(results, mode=mode)
    payload = {"mode": mode, "overall": rows[-1], "by_batch": rows[:-1]}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    behavior_payload = {
        "overall": _summarize_behavior_stats(results),
        "by_task": [
            result.get("behavior_stats") or _behavior_hit_stats(result)
            for result in results
        ],
    }
    (output_dir / "behavior_stats.json").write_text(
        json.dumps(behavior_payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    write_summary_table(rows, output_dir / "summary.csv")
    try:
        write_plots(rows, output_dir / "figures")
        write_strategy_library_plot(rows, output_dir / "figures")
        _write_proposal_category_outputs(output_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to render figures: %s", exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emergent efficiency SWE experiment runner")
    parser.add_argument(
        "--mode",
        choices=[
            "normal-evolve",
            "cost-aware",
            "predefined-control",
            "predefined-cost-aware",
            "category-seeded-cost-aware",
        ],
        default="cost-aware",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--dataset", default="MariusHobbhahn/swe-bench-verified-mini")
    parser.add_argument("--seed-workspace", default="seed_workspaces/swe")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Base output path. By default the runner appends a timestamped run "
            "directory/name to avoid overwrites."
        ),
    )
    parser.add_argument(
        "--fixed-output-dir",
        action="store_true",
        help="Use --output-dir exactly as provided instead of appending a timestamp.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Only useful with --fixed-output-dir; removes an existing output directory.",
    )
    parser.add_argument("--model-id", default="us.anthropic.claude-opus-4-6-v1")
    parser.add_argument("--evolver-model", default="")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--evolver-max-tokens", type=int, default=8192)
    parser.add_argument("--evolver-temperature", type=float, default=0.2)
    parser.add_argument(
        "--evolver-call-retries",
        type=int,
        default=4,
        help="Additional retries for the cost-aware proposer/evolver LLM call.",
    )
    parser.add_argument(
        "--evolver-retry-initial-delay",
        type=float,
        default=2.0,
        help="Initial proposer retry delay in seconds.",
    )
    parser.add_argument(
        "--evolver-retry-max-delay",
        type=float,
        default=30.0,
        help="Maximum proposer retry delay in seconds.",
    )
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--window-size", type=int, default=40)
    parser.add_argument("--max-proposals-per-batch", type=int, default=2)
    parser.add_argument(
        "--category-seed",
        choices=["none", "basic-efficiency"],
        default="basic-efficiency",
        help="Category taxonomy exposed only in category-seeded-cost-aware mode.",
    )
    parser.add_argument("--dry-run-proposer", action="store_true", help="Use a deterministic dummy proposal; no evolver LLM call.")
    parser.add_argument(
        "--predefined-strategies",
        nargs="*",
        default=[
            "observation_masking",
            "plan_template_cache",
            "adaptive_tool_trimming",
            "stagnation_rescue_stop",
            "skill_compilation",
        ],
        help="Strategies used by predefined-control and predefined-cost-aware modes.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    requested_output_dir = Path(args.output_dir)
    output_dir = _resolve_output_dir(args)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"Output directory exists and is not empty: {output_dir}. "
            "Use --fixed-output-dir --overwrite for an intentional replacement, "
            "or choose a different --output-dir prefix."
        )
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    workspace_dir = output_dir / "workspace"
    seed_workspace = Path(args.seed_workspace)
    hide_efficiency = args.mode in {
        "normal-evolve",
        "cost-aware",
        "category-seeded-cost-aware",
    }
    _copy_seed_workspace(seed_workspace, workspace_dir, hide_efficiency=hide_efficiency)

    runtime_config = None
    if args.mode in {"predefined-control", "predefined-cost-aware"}:
        runtime_config = _configure_predefined_control(
            workspace_dir,
            args.predefined_strategies,
            experiment=f"emergent_efficiency_{args.mode.replace('-', '_')}",
        )

    leakage = _strategy_leakage_scan(workspace_dir) if hide_efficiency else {"passed": True, "hits": []}
    emergence_dir = output_dir / "emergence"
    emergence_dir.mkdir(parents=True, exist_ok=True)
    (emergence_dir / "leakage_audit.json").write_text(
        json.dumps(leakage, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if hide_efficiency and not leakage.get("passed"):
        raise SystemExit(
            f"Leakage audit failed; predefined names found in workspace. See {emergence_dir / 'leakage_audit.json'}"
        )

    from agent_evolve.benchmarks.swe_verified_mini.benchmark import SweVerifiedMiniBenchmark

    benchmark = SweVerifiedMiniBenchmark(dataset_name=args.dataset, shuffle=False)
    tasks = benchmark.get_tasks(split="test", limit=args.limit)
    task_by_id = {task.id: task for task in tasks}
    batches = [tasks[i : i + args.batch_size] for i in range(0, len(tasks), args.batch_size)]

    print(f"Mode: {args.mode}")
    if args.mode == "category-seeded-cost-aware":
        print(f"Category seed: {args.category_seed}")
    print(f"Tasks: {len(tasks)} in {len(batches)} batch(es)")
    print(f"Requested output: {requested_output_dir.resolve()}")
    print(f"Output: {output_dir.resolve()}")
    print(f"Workspace: {workspace_dir.resolve()}")
    print()

    all_results: list[dict[str, Any]] = []
    for batch_idx, batch_tasks in enumerate(batches, start=1):
        print(f"=== Batch {batch_idx}/{len(batches)} ===")
        batch_results = _solve_batch(
            batch_tasks=batch_tasks,
            batch_num=batch_idx,
            workspace_dir=workspace_dir,
            output_dir=output_dir,
            args=args,
            runtime_config=runtime_config,
        )
        all_results.extend(batch_results)

        patch_dir = output_dir / "patches"
        patch_dir.mkdir(parents=True, exist_ok=True)
        for result in batch_results:
            if result.get("patch"):
                patch_path = patch_dir / f"{str(result['instance_id']).replace('/', '_')}.diff"
                patch_path.write_text(str(result["patch"]), encoding="utf-8")

        observations = _results_to_observations(batch_results, task_by_id)
        _write_workspace_evolution_records(
            workspace_dir=workspace_dir,
            batch_results=batch_results,
            observations=observations,
            batch_num=batch_idx,
        )
        obs_dir = output_dir / "emergence" / "observations"
        obs_dir.mkdir(parents=True, exist_ok=True)
        obs_path = obs_dir / f"batch_{batch_idx:02d}.json"
        obs_path.write_text(
            json.dumps(
                [
                    {
                        "task_id": obs.task.id,
                        "success": obs.feedback.success,
                        "score": obs.feedback.score,
                        "patch_len": len(obs.trajectory.output or ""),
                        "steps": len(obs.trajectory.steps or []),
                    }
                    for obs in observations
                ],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        if batch_idx < len(batches):
            normal_record = _run_normal_evolution(
                workspace_dir=workspace_dir,
                output_dir=output_dir,
                observations=observations,
                batch_num=batch_idx,
                args=args,
            )
            print(
                "  Normal evolution: "
                f"solver_proposals={normal_record['solver_proposals']} "
                f"result={normal_record['result']}"
            )

        if args.mode in {
            "cost-aware",
            "predefined-cost-aware",
            "category-seeded-cost-aware",
        } and batch_idx < len(batches):
            report = propose_emergent_efficiency(
                batch_results=batch_results,
                batch_num=batch_idx,
                max_steps=args.max_steps,
                args=args,
            )
            _write_proposals(output_dir, report)
            applied = _apply_proposals(
                workspace_dir,
                report,
                output_dir=output_dir,
                max_apply=args.max_proposals_per_batch,
            )
            print(
                f"  Emergent proposals: {len(report.get('proposals', []))}, "
                f"applied: {sum(1 for a in applied if a.get('status') == 'applied')}"
            )

        if batch_idx < len(batches):
            _record_workspace_snapshot(workspace_dir, output_dir, batch_idx)

        _write_run_outputs(output_dir, all_results, mode=args.mode)
        batch_summary = summarize_results(batch_results)
        print(
            f"  Batch summary: success={batch_summary['success']}/{batch_summary['tasks']} "
            f"avg_cost={batch_summary['avg_cost']} cost_of_pass={batch_summary['cost_of_pass']}"
        )
        print()

    final_summary = summarize_results(all_results)
    print("=" * 60)
    print(
        f"FINAL {args.mode}: success={final_summary['success']}/{final_summary['tasks']} "
        f"success_rate={final_summary['success_rate']:.3f} "
        f"avg_cost={final_summary['avg_cost']} cost_of_pass={final_summary['cost_of_pass']} "
        f"zero_token_tasks={final_summary['zero_token_tasks']}"
    )
    print(f"Results: {output_dir / 'results.json'}")
    print(f"Summary: {output_dir / 'summary.csv'}")
    print(f"Figures: {output_dir / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
