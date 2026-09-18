#!/usr/bin/env python3
"""Run emergent efficiency experiments on Terminal-Bench 2.0.

This mirrors the SWE emergent-efficiency experiment matrix on Terminal-Bench:

* normal-evolve
* cost-aware
* predefined-control
* predefined-cost-aware
* category-seeded-cost-aware

The runner intentionally reuses the Terminal-Bench task loader, Docker
execution, ReAct solver, and workspace protocol already used by
examples/tb_examples/batch_evolve_terminal.py.  It adds the cost/behavior
feedback loop and proposal adoption pipeline from the SWE runner, adapted to
Terminal-Bench result schemas and tool execution contracts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import random
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("BYPASS_TOOL_CONSENT", "true")

from agent_evolve.agents.terminal.dataset import TB2Task, load_all_tasks
from agent_evolve.config import EvolveConfig
from agent_evolve.contract.workspace import AgentWorkspace
from agent_evolve.efficiency.profiler import normalize_usage, result_strategy_hits, summarize_results
from agent_evolve.efficiency.strategies import DEFAULT_STRATEGIES, load_active_config
from agent_evolve.efficiency.visualization import (
    write_plots,
    write_proposal_category_plots,
    write_strategy_library_plot,
    write_summary_table,
)
from agent_evolve.types import Feedback, Observation, Task, Trajectory

logger = logging.getLogger("tb_emergent_efficiency")

DEFAULT_OUTPUT_ROOT = "logs/tb_emergent_efficiency"
DEFAULT_MODEL_ID = "deepseek:deepseek-v4-flash"


def _load_swe_runner_module():
    path = REPO_ROOT / "examples" / "swe_examples_new" / "emergent_efficiency" / "run_emergent_efficiency.py"
    spec = importlib.util.spec_from_file_location("_swe_emergent_efficiency_runner", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load SWE emergent runner from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_swe = _load_swe_runner_module()


EMERGENCE_SYSTEM_PROMPT_TB = """\
You are an experiment assistant for a Terminal-Bench solving agent.

Goal: discover cost-saving behavior changes from evidence, without using any
named predefined strategy library.

You may propose ordinary reusable skills, short prompt fragments, compact
memory notes, or deterministic Terminal-Bench workspace tools. Each proposal
must preserve correctness first. Avoid shortcuts that merely stop early, skip
investigation, or reduce validation.

For kind=tool, content must be complete Python source for tools/<name>.py and
must define execute(container_name, tool_input, log). The function may call
subprocess.run(["docker", "exec", container_name, ...]) and must return a
short string. Include optional "parameters" and "required" fields using JSON
schema-style parameter metadata.

Output JSON only:
{
  "proposals": [
    {
      "name": "short_generic_name",
      "kind": "skill" | "prompt_fragment" | "tool" | "memory",
      "description": "one sentence with trigger and do-not-trigger conditions",
      "trigger": "when to use it",
      "do_not_trigger": "when not to use it",
      "content": "procedure under 350 words, or complete Terminal-Bench tool Python source for kind=tool",
      "parameters": {"optional_arg": {"type": "string", "description": "optional for kind=tool"}},
      "required": ["optional_arg"],
      "expected_savings": "why this should reduce cost",
      "correctness_risk": "what could go wrong",
      "confidence": "HIGH" | "MEDIUM" | "LOW",
      "evidence": ["task_id: observed waste -> proposed fix"]
    }
  ]
}

Return {"proposals": []} when there is no cross-task efficiency pattern.
"""


PREDEFINED_COST_AWARE_SYSTEM_PROMPT_TB = EMERGENCE_SYSTEM_PROMPT_TB.replace(
    "without using any\nnamed predefined strategy library.",
    "while predefined runtime efficiency strategies are already active.",
).replace(
    "why this should reduce cost",
    "why this should reduce cost beyond active strategies",
).replace(
    "observed waste -> proposed fix",
    "observed waste -> proposed incremental fix",
)


CATEGORY_SEEDED_SYSTEM_PROMPT_TB = EMERGENCE_SYSTEM_PROMPT_TB.replace(
    "without using any\nnamed predefined strategy library.",
    "with only broad M1-M7 efficiency categories, not predefined strategy names or implementations.",
).replace(
    '"description": "one sentence with trigger and do-not-trigger conditions",',
    '"categories": ["M1 Context"],\n      "description": "one sentence with trigger and do-not-trigger conditions",',
).replace(
    '"expected_savings": "why this should reduce cost",',
    '"expected_savings": "which category metric should improve and why",',
)

_swe.EMERGENCE_SYSTEM_PROMPT = EMERGENCE_SYSTEM_PROMPT_TB
_swe.PREDEFINED_COST_AWARE_SYSTEM_PROMPT = PREDEFINED_COST_AWARE_SYSTEM_PROMPT_TB
_swe.CATEGORY_SEEDED_SYSTEM_PROMPT = CATEGORY_SEEDED_SYSTEM_PROMPT_TB


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _json_default(value: Any) -> str:
    return str(value)


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


def _slug(value: str, *, default: str = "proposal", limit: int = 80) -> str:
    return _swe._slug(value, default=default, limit=limit)


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    raw_output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT
    output_dir = Path(raw_output_dir)
    if args.fixed_output_dir:
        return output_dir
    stamp = _timestamp()
    mode_slug = args.mode.replace("-", "_")
    if raw_output_dir == DEFAULT_OUTPUT_ROOT:
        return output_dir / f"{mode_slug}_{stamp}"
    return output_dir.with_name(f"{output_dir.name}_{stamp}")


def _task_to_core_task(task: TB2Task) -> Task:
    metadata = dict(task.metadata)
    metadata.update(
        {
            "task_name": task.name,
            "docker_image": task.docker_image,
            "test_sh_path": task.test_sh_path,
            "test_py_path": task.test_py_path,
            "files": task.files,
            "benchmark": "terminal_bench_2",
        }
    )
    return Task(id=task.name, input=task.prompt, metadata=metadata)


def _filter_tasks(tasks: list[TB2Task], args: argparse.Namespace) -> list[TB2Task]:
    if args.tasks:
        names = {name.strip() for name in args.tasks.split(",") if name.strip()}
        tasks = [task for task in tasks if task.name in names]
    if args.exclude:
        excluded = {name.strip() for name in args.exclude.split(",") if name.strip()}
        tasks = [task for task in tasks if task.name not in excluded]
    if args.category:
        tasks = [task for task in tasks if task.metadata.get("category") == args.category]
    if args.difficulty:
        tasks = [task for task in tasks if task.metadata.get("difficulty") == args.difficulty]
    if args.shuffle:
        rng = random.Random(args.shuffle_seed)
        rng.shuffle(tasks)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    return tasks


def _tool_action(tool_name: str, arguments: Any) -> str:
    if tool_name == "submit":
        return "submit"
    if tool_name == "read_skill":
        return "read_skill"
    if tool_name == "python":
        code = str((arguments or {}).get("code", "") if isinstance(arguments, dict) else arguments).lower()
        if any(marker in code for marker in ("open(", "write_text", ".write(", "pathlib", "shutil.copy")):
            return "write_file"
        if any(marker in code for marker in ("pytest", "unittest", "subprocess.run", "assert ")):
            return "run_test"
        return "python"
    if tool_name == "bash":
        cmd = str((arguments or {}).get("cmd", "") if isinstance(arguments, dict) else arguments).lower()
        edit_markers = (
            "cat >",
            "cat <<",
            "tee ",
            "sed -i",
            "python - <<",
            "python3 - <<",
            "perl -pi",
            "apply_patch",
            "> ",
            ">> ",
        )
        test_markers = (
            "pytest",
            "tox ",
            " unittest",
            "npm test",
            "go test",
            "cargo test",
            "mvn test",
            "/tests/test.sh",
            "bash /tests",
        )
        if any(marker in cmd for marker in test_markers):
            return "run_test"
        if any(marker in cmd for marker in edit_markers):
            return "write_file"
        return "bash"
    return tool_name


def _arguments_summary(arguments: Any) -> str:
    if isinstance(arguments, dict):
        for key in ("cmd", "code", "answer", "name"):
            if key in arguments:
                return str(arguments.get(key, ""))[:600]
        return json.dumps(arguments, ensure_ascii=False, sort_keys=True)[:600]
    return str(arguments or "")[:600]


def _conversation_to_steps(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Terminal-Bench conversations into profiler-compatible tool steps."""
    calls: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for entry in conversation:
        if not isinstance(entry, dict):
            continue
        if entry.get("role") == "assistant":
            for call in entry.get("tool_calls", []) or []:
                if not isinstance(call, dict):
                    continue
                tool_name = str(call.get("function") or call.get("name") or "")
                arguments = call.get("arguments", {})
                call_id = str(call.get("id") or "")
                step = {
                    "tool": tool_name,
                    "action": _tool_action(tool_name, arguments),
                    "input": arguments,
                    "input_summary": _arguments_summary(arguments),
                    "assistant_context": str(entry.get("content") or "")[:1000],
                }
                calls.append(step)
                if call_id:
                    by_id[call_id] = step
        elif entry.get("role") == "tool":
            call_id = str(entry.get("tool_call_id") or "")
            step = by_id.get(call_id)
            if step is None:
                tool_name = str(entry.get("function") or "")
                step = {
                    "tool": tool_name,
                    "action": _tool_action(tool_name, {}),
                    "input": {},
                    "input_summary": "",
                }
                calls.append(step)
            content = str(entry.get("content") or "")
            step["result_snippet"] = content[:1600]
            step["output_chars"] = len(content)
    return calls


def _usage_quality(usage: dict[str, Any]) -> str:
    if any(_safe_int(usage.get(key)) > 0 for key in ("input_tokens", "output_tokens", "total_tokens")):
        return "ok"
    return "missing"


def _write_artifact(path: Path, payload: Any, *, text: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if text:
        path.write_text(str(payload), encoding="utf-8")
    else:
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
    return str(path)


def _build_system_prompt_for_terminal(agent: Any, workspace_dir: Path) -> str:
    workspace = AgentWorkspace(workspace_dir)
    parts: list[str] = [agent.system_prompt.strip()]

    efficiency_prompt = workspace_dir / "prompts" / "efficiency.md"
    if efficiency_prompt.exists():
        text = efficiency_prompt.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            parts.append(text)

    fragments = []
    for name in workspace.list_fragments():
        text = workspace.read_fragment(name).strip()
        if text:
            fragments.append(f"### {name}\n{text}")
    if fragments:
        parts.append("## Prompt Fragments\n" + "\n\n".join(fragments))

    strategy_memories = workspace.read_memories(category="strategy", limit=20)
    if strategy_memories:
        memory_lines = []
        for entry in strategy_memories:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or entry.get("type") or "memory")
            trigger = str(entry.get("trigger") or entry.get("description") or "")
            content = str(entry.get("content") or "")[:900]
            memory_lines.append(f"- {name}: {trigger}\n  {content}")
        if memory_lines:
            parts.append("## Efficiency Memory\n" + "\n".join(memory_lines))

    if agent.skills:
        parts.append(
            "## Available Skills\n"
            "You have specialized skills available. Call `read_skill(name)` to load the full content "
            "before tackling a relevant challenge."
        )
        parts.extend(f"- **{skill.name}**: {skill.description}" for skill in agent.skills)

    return "\n\n".join(part for part in parts if part)


def _normalize_tb_result(
    *,
    raw: dict[str, Any],
    task: TB2Task,
    workspace_dir: Path,
    output_dir: Path,
    system_prompt_text: str,
    model_id: str,
    batch_num: int,
    mode: str,
) -> dict[str, Any]:
    conversation = raw.get("conversation", []) if isinstance(raw.get("conversation"), list) else []
    tool_steps = _conversation_to_steps(conversation)
    active_config = load_active_config(workspace_dir)
    active_strategies = list(active_config.get("strategies", []) or [])
    usage = normalize_usage(
        raw.get("usage", {}),
        model_id=model_id,
        usage_record_quality=_usage_quality(raw.get("usage", {})),
        usage_source="terminal_bench_react_solver",
        usage_estimated=False,
    )
    strategy_hits = result_strategy_hits({"steps": tool_steps, "strategies": active_strategies})
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
    summary_step = {
        "usage": usage,
        "active_efficiency_strategies": active_strategies,
        "efficiency_config_id": active_config.get("metadata", {}).get("efficiency_config_id", ""),
        "route_model_calls": [],
    }
    steps = tool_steps + [summary_step]
    safe_name = task.name.replace("/", "_")
    conversation_path = _write_artifact(
        output_dir / "conversations" / f"{safe_name}.json",
        conversation,
    )
    system_prompt_path = _write_artifact(
        output_dir / "system_prompts" / f"{safe_name}.md",
        system_prompt_text,
        text=True,
    )
    passed = bool(raw.get("passed", False))
    result = {
        "instance_id": task.name,
        "task_name": task.name,
        "success": passed,
        "score": 1.0 if passed else 0.0,
        "passed": passed,
        "status": raw.get("status", "passed" if passed else "failed"),
        "feedback_detail": raw.get("eval_output", ""),
        "eval_output": raw.get("eval_output", ""),
        "elapsed": round(_safe_float(raw.get("total_time")), 1),
        "solve_time": round(_safe_float(raw.get("solve_time")), 1),
        "patch_len": 0,
        "patch": "",
        "skill_proposal": raw.get("skill_draft") or "",
        "skill_draft": raw.get("skill_draft") or "",
        "num_tool_calls": len(tool_steps),
        "num_turns": _safe_int(raw.get("conversation_turns")) or len(
            [entry for entry in conversation if isinstance(entry, dict) and entry.get("role") == "assistant"]
        ),
        "cumulative_input_tokens": usage.get("input_tokens", 0),
        "cumulative_output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "cumulative_cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
        "cumulative_cache_write_input_tokens": usage.get("cache_write_input_tokens", 0),
        "cost_units": usage.get("cost_units", 0.0),
        "main_cost_units": usage.get("main_cost_units", usage.get("cost_units", 0.0)),
        "route_model_cost_units": usage.get("route_model_cost_units", 0.0),
        "main_model_id": usage.get("model_id", model_id),
        "main_model_provider": usage.get("provider", ""),
        "main_route_level": usage.get("route_level", ""),
        "input_price_per_1m": usage.get("input_price_per_1m", 0.0),
        "output_price_per_1m": usage.get("output_price_per_1m", 0.0),
        "cache_read_price_per_1m": usage.get("cache_read_price_per_1m", 0.0),
        "cache_write_price_per_1m": usage.get("cache_write_price_per_1m", 0.0),
        "cost_currency": usage.get("cost_currency", ""),
        "cost_price_source": usage.get("cost_price_source", ""),
        "cost_priced": bool(usage.get("cost_priced", False)),
        "usage_record_quality": usage.get("usage_record_quality", _usage_quality(usage)),
        "usage_source": usage.get("usage_source", "terminal_bench_react_solver"),
        "usage_estimated": bool(usage.get("usage_estimated", False)),
        "max_input_tokens_per_turn": 0,
        "active_efficiency_strategies": active_strategies,
        "strategies": active_strategies,
        "skill_usage_counts": skill_usage_counts,
        "emergent_skill_usage_counts": emergent_skill_usage_counts,
        "efficiency_config_id": summary_step["efficiency_config_id"],
        "route_model_calls": [],
        "conversation_path": conversation_path,
        "system_prompt_path": system_prompt_path,
        "steps": steps,
        "metadata": {
            **task.metadata,
            "task_name": task.name,
            "docker_image": task.docker_image,
            "benchmark": "terminal_bench_2",
        },
        "batch": batch_num,
        "mode": mode,
    }
    result = _swe._attach_usage_fields(result)
    result["behavior_stats"] = _swe._behavior_hit_stats(result)
    return result


def _solve_terminal_batch(
    *,
    batch_tasks: list[TB2Task],
    batch_num: int,
    workspace_dir: Path,
    output_dir: Path,
    system_prompt_text: str,
    agent: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    from examples.tb_examples.batch_evolve_terminal import _solve_one_task

    raw_output = output_dir / "raw_tb_results.jsonl"
    raw_errors = output_dir / "raw_tb_errors.jsonl"
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(
                _solve_one_task,
                task,
                args.model_id,
                args.region,
                args.max_tokens,
                str(output_dir / "task_logs"),
                str(raw_output),
                str(raw_errors),
                args.solver,
                system_prompt_text,
                agent,
                args.propose_skill,
            ): task
            for task in batch_tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                raw = future.result()
            except Exception as exc:  # noqa: BLE001
                raw = {
                    "task_name": task.name,
                    "passed": False,
                    "eval_output": f"ERROR: {exc}",
                    "usage": {},
                    "solve_time": 0,
                    "total_time": 0,
                    "conversation_turns": 0,
                    "conversation": [],
                    "metadata": task.metadata,
                    "status": "error",
                }
            result = _normalize_tb_result(
                raw=raw,
                task=task,
                workspace_dir=workspace_dir,
                output_dir=output_dir,
                system_prompt_text=system_prompt_text,
                model_id=args.model_id,
                batch_num=batch_num,
                mode=args.mode,
            )
            results.append(result)
            status = "PASS" if result.get("success") else "FAIL"
            print(
                f"  {status} {task.name} "
                f"cost={_safe_float(result.get('cost_units')):,.0f} "
                f"tokens={_safe_int(result.get('total_tokens')):,d} "
                f"calls={_safe_int(result.get('num_tool_calls'))}"
            )
    return results


def _results_to_observations(results: list[dict[str, Any]], tasks_by_id: dict[str, Task]) -> list[Observation]:
    observations: list[Observation] = []
    for result in results:
        task = tasks_by_id.get(str(result.get("instance_id", "")))
        if task is None:
            continue
        feedback = Feedback(
            success=bool(result.get("success")),
            score=float(result.get("score", 0.0) or 0.0),
            detail=str(result.get("feedback_detail") or ""),
            raw={"benchmark": "terminal_bench_2"},
        )
        trajectory = Trajectory(
            task_id=task.id,
            output=str(result.get("feedback_detail") or ""),
            steps=result.get("steps", []) if isinstance(result.get("steps"), list) else [],
            conversation=[],
        )
        conv_path = Path(str(result.get("conversation_path") or ""))
        if conv_path.exists():
            try:
                trajectory.conversation = json.loads(conv_path.read_text(encoding="utf-8"))
            except Exception:
                trajectory.conversation = []
        trajectory._skill_proposal = str(result.get("skill_proposal") or "")
        observations.append(Observation(task=task, trajectory=trajectory, feedback=feedback))
    return observations


def _write_workspace_evolution_records(
    *,
    workspace_dir: Path,
    output_dir: Path,
    batch_results: list[dict[str, Any]],
    observations: list[Observation],
    batch_num: int,
) -> None:
    evolution_dir = workspace_dir / "evolution"
    trajectories_dir = evolution_dir / "trajectories"
    observations_dir = evolution_dir / "observations"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    observations_dir.mkdir(parents=True, exist_ok=True)

    batch_slug = f"batch_{batch_num:02d}"
    trajectory_index: list[dict[str, Any]] = []
    for result in batch_results:
        task_id = str(result.get("instance_id", ""))
        safe_task_id = task_id.replace("/", "__")
        path = trajectories_dir / f"{batch_slug}__{safe_task_id}.json"
        payload = dict(result)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        trajectory_index.append(
            {
                "task_id": task_id,
                "path": str(path.relative_to(evolution_dir)),
                "success": bool(result.get("success")),
                "score": result.get("score", 0.0),
                "steps": len(result.get("steps", []) if isinstance(result.get("steps"), list) else []),
                "conversation_path": result.get("conversation_path", ""),
                "system_prompt_path": result.get("system_prompt_path", ""),
            }
        )

    observation_payload = []
    for obs in observations:
        observation_payload.append(
            {
                "task": {"id": obs.task.id, "input": obs.task.input, "metadata": obs.task.metadata},
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
                    "conversation": obs.trajectory.conversation,
                    "skill_proposal": getattr(obs.trajectory, "_skill_proposal", ""),
                },
            }
        )
    (observations_dir / f"{batch_slug}.json").write_text(
        json.dumps(observation_payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    with (evolution_dir / "trajectory_index.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"batch": batch_num, "trajectories": trajectory_index}, ensure_ascii=False) + "\n")


def _tool_schema(value: Any) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict):
        return {}
    schema: dict[str, dict[str, str]] = {}
    for name, raw in value.items():
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", str(name)):
            continue
        if isinstance(raw, dict):
            typ = str(raw.get("type") or "string")
            desc = str(raw.get("description") or "")
        else:
            typ = "string"
            desc = str(raw or "")
        if typ not in {"string", "integer", "number", "boolean", "array", "object"}:
            typ = "string"
        schema[str(name)] = {"type": typ, "description": desc[:240]}
    return schema


def _tb_tool_validation_error(content: str) -> str:
    text = str(content or "").strip()
    if len(text) < 120:
        return "tool content too short"
    if "def execute(" not in text:
        return "tool content must define execute(container_name, tool_input, log)"
    if "container_name" not in text or "tool_input" not in text:
        return "execute signature must accept container_name and tool_input"
    return ""


def _apply_tb_proposals(
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
        if not isinstance(proposal, dict):
            continue
        confidence = str(proposal.get("confidence") or "MEDIUM").upper()
        categories = _swe._proposal_categories(proposal)
        category_status = "missing_categories" if category_seeded and not categories else "ok"
        if confidence == "LOW":
            applied.append(
                {
                    "index": index,
                    "status": "skipped_low_confidence",
                    "categories": categories,
                    "category_status": category_status,
                    "proposal": proposal,
                }
            )
            continue
        name = _slug(str(proposal.get("name") or f"proposal_{index}"), default=f"proposal_{index}")
        kind = _swe._normalize_kind(str(proposal.get("kind") or "skill"))
        content = _swe._strip_code_fence(str(proposal.get("content") or ""))
        if not content:
            applied.append(
                {
                    "index": index,
                    "status": "skipped_empty_content",
                    "categories": categories,
                    "category_status": category_status,
                    "proposal": proposal,
                }
            )
            continue

        description = str(proposal.get("description") or proposal.get("trigger") or content[:120]).strip()
        proposal_id = f"emergent_gen{int(proposal_report.get('batch', 0)):02d}_{index:02d}_{name}"
        leaked_names = _swe._proposal_contains_predefined_name(proposal)

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
            workspace.write_skill(
                proposal_id,
                (
                    "---\n"
                    f"name: {proposal_id}\n"
                    f"description: {description[:180]}\n"
                    "---\n\n"
                    "# Terminal-Bench Emergent Efficiency Skill\n\n"
                    f"Categories: {', '.join(categories) if categories else 'none'}\n\n"
                    f"Trigger: {proposal.get('trigger', '')}\n\n"
                    f"Do not trigger: {proposal.get('do_not_trigger', '')}\n\n"
                    f"{content.strip()}\n\n"
                    f"Expected savings: {proposal.get('expected_savings', '')}\n\n"
                    f"Correctness risk: {proposal.get('correctness_risk', '')}\n"
                ),
            )
            target = f"skills/{proposal_id}/SKILL.md"
        elif kind == "tool":
            validation_error = _tb_tool_validation_error(content)
            if validation_error:
                applied.append(
                    {
                        "index": index,
                        "status": f"skipped_invalid_tool:{validation_error}",
                        "kind": kind,
                        "categories": categories,
                        "category_status": category_status,
                        "proposal": proposal,
                    }
                )
                continue
            tool_name = _slug(str(proposal.get("tool_name") or name), default=f"tool_{index}")
            registry = workspace.read_tool_registry()
            if any(entry.get("name") == tool_name for entry in registry):
                applied.append(
                    {
                        "index": index,
                        "status": "skipped_duplicate_tool",
                        "proposal_id": proposal_id,
                        "kind": kind,
                        "categories": categories,
                        "category_status": category_status,
                        "target": f"tools/{tool_name}.py",
                        "proposal": proposal,
                    }
                )
                continue
            workspace.write_tool(tool_name, content)
            registry.append(
                {
                    "name": tool_name,
                    "file": f"{tool_name}.py",
                    "description": description[:180],
                    "parameters": _tool_schema(proposal.get("parameters")),
                    "required": [
                        str(item)
                        for item in proposal.get("required", [])
                        if isinstance(item, str) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", item)
                    ],
                }
            )
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
            applied.append(
                {
                    "index": index,
                    "status": f"skipped_unknown_kind:{kind}",
                    "kind": kind,
                    "categories": categories,
                    "category_status": category_status,
                    "proposal": proposal,
                }
            )
            continue

        applied.append(
            {
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
        )

    adoption_path = output_dir / "emergence" / "proposal_adoption.json"
    adoption_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if adoption_path.exists():
        try:
            loaded = json.loads(adoption_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                existing = loaded
        except Exception:
            existing = []
    existing.extend(applied)
    adoption_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    return applied


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
    return _swe._summarize_proposal_categories(adoptions)


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


def _strategies_for(results: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for result in results:
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


def _make_summary_rows(results: list[dict[str, Any]], *, mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch in sorted({_safe_int(result.get("batch")) for result in results}):
        batch_results = [result for result in results if _safe_int(result.get("batch")) == batch]
        row = {"group": f"{mode}_batch_{batch:02d}", "strategies": _strategies_for(batch_results)}
        row.update(summarize_results(batch_results))
        row.update(_swe._summarize_behavior_stats(batch_results))
        rows.append(row)
    overall = {"group": f"{mode}_overall", "strategies": _strategies_for(results)}
    overall.update(summarize_results(results))
    overall.update(_swe._summarize_behavior_stats(results))
    rows.append(overall)
    return rows


def _write_run_outputs(output_dir: Path, results: list[dict[str, Any]], *, mode: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    rows = _make_summary_rows(results, mode=mode)
    payload = {"mode": mode, "benchmark": "terminal_bench_2", "overall": rows[-1], "by_batch": rows[:-1]}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    behavior_payload = {
        "overall": _swe._summarize_behavior_stats(results),
        "by_task": [result.get("behavior_stats") or _swe._behavior_hit_stats(result) for result in results],
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


def _run_normal_evolution(
    *,
    workspace_dir: Path,
    output_dir: Path,
    observations: list[Observation],
    batch_num: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    record = _swe._run_normal_evolution(
        workspace_dir=workspace_dir,
        output_dir=output_dir,
        observations=observations,
        batch_num=batch_num,
        args=args,
    )
    return record


def _copy_seed_workspace(seed_workspace: Path, workspace_dir: Path, *, hide_efficiency: bool) -> None:
    _swe._copy_seed_workspace(seed_workspace, workspace_dir, hide_efficiency=hide_efficiency)


def _configure_predefined_control(workspace_dir: Path, strategies: list[str], *, experiment: str) -> dict[str, Any]:
    return _swe._configure_predefined_control(workspace_dir, strategies, experiment=experiment)


def _write_experiment_manifest(output_dir: Path, args: argparse.Namespace, tasks: list[TB2Task]) -> None:
    payload = {
        "benchmark": "terminal_bench_2",
        "mode": args.mode,
        "tasks": [task.name for task in tasks],
        "task_count": len(tasks),
        "batch_size": args.batch_size,
        "parallel": args.parallel,
        "solver": args.solver,
        "model_id": args.model_id,
        "evolver_model": args.evolver_model or args.model_id,
        "seed_workspace": args.seed_workspace,
        "predefined_strategies": args.predefined_strategies,
        "category_seed": args.category_seed,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emergent efficiency Terminal-Bench 2.0 experiment runner")
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
    parser.add_argument("--challenges-dir", default=None, help="Path to Terminal-Bench challenges directory")
    parser.add_argument("--tasks", default=None, help="Comma-separated task names to run")
    parser.add_argument("--exclude", default=None, help="Comma-separated task names to skip")
    parser.add_argument("--category", default=None, help="Filter by Terminal-Bench category")
    parser.add_argument("--difficulty", default=None, help="Filter by difficulty")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--solver", choices=["react", "strands"], default="react")
    parser.add_argument("--seed-workspace", default="seed_workspaces/terminal")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fixed-output-dir", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Only with --fixed-output-dir; removes existing output dir")
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Solver model id. Default: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument("--evolver-model", default="")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--evolver-max-tokens", type=int, default=8192)
    parser.add_argument("--evolver-temperature", type=float, default=0.2)
    parser.add_argument("--evolver-call-retries", type=int, default=4)
    parser.add_argument("--evolver-retry-initial-delay", type=float, default=2.0)
    parser.add_argument("--evolver-retry-max-delay", type=float, default=30.0)
    parser.add_argument("--max-steps", type=int, default=500, help="Tool-call limit used for waste-signal thresholds")
    parser.add_argument("--max-proposals-per-batch", type=int, default=2)
    parser.add_argument(
        "--category-seed",
        choices=["none", "basic-efficiency"],
        default="basic-efficiency",
        help="Taxonomy exposed only in category-seeded-cost-aware mode",
    )
    parser.add_argument("--dry-run-proposer", action="store_true", help="Use deterministic dummy proposal")
    parser.add_argument("--propose-skill", action="store_true", default=True, help="Ask solver for skill drafts")
    parser.add_argument("--no-propose-skill", dest="propose_skill", action="store_false")
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
        help="Strategies used by predefined-control and predefined-cost-aware modes",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.parallel <= 0:
        raise SystemExit("--parallel must be positive")
    if args.max_proposals_per_batch < 0:
        raise SystemExit("--max-proposals-per-batch must be non-negative")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    for name in ("botocore", "urllib3", "httpcore", "httpx", "strands.models", "strands.tools", "strands.telemetry"):
        logging.getLogger(name).setLevel(logging.WARNING)

    output_dir = _resolve_output_dir(args)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"Output directory exists and is not empty: {output_dir}. "
            "Use --fixed-output-dir --overwrite for an intentional replacement, or choose a different prefix."
        )
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    workspace_dir = output_dir / "workspace"
    seed_workspace = Path(args.seed_workspace)
    hide_efficiency = args.mode in {"normal-evolve", "cost-aware", "category-seeded-cost-aware"}
    _copy_seed_workspace(seed_workspace, workspace_dir, hide_efficiency=hide_efficiency)

    runtime_config = None
    if args.mode in {"predefined-control", "predefined-cost-aware"}:
        runtime_config = _configure_predefined_control(
            workspace_dir,
            args.predefined_strategies,
            experiment=f"tb_emergent_efficiency_{args.mode.replace('-', '_')}",
        )
    _ = runtime_config

    leakage = _swe._strategy_leakage_scan(workspace_dir) if hide_efficiency else {"passed": True, "hits": []}
    emergence_dir = output_dir / "emergence"
    emergence_dir.mkdir(parents=True, exist_ok=True)
    (emergence_dir / "leakage_audit.json").write_text(
        json.dumps(leakage, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if hide_efficiency and not leakage.get("passed"):
        raise SystemExit(f"Leakage audit failed. See {emergence_dir / 'leakage_audit.json'}")

    tasks = _filter_tasks(load_all_tasks(args.challenges_dir), args)
    if not tasks:
        raise SystemExit("No Terminal-Bench tasks selected.")
    batches = [tasks[i : i + args.batch_size] for i in range(0, len(tasks), args.batch_size)]
    core_tasks = [_task_to_core_task(task) for task in tasks]
    tasks_by_id = {task.id: task for task in core_tasks}
    _write_experiment_manifest(output_dir, args, tasks)

    from agent_evolve.agents.terminal.agent import TerminalAgent

    agent = TerminalAgent(
        workspace_dir=workspace_dir,
        model_id=args.model_id,
        region=args.region,
        max_tokens=args.max_tokens,
    )

    print(f"Mode: {args.mode}")
    if args.mode == "category-seeded-cost-aware":
        print(f"Category seed: {args.category_seed}")
    print(f"Benchmark: Terminal-Bench 2.0")
    print(f"Tasks: {len(tasks)} in {len(batches)} batch(es)")
    print(f"Output: {output_dir.resolve()}")
    print(f"Workspace: {workspace_dir.resolve()}")
    print()

    all_results: list[dict[str, Any]] = []
    for batch_idx, batch_tasks in enumerate(batches, start=1):
        print(f"=== Batch {batch_idx}/{len(batches)} ===")
        agent.reload_from_fs()
        system_prompt_text = _build_system_prompt_for_terminal(agent, workspace_dir)
        batch_results = _solve_terminal_batch(
            batch_tasks=batch_tasks,
            batch_num=batch_idx,
            workspace_dir=workspace_dir,
            output_dir=output_dir,
            system_prompt_text=system_prompt_text,
            agent=agent,
            args=args,
        )
        all_results.extend(batch_results)

        observations = _results_to_observations(batch_results, tasks_by_id)
        _write_workspace_evolution_records(
            workspace_dir=workspace_dir,
            output_dir=output_dir,
            batch_results=batch_results,
            observations=observations,
            batch_num=batch_idx,
        )
        obs_dir = output_dir / "emergence" / "observations"
        obs_dir.mkdir(parents=True, exist_ok=True)
        (obs_dir / f"batch_{batch_idx:02d}.json").write_text(
            json.dumps(
                [
                    {
                        "task_id": obs.task.id,
                        "success": obs.feedback.success,
                        "score": obs.feedback.score,
                        "steps": len(obs.trajectory.steps or []),
                        "conversation_turns": len(obs.trajectory.conversation or []),
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
            agent.reload_from_fs()

        if args.mode in {"cost-aware", "predefined-cost-aware", "category-seeded-cost-aware"} and batch_idx < len(batches):
            report = _swe.propose_emergent_efficiency(
                batch_results=batch_results,
                batch_num=batch_idx,
                max_steps=args.max_steps,
                args=args,
            )
            _swe._write_proposals(output_dir, report)
            applied = _apply_tb_proposals(
                workspace_dir,
                report,
                output_dir=output_dir,
                max_apply=args.max_proposals_per_batch,
            )
            print(
                f"  Emergent proposals: {len(report.get('proposals', []))}, "
                f"applied: {sum(1 for item in applied if item.get('status') == 'applied')}"
            )
            agent.reload_from_fs()

        if batch_idx < len(batches):
            _swe._record_workspace_snapshot(workspace_dir, output_dir, batch_idx)

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
        f"Final: success={final_summary['success']}/{final_summary['tasks']} "
        f"success_rate={final_summary['success_rate']:.2%} "
        f"avg_cost={final_summary['avg_cost']} cost_of_pass={final_summary['cost_of_pass']}"
    )
    print(f"Summary: {output_dir / 'summary.json'}")
    print(f"Results: {output_dir / 'results.json'}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
