#!/usr/bin/env python3
"""Batched VisualWebArena experiments with adaptive and cost-aware evolution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import logging
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from common import aggregate, append, digest, dump, load_prices, read_jsonl, run_logged

MODES = (
    "normal-evolve",
    "cost-aware",
    "predefined-control",
    "predefined-cost-aware",
    "category-seeded-cost-aware",
    "static-control",
)
COST_MODES = ("cost-aware", "predefined-cost-aware", "category-seeded-cost-aware")
PREDEFINED = (
    "Use the current screenshot to ground every target before acting. "
    "Reuse the latest observation only until the page changes. "
    "After navigation or editing, inspect the resulting page before deciding the next action. "
    "Avoid repeated navigation to an unchanged page. Never omit required visual checks."
)
PROPOSER_SYSTEM = """Analyze VisualWebArena browser trajectories and return JSON only:
{"proposals":[{"name":"generic_name","kind":"skill|prompt_fragment|memory",
"description":"when this applies","trigger":"condition","do_not_trigger":"exception",
"content":"short reusable procedure","expected_savings":"evidence-based hypothesis",
"correctness_risk":"risk and verification","confidence":"HIGH|MEDIUM|LOW",
"evidence":["task_id: observed behavior"],"categories":[]}]}
Optimize browser navigation and observation processing while preserving official task correctness.
Use screenshots, image goals and page-state evidence. Do not assume element IDs remain stable after
navigation. Never use evaluation references, reward code, saved answers or task-specific solutions.
Only propose skills, prompt fragments or memory: this backend has no generated-tool execution API.
Distinguish model output tokens from tool output entering future model inputs. Missing cost is unknown,
not zero. Return {"proposals":[]} when no reusable pattern is supported."""


def positive(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=MODES, default="cost-aware")
    p.add_argument("--vwa-repo", type=Path, default=os.environ.get("VWA_REPO"))
    p.add_argument("--vwa-python", default=os.environ.get("VWA_PYTHON", sys.executable))
    p.add_argument(
        "--tasks-dir",
        type=Path,
        help="Generated official task directory; filenames may repeat across sites",
    )
    p.add_argument(
        "--task-ids", help="Comma-separated IDs from task-list.json; selection preserves this order"
    )
    p.add_argument("--list-tasks", action="store_true")
    p.add_argument("--limit", type=positive, default=10)
    p.add_argument("--batch-size", type=positive, default=2)
    p.add_argument(
        "--parallel",
        type=int,
        choices=[1],
        default=1,
        help="Shared websites require serial execution",
    )
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seed-workspace", type=Path, default=HERE / "seed_workspace")
    p.add_argument(
        "--output-dir", type=Path, default=ROOT / "logs/visualwebarena_emergent_efficiency"
    )
    p.add_argument("--run-name", default="vwa")
    p.add_argument("--model-id", default=os.environ.get("VWA_MODEL", ""))
    p.add_argument(
        "--evolver-model", default=os.environ.get("VWA_EVOLVER_MODEL", "deepseek:deepseek-v4-flash")
    )
    p.add_argument("--region", default="us-west-2")
    p.add_argument("--max-steps", type=positive, default=30)
    p.add_argument("--max-tokens", type=positive, default=2048)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--max-obs-length", type=positive, default=3840)
    p.add_argument(
        "--observation-type",
        choices=["image_som", "accessibility_tree_with_captioner"],
        default="image_som",
    )
    p.add_argument(
        "--instruction-path",
        type=Path,
        help="Official prompt JSON, relative to --vwa-repo or absolute",
    )
    p.add_argument("--eval-captioning-model-device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--viewport-width", type=positive, default=1280)
    p.add_argument("--viewport-height", type=positive, default=2048)
    p.add_argument("--sleep-after-execution", type=float, default=2.5)
    p.add_argument("--request-timeout", type=positive, default=120)
    p.add_argument("--task-timeout", type=positive, default=1800)
    p.add_argument("--trajectory-text-chars", type=positive, default=12000)
    p.add_argument(
        "--workspace-context-chars",
        type=positive,
        default=30000,
        help="Fail instead of silently truncating skill bodies",
    )
    p.add_argument(
        "--reset-command", help="argv string, no shell; called before EACH task, cwd=VWA repo"
    )
    p.add_argument(
        "--allow-unreset-sites",
        action="store_true",
        help="Explicitly accept shared website state across tasks",
    )
    p.add_argument(
        "--price-file", type=Path, help="Explicit USD prices; absent costs remain unknown"
    )
    p.add_argument("--evolver-max-tokens", type=positive, default=8192)
    p.add_argument("--evolver-temperature", type=float, default=0.2)
    p.add_argument("--max-skills", type=positive, default=10)
    p.add_argument("--max-proposals-per-batch", type=positive, default=2)
    p.add_argument(
        "--include-feedback",
        action="store_true",
        help="Otherwise ordinary evolution uses trajectory-only",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Synthetic tasks, no websites, browser or LLM calls"
    )
    p.add_argument(
        "--dry-run-proposer",
        action="store_true",
        help="Skip BOTH ordinary evolution and paid efficiency proposer",
    )
    args = p.parse_args(argv)
    if args.run_name in (".", "..") or not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name):
        p.error("--run-name must be a simple filename component")
    if args.sleep_after_execution < 0:
        p.error("--sleep-after-execution cannot be negative")
    if not args.dry_run and not args.tasks_dir:
        p.error("--tasks-dir is required for real generated VWA task configs")
    if not args.dry_run and not args.list_tasks:
        if not args.vwa_repo or not (args.vwa_repo / "run.py").is_file():
            p.error("--vwa-repo must contain official run.py")
        if not args.model_id:
            p.error("--model-id must name a vision-capable Chat Completions model")
        if not os.environ.get("VWA_API_KEY"):
            p.error("Set VWA_API_KEY (never put credentials in command flags)")
        required = (
            "REDDIT",
            "SHOPPING",
            "WIKIPEDIA",
            "HOMEPAGE",
            "CLASSIFIEDS",
            "CLASSIFIEDS_RESET_TOKEN",
        )
        missing = [key for key in required if not os.environ.get(key)]
        if missing:
            p.error("Missing official VWA environment variables: " + ", ".join(missing))
        if not args.reset_command and not args.allow_unreset_sites:
            p.error(
                "Provide --reset-command, or explicitly accept state contamination with --allow-unreset-sites"
            )
    return args


def load_tasks(args) -> list[dict]:
    if args.dry_run and args.tasks_dir is None:
        tasks = json.loads((HERE / "smoke_tasks.json").read_text())
    else:
        root = args.tasks_dir.resolve()
        if not root.is_dir():
            raise ValueError(f"Task directory does not exist: {root}")
        tasks = []
        for path in sorted(root.rglob("*.json"), key=lambda p: p.relative_to(root).as_posix()):
            obj = json.loads(path.read_text())
            if not isinstance(obj, dict) or "intent" not in obj or "task_id" not in obj:
                continue  # Ignore raw templates and metadata files.
            if not isinstance(obj.get("eval"), dict):
                raise TypeError(f"Not a generated official task config (missing eval): {path}")
            identity = path.relative_to(root).with_suffix("").as_posix()
            if not re.fullmatch(r"[A-Za-z0-9_./-]+", identity):
                raise ValueError(f"Unsupported task filename: {path}")
            tasks.append(
                {
                    "id": identity,
                    "intent": obj["intent"],
                    "config_file": str(path),
                    "checksum": digest(path),
                    "sites": obj.get("sites", []),
                }
            )
    by_id = {t["id"]: t for t in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("Duplicate task identity")
    if args.task_ids:
        selected = args.task_ids.split(",")
        if len(selected) != len(set(selected)):
            raise ValueError("Duplicate task IDs are not allowed")
        if set(selected) - set(by_id):
            raise ValueError("Unknown task IDs: " + ", ".join(sorted(set(selected) - set(by_id))))
        tasks = [by_id[key] for key in selected]
    if args.shuffle:
        random.Random(args.seed).shuffle(tasks)
    if args.list_tasks:
        return tasks
    tasks = tasks[: args.limit]
    if not tasks:
        raise ValueError("No generated task configs found")
    return tasks


def workspace_context(workspace: Path, max_chars: int) -> tuple[str, list[str]]:
    files = [workspace / "prompts/system.md"]
    files += sorted((workspace / "prompts/fragments").glob("*.md"))
    files += sorted(p for p in (workspace / "skills").rglob("SKILL.md") if "_drafts" not in p.parts)
    files += sorted((workspace / "memory").glob("*.jsonl"))
    context, included = [], []
    for path in files:
        if path.is_file():
            if path.is_symlink() or not path.resolve().is_relative_to(workspace.resolve()):
                raise ValueError(f"Workspace artifact escapes root: {path}")
            body = path.read_text().strip()
            if body:
                included.append(path.relative_to(workspace).as_posix())
                context.append(f"## {included[-1]}\n{body}")
    text = "\n\n".join(context)
    if len(text) > max_chars:
        raise ValueError(
            f"Workspace context has {len(text)} chars; budget is {max_chars}. Reduce skills or raise budget."
        )
    return text, included


def enforce_skill_budget(workspace: Path, before: list[str], cap: int, out: Path, batch: int):
    files = {
        p.relative_to(workspace).as_posix(): p
        for p in (workspace / "skills").rglob("SKILL.md")
        if "_drafts" not in p.parts
    }
    kept = [p for p in before if p in files][:cap]
    kept += [p for p in sorted(files) if p not in kept][: max(0, cap - len(kept))]
    removed = []
    for name, path in files.items():
        if name not in kept:
            if not path.resolve().is_relative_to(workspace.resolve()):
                raise ValueError("Refusing to prune artifact outside workspace")
            path.unlink()  # Only remove newly over-budget SKILL.md artifacts, never whole trees.
            removed.append(name)
    append(
        out / "emergence/skill_budget.jsonl",
        {"batch": batch, "kept": kept, "removed": removed, "limit": cap},
    )


class MeteredProvider:
    """Record completion or converse-loop aggregates, without claiming cache precision."""

    def __init__(self, provider, ledger: Path, phase: str, model: str):
        self.provider, self.ledger, self.phase, self.model = provider, ledger, phase, model

    def __getattr__(self, name):
        method = getattr(self.provider, name)
        if name not in ("complete", "complete_with_tools", "converse_loop"):
            return method

        def metered(*a, **kw):
            started = time.monotonic()
            try:
                response = method(*a, **kw)
            except Exception as exc:
                append(
                    self.ledger,
                    {
                        "event": "usage",
                        "phase": self.phase,
                        "model_id": self.model,
                        "usage_record_quality": "request_failed",
                        "request_failed": True,
                        "cost_priced": False,
                        "error_type": type(exc).__name__,
                        "elapsed": time.monotonic() - started,
                    },
                )
                raise
            usage = getattr(response, "usage", {}) or {}
            # Provider adapters aggregate input/output with heterogeneous cache conventions.
            # Preserve their exact raw usage, and leave monetary cost unknown rather than
            # applying a price to a possibly cache-inclusive input field.
            inp = int(usage.get("input_tokens", usage.get("inputTokens", 0)) or 0)
            out = int(usage.get("output_tokens", usage.get("outputTokens", 0)) or 0)
            append(
                self.ledger,
                {
                    "event": "usage",
                    "phase": self.phase,
                    "model_id": self.model,
                    "input_tokens": inp,
                    "output_tokens": out,
                    "total_tokens": inp + out,
                    "raw_provider_usage": usage,
                    "usage_record_quality": "aggregate_cache_unknown",
                    "cost_units": None,
                    "cost_priced": False,
                    "elapsed": time.monotonic() - started,
                    "call_scope": "converse_loop_aggregate"
                    if name == "converse_loop"
                    else "completion",
                },
            )
            return response

        return metered


@contextmanager
def meter_adaptive_factory(provider):
    # build_evolution_prompt may create trajectory judges through this same factory.
    # Temporarily route those calls through the same phase ledger as the evolver.
    tools = importlib.import_module("agent_evolve.algorithms.adaptive_skill.tools")
    original = tools.create_default_llm
    tools.create_default_llm = lambda config: provider
    try:
        yield
    finally:
        tools.create_default_llm = original


def evolve(workspace: Path, batch_rows: list[dict], args, out: Path, batch: int):
    from agent_evolve.algorithms.adaptive_skill import AdaptiveSkillEngine
    from agent_evolve.algorithms.adaptive_skill.tools import create_default_llm
    from agent_evolve.config import EvolveConfig
    from agent_evolve.contract.workspace import AgentWorkspace

    config = EvolveConfig(
        evolver_model=args.evolver_model,
        evolver_max_tokens=args.evolver_max_tokens,
        evolver_temperature=args.evolver_temperature,
        trajectory_only=not args.include_feedback,
        evolve_prompts=True,
        evolve_skills=True,
        evolve_memory=True,
        evolve_tools=False,
        extra={
            "region": args.region,
            "max_skills": args.max_skills,
            "protect_skills": True,
            "solver_proposed": False,
        },
    )
    provider = MeteredProvider(
        create_default_llm(config), out / "usage.jsonl", "normal_evolution", args.evolver_model
    )
    logs = []
    for row in batch_rows:
        conversation = [{"role": "user", "content": row["task_input"]}]
        for step in row["steps"]:
            screenshot = step.get("screenshot")
            observation = step["observation"] + (
                f"\nScreenshot artifact: {screenshot}" if screenshot else ""
            )
            conversation.extend(
                [
                    {"role": "user", "content": observation},
                    {"role": "assistant", "content": step["action"]},
                ]
            )
        logs.append(
            {
                "task_id": row["instance_id"],
                "task_input": row["task_input"],
                "conversation": conversation,
                "steps": row["steps"],
                "success": row["success"],
                "score": row["score"],
                "feedback_detail": row["status"],
            }
        )
    dump(out / "emergence/observations" / f"batch_{batch:02d}.json", logs)
    started = time.monotonic()
    with meter_adaptive_factory(provider):
        result = AdaptiveSkillEngine(config, llm=provider).evolve(
            workspace=AgentWorkspace(workspace), observation_logs=logs, evo_number=batch
        )
    append(
        out / "emergence/normal_evolution.jsonl",
        {
            "batch": batch,
            "result": result,
            "engine": "adaptive_skill",
            "elapsed": time.monotonic() - started,
            "trajectory_only": not args.include_feedback,
            "visual_evidence": "text_and_screenshot_paths_not_image_bytes",
        },
    )


def propose(workspace: Path, rows: list[dict], args, out: Path, batch: int):
    from agent_evolve.algorithms.adaptive_skill.tools import create_default_llm
    from agent_evolve.config import EvolveConfig

    shared = importlib.import_module(
        "examples.swe_examples_new.emergent_efficiency.run_emergent_efficiency"
    )
    taxonomy = (
        shared._category_taxonomy_payload() if args.mode == "category-seeded-cost-aware" else []
    )
    feedback = [
        {
            k: r.get(k)
            for k in (
                "instance_id",
                "success",
                "score",
                "cost_units",
                "total_tokens",
                "cumulative_input_tokens",
                "cumulative_cache_read_input_tokens",
                "cumulative_output_tokens",
                "num_tool_calls",
                "steps",
                "usage_record_quality",
            )
        }
        for r in rows
    ]
    system = PROPOSER_SYSTEM + (
        "\nUse broad categories only: " + json.dumps(taxonomy) if taxonomy else ""
    )
    if args.mode == "predefined-cost-aware":
        system += "\nThe fixed browser efficiency prompt is already active: " + PREDEFINED
    from agent_evolve.llm.base import LLMMessage

    config = EvolveConfig(evolver_model=args.evolver_model, extra={"region": args.region})
    provider = MeteredProvider(
        create_default_llm(config), out / "usage.jsonl", "efficiency_proposer", args.evolver_model
    )
    response = provider.complete(
        [
            LLMMessage(role="system", content=system),
            LLMMessage(role="user", content=json.dumps(feedback, ensure_ascii=False)),
        ],
        max_tokens=args.evolver_max_tokens,
        temperature=args.evolver_temperature,
    )
    parsed = shared._extract_json_object(response.content)
    proposals = parsed.get("proposals", []) if isinstance(parsed, dict) else []
    if not isinstance(proposals, list):
        proposals = []
    accepted, rejected = [], []
    for prop in proposals:
        if not isinstance(prop, dict) or prop.get("kind") not in (
            "skill",
            "prompt_fragment",
            "memory",
        ):
            rejected.append({"status": "rejected_unsupported_kind", "proposal": prop})
        elif not isinstance(prop.get("content"), str) or not prop["content"].strip():
            rejected.append({"status": "rejected_empty_content", "proposal": prop})
        else:
            accepted.append(prop)
    report = {
        "batch": batch,
        "mode": args.mode,
        "category_seeded": bool(taxonomy),
        "category_taxonomy": taxonomy,
        "raw_response": response.content,
        "parse_status": "parsed" if parsed else "parse_failed",
        "proposals": accepted,
    }
    shared._write_proposals(out, report)
    shared._apply_proposals(
        workspace, report, output_dir=out, max_apply=args.max_proposals_per_batch
    )
    if rejected:
        dump(out / "emergence" / f"batch_{batch:02d}_rejected.json", rejected)


def solve(task: dict, batch: int, workspace: Path, args, out: Path, prices: dict | None):
    slug = (
        re.sub(r"[^A-Za-z0-9_.-]", "_", task["id"])
        + "_"
        + hashlib.sha256(task["id"].encode()).hexdigest()[:8]
    )
    td = out / "tasks" / slug
    td.mkdir(parents=True)
    context, included = workspace_context(workspace, args.workspace_context_chars)
    (td / "workspace_context.md").write_text(context)
    dump(
        td / "context_manifest.json",
        {
            "included": included,
            "chars": len(context),
            "sha256": hashlib.sha256(context.encode()).hexdigest(),
        },
    )
    if args.dry_run:
        raw = {
            "status": "dry_run",
            "success": False,
            "score": None,
            "synthetic": True,
            "elapsed": 0,
            "num_tool_calls": 0,
            "num_turns": 0,
            "steps": [],
            "usage": {},
        }
        dump(td / "worker_result.json", raw)
        code, timed_out, elapsed = 0, False, 0
    else:
        repo = args.vwa_repo.resolve()
        default = (
            "p_som_cot_id_actree_3s.json"
            if args.observation_type == "image_som"
            else "p_cot_id_actree_3s.json"
        )
        prompt_path = args.instruction_path or Path("agent/prompts/jsons") / default
        if not prompt_path.is_absolute():
            prompt_path = repo / prompt_path
        prompt = json.loads(prompt_path.read_text())
        if not isinstance(prompt.get("intro"), str):
            raise ValueError("Official prompt must contain string intro")
        # Complete skill bodies travel with each job; no inaccessible host-only skill paths.
        prompt["intro"] += "\n\n# A-Evolve browser workspace\n" + context
        dump(td / "instruction.json", prompt)
        job = dict(
            vwa_repo=str(repo),
            config_file=task["config_file"],
            result_dir=str(td),
            instruction_path=str(td / "instruction.json"),
            model=args.model_id,
            observation_type=args.observation_type,
            action_set_tag="som"
            if args.observation_type == "image_som"
            else "id_accessibility_tree",
            prices=prices,
            **{
                k: getattr(args, k)
                for k in (
                    "max_steps",
                    "max_tokens",
                    "temperature",
                    "max_obs_length",
                    "eval_captioning_model_device",
                    "sleep_after_execution",
                    "viewport_width",
                    "viewport_height",
                    "request_timeout",
                    "trajectory_text_chars",
                )
            },
        )
        dump(td / "job.json", job)
        env = os.environ.copy()
        env["DATASET"] = "visualwebarena"
        if args.reset_command:
            reset_code, reset_timeout, _ = run_logged(
                shlex.split(args.reset_command),
                cwd=repo,
                env=env,
                timeout=args.task_timeout,
                stdout=td / "reset.stdout.log",
                stderr=td / "reset.stderr.log",
            )
            if reset_code or reset_timeout:
                raise RuntimeError(f"Website reset failed; refusing to execute task {task['id']}")
        code, timed_out, elapsed = run_logged(
            [args.vwa_python, str(HERE / "vwa_worker.py"), "--job", str(td / "job.json")],
            cwd=repo,
            env=env,
            timeout=args.task_timeout,
            stdout=td / "stdout.log",
            stderr=td / "stderr.log",
        )
        result_path = td / "worker_result.json"
        raw = (
            json.loads(result_path.read_text())
            if result_path.exists()
            else {
                "success": False,
                "score": None,
                "status": "worker_failed",
                "steps": [],
                "usage": {},
            }
        )
        if timed_out or code:
            raw.update(
                success=False, score=None, status="timeout" if timed_out else "worker_failed"
            )
    usage = raw.get("usage", {})
    row = {
        "instance_id": task["id"],
        "task_name": task["id"],
        "task_input": task["intent"],
        "batch": batch,
        "mode": args.mode,
        "success": raw["success"],
        "score": raw.get("score"),
        "status": raw["status"],
        "synthetic": args.dry_run,
        "elapsed": elapsed,
        "solve_time": elapsed,
        "num_tool_calls": raw.get("num_tool_calls", 0),
        "num_turns": raw.get("num_turns", 0),
        "steps": raw.get("steps", []),
        "cost_units": usage.get("cost_units"),
        "total_tokens": usage.get("total_tokens", 0),
        "cumulative_input_tokens": usage.get("input_tokens", 0),
        "cumulative_output_tokens": usage.get("output_tokens", 0),
        "cumulative_cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
        "usage_record_quality": "synthetic"
        if args.dry_run
        else ("ok" if usage.get("usage_complete") else "missing"),
        "usage_estimated": False,
        "cost_priced": usage.get("cost_priced", False),
        "active_efficiency_strategies": ["fixed_browser_prompt"]
        if args.mode.startswith("predefined-")
        else [],
        "workspace_artifacts_injected": included,
        "raw_result_path": str(td / "worker_result.json"),
        "source_config_sha256": task.get("checksum"),
        "returncode": code,
        "timed_out": timed_out,
    }
    return row


def summarize(rows: list[dict]) -> dict:
    evaluated = [r for r in rows if r["score"] is not None and not r["synthetic"]]
    cost_known = bool(evaluated) and all(r.get("cost_priced") for r in rows)
    success = sum(r["success"] for r in evaluated)
    cost = sum(r.get("cost_units") or 0 for r in rows)
    return {
        "tasks": len(rows),
        "evaluated_tasks": len(evaluated),
        "success": success,
        "success_rate": success / len(rows)
        if rows and not any(r["synthetic"] for r in rows)
        else None,
        "score_sum": sum(r["score"] for r in evaluated),
        "synthetic_tasks": sum(r["synthetic"] for r in rows),
        "tokens": sum(r["total_tokens"] for r in rows),
        "total_cost": cost if cost_known else None,
        "priced_partial_cost": cost,
        "cost_of_pass": cost / success if cost_known and success else None,
        "unpriced_tasks": sum(not r.get("cost_priced") for r in rows),
        "tool_calls": sum(r["num_tool_calls"] for r in rows),
        "wall_time": sum(r["elapsed"] for r in rows),
    }


def write_outputs(out: Path, rows: list[dict], args, elapsed: float, *, render_figures=False):
    dump(out / "results.json", rows)
    batches = [
        dict(batch=b, **summarize([r for r in rows if r["batch"] == b]))
        for b in sorted({r["batch"] for r in rows})
    ]
    dump(
        out / "summary.json",
        {
            "mode": args.mode,
            "benchmark": "visualwebarena",
            "overall": summarize(rows),
            "by_batch": batches,
            "experiment_elapsed": elapsed,
        },
    )
    table = [dict(group=f"batch_{b['batch']:02d}", **b) for b in batches]
    table.append(dict(group="overall", batch="", **summarize(rows)))
    with (out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[-1]))
        writer.writeheader()
        writer.writerows(table)
    ledger = read_jsonl(out / "usage.jsonl")
    for p in (out / "tasks").glob("*/usage.jsonl"):
        ledger += read_jsonl(p)
    accounting = {
        phase: aggregate([x for x in ledger if x.get("phase") == phase])
        for phase in ("solver", "judge", "normal_evolution", "efficiency_proposer")
    }
    dump(
        out / "accounting.json",
        {
            "phases": accounting,
            "total": aggregate(ledger),
            "local_captioner_compute_included": False,
            "note": "Provider evolution aggregates have unknown cache accounting; no end-to-end monetary claim.",
        },
    )
    if not render_figures:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].bar([x["batch"] for x in batches], [x["tokens"] for x in batches])
        axes[0].set_title("Tokens per batch")
        axes[1].bar([x["batch"] for x in batches], [x["tool_calls"] for x in batches])
        axes[1].set_title("Browser action decisions")
        axes[2].bar([x["batch"] for x in batches], [x["wall_time"] for x in batches])
        axes[2].set_title("Task elapsed sum (s)")
        for ax in axes:
            ax.set_xlabel("Batch")
        fig.suptitle("SYNTHETIC smoke run" if args.dry_run else f"VisualWebArena: {args.mode}")
        fig.tight_layout()
        (out / "figures").mkdir(exist_ok=True)
        fig.savefig(out / "figures/batch_overview.png", dpi=160)
        plt.close(fig)
    except ImportError:
        logging.getLogger(__name__).warning("matplotlib not available; numeric results still saved")


def main(argv=None) -> int:
    args = parse_args(argv)
    tasks = load_tasks(args)
    if args.list_tasks:
        print(json.dumps(tasks, ensure_ascii=False, indent=2))
        return 0
    prices = load_prices(str(args.price_file) if args.price_file else None)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    out = (args.output_dir / f"{args.run_name}_{args.mode}_{stamp}").resolve()
    out.mkdir(parents=True, exist_ok=False)
    workspace = out / "workspace"
    shutil.copytree(args.seed_workspace, workspace)
    if args.mode.startswith("predefined-"):
        p = workspace / "prompts/fragments/fixed_browser_efficiency.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(PREDEFINED)
    initial = [
        p.relative_to(workspace).as_posix() for p in (workspace / "skills").rglob("SKILL.md")
    ]
    if len(initial) > args.max_skills:
        raise ValueError("Seed skills exceed --max-skills")
    dump(out / "task-list.json", tasks)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=False, capture_output=True, text=True
    )
    vwa_revision = (
        subprocess.run(
            ["git", "-C", str(args.vwa_repo), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if args.vwa_repo
        else None
    )
    manifest = {
        "benchmark": "visualwebarena",
        "runner": "native_vwa_adaptive_efficiency",
        "created_at": datetime.now(UTC).isoformat(),
        "mode": args.mode,
        "synthetic": args.dry_run,
        "configuration": vars(args),
        "repo_revision": revision.stdout.strip(),
        "vwa_revision": vwa_revision.stdout.strip() if vwa_revision else None,
        "seed_sha256": {
            str(p.relative_to(workspace)): digest(p) for p in workspace.rglob("*") if p.is_file()
        },
        "runner_sources_sha256": {p.name: digest(p) for p in HERE.glob("*.py")},
        "price_profile": prices,
        "price_file_sha256": digest(args.price_file) if args.price_file else None,
        "state_isolation": "external_reset_per_task"
        if args.reset_command
        else "unreset_shared_state",
        "normal_evolution_engine": "adaptive_skill",
        "prompt_delivery": "full_skill_bodies_inline",
        "predefined_control": "fixed_browser_prompt_only_not_terminal_runtime_strategies",
    }
    dump(out / "experiment_manifest.json", manifest)
    started = time.monotonic()
    rows = []
    failed = False
    try:
        for offset in range(0, len(tasks), args.batch_size):
            batch = offset // args.batch_size + 1
            current = []
            for task in tasks[offset : offset + args.batch_size]:
                row = solve(task, batch, workspace, args, out, prices)
                rows.append(row)
                current.append(row)
                write_outputs(out, rows, args, time.monotonic() - started)
            if offset + args.batch_size < len(tasks) and args.mode not in (
                "static-control",
                "predefined-control",
            ):
                before = [
                    p.relative_to(workspace).as_posix()
                    for p in (workspace / "skills").rglob("SKILL.md")
                ]
                if args.dry_run or args.dry_run_proposer:
                    append(
                        out / "emergence/normal_evolution.jsonl",
                        {"batch": batch, "status": "skipped_dry_run"},
                    )
                    if args.mode in COST_MODES:
                        dump(
                            out / "emergence" / f"batch_{batch:02d}_proposal_report.json",
                            {"batch": batch, "status": "skipped_dry_run", "proposals": []},
                        )
                else:
                    evolve(workspace, current, args, out, batch)
                    enforce_skill_budget(workspace, before, args.max_skills, out, batch)
                    before = [
                        p.relative_to(workspace).as_posix()
                        for p in (workspace / "skills").rglob("SKILL.md")
                    ]
                    if args.mode in COST_MODES:
                        propose(workspace, current, args, out, batch)
                        enforce_skill_budget(workspace, before, args.max_skills, out, batch)
                context, included = workspace_context(workspace, args.workspace_context_chars)
                dump(
                    out / "emergence/workspace_snapshots" / f"batch_{batch:02d}.json",
                    {"artifacts": included, "context": context},
                )
        failed = any(
            r["status"] in ("timeout", "worker_failed", "evaluation_missing") for r in rows
        )
    except Exception as exc:
        failed = True
        dump(out / "runner_error.json", {"error_type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        write_outputs(out, rows, args, time.monotonic() - started, render_figures=True)
        dump(
            out / "run_status.json",
            {
                "status": "failed" if failed else "completed",
                "planned_tasks": len(tasks),
                "completed_tasks": len(rows),
                "synthetic": args.dry_run,
            },
        )
        print(f"Artifacts: {out}", flush=True)
    return 2 if failed else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
