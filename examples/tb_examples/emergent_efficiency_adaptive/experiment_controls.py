"""Task selection and failure-aware efficiency feedback for the adaptive TB runner."""

from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .trace_stats import parse_session

COST_MODES = {"cost-aware", "predefined-cost-aware", "category-seeded-cost-aware"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def prior_task_names(roots):
    names, files = set(), []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        candidates = (
            [root]
            if root.is_file()
            else [root / "results.json", *sorted(root.glob("*/results.json"))]
        )
        for path in candidates:
            if not path.is_file():
                continue
            manifest = path.parent / "experiment_manifest.json"
            meta = read_json(manifest) if manifest.exists() else {}
            if meta.get("dry_run_harbor") or meta.get("synthetic"):
                continue
            if meta.get("benchmark") not in (
                None,
                "terminal_bench_2_1",
                "terminal_bench_2",
                "terminal_bench",
            ):
                continue
            rows = read_json(path)
            if isinstance(rows, dict):
                rows = rows.get("results")
            if not isinstance(rows, list):
                raise TypeError(f"Cannot inspect task history: {path}")
            files.append(str(path.resolve()))
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if row.get("synthetic") or row.get("status") in ("dry_run", "synthetic"):
                    continue
                identity = row.get("instance_id") or row.get("task_name")
                if isinstance(identity, str) and identity:
                    names.add(identity.removeprefix("terminal-bench/"))
    return names, files


def select_tasks(tasks, args, filter_tasks):
    filters = copy.copy(args)
    filters.limit = None  # History must be excluded BEFORE applying the task count.
    candidates = filter_tasks(list(tasks), filters)
    selection_path = Path(args.selection_file) if args.selection_file else None
    if selection_path and selection_path.exists() and selection_path.stat().st_size:
        report = read_json(selection_path)
        selected = report.get("tasks")
        if report.get("benchmark") != "terminal_bench_2_1" or not isinstance(selected, list):
            raise ValueError("Invalid frozen Terminal-Bench selection")
        if len(set(selected)) != len(selected):
            raise ValueError("Frozen selection contains duplicate tasks")
        by_name = {task.name: task for task in candidates}
        missing = set(selected) - set(by_name)
        if missing:
            raise ValueError(f"Frozen tasks missing or excluded by filters: {sorted(missing)}")
        return [by_name[name] for name in selected], {**report, "selection_reused": True}
    roots = args.history_root or ["logs/tb_emergent_efficiency_adaptive_harbor"]
    seen, files = prior_task_names(roots) if not args.allow_rerun else (set(), [])
    excluded = [task.name for task in candidates if task.name in seen]
    selected = [task for task in candidates if task.name not in seen]
    if args.limit is not None:
        selected = selected[: args.limit]
    report = {
        "benchmark": "terminal_bench_2_1",
        "created_at": datetime.now(UTC).isoformat(),
        "tasks": [task.name for task in selected],
        "history_roots": roots,
        "history_files": files,
        "excluded_previously_run": excluded,
        "allow_rerun": args.allow_rerun,
        "selection_reused": False,
        "note": "Matrix modes intentionally share this frozen list; completed and failed historical attempts are excluded.",
    }
    if selection_path:
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        selection_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return selected, report


def analyze_result(result):
    """Analyze a separate copy; never modify the original result or observer input."""
    result = dict(result)
    raw_path = result.get("raw_harbor_result_path")
    if not raw_path or not Path(raw_path).is_file():
        return result
    raw = read_json(raw_path)
    artifact = (raw.get("usage") or {}).get("usage_artifact_path")
    if artifact and Path(artifact).is_file():
        trace = parse_session(artifact)
        result["token_distribution"] = {
            "source": artifact,
            "cumulative_samples": trace["timeline"],
            "declared_early_stop": trace["declared_stop"],
        }
        result["failure_analysis_steps"] = trace["steps"]
    return result


def failure_feedback(results):
    groups = {}
    for name, success in (("successful", True), ("failed", False)):
        rows = [r for r in results if bool(r.get("success")) is success]
        totals = {
            key: sum(int(r.get(field) or 0) for r in rows)
            for key, field in (
                ("tokens", "total_tokens"),
                ("uncached_input", "cumulative_input_tokens"),
                ("cached_input", "cumulative_cache_read_input_tokens"),
                ("output", "cumulative_output_tokens"),
            )
        }
        groups[name] = {"tasks": len(rows), **totals}
        groups[name]["token_shares"] = {
            k: v / totals["tokens"] if totals["tokens"] else None
            for k, v in totals.items()
            if k != "tokens"
        }
    failures = []
    for row in sorted(
        (r for r in results if not r.get("success")),
        key=lambda r: r.get("total_tokens", 0),
        reverse=True,
    ):
        samples = (row.get("token_distribution") or {}).get("cumulative_samples", [])
        quartiles = []
        previous = 0
        for i in range(1, 5):
            if samples:
                current = samples[(len(samples) * i + 3) // 4 - 1]["total_tokens"]
                quartiles.append(current - previous)
                previous = current
        failures.append(
            {
                "task_id": row.get("instance_id") or row.get("task_name"),
                "status": row.get("status"),
                "tokens": row.get("total_tokens", 0),
                "input_tokens": row.get("cumulative_input_tokens", 0),
                "cache_read_tokens": row.get("cumulative_cache_read_input_tokens", 0),
                "output_tokens": row.get("cumulative_output_tokens", 0),
                "token_quartiles_by_model_updates": quartiles or None,
                "trajectory_available": bool(samples),
                "usage_record_quality": row.get("usage_record_quality", "unknown"),
                "declared_early_stop": (row.get("token_distribution") or {}).get(
                    "declared_early_stop", False
                ),
                "recent_steps": row.get("failure_analysis_steps", row.get("steps", []))[-6:],
            }
        )
    return {
        "groups": groups,
        "failed_tasks": failures,
        "notes": [
            "Quartiles split cumulative model-usage updates, NOT elapsed time or exact per-command cost.",
            "Reasoning tokens are already part of output; cache reads are included once.",
            "Missing trace/usage is unknown, not evidence of no work or no failure.",
            "A failed evaluation does not prove that a task is impossible; distinguish infrastructure failures.",
        ],
    }


@contextmanager
def failure_aware_proposer(shared, feedback):
    original_feedback = shared._build_cost_feedback
    prompts = {
        name: getattr(shared, name)
        for name in (
            "EMERGENCE_SYSTEM_PROMPT",
            "PREDEFINED_COST_AWARE_SYSTEM_PROMPT",
            "CATEGORY_SEEDED_SYSTEM_PROMPT",
        )
    }
    supplement = """\nADDITIONAL FOCUS ONLY: retain all existing efficiency objectives, strategies and proposal
kinds described above. Failure analysis supplements rather than replaces the original method.
Also inspect failure_token_distribution: uncached input, cache reads, output and consumption
in later model updates. Look for avoidable token use on failed attempts while continuing to
consider all other original efficiency opportunities, including successful-task improvements.
As ONE OPTIONAL efficiency strategy, consider feature-based early stopping when observed
obstacles remain after reasonable alternative recovery attempts and further work has little
prospect of progress. Examples include unavailable essential resources and repeatedly unchanged
failure states. Propose reusable trigger, do-not-trigger, bounded-recovery and diagnostic rules
through the SAME proposal/application mechanism used by other efficiency strategies.
Let the solving agent decide whether the strategy applies. Do not add an external interrupt,
fixed global token cap or task-ID blacklist. A failed benchmark run or high cache use alone
is not proof of impossibility. Do not force every proposal to be early stopping, and do not
replace the existing optimization methods. Preserve partial work, report unresolved obstacles
honestly and leave success determination to the original official verifier.
"""

    def build(*a, **kw):
        return (
            original_feedback(*a, **kw)
            + "\n\n## failure_token_distribution\n"
            + json.dumps(feedback, ensure_ascii=False)
        )

    shared._build_cost_feedback = build
    for name, text in prompts.items():
        setattr(shared, name, text + supplement)
    try:
        yield
    finally:
        shared._build_cost_feedback = original_feedback
        for name, text in prompts.items():
            setattr(shared, name, text)
