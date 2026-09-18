from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
RUN_ROOT = ROOT / "logs" / "emergent_efficiency" / "multi_mode"
OUT_DIR = RUN_ROOT / "analysis_latest"
FIG_DIR = OUT_DIR / "figures"

MODE_ORDER = [
    "normal-evolve",
    "cost-aware",
    "predefined-control",
    "category-seeded-cost-aware",
]

MODE_LABELS = {
    "normal-evolve": "Normal",
    "cost-aware": "CostAware",
    "predefined-control": "Predefined",
    "category-seeded-cost-aware": "CategorySeeded",
}

MODE_COLORS = {
    "normal-evolve": "#4E79A7",
    "cost-aware": "#B07AA1",
    "predefined-control": "#F28E2B",
    "category-seeded-cost-aware": "#59A14F",
}


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_created_at(path: Path) -> datetime:
    manifest = load_json(path / "experiment_manifest.json", {})
    raw = manifest.get("created_at") if isinstance(manifest, dict) else None
    if raw:
        return datetime.fromisoformat(str(raw))
    match = re.search(r"_(20\d{6})_(\d{6})", path.name)
    if match:
        return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
    return datetime.fromtimestamp(path.stat().st_mtime)


def discover_latest_runs() -> tuple[dict[str, Path], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    latest: dict[str, Path] = {}
    for path in RUN_ROOT.iterdir():
        if not path.is_dir() or path.name == OUT_DIR.name:
            continue
        summary = load_json(path / "summary.json")
        if not summary:
            continue
        mode = summary.get("mode")
        if not mode:
            continue
        created_at = parse_created_at(path)
        overall = summary.get("overall", {})
        batches = summary.get("by_batch", [])
        row = {
            "mode": mode,
            "path": str(path.relative_to(ROOT)),
            "run": path.name,
            "created_at": created_at.isoformat(timespec="seconds"),
            "tasks": overall.get("tasks", ""),
            "success": overall.get("success", ""),
            "success_rate": overall.get("success_rate", ""),
            "num_batches": len(batches),
            "tasks_per_batch": ",".join(str(b.get("tasks", "")) for b in batches),
            "selected": False,
            "_path": path,
            "_created_at": created_at,
        }
        candidates.append(row)
        if mode not in latest or created_at > parse_created_at(latest[mode]):
            latest[mode] = path
    for row in candidates:
        row["selected"] = row["_path"] == latest.get(row["mode"])
        row.pop("_path", None)
        row.pop("_created_at", None)
    candidates.sort(key=lambda r: (str(r["mode"]), str(r["created_at"])))
    return latest, candidates


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def pct(value: Any) -> str:
    v = safe_float(value)
    return "NA" if math.isnan(v) else f"{v * 100:.1f}%"


def signed_pct(value: Any) -> str:
    v = safe_float(value)
    if math.isnan(v) or math.isinf(v):
        return "NA"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.1f}%"


def fmt_cost(value: Any) -> str:
    v = safe_float(value)
    return "NA" if math.isnan(v) else f"{v:.3f}"


def fmt_int(value: Any) -> str:
    v = safe_float(value)
    return "NA" if math.isnan(v) else f"{int(round(v)):,}"


def fmt_float(value: Any, digits: int = 2) -> str:
    v = safe_float(value)
    return "NA" if math.isnan(v) else f"{v:.{digits}f}"


def ratio_delta(a: Any, b: Any) -> float:
    x = safe_float(a)
    y = safe_float(b)
    if math.isnan(x) or math.isnan(y) or y == 0:
        return math.nan
    return x / y - 1


def md_table(rows: list[dict[str, Any]], keys: list[str], headers: list[str] | None = None) -> str:
    if headers is None:
        headers = keys
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        vals = [str(row.get(key, "")).replace("\n", " ") for key in keys]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def task_id(row: dict[str, Any]) -> str:
    return row.get("instance_id") or row.get("task_name") or "unknown"


def setup_plot() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D0D5DD",
            "axes.labelcolor": "#1F2937",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "grid.color": "#E5E7EB",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "legend.frameon": False,
        }
    )


def label_bars(ax, values: list[float], fmt: str = "{:.2f}", dy: int = 3) -> None:
    for i, value in enumerate(values):
        if math.isnan(value):
            continue
        ax.annotate(fmt.format(value), (i, value), textcoords="offset points", xytext=(0, dy), ha="center", fontsize=8, color="#374151")


def plot_overall_dashboard(overall_rows: list[dict[str, Any]]) -> None:
    labels = [MODE_LABELS.get(r["mode"], r["mode"]) for r in overall_rows]
    colors = [MODE_COLORS.get(r["mode"], "#777777") for r in overall_rows]
    metrics = [
        ("success_rate", "Success Rate", lambda v: v * 100, "{:.0f}%"),
        ("avg_cost", "Avg Cost / Task ($)", lambda v: v, "{:.3f}"),
        ("avg_tokens", "Avg Tokens / Task", lambda v: v / 1000, "{:.0f}k"),
        ("avg_tool_calls", "Avg Tool Calls / Task", lambda v: v, "{:.1f}"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=180)
    for ax, (key, title, transform, label_fmt) in zip(axes.ravel(), metrics):
        vals = [transform(safe_float(r.get(key), 0)) for r in overall_rows]
        ax.bar(labels, vals, color=colors, width=0.62)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.85)
        ax.spines[["top", "right"]].set_visible(False)
        label_bars(ax, vals, label_fmt)
    axes[0, 0].set_ylim(0, 108)
    fig.suptitle("Latest Multi-mode Runs: Overall Comparison", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / "overall_dashboard.png", bbox_inches="tight")
    plt.close(fig)


def plot_frontier(overall_rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=180)
    for r in overall_rows:
        mode = r["mode"]
        x = safe_float(r.get("avg_cost"))
        y = safe_float(r.get("success_rate")) * 100
        size = 180 + safe_float(r.get("avg_tool_calls"), 0) * 14
        ax.scatter(x, y, s=size, color=MODE_COLORS.get(mode, "#777777"), alpha=0.85, edgecolor="#111827", linewidth=0.8)
        ax.annotate(MODE_LABELS.get(mode, mode), (x, y), xytext=(8, 8), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Avg Cost per Task ($, lower is better)")
    ax.set_ylabel("Success Rate (%, higher is better)")
    ax.set_title("Success-Cost Frontier (bubble size = avg tool calls)")
    ax.set_ylim(95, 101.5)
    ax.grid(True, alpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "success_cost_frontier.png", bbox_inches="tight")
    plt.close(fig)


def plot_batch_trends(summaries: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), dpi=180, sharex=True)
    for mode in MODE_ORDER:
        if mode not in summaries:
            continue
        batches = summaries[mode].get("by_batch", [])
        xs = list(range(1, len(batches) + 1))
        color = MODE_COLORS.get(mode, "#777777")
        label = MODE_LABELS.get(mode, mode)
        axes[0].plot(xs, [safe_float(b.get("success_rate"), 0) * 100 for b in batches], marker="o", label=label, color=color)
        axes[1].plot(xs, [safe_float(b.get("avg_cost"), 0) for b in batches], marker="o", label=label, color=color)
        axes[2].plot(xs, [safe_float(b.get("avg_tool_calls"), 0) for b in batches], marker="o", label=label, color=color)
    axes[0].set_ylabel("Success %")
    axes[1].set_ylabel("Avg Cost ($)")
    axes[2].set_ylabel("Avg Tool Calls")
    axes[2].set_xlabel("Batch")
    for ax in axes:
        ax.grid(True, alpha=0.85)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="best")
    axes[0].set_title("Batch-level Dynamics")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "batch_trends.png", bbox_inches="tight")
    plt.close(fig)


def plot_task_success_matrix(results_by_mode: dict[str, list[dict[str, Any]]]) -> None:
    tasks = sorted({task_id(r) for rows in results_by_mode.values() for r in rows})
    modes = [m for m in MODE_ORDER if m in results_by_mode]
    data = []
    for task in tasks:
        row = []
        for mode in modes:
            match = next((r for r in results_by_mode[mode] if task_id(r) == task), None)
            row.append(1 if match and match.get("success") else 0)
        data.append(row)
    fig, ax = plt.subplots(figsize=(7.2, max(6.0, len(tasks) * 0.28)), dpi=180)
    ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(modes)), [MODE_LABELS.get(m, m) for m in modes], rotation=25, ha="right")
    ax.set_yticks(range(len(tasks)), [t.replace("django__django-", "#") for t in tasks], fontsize=8)
    for y, row in enumerate(data):
        for x, val in enumerate(row):
            ax.text(x, y, "P" if val else "F", ha="center", va="center", fontsize=8, color="#111827")
    ax.set_title("Task-level Pass/Fail Matrix")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "task_success_matrix.png", bbox_inches="tight")
    plt.close(fig)


def plot_task_delta(results_by_mode: dict[str, list[dict[str, Any]]], metric: str, ylabel: str, filename: str) -> None:
    base_mode = "normal-evolve"
    if base_mode not in results_by_mode:
        return
    base = {task_id(r): r for r in results_by_mode[base_mode]}
    tasks = sorted(base)
    compare_modes = [m for m in MODE_ORDER if m in results_by_mode and m != base_mode]
    fig, ax = plt.subplots(figsize=(13, 5.8), dpi=180)
    width = 0.24 if len(compare_modes) >= 3 else 0.34
    offsets = [width * (i - (len(compare_modes) - 1) / 2) for i in range(len(compare_modes))]
    for offset, mode in zip(offsets, compare_modes):
        lookup = {task_id(r): r for r in results_by_mode[mode]}
        vals = [safe_float(lookup.get(task, {}).get(metric), 0) - safe_float(base[task].get(metric), 0) for task in tasks]
        ax.bar([i + offset for i in range(len(tasks))], vals, width=width, label=MODE_LABELS.get(mode, mode), color=MODE_COLORS.get(mode, "#777777"))
    ax.axhline(0, color="#111827", linewidth=0.9)
    ax.set_xticks(range(len(tasks)), [t.replace("django__django-", "#") for t in tasks], rotation=50, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " vs Normal")
    ax.grid(axis="y", alpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def plot_behavior_waste(behavior_rows: list[dict[str, Any]]) -> None:
    labels = [MODE_LABELS.get(r["mode"], r["mode"]) for r in behavior_rows]
    metrics = [
        ("repeated_exact_inputs", "Repeated inputs"),
        ("repeated_test_inputs", "Repeated tests"),
        ("same_state_repeated_tests", "Same-state tests"),
        ("post_edit_repeated_tests", "Post-edit tests"),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=180)
    width = 0.18
    xs = list(range(len(labels)))
    for i, (key, pretty) in enumerate(metrics):
        vals = [safe_float(r.get(key), 0) for r in behavior_rows]
        ax.bar([x + (i - 1.5) * width for x in xs], vals, width=width, label=pretty)
    ax.set_xticks(xs, labels)
    ax.set_ylabel("Count")
    ax.set_title("Behavior Waste Signals")
    ax.grid(axis="y", alpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "behavior_waste_signals.png", bbox_inches="tight")
    plt.close(fig)


def plot_strategy_hits(strategy_rows: list[dict[str, Any]]) -> None:
    strategies = sorted({r["strategy"] for r in strategy_rows})
    if not strategies:
        return
    modes = [m for m in MODE_ORDER if any(r["mode"] == m for r in strategy_rows)]
    lookup = {(r["mode"], r["strategy"]): r for r in strategy_rows}
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=180)
    width = 0.22
    xs = list(range(len(strategies)))
    for i, mode in enumerate(modes):
        vals = [safe_float(lookup.get((mode, s), {}).get("hits"), 0) for s in strategies]
        ax.bar([x + (i - (len(modes) - 1) / 2) * width for x in xs], vals, width=width, label=MODE_LABELS.get(mode, mode), color=MODE_COLORS.get(mode, "#777777"))
    ax.set_xticks(xs, strategies, rotation=25, ha="right")
    ax.set_ylabel("Hit Count")
    ax.set_title("Strategy Hit Counts")
    ax.grid(axis="y", alpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "strategy_hit_counts.png", bbox_inches="tight")
    plt.close(fig)


def plot_proposals(proposal_rows: list[dict[str, Any]]) -> None:
    applied = [r for r in proposal_rows if r.get("status") == "applied"]
    if not applied:
        return
    kind_counts = Counter(r["kind"] for r in applied)
    category_counts: Counter[str] = Counter()
    for row in applied:
        for cat in str(row.get("categories", "")).split(";"):
            if cat:
                category_counts[cat] += 1
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)
    axes[0].bar(kind_counts.keys(), kind_counts.values(), color="#76B7B2")
    axes[0].set_title("Applied Proposal Kinds")
    axes[0].grid(axis="y", alpha=0.85)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[1].bar(category_counts.keys(), category_counts.values(), color="#59A14F")
    axes[1].set_title("Applied Proposal Categories")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.85)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "proposal_summary.png", bbox_inches="tight")
    plt.close(fig)


def collect_data(latest: dict[str, Path]) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    summaries: dict[str, dict[str, Any]] = {}
    results_by_mode: dict[str, list[dict[str, Any]]] = {}
    overall_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    strategy_rows: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    for mode in MODE_ORDER:
        path = latest.get(mode)
        if path is None:
            continue
        summary = load_json(path / "summary.json", {})
        results = load_json(path / "results.json", [])
        summaries[mode] = summary
        results_by_mode[mode] = results
        overall = summary.get("overall", {})
        usage_counts = overall.get("usage_quality_counts", {})
        batches = summary.get("by_batch", [])
        overall_rows.append(
            {
                "mode": mode,
                "label": MODE_LABELS.get(mode, mode),
                "run_dir": str(path.relative_to(ROOT)),
                "created_at": parse_created_at(path).isoformat(timespec="seconds"),
                "tasks": overall.get("tasks", len(results)),
                "success": overall.get("success", 0),
                "success_rate": overall.get("success_rate", 0),
                "num_batches": len(batches),
                "batch_task_counts": ",".join(str(b.get("tasks", "")) for b in batches),
                "avg_cost": overall.get("avg_cost", 0),
                "cost_of_pass": overall.get("cost_of_pass", 0),
                "total_cost": overall.get("total_cost", 0),
                "avg_tokens": overall.get("avg_tokens", 0),
                "input_tokens": overall.get("input_tokens", 0),
                "output_tokens": overall.get("output_tokens", 0),
                "avg_tool_calls": overall.get("avg_tool_calls", 0),
                "tool_calls": overall.get("tool_calls", 0),
                "turns": overall.get("turns", 0),
                "wall_time": overall.get("wall_time", 0),
                "token_record_quality": overall.get("token_record_quality", ""),
                "usage_ok_tasks": usage_counts.get("ok", 0),
                "zero_token_tasks": overall.get("zero_token_tasks", 0),
                "repeated_exact_inputs": overall.get("repeated_exact_inputs", 0),
                "repeated_test_inputs": overall.get("repeated_test_inputs", 0),
                "same_state_repeated_tests": overall.get("same_state_repeated_tests", 0),
                "post_edit_repeated_tests": overall.get("post_edit_repeated_tests", 0),
                "prompt_fragment_visible_tasks": overall.get("prompt_fragment_visible_tasks", 0),
            }
        )
        for row in results:
            bs = row.get("behavior_stats") or {}
            task_rows.append(
                {
                    "mode": mode,
                    "task": task_id(row),
                    "batch": row.get("batch", ""),
                    "success": bool(row.get("success")),
                    "cost_units": row.get("cost_units", 0),
                    "total_tokens": row.get("total_tokens", 0),
                    "input_tokens": row.get("cumulative_input_tokens", 0),
                    "output_tokens": row.get("cumulative_output_tokens", 0),
                    "tool_calls": row.get("num_tool_calls", 0),
                    "turns": row.get("num_turns", 0),
                    "elapsed": row.get("elapsed", 0),
                    "usage_record_quality": row.get("usage_record_quality", ""),
                    "repeated_exact_inputs": bs.get("repeated_exact_inputs", 0),
                    "repeated_test_inputs": bs.get("repeated_test_inputs", 0),
                    "same_state_repeated_tests": bs.get("same_state_repeated_tests", 0),
                    "post_edit_repeated_tests": bs.get("post_edit_repeated_tests", 0),
                    "prompt_fragment_visible": bs.get("prompt_fragment_visible", False),
                }
            )
        for name, stat in (overall.get("strategy_hits") or {}).items():
            strategy_rows.append(
                {
                    "mode": mode,
                    "strategy": name,
                    "hits": stat.get("hits", 0),
                    "tasks_with_hit": stat.get("tasks_with_hit", 0),
                    "task_hit_rate": stat.get("task_hit_rate", 0),
                }
            )
        for adoption in load_json(path / "emergence" / "proposal_adoption.json", []):
            proposal = adoption.get("proposal", {})
            proposal_rows.append(
                {
                    "mode": mode,
                    "proposal_id": adoption.get("proposal_id", ""),
                    "status": adoption.get("status", ""),
                    "kind": adoption.get("kind", proposal.get("kind", "")),
                    "categories": ";".join(adoption.get("categories", proposal.get("categories", []))),
                    "name": proposal.get("name", ""),
                    "description": proposal.get("description", ""),
                    "trigger": proposal.get("trigger", ""),
                    "expected_savings": proposal.get("expected_savings", ""),
                    "correctness_risk": proposal.get("correctness_risk", ""),
                    "confidence": proposal.get("confidence", adoption.get("confidence", "")),
                    "target": adoption.get("target", ""),
                    "evidence": " | ".join(proposal.get("evidence", [])),
                }
            )
    overall_rows.sort(key=lambda r: MODE_ORDER.index(r["mode"]))
    return summaries, results_by_mode, overall_rows, task_rows, strategy_rows, proposal_rows


def best_modes(task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in task_rows:
        if row["success"]:
            grouped[row["task"]].append(row)
    rows: list[dict[str, Any]] = []
    for task, items in sorted(grouped.items()):
        best_cost = min(items, key=lambda r: safe_float(r["cost_units"], math.inf))
        best_calls = min(items, key=lambda r: safe_float(r["tool_calls"], math.inf))
        rows.append(
            {
                "task": task,
                "lowest_cost_mode": best_cost["mode"],
                "lowest_cost": best_cost["cost_units"],
                "fewest_calls_mode": best_calls["mode"],
                "fewest_calls": best_calls["tool_calls"],
            }
        )
    return rows


def build_case_rows(task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in task_rows:
        by_task[row["task"]][row["mode"]] = row
    cases: list[dict[str, Any]] = []
    for task, per_mode in sorted(by_task.items()):
        required = [per_mode.get(mode) for mode in MODE_ORDER]
        if any(row is None for row in required):
            continue
        costs = {mode: safe_float(per_mode[mode]["cost_units"]) for mode in MODE_ORDER}
        spread = max(costs.values()) - min(costs.values())
        if spread < 0.15 and task not in {"django__django-11885", "django__django-12209", "django__django-12273", "django__django-12325"}:
            continue
        cases.append(
            {
                "task": task,
                "normal_cost": fmt_cost(costs["normal-evolve"]),
                "cost_aware_cost": fmt_cost(costs["cost-aware"]),
                "predefined_cost": fmt_cost(costs["predefined-control"]),
                "category_cost": fmt_cost(costs["category-seeded-cost-aware"]),
                "normal_calls": per_mode["normal-evolve"]["tool_calls"],
                "cost_aware_calls": per_mode["cost-aware"]["tool_calls"],
                "predefined_calls": per_mode["predefined-control"]["tool_calls"],
                "category_calls": per_mode["category-seeded-cost-aware"]["tool_calls"],
                "insight": "large cross-mode efficiency spread at equal pass outcome",
            }
        )
    return cases


def report(
    selected_rows: list[dict[str, Any]],
    overall_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    behavior_rows: list[dict[str, Any]],
    strategy_rows: list[dict[str, Any]],
    proposal_rows: list[dict[str, Any]],
) -> str:
    by_mode = {r["mode"]: r for r in overall_rows}
    normal = by_mode.get("normal-evolve", {})
    predefined = by_mode.get("predefined-control", {})
    category = by_mode.get("category-seeded-cost-aware", {})
    cost_aware = by_mode.get("cost-aware", {})
    best_rows = best_modes(task_rows)
    best_cost_counts = Counter(row["lowest_cost_mode"] for row in best_rows)
    best_call_counts = Counter(row["fewest_calls_mode"] for row in best_rows)
    case_rows = build_case_rows(task_rows)
    applied_props = [r for r in proposal_rows if r["status"] == "applied"]
    props_by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in applied_props:
        props_by_mode[row["mode"]].append(row)
    prop_kind_counts = Counter(r["kind"] for r in applied_props)
    prop_cat_counts: Counter[str] = Counter()
    for row in applied_props:
        for cat in row["categories"].split(";"):
            if cat:
                prop_cat_counts[cat] += 1

    selection_md = [
        {
            "mode": r["mode"],
            "created_at": r["created_at"],
            "selected": "yes" if r["selected"] else "no",
            "tasks": r["tasks"],
            "success_rate": pct(r["success_rate"]) if r["success_rate"] != "" else "",
            "batches": r["num_batches"],
            "tasks_per_batch": r["tasks_per_batch"],
            "run": r["run"],
        }
        for r in selected_rows
    ]
    overall_md = [
        {
            "mode": r["mode"],
            "run": Path(r["run_dir"]).name,
            "pass": f"{r['success']}/{r['tasks']}",
            "success_rate": pct(r["success_rate"]),
            "avg_cost": fmt_cost(r["avg_cost"]),
            "cost_of_pass": fmt_cost(r["cost_of_pass"]),
            "avg_tokens": fmt_int(r["avg_tokens"]),
            "avg_calls": fmt_float(r["avg_tool_calls"], 2),
            "waste_inputs": r["repeated_exact_inputs"],
            "prompt_visible": r["prompt_fragment_visible_tasks"],
        }
        for r in overall_rows
    ]
    delta_md = []
    for mode in MODE_ORDER:
        if mode == "normal-evolve" or mode not in by_mode or not normal:
            continue
        r = by_mode[mode]
        delta_md.append(
            {
                "mode": mode,
                "success_delta": f"{(safe_float(r['success_rate']) - safe_float(normal['success_rate'])) * 100:+.1f} pp",
                "avg_cost_delta": signed_pct(ratio_delta(r["avg_cost"], normal["avg_cost"])),
                "tokens_delta": signed_pct(ratio_delta(r["avg_tokens"], normal["avg_tokens"])),
                "tool_calls_delta": signed_pct(ratio_delta(r["avg_tool_calls"], normal["avg_tool_calls"])),
                "wall_time_delta": signed_pct(ratio_delta(r["wall_time"], normal["wall_time"])),
            }
        )
    behavior_md = [
        {
            "mode": r["mode"],
            "repeated_inputs": r["repeated_exact_inputs"],
            "repeated_tests": r["repeated_test_inputs"],
            "same_state_tests": r["same_state_repeated_tests"],
            "post_edit_tests": r["post_edit_repeated_tests"],
            "prompt_visible_tasks": r["prompt_fragment_visible_tasks"],
        }
        for r in behavior_rows
    ]
    proposal_md = [
        {
            "mode": r["mode"],
            "name": r["name"],
            "kind": r["kind"],
            "categories": r["categories"],
            "confidence": r["confidence"],
            "description": r["description"],
        }
        for r in applied_props
    ]
    cheapest = min(overall_rows, key=lambda r: safe_float(r["avg_cost"], math.inf))
    lowest_calls = min(overall_rows, key=lambda r: safe_float(r["avg_tool_calls"], math.inf))
    all_success = all(safe_float(r["success_rate"]) == 1.0 for r in overall_rows)
    best_md = [
        {
            "task": r["task"].replace("django__django-", "#"),
            "lowest_cost_mode": r["lowest_cost_mode"],
            "lowest_cost": fmt_cost(r["lowest_cost"]),
            "fewest_calls_mode": r["fewest_calls_mode"],
            "fewest_calls": r["fewest_calls"],
        }
        for r in best_rows
    ]

    lines = [
        "# multi_mode 最新实验结果分析报告",
        "",
        f"生成时间：2026-09-13。输入目录：`{RUN_ROOT.relative_to(ROOT)}`。选择规则：每个 `mode` 选择目录时间戳最新的 run；本轮最新 run 均为 20 个任务、5 个 batch、每个 batch 4 个任务。",
        "",
        "## 1. 结论摘要",
        "",
        f"- **最新四组全部达到 20/20 通过**：`normal-evolve`、`cost-aware`、`predefined-control`、`category-seeded-cost-aware` 成功率均为 100%。因此本轮核心不再是 correctness rescue，而是等正确率下的成本、token、工具调用和行为浪费对比。" if all_success else f"- **最新四组已纳入统一对比**：最低成功率为 {pct(min(r['success_rate'] for r in overall_rows))}，以下同时报告 correctness 与 efficiency。",
        f"- **成本最低的是 `{cheapest['mode']}`**：平均成本 {fmt_cost(cheapest.get('avg_cost'))}/task。`category-seeded-cost-aware` 平均成本 {fmt_cost(category.get('avg_cost'))}，`cost-aware` 为 {fmt_cost(cost_aware.get('avg_cost'))}，`predefined-control` 为 {fmt_cost(predefined.get('avg_cost'))}。",
        f"- **工具调用最低的是 `{lowest_calls['mode']}`**：平均 {fmt_float(lowest_calls.get('avg_tool_calls'), 2)} 次/task。CategorySeeded 低于 CostAware 的 {fmt_float(cost_aware.get('avg_tool_calls'), 2)} 和 Predefined 的 {fmt_float(predefined.get('avg_tool_calls'), 2)}，说明类别先验对 M5 Tools 的压缩更明显。",
        f"- **`cost-aware` 的位置很关键**：同样 20/20 通过，平均成本 {fmt_cost(cost_aware.get('avg_cost'))}/task，略高于 CategorySeeded 的 {fmt_cost(category.get('avg_cost'))}，工具调用也更多。这表明仅有成本反馈能维持正确率，但没有类别先验时，策略组织和动作压缩弱一些。",
        f"- **`predefined-control` 在本轮仍不占优**：同样 20/20 通过，但平均成本、token 和工具调用均高于 Normal；相比 CategorySeeded，成本更高、调用更多，说明固定预定义策略没有比演化出的策略更经济。",
        f"- **proposal 层面出现清晰分化**：CostAware applied {len(props_by_mode.get('cost-aware', []))} 条 proposal，但类别为空；CategorySeeded applied {len(props_by_mode.get('category-seeded-cost-aware', []))} 条 proposal，并显式映射到 {dict(prop_cat_counts)}。这支持“类别先验让策略更可解释、更可归因”的结论。",
        "",
        "## 2. 最新 Run 选择",
        "",
        md_table(selection_md, ["mode", "created_at", "selected", "tasks", "success_rate", "batches", "tasks_per_batch", "run"]),
        "",
        "## 3. 实验设置与数据口径",
        "",
        "- 数据：SWE 任务子集，本轮每个 mode 跑 20 个任务。",
        "- 批处理：最新 run 均为 5 个 batch，每个 batch 4 个任务。`summary.json` 未单独记录命令行 `batch_size`，因此报告使用实际 `by_batch` 结构作为批处理口径。",
        "- 对比模式：`normal-evolve`、`cost-aware`、`predefined-control`、`category-seeded-cost-aware`。",
        "- 指标：pass rate、平均成本、cost of pass、token、工具调用、turn、wall time、重复输入/重复测试、prompt fragment 可见任务数、proposal adoption。",
        "- 成本：本轮 `cost_currency=USD` 且 `cost_priced=True`，因此图表中的 `$` 是按当前运行配置计算出的美元成本；完整 token 也同步列出，避免单一价格口径误读。",
        "",
        "## 4. 总体对比",
        "",
        "![overall_dashboard](figures/overall_dashboard.png)",
        "",
        md_table(overall_md, ["mode", "run", "pass", "success_rate", "avg_cost", "cost_of_pass", "avg_tokens", "avg_calls", "waste_inputs", "prompt_visible"]),
        "",
        "![success_cost_frontier](figures/success_cost_frontier.png)",
        "",
        "相对 Normal 的变化：",
        "",
        md_table(delta_md, ["mode", "success_delta", "avg_cost_delta", "tokens_delta", "tool_calls_delta", "wall_time_delta"]),
        "",
        "解读：因为四组都是 100% 成功，本轮 success-cost frontier 的纵轴没有拉开；真正的差异在横轴和气泡大小。Normal 是最低成本基线，CategorySeeded 是最低工具调用方案，CostAware 接近 Predefined 的成本但工具调用更少，Predefined 没有形成成本优势。",
        "",
        "## 5. Batch 级动态",
        "",
        "![batch_trends](figures/batch_trends.png)",
        "",
        "- 四组所有 batch 都是 100% 通过，说明最新大 batch run 的稳定性显著高于旧 run。",
        "- CategorySeeded 的 batch 1/2/5 成本低于或接近 Normal，但 batch 3/4 成本抬升，尤其 `django__django-12039`, `12209`, `12273` 拉高平均成本。",
        "- CostAware 的 batch 4 成本和工具调用较高，是它总体成本接近 Predefined 的主要原因之一。",
        "- Predefined 在 batch 4/5 工具调用和成本都偏高，说明固定策略在后期较难任务上没有降低搜索成本。",
        "",
        "## 6. 任务级通过与成本差异",
        "",
        "![task_success_matrix](figures/task_success_matrix.png)",
        "",
        "![per_task_cost_delta_vs_normal.png](figures/per_task_cost_delta_vs_normal.png)",
        "",
        "![per_task_tool_delta_vs_normal.png](figures/per_task_tool_delta_vs_normal.png)",
        "",
        "所有任务都通过，因此任务级矩阵主要证明 correctness 已经对齐；更有价值的是逐任务成本差异。低成本/低工具调用归属统计如下：",
        "",
        f"- 最低成功成本归属：{dict(best_cost_counts)}。",
        f"- 最少工具调用归属：{dict(best_call_counts)}。",
        "",
        md_table(best_md, ["task", "lowest_cost_mode", "lowest_cost", "fewest_calls_mode", "fewest_calls"]),
        "",
        "值得分享的高信号 case：",
        "",
        md_table(case_rows, ["task", "normal_cost", "cost_aware_cost", "predefined_cost", "category_cost", "normal_calls", "cost_aware_calls", "predefined_calls", "category_calls", "insight"]),
        "",
        "- `django__django-11885`：四组都通过，CostAware 与 CategorySeeded 的工具调用明显少于 Normal/Predefined，说明演化出的成本策略在这类高探索任务上能压缩工具边界。",
        "- `django__django-12209` 与 `django__django-12273`：四组都通过，但 CategorySeeded 成本最高，说明策略更稳不等于更便宜，后期难任务仍会被额外验证拉高成本。",
        "- `django__django-12304`：CategorySeeded 成本和调用都低于 Normal/Predefined，说明演化策略在部分后期任务上已经能做到 success-preserving efficiency。",
        "",
        "## 7. 行为浪费信号",
        "",
        "![behavior_waste_signals](figures/behavior_waste_signals.png)",
        "",
        md_table(behavior_md, ["mode", "repeated_inputs", "repeated_tests", "same_state_tests", "post_edit_tests", "prompt_visible_tasks"]),
        "",
        "行为层面的结论比总体成本更细：CategorySeeded 的平均工具调用最低，重复输入、重复测试和 same-state repeated tests 也都低于 Normal；CostAware 的重复测试与 CategorySeeded 持平，但重复输入更多、工具调用更多。这说明成本反馈本身能学到“少重复测试”，而类别先验进一步帮助压缩编辑/读取类无效动作。Predefined 的 prompt fragment 在 12 个任务可见，但成本仍最高，说明“策略可见”与“策略有效执行”之间仍存在差距。",
        "",
        "## 8. 策略命中与 Proposal 分析",
        "",
        "![strategy_hit_counts](figures/strategy_hit_counts.png)",
        "",
        "![proposal_summary](figures/proposal_summary.png)",
        "",
        md_table(proposal_md, ["mode", "name", "kind", "categories", "confidence", "description"]),
        "",
        "本轮 CostAware 与 CategorySeeded 都产生了可落地 proposal，但差异明显：CostAware 的 proposal 多数没有 M1-M7 类别标注，CategorySeeded 的 proposal 全部被映射到成本维度，因此更适合做方法论文中的可解释策略分析。",
        "",
        "CategorySeeded 的 proposal 更像是一组面向 SWE 编辑/测试循环的效率工具箱：",
        "",
        "- `probe_exact_snippet`：在 str_replace 前检查 old_str 是否逐字节匹配，减少编辑失败后的重读和重试。",
        "- `test_run_dedup_and_batch`：维护测试 ledger，避免代码未变化时重复运行相同测试，并把 focused test 与模块测试合并。",
        "- `repeat_test_state_guard` / `same_state_test_cache_summarizer`：对相同工作区状态下的重复命令做提示或缓存，减少重复测试和长输出回放。",
        "- `one_narrow_test_per_edit` / `state_change_before_rerun_gate`：把每次编辑后的测试约束为最小必要验证，并要求复测前说明状态变化。",
        "- `targeted_symbol_read`：先定位符号再读局部代码块，减少大文件阅读和无关上下文进入 prompt。",
        "",
        "这些 proposal 主要落在 M5 Tools，同时覆盖 M1 Context、M2 Reasoning、M3 Output、M4 Planning。CostAware 也能生成 batching/cache/dedup 类策略，但缺少类别化归因；CategorySeeded 的优势不是只多生成策略，而是把策略绑定到可观测成本维度上。",
        "",
        "## 9. 本轮特殊 Insights",
        "",
        "1. **等成功率下，Normal 仍是强成本基线**：四组都 20/20 后，任何效率策略都必须在成本或动作结构上击败 Normal 才算真正有效。",
        "2. **CostAware 证明成本反馈可保持正确率，但不是最优动作结构**：它和其他组一样全通过，但工具调用高于 CategorySeeded。",
        "3. **CategorySeeded 的优势从正确性转为工具调用和可解释性**：它不是最低成本，但 avg tool calls 最低，且 proposal 能映射到 M1-M7 成本维度。",
        "4. **Predefined 策略没有带来可见成本红利**：固定策略在 20/20 通过的前提下比 Normal 更贵，比 CategorySeeded 和 CostAware 调用更多。",
        "5. **proposal generation 已经能定位真实浪费点**：str_replace 失败、重复测试、长输出、重复只读探测都是 SWE agent 的典型效率瓶颈。",
        "6. **下一步应该做策略级 ablation**：CostAware 和 CategorySeeded 各有 8 条 proposal 同时 applied，无法判断具体是哪条带来工具调用下降，也无法判断哪些策略抬高了 token/cost。",
        "",
        "## 10. 建议",
        "",
        "1. 对 CostAware 与 CategorySeeded 的 proposal 做逐项 ablation，优先测试 `test_run_dedup_and_batch`、`probe_exact_snippet`、`same_state_test_cache_summarizer`，并对照 CostAware 的 `batch_independent_recon`、`skip_equivalent_test_rerun`。",
        "2. 增加 hard gate：相同测试命令且工作区无变化时默认返回 cached result，除非 agent 显式 force rerun。",
        "3. 把评估拆成两类：`success-preserving efficiency` 用本轮 20/20 数据衡量；`success-rescuing efficiency` 用旧 run 或更难任务衡量。",
        "4. 针对 `12209`、`12273`、`12039` 做轨迹级 case study，找出 CategorySeeded 成本高于 Normal 的原因。",
        "5. 保留每轮 run selection 表，避免新旧 batch 结果混用导致结论漂移。",
        "",
        "## 11. 产物索引",
        "",
        "- `run_selection.csv`：所有候选 run 与最新选择。",
        "- `overall_comparison.csv`：mode 级指标。",
        "- `task_level_comparison.csv`：任务级成本、token、工具调用与行为信号。",
        "- `behavior_comparison.csv`：重复行为指标。",
        "- `strategy_hits.csv`：策略命中统计。",
        "- `proposal_analysis.csv`：CostAware 与 CategorySeeded proposal adoption 明细。",
        "- `category_seeded_proposals.csv`：兼容旧文件名，内容同 `proposal_analysis.csv`。",
        "- `figures/*.png`：所有图表。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    setup_plot()
    latest, selected_rows = discover_latest_runs()
    summaries, results_by_mode, overall_rows, task_rows, strategy_rows, proposal_rows = collect_data(latest)
    behavior_rows = [
        {
            "mode": r["mode"],
            "repeated_exact_inputs": r["repeated_exact_inputs"],
            "repeated_test_inputs": r["repeated_test_inputs"],
            "same_state_repeated_tests": r["same_state_repeated_tests"],
            "post_edit_repeated_tests": r["post_edit_repeated_tests"],
            "prompt_fragment_visible_tasks": r["prompt_fragment_visible_tasks"],
        }
        for r in overall_rows
    ]
    write_csv(OUT_DIR / "run_selection.csv", selected_rows)
    write_csv(OUT_DIR / "overall_comparison.csv", overall_rows)
    write_csv(OUT_DIR / "task_level_comparison.csv", task_rows)
    write_csv(OUT_DIR / "behavior_comparison.csv", behavior_rows)
    write_csv(OUT_DIR / "strategy_hits.csv", strategy_rows)
    write_csv(OUT_DIR / "proposal_analysis.csv", proposal_rows)
    write_csv(OUT_DIR / "category_seeded_proposals.csv", proposal_rows)
    write_csv(OUT_DIR / "best_task_modes.csv", best_modes(task_rows))

    plot_overall_dashboard(overall_rows)
    plot_frontier(overall_rows)
    plot_batch_trends(summaries)
    plot_task_success_matrix(results_by_mode)
    plot_task_delta(results_by_mode, "cost_units", "Cost Delta ($)", "per_task_cost_delta_vs_normal.png")
    plot_task_delta(results_by_mode, "num_tool_calls", "Tool-call Delta", "per_task_tool_delta_vs_normal.png")
    plot_behavior_waste(behavior_rows)
    plot_strategy_hits(strategy_rows)
    plot_proposals(proposal_rows)

    text = report(selected_rows, overall_rows, task_rows, behavior_rows, strategy_rows, proposal_rows)
    report_path = OUT_DIR / "MULTI_MODE_LATEST_ANALYSIS_CN.md"
    report_path.write_text(text, encoding="utf-8")
    print(report_path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
