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
RUN_ROOT = ROOT / "logs" / "tb_emergent_efficiency"
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
    "cost-aware": "Cost-aware",
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
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def parse_created_at(path: Path, manifest: dict[str, Any] | None) -> datetime:
    raw = (manifest or {}).get("created_at")
    if raw:
        return datetime.fromisoformat(str(raw))
    match = re.search(r"_(20\d{6})_(\d{6})", path.name)
    if match:
        return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
    return datetime.fromtimestamp(path.stat().st_mtime)


def discover_latest_runs() -> tuple[dict[str, Path], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for path in RUN_ROOT.iterdir():
        if not path.is_dir() or path.name == OUT_DIR.name:
            continue
        summary = load_json(path / "summary.json")
        if not summary:
            continue
        manifest = load_json(path / "experiment_manifest.json", {})
        mode = summary.get("mode") or manifest.get("mode")
        if not mode:
            continue
        created_at = parse_created_at(path, manifest)
        overall = summary.get("overall", {})
        candidates.append(
            {
                "mode": mode,
                "path": str(path.relative_to(ROOT)),
                "created_at": created_at.isoformat(timespec="seconds"),
                "tasks": overall.get("tasks", manifest.get("task_count", "")),
                "success_rate": overall.get("success_rate", ""),
                "selected": False,
                "_path": path,
                "_created_at": created_at,
            }
        )
    latest: dict[str, Path] = {}
    for row in candidates:
        mode = row["mode"]
        current = latest.get(mode)
        if current is None:
            latest[mode] = row["_path"]
            continue
        current_manifest = load_json(current / "experiment_manifest.json", {})
        if row["_created_at"] > parse_created_at(current, current_manifest):
            latest[mode] = row["_path"]
    for row in candidates:
        row["selected"] = row["_path"] == latest.get(row["mode"])
        row.pop("_path", None)
        row.pop("_created_at", None)
    candidates.sort(key=lambda r: (str(r["mode"]), str(r["created_at"])))
    return latest, candidates


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v


def fmt_num(value: Any, digits: int = 1) -> str:
    v = safe_float(value)
    if math.isnan(v) or math.isinf(v):
        return "NA"
    return f"{v:,.{digits}f}"


def fmt_int(value: Any) -> str:
    v = safe_float(value)
    if math.isnan(v) or math.isinf(v):
        return "NA"
    return f"{int(round(v)):,}"


def pct(value: Any) -> str:
    v = safe_float(value)
    if math.isnan(v):
        return "NA"
    return f"{v * 100:.1f}%"


def task_id(row: dict[str, Any]) -> str:
    meta = row.get("metadata") or {}
    return (
        row.get("task_name")
        or row.get("instance_id")
        or meta.get("task_name")
        or meta.get("name")
        or meta.get("instance_id")
        or "unknown"
    )


def short_failure(row: dict[str, Any], limit: int = 190) -> str:
    if row.get("success"):
        return ""
    raw = row.get("feedback_detail") or row.get("eval_output") or row.get("status") or ""
    text = re.sub(r"\s+", " ", str(raw)).strip()
    return text[:limit] + ("..." if len(text) > limit else "")


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


def md_table(rows: list[dict[str, Any]], keys: list[str], headers: list[str] | None = None) -> str:
    if headers is None:
        headers = keys
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        vals = [str(row.get(key, "")).replace("\n", " ") for key in keys]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


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


def label_bars(ax, values: list[float], fmt: str = "{:.1f}", dy: int = 3) -> None:
    for i, value in enumerate(values):
        if math.isnan(value):
            continue
        ax.annotate(
            fmt.format(value),
            (i, value),
            textcoords="offset points",
            xytext=(0, dy),
            ha="center",
            fontsize=8,
            color="#374151",
        )


def plot_overall_dashboard(overall_rows: list[dict[str, Any]]) -> None:
    labels = [MODE_LABELS.get(r["mode"], r["mode"]) for r in overall_rows]
    colors = [MODE_COLORS.get(r["mode"], "#777777") for r in overall_rows]
    metrics = [
        ("success_rate", "Success Rate", lambda v: v * 100, "{:.0f}%"),
        ("cost_of_pass", "Cost of Pass", lambda v: v / 1_000_000, "{:.2f}M"),
        ("avg_cost", "Avg Cost / Task", lambda v: v / 1_000_000, "{:.2f}M"),
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
    axes[0, 0].set_ylim(0, 105)
    fig.suptitle("Terminal-Bench 2 Latest Runs: Overall Comparison", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / "overall_dashboard.png", bbox_inches="tight")
    plt.close(fig)


def plot_frontier(overall_rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=180)
    for r in overall_rows:
        mode = r["mode"]
        x = safe_float(r.get("cost_of_pass")) / 1_000_000
        y = safe_float(r.get("success_rate")) * 100
        size = 120 + min(900, safe_float(r.get("avg_tool_calls"), 0) * 8)
        ax.scatter(
            x,
            y,
            s=size,
            color=MODE_COLORS.get(mode, "#777777"),
            alpha=0.85,
            edgecolor="#111827",
            linewidth=0.8,
        )
        ax.annotate(MODE_LABELS.get(mode, mode), (x, y), xytext=(8, 7), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Cost of Pass (M token-equivalent, lower is better)")
    ax.set_ylabel("Success Rate (%, higher is better)")
    ax.set_title("Success-Cost Frontier (bubble size = avg tool calls)")
    ax.grid(True, alpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 100)
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
        label = MODE_LABELS.get(mode, mode)
        color = MODE_COLORS.get(mode, "#777777")
        axes[0].plot(xs, [safe_float(b.get("success_rate"), 0) * 100 for b in batches], marker="o", label=label, color=color)
        axes[1].plot(xs, [safe_float(b.get("avg_cost"), 0) / 1_000_000 for b in batches], marker="o", label=label, color=color)
        axes[2].plot(xs, [safe_float(b.get("avg_tool_calls"), 0) for b in batches], marker="o", label=label, color=color)
    axes[0].set_ylabel("Success %")
    axes[1].set_ylabel("Avg Cost (M)")
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
    data: list[list[int]] = []
    for task in tasks:
        row: list[int] = []
        for mode in modes:
            match = next((r for r in results_by_mode[mode] if task_id(r) == task), None)
            row.append(1 if match and match.get("success") else 0)
        data.append(row)
    fig, ax = plt.subplots(figsize=(7.4, max(5.2, len(tasks) * 0.34)), dpi=180)
    ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(modes)), [MODE_LABELS.get(m, m) for m in modes], rotation=25, ha="right")
    ax.set_yticks(range(len(tasks)), tasks, fontsize=8)
    for y, row in enumerate(data):
        for x, val in enumerate(row):
            ax.text(x, y, "P" if val else "F", ha="center", va="center", fontsize=8, color="#111827")
    ax.set_title("Task-level Pass/Fail Matrix")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "task_success_matrix.png", bbox_inches="tight")
    plt.close(fig)


def plot_cost_delta(results_by_mode: dict[str, list[dict[str, Any]]]) -> None:
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
        vals = []
        for task in tasks:
            rr = lookup.get(task)
            vals.append((safe_float(rr.get("cost_units"), 0) - safe_float(base[task].get("cost_units"), 0)) / 1_000_000 if rr else math.nan)
        ax.bar(
            [i + offset for i in range(len(tasks))],
            vals,
            width=width,
            label=MODE_LABELS.get(mode, mode),
            color=MODE_COLORS.get(mode, "#777777"),
        )
    ax.axhline(0, color="#111827", linewidth=0.9)
    ax.set_xticks(range(len(tasks)), tasks, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Cost Delta vs Normal (M token-equivalent)")
    ax.set_title("Per-task Cost Delta vs Normal")
    ax.grid(axis="y", alpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_task_cost_delta_vs_normal.png", bbox_inches="tight")
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
    modes = [m for m in MODE_ORDER if any(r["mode"] == m for r in strategy_rows)]
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=180)
    width = 0.18
    xs = list(range(len(strategies)))
    for i, mode in enumerate(modes):
        lookup = {(r["mode"], r["strategy"]): r for r in strategy_rows}
        vals = [safe_float(lookup.get((mode, s), {}).get("hits"), 0) for s in strategies]
        ax.bar(
            [x + (i - (len(modes) - 1) / 2) * width for x in xs],
            vals,
            width=width,
            label=MODE_LABELS.get(mode, mode),
            color=MODE_COLORS.get(mode, "#777777"),
        )
    ax.set_xticks(xs, strategies, rotation=25, ha="right")
    ax.set_ylabel("Hit Count")
    ax.set_title("Strategy Hit Counts")
    ax.grid(axis="y", alpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "strategy_hit_counts.png", bbox_inches="tight")
    plt.close(fig)


def plot_usage_quality(overall_rows: list[dict[str, Any]]) -> None:
    labels = [MODE_LABELS.get(r["mode"], r["mode"]) for r in overall_rows]
    ok = [safe_float(r.get("usage_ok_tasks"), 0) for r in overall_rows]
    missing = [safe_float(r.get("usage_missing_tasks"), 0) for r in overall_rows]
    partial = [safe_float(r.get("usage_estimated_tasks"), 0) for r in overall_rows]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=180)
    ax.bar(labels, ok, color="#59A14F", label="ok")
    ax.bar(labels, missing, bottom=ok, color="#E15759", label="missing")
    ax.bar(labels, partial, bottom=[a + b for a, b in zip(ok, missing)], color="#F28E2B", label="estimated")
    ax.set_ylabel("Task Count")
    ax.set_title("Usage Record Quality")
    ax.grid(axis="y", alpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "usage_quality.png", bbox_inches="tight")
    plt.close(fig)


def plot_proposals(proposal_rows: list[dict[str, Any]]) -> None:
    applied = [r for r in proposal_rows if r.get("status") == "applied"]
    if not applied:
        return
    kind_counts = Counter(r.get("kind", "unknown") for r in applied)
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
        manifest = load_json(path / "experiment_manifest.json", {})
        results = load_json(path / "results.json", [])
        summaries[mode] = summary
        results_by_mode[mode] = results
        overall = summary.get("overall", {})
        usage_counts = overall.get("usage_quality_counts", {})
        overall_rows.append(
            {
                "mode": mode,
                "label": MODE_LABELS.get(mode, mode),
                "run_dir": str(path.relative_to(ROOT)),
                "created_at": manifest.get("created_at", ""),
                "benchmark": manifest.get("benchmark", summary.get("benchmark", "")),
                "model": manifest.get("model_id", ""),
                "solver": manifest.get("solver", ""),
                "task_count": overall.get("tasks", len(results)),
                "success": overall.get("success", 0),
                "success_rate": overall.get("success_rate", 0),
                "avg_cost": overall.get("avg_cost", 0),
                "avg_tokens": overall.get("avg_tokens", 0),
                "cost_of_pass": overall.get("cost_of_pass", 0),
                "total_cost": overall.get("total_cost", 0),
                "input_tokens": overall.get("input_tokens", 0),
                "output_tokens": overall.get("output_tokens", 0),
                "avg_tool_calls": overall.get("avg_tool_calls", 0),
                "tool_calls": overall.get("tool_calls", 0),
                "turns": overall.get("turns", 0),
                "wall_time": overall.get("wall_time", 0),
                "zero_token_tasks": overall.get("zero_token_tasks", 0),
                "zero_cost_tasks": overall.get("zero_cost_tasks", 0),
                "token_record_quality": overall.get("token_record_quality", ""),
                "usage_ok_tasks": usage_counts.get("ok", 0),
                "usage_missing_tasks": usage_counts.get("missing", 0),
                "usage_estimated_tasks": usage_counts.get("estimated", 0),
                "prompt_fragment_visible_tasks": overall.get("prompt_fragment_visible_tasks", 0),
                "repeated_exact_inputs": overall.get("repeated_exact_inputs", 0),
                "repeated_test_inputs": overall.get("repeated_test_inputs", 0),
                "same_state_repeated_tests": overall.get("same_state_repeated_tests", 0),
                "post_edit_repeated_tests": overall.get("post_edit_repeated_tests", 0),
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
                    "score": row.get("score", 0),
                    "cost_units": row.get("cost_units", 0),
                    "total_tokens": row.get("total_tokens", 0),
                    "input_tokens": row.get("cumulative_input_tokens", row.get("input_tokens", 0)),
                    "output_tokens": row.get("cumulative_output_tokens", row.get("output_tokens", 0)),
                    "tool_calls": row.get("num_tool_calls", row.get("tool_calls", 0)),
                    "turns": row.get("num_turns", row.get("turns", 0)),
                    "elapsed": row.get("elapsed", row.get("solve_time", 0)),
                    "usage_record_quality": row.get("usage_record_quality", ""),
                    "status": row.get("status", ""),
                    "repeated_exact_inputs": bs.get("repeated_exact_inputs", 0),
                    "repeated_test_inputs": bs.get("repeated_test_inputs", 0),
                    "same_state_repeated_tests": bs.get("same_state_repeated_tests", 0),
                    "post_edit_repeated_tests": bs.get("post_edit_repeated_tests", 0),
                    "prompt_fragment_visible": bs.get("prompt_fragment_visible", False),
                    "failure_excerpt": short_failure(row),
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
        if not load_json(path / "emergence" / "proposal_adoption.json", []):
            proposals = read_jsonl(path / "emergence" / "proposals.jsonl")
            if proposals:
                proposal_rows.append(
                    {
                        "mode": mode,
                        "proposal_id": "",
                        "status": "none_applied",
                        "kind": "",
                        "categories": "",
                        "name": "",
                        "description": f"{len(proposals)} proposal rounds recorded, no adoption records.",
                        "trigger": "",
                        "expected_savings": "",
                        "correctness_risk": "",
                        "confidence": "",
                        "target": "",
                        "evidence": "",
                    }
                )
    overall_rows.sort(key=lambda r: MODE_ORDER.index(r["mode"]) if r["mode"] in MODE_ORDER else 999)
    return summaries, results_by_mode, overall_rows, task_rows, strategy_rows, proposal_rows


def build_failure_summary(task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in task_rows:
        if row["success"]:
            continue
        text = str(row.get("failure_excerpt", ""))
        if "No such file or directory" in text or "cannot open file" in text:
            kind = "missing artifact/file"
        elif "AssertionError" in text:
            kind = "assertion mismatch"
        elif "does not properly cancel" in text or "asyncio.gather" in text:
            kind = "async cancellation logic"
        elif "accuracy" in text or "Solver file" in text or "training_output" in text:
            kind = "training/config output mismatch"
        elif safe_float(row.get("cost_units"), 0) == 0 and str(row.get("usage_record_quality")) == "missing":
            kind = "early failure / missing usage"
        else:
            kind = "test failure"
        rows.append(
            {
                "mode": row["mode"],
                "task": row["task"],
                "failure_kind": kind,
                "cost_units": row["cost_units"],
                "tool_calls": row["tool_calls"],
                "excerpt": text,
            }
        )
    return rows


def build_report(
    selected_rows: list[dict[str, Any]],
    overall_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    strategy_rows: list[dict[str, Any]],
    proposal_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
) -> str:
    by_mode = {row["mode"]: row for row in overall_rows}
    normal = by_mode.get("normal-evolve", {})
    costaware = by_mode.get("cost-aware", {})
    predefined = by_mode.get("predefined-control", {})
    category = by_mode.get("category-seeded-cost-aware", {})
    best_success = max(overall_rows, key=lambda r: safe_float(r.get("success_rate"), -1))
    best_avg_cost = min(overall_rows, key=lambda r: safe_float(r.get("avg_cost"), math.inf))
    best_cop = min(overall_rows, key=lambda r: safe_float(r.get("cost_of_pass"), math.inf))
    best_calls = min(overall_rows, key=lambda r: safe_float(r.get("avg_tool_calls"), math.inf))

    overall_md = [
        {
            "mode": row["mode"],
            "run": Path(row["run_dir"]).name,
            "pass": f"{row['success']}/{row['task_count']}",
            "success_rate": pct(row["success_rate"]),
            "avg_cost": fmt_int(row["avg_cost"]),
            "cost_of_pass": fmt_int(row["cost_of_pass"]),
            "avg_tokens": fmt_int(row["avg_tokens"]),
            "avg_calls": fmt_num(row["avg_tool_calls"], 1),
            "usage": row["token_record_quality"],
        }
        for row in overall_rows
    ]
    selection_md = [
        {
            "mode": row["mode"],
            "created_at": row["created_at"],
            "selected": "yes" if row["selected"] else "no",
            "tasks": row["tasks"],
            "success_rate": pct(row["success_rate"]) if row["success_rate"] != "" else "",
            "run": Path(row["path"]).name,
        }
        for row in selected_rows
    ]
    failure_counts = Counter(r["failure_kind"] for r in failure_rows)
    fail_md = [{"failure_kind": k, "count": v} for k, v in failure_counts.most_common()]
    proposal_md = [
        {
            "mode": r["mode"],
            "status": r["status"],
            "kind": r["kind"],
            "categories": r["categories"] or "(none)",
            "name": r["name"] or r["description"],
            "confidence": r["confidence"],
        }
        for r in proposal_rows
    ]
    task_lookup = defaultdict(dict)
    for row in task_rows:
        task_lookup[row["task"]][row["mode"]] = row

    task_matrix_rows: list[dict[str, Any]] = []
    for task, per_mode in sorted(task_lookup.items()):
        task_matrix_rows.append(
            {
                "task": task,
                "normal": f"{'P' if per_mode.get('normal-evolve', {}).get('success') else 'F'} / {fmt_int(per_mode.get('normal-evolve', {}).get('cost_units', 0))}",
                "cost-aware": f"{'P' if per_mode.get('cost-aware', {}).get('success') else 'F'} / {fmt_int(per_mode.get('cost-aware', {}).get('cost_units', 0))}",
                "predefined": f"{'P' if per_mode.get('predefined-control', {}).get('success') else 'F'} / {fmt_int(per_mode.get('predefined-control', {}).get('cost_units', 0))}",
                "category": f"{'P' if per_mode.get('category-seeded-cost-aware', {}).get('success') else 'F'} / {fmt_int(per_mode.get('category-seeded-cost-aware', {}).get('cost_units', 0))}",
            }
        )

    best_task_cost_rows: list[dict[str, Any]] = []
    for task, per_mode in sorted(task_lookup.items()):
        successes = [(mode, row) for mode, row in per_mode.items() if row.get("success")]
        if not successes:
            continue
        mode, row = min(successes, key=lambda item: safe_float(item[1].get("cost_units"), math.inf))
        best_task_cost_rows.append({"task": task, "lowest_success_cost_mode": mode, "cost": fmt_int(row.get("cost_units", 0))})

    behavior_md = [
        {
            "mode": row["mode"],
            "repeated_inputs": row["repeated_exact_inputs"],
            "repeated_tests": row["repeated_test_inputs"],
            "same_state_tests": row["same_state_repeated_tests"],
            "post_edit_tests": row["post_edit_repeated_tests"],
            "prompt_visible_tasks": row["prompt_fragment_visible_tasks"],
        }
        for row in overall_rows
    ]
    props_by_mode = defaultdict(list)
    for row in proposal_rows:
        if row.get("status") == "applied":
            props_by_mode[row["mode"]].append(row)
    cat_prop_categories = Counter()
    for row in props_by_mode.get("category-seeded-cost-aware", []):
        for cat in row.get("categories", "").split(";"):
            if cat:
                cat_prop_categories[cat] += 1

    lines = [
        "# Terminal-Bench 2 Emergent Efficiency 最新实验分析报告",
        "",
        f"生成时间：2026-09-13。输入目录：`{RUN_ROOT.relative_to(ROOT)}`。选择规则：对每个 `mode` 使用 `experiment_manifest.created_at` 最新的 run；缺失时回退到目录时间戳。",
        "",
        "> 重要口径：本批实验的 `cost_units` 全部来自 `token_equivalent_fallback`，因此图表中的 cost 是 token-equivalent 成本，不是实际美元账单。本轮最新 `cost-aware` 已修复旧 run 的 usage missing 问题，10/10 任务 usage 均为 ok。",
        "",
        "## 1. 结论摘要",
        "",
        f"- **正确率最高的是 `{best_success['mode']}`**：{best_success['success']}/{best_success['task_count']} 通过，成功率 {pct(best_success['success_rate'])}。它仍是最强 correctness baseline，但平均成本为 {fmt_int(best_success['avg_cost'])} token-equivalent/task。",
        f"- **成本和工具调用最优的是 `category-seeded-cost-aware`**：平均成本 {fmt_int(category.get('avg_cost'))}、cost of pass {fmt_int(category.get('cost_of_pass'))}、平均工具调用 {fmt_num(category.get('avg_tool_calls'), 1)}，均为四组最低；但它只有 {category.get('success', 0)}/{category.get('task_count', 0)} 通过，低于 Normal 和 Predefined。",
        f"- **`predefined-control` 是本轮更平衡的非 Normal 方案**：8/10 通过，高于 CostAware/CategorySeeded 的 7/10；cost of pass {fmt_int(predefined.get('cost_of_pass'))}，低于 Normal 的 {fmt_int(normal.get('cost_of_pass'))}。",
        f"- **`cost-aware` 最新 run 已经可比较**：7/10 通过，usage 全 ok；但平均成本 {fmt_int(costaware.get('avg_cost'))}、工具调用 {fmt_num(costaware.get('avg_tool_calls'), 1)} 均高于 CategorySeeded，说明仅使用成本摘要能改善旧异常 run，但没有类别先验时动作压缩不够强。",
        f"- **proposal 生成呈现两层差异**：CostAware applied {len(props_by_mode.get('cost-aware', []))} 条 proposal，但无 M1-M7 类别标注；CategorySeeded applied {len(props_by_mode.get('category-seeded-cost-aware', []))} 条 proposal，并覆盖 {dict(cat_prop_categories)}，因此更适合做可解释的效率策略分析。",
        "",
        "## 2. 最新 Run 选择",
        "",
        md_table(selection_md, ["mode", "created_at", "selected", "tasks", "success_rate", "run"]),
        "",
        "## 3. 实验设置",
        "",
        "- Benchmark：Terminal-Bench 2 子集，10 个任务。",
        "- 任务集合：`adaptive-rejection-sampler`, `bn-fit-modify`, `break-filter-js-from-html`, `build-cython-ext`, `build-pmars`, `build-pov-ray`, `caffe-cifar-10`, `cancel-async-tasks`, `chess-best-move`, `circuit-fibsqrt`。",
        "- Solver：`react`；主模型和 evolver 模型均为 `deepseek:deepseek-v4-flash`。",
        "- 调度：最新 matrix run 均为 `batch_size=2`, `parallel=2`，共 5 个 batch，每个 batch 2 个任务。",
        "- 成本观测：记录 input/output token、工具调用、turn 数、wall time、重复命令/重复测试等行为信号；成本单位是 token-equivalent fallback。",
        "",
        "## 4. 总体对比",
        "",
        "![overall_dashboard](figures/overall_dashboard.png)",
        "",
        md_table(overall_md, ["mode", "run", "pass", "success_rate", "avg_cost", "cost_of_pass", "avg_tokens", "avg_calls", "usage"]),
        "",
        "![success_cost_frontier](figures/success_cost_frontier.png)",
        "",
        f"解读：Normal 位于最高成功率但高成本区域；CategorySeeded 位于最低成本/最低工具调用区域，但正确率只有 70%；Predefined 在成功率和成本之间更均衡；CostAware 与 CategorySeeded 成功率相同，但成本高约 {((safe_float(costaware.get('avg_cost')) / safe_float(category.get('avg_cost'))) - 1) * 100:.1f}%。",
        "",
        "## 5. Batch 动态",
        "",
        "![batch_trends](figures/batch_trends.png)",
        "",
        "- Normal：batch 1 为 1/2，其余 batch 均 2/2；batch 2 和 batch 4 成本峰值明显，是 9/10 正确率背后的主要成本来源。",
        "- CostAware：batch 1 为 1/2，batch 2/3/5 为 2/2，batch 4 为 0/2；batch 2 和 batch 5 成本较高。",
        "- Predefined：batch 1 为 1/2，batch 4 为 1/2，其余 batch 为 2/2；是除 Normal 外唯一达到 8/10 的模式。",
        "- CategorySeeded：batch 1 为 1/2，batch 4 为 0/2，其余 batch 为 2/2；成本曲线整体低于 CostAware/Normal。",
        "",
        "## 6. 任务级结果",
        "",
        "![task_success_matrix](figures/task_success_matrix.png)",
        "",
        "![per_task_cost_delta_vs_normal](figures/per_task_cost_delta_vs_normal.png)",
        "",
        md_table(task_matrix_rows, ["task", "normal", "cost-aware", "predefined", "category"]),
        "",
        "每个任务的最低成功成本归属：",
        "",
        md_table(best_task_cost_rows, ["task", "lowest_success_cost_mode", "cost"]),
        "",
        "关键案例：",
        "",
        "- `adaptive-rejection-sampler`：四组均失败，错误仍集中在 `ars.r` 缺失/入口文件命名问题，是最稳定的 hard case。",
        "- `cancel-async-tasks`：Normal 和 Predefined 通过，但 CostAware/CategorySeeded 失败；说明成本策略可能过早停止或过早收敛，未完整修复 asyncio cancellation 语义。",
        "- `caffe-cifar-10`：只有 Normal 通过，其他三组失败；这是区分高成本强探索和效率策略压缩的关键任务。",
        "- `build-cython-ext`：四组均通过，但 CategorySeeded 成本 2.20M、Predefined 2.60M，显著低于 Normal 13.10M 和 CostAware 10.60M，是效率策略最明显的成功案例。",
        "- `circuit-fibsqrt`：四组均通过，Normal 成本最低；说明效率策略并非每个任务都占优，部分任务 Normal 的自然搜索路径更短。",
        "",
        "## 7. 行为浪费与策略命中",
        "",
        "![behavior_waste_signals](figures/behavior_waste_signals.png)",
        "",
        "![strategy_hit_counts](figures/strategy_hit_counts.png)",
        "",
        "![usage_quality](figures/usage_quality.png)",
        "",
        md_table(behavior_md, ["mode", "repeated_inputs", "repeated_tests", "same_state_tests", "post_edit_tests", "prompt_visible_tasks"]),
        "",
        "行为层面的结论：CategorySeeded 的重复输入和平均工具调用最低，说明类别先验确实压缩了部分无效动作；CostAware 的 prompt fragment 在 8 个任务可见，但重复输入和调用仍高于 CategorySeeded；Predefined 的具体预定义策略仍大多没有 hit，更多像静态提示而不是强约束 runtime 行为。",
        "",
        "## 8. Emergent Strategy / Proposal 分析",
        "",
        "![proposal_summary](figures/proposal_summary.png)",
        "",
        md_table(proposal_md, ["mode", "status", "kind", "categories", "name", "confidence"]),
        "",
        "本轮 CostAware 与 CategorySeeded 都能从成本摘要生成策略，但侧重点不同：",
        "",
        "- CostAware 生成 `artifact_edit_checkpoint`, `bounded_recon_then_targeted_iteration` 等策略，重点是避免长时间只诊断不写文件，把搜索拉回 edit/test loop。",
        "- CategorySeeded 生成 `edit_first_checkpoint`, `reuse_prior_probe_output` 等策略，并且显式落到 M1-M5 成本类别上，便于解释策略针对的是推理、计划、工具调用、输出还是上下文。",
        "- 从结果看，CategorySeeded 的策略更能压低平均成本和工具调用；CostAware 的策略改善了旧 run 的异常问题，但总体成本仍偏高。",
        "",
        "## 9. 失败类型概览",
        "",
        md_table(fail_md, ["failure_kind", "count"]),
        "",
        "失败集中在三类：文件/入口缺失、异步取消语义、训练/构建配置输出。效率策略要继续提升 TB2，需要避免在这些任务上过早停止，同时保留足够诊断证据以完成语义修复。",
        "",
        "## 10. 建议",
        "",
        "1. 对 `adaptive-rejection-sampler` 做单独 case study，检查任务期望入口、文件名和执行路径，优先修复所有模式共同失败的问题。",
        "2. 对 `cancel-async-tasks` 比较 Normal/Predefined 的成功轨迹与 CostAware/CategorySeeded 的失败轨迹，定位效率策略在哪一步过早收敛。",
        "3. 将 CategorySeeded 的 5 条 proposal 做 ablation，重点测试 `edit_first_checkpoint` 和 `reuse_prior_probe_output` 是否直接带来成本下降。",
        "4. 对 CostAware 增加类别化输出要求，让其 proposal 也标注 M1-M7，便于和 CategorySeeded 公平比较策略可解释性。",
        "5. 报告主指标建议同时使用 success rate、cost of pass、avg tool calls；单看平均成本会高估低成功率模式。",
        "",
        "## 11. 产物索引",
        "",
        "- `run_selection.csv`：所有候选 run 与最新选择。",
        "- `overall_comparison.csv`：mode 级汇总。",
        "- `task_level_comparison.csv`：任务级 pass/cost/tool/失败摘要。",
        "- `behavior_comparison.csv`：重复行为指标。",
        "- `strategy_hits.csv`：策略命中统计。",
        "- `proposal_analysis.csv`：proposal adoption 与内容摘要。",
        "- `failure_summary.csv`：失败类型归纳。",
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
        }
        for r in overall_rows
    ]
    failure_rows = build_failure_summary(task_rows)

    write_csv(OUT_DIR / "run_selection.csv", selected_rows)
    write_csv(OUT_DIR / "overall_comparison.csv", overall_rows)
    write_csv(OUT_DIR / "task_level_comparison.csv", task_rows)
    write_csv(OUT_DIR / "behavior_comparison.csv", behavior_rows)
    write_csv(OUT_DIR / "strategy_hits.csv", strategy_rows)
    write_csv(OUT_DIR / "proposal_analysis.csv", proposal_rows)
    write_csv(OUT_DIR / "failure_summary.csv", failure_rows)

    plot_overall_dashboard(overall_rows)
    plot_frontier(overall_rows)
    plot_batch_trends(summaries)
    plot_task_success_matrix(results_by_mode)
    plot_cost_delta(results_by_mode)
    plot_behavior_waste(behavior_rows)
    plot_strategy_hits(strategy_rows)
    plot_usage_quality(overall_rows)
    plot_proposals(proposal_rows)

    report = build_report(selected_rows, overall_rows, task_rows, strategy_rows, proposal_rows, failure_rows)
    (OUT_DIR / "TB2_LATEST_ANALYSIS_CN.md").write_text(report, encoding="utf-8")
    print(f"Wrote {OUT_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
