#!/usr/bin/env python3
"""Generate matplotlib figures for the emergent efficiency documentation."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/a_evolve_matplotlib_cache")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
REPORT_DIR = ROOT / "logs" / "emergent_efficiency" / "five_run_emergent_efficiency_report_20260903_212446"
LOG_ROOT = ROOT / "logs" / "emergent_efficiency"
OUT = Path(__file__).resolve().parent / "figures"

GROUPS = [
    ("A", "Normal", "normal_evolve_s4", "#4C78A8"),
    ("B", "Cost-aware", "cost_aware_s4", "#F58518"),
    ("C", "Predefined", "predefined_s4", "#54A24B"),
    ("D", "Pre+Cost", "predefined_cost_aware_20260903_134010_785227", "#B279A2"),
    ("E", "Category", "category_seeded_cost_aware_20260903_195310_364868", "#E45756"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(value: str, default: float = 0.0) -> float:
    try:
        if str(value).lower() == "inf":
            return math.inf
        return float(value)
    except (TypeError, ValueError):
        return default


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_axis_off()
    boxes = [
        ("Seed\nworkspace", 0.04, 0.62, "#DCEBFA"),
        ("Task loader\nSWE mini", 0.20, 0.62, "#E8F4E1"),
        ("Parallel solve\nSweAgent", 0.36, 0.62, "#FDEBD0"),
        ("Docker eval\nPASS/FAIL", 0.52, 0.62, "#EDE7F6"),
        ("Usage profiler\ncost + behavior", 0.68, 0.62, "#FADBD8"),
        ("Run outputs\npatches/results", 0.84, 0.62, "#EAECEE"),
        ("Guided synthesis\nnormal evolution", 0.28, 0.20, "#D5F5E3"),
        ("Cost-aware proposer\nB/D/E only", 0.52, 0.20, "#FCF3CF"),
        ("Applied artifacts\nskills / prompts\ntools / memory", 0.755, 0.20, "#D6EAF8"),
    ]
    for label, x, y, color in boxes:
        width = 0.14 if label.startswith("Applied artifacts") else 0.12
        rect = plt.Rectangle((x, y), width, 0.16, facecolor=color, edgecolor="#333333", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + width / 2, y + 0.08, label, ha="center", va="center", fontsize=9.5, weight="bold")
    arrows = [
        ((0.16, 0.70), (0.20, 0.70)),
        ((0.32, 0.70), (0.36, 0.70)),
        ((0.48, 0.70), (0.52, 0.70)),
        ((0.64, 0.70), (0.68, 0.70)),
        ((0.80, 0.70), (0.84, 0.70)),
        ((0.58, 0.62), (0.36, 0.36)),
        ((0.74, 0.62), (0.58, 0.36)),
        ((0.64, 0.28), (0.76, 0.28)),
        ((0.40, 0.28), (0.76, 0.28)),
        ((0.82, 0.36), (0.10, 0.62)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#333333"})
    ax.text(0.5, 0.94, "Emergent Efficiency Pipeline", ha="center", va="center", fontsize=18, weight="bold")
    ax.text(0.5, 0.05, "Workspace is the interface: agent reads it, evolver/proposer mutates it between batches.", ha="center", fontsize=11)
    save(fig, "pipeline_architecture.png")


def plot_design_matrix() -> None:
    rows = ["A Normal", "B Cost-aware", "C Predefined", "D Pre+Cost", "E Category"]
    cols = ["Normal\nevolution", "Cost\nsignals", "Predefined\nstrategies", "M1-M7\ncategories", "Writes\nworkspace"]
    data = [
        [1, 0, 0, 0, 1],
        [1, 1, 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 1, 0, 1, 1],
    ]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.imshow(data, cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    for i, row in enumerate(data):
        for j, value in enumerate(row):
            ax.text(j, i, "yes" if value else "no", ha="center", va="center", fontsize=10, weight="bold")
    ax.set_title("Experiment Design Matrix", fontsize=15, weight="bold")
    save(fig, "experiment_design_matrix.png")


def plot_overall() -> None:
    rows = read_csv(REPORT_DIR / "overall_comparison.csv")
    labels = [r["label"].split("_", 1)[0] for r in rows]
    names = ["A Normal", "B Cost-aware", "C Predefined", "D Pre+Cost", "E Category"]
    colors = [GROUPS[i][3] for i in range(len(rows))]
    success = [num(r["success_rate"]) for r in rows]
    cost = [num(r["avg_cost"]) for r in rows]
    cost_pass = [num(r["cost_of_pass"]) for r in rows]
    tokens = [num(r["avg_tokens"]) / 1000 for r in rows]
    tools = [num(r["avg_tool_calls"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.scatter(cost_pass, success, s=[180 + t * 2 for t in tools], c=colors, edgecolor="#333333")
    for label, x, y in zip(labels, cost_pass, success):
        ax.annotate(label, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=10, weight="bold")
    ax.set_xlabel("Cost of pass ($ / solved task)")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0.72, 1.04)
    ax.grid(alpha=0.25)
    ax.set_title("Efficiency Frontier: success vs cost of pass", fontsize=14, weight="bold")
    save(fig, "success_cost_frontier.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].bar(labels, tokens, color=colors)
    axes[0].set_ylabel("Avg tokens / task (K)")
    axes[0].set_title("Token load")
    axes[1].bar(labels, cost, color=colors)
    axes[1].set_ylabel("Avg cost / task ($)")
    axes[1].set_title("Cost")
    axes[2].bar(labels, tools, color=colors)
    axes[2].set_ylabel("Avg tool calls / task")
    axes[2].set_title("Tool calls")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="x", rotation=20)
    save(fig, "tokens_cost_tools.png")

    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], cost, width=0.4, label="avg cost", color="#72B7B2")
    ax.bar([i + 0.2 for i in x], cost_pass, width=0.4, label="cost of pass", color="#E45756")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("$")
    ax.set_title("Cost metrics by group", fontsize=14, weight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save(fig, "cost_metrics.png")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    y = range(len(labels))
    ax.barh(y, [num(r["delta_avg_tokens_pct_vs_A"]) for r in rows], color=colors)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(list(y), names)
    ax.set_xlabel("Delta avg tokens vs A (%)")
    ax.set_title("Token reduction relative to normal evolution", fontsize=14, weight="bold")
    ax.grid(axis="x", alpha=0.25)
    save(fig, "delta_tokens_vs_a.png")


def plot_task_matrix() -> None:
    rows = read_csv(REPORT_DIR / "task_matrix.csv")
    group_keys = [
        "A_normal_self_evolve",
        "B_cost_aware_emergence",
        "C_predefined_control",
        "D_predefined_cost_aware_latest",
        "E_category_seeded_latest",
    ]
    data = []
    for g in group_keys:
        data.append([1 if str(r.get(f"{g}_success", "")).lower() == "true" else 0 for r in rows])
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(group_keys)), ["A", "B", "C", "D", "E"])
    ax.set_xticks(range(len(rows)), [r["instance_id"].replace("django__django-", "") for r in rows], rotation=45)
    for i, row in enumerate(data):
        for j, value in enumerate(row):
            ax.text(j, i, "P" if value else "F", ha="center", va="center", fontsize=9, weight="bold")
    ax.set_title("Per-task pass/fail matrix", fontsize=14, weight="bold")
    ax.set_xlabel("Django instance suffix")
    save(fig, "task_success_matrix.png")


def read_group_summaries() -> dict[str, list[dict[str, str]]]:
    out = {}
    for letter, _, run_dir, _ in GROUPS:
        rows = read_csv(LOG_ROOT / run_dir / "summary.csv")
        out[letter] = [r for r in rows if "_batch_" in r["group"]]
    return out


def plot_batch_trends() -> None:
    summaries = read_group_summaries()
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9), sharex=True)
    for letter, name, _, color in GROUPS:
        rows = summaries[letter]
        batches = list(range(1, len(rows) + 1))
        axes[0].plot(batches, [num(r["success_rate"]) for r in rows], marker="o", label=letter, color=color)
        axes[1].plot(batches, [num(r["avg_cost"]) for r in rows], marker="o", label=letter, color=color)
        axes[2].plot(batches, [num(r["avg_tool_calls"]) for r in rows], marker="o", label=letter, color=color)
    axes[0].set_ylabel("Success rate")
    axes[1].set_ylabel("Avg cost")
    axes[2].set_ylabel("Avg tool calls")
    axes[2].set_xlabel("Batch")
    axes[0].set_title("Batch-level trends", fontsize=14, weight="bold")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(ncol=5, fontsize=9)
    save(fig, "batch_trends.png")


def plot_behavior_and_proposals() -> None:
    rows = read_csv(REPORT_DIR / "overall_comparison.csv")
    labels = [r["label"].split("_", 1)[0] for r in rows]
    colors = [GROUPS[i][3] for i in range(len(rows))]
    metrics = ["repeated_exact_inputs", "repeated_test_inputs", "same_state_repeated_tests"]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    width = 0.24
    x = list(range(len(labels)))
    for k, metric in enumerate(metrics):
        ax.bar([i + (k - 1) * width for i in x], [num(r[metric]) for r in rows], width=width, label=metric)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Event count")
    ax.set_title("Repeated-action behavior signals", fontsize=14, weight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save(fig, "behavior_signals.png")

    proposals = read_csv(REPORT_DIR / "proposal_catalog.csv")
    kind_counts: dict[str, int] = {}
    cat_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for row in proposals:
        status_counts[row["adoption_status"] or "not_applied"] = status_counts.get(row["adoption_status"] or "not_applied", 0) + 1
        if row["adoption_status"] == "applied":
            kind_counts[row["kind"]] = kind_counts.get(row["kind"], 0) + 1
            for cat in row["categories"].split(";"):
                cat = cat.strip()
                if cat:
                    cat_counts[cat] = cat_counts.get(cat, 0) + 1

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, counts, title, color in [
        (axes[0], status_counts, "Proposal status", "#4C78A8"),
        (axes[1], kind_counts, "Applied artifact kind", "#F58518"),
        (axes[2], cat_counts, "Applied M-categories", "#54A24B"),
    ]:
        names = list(counts) or ["none"]
        values = [counts.get(n, 0) for n in names]
        ax.bar(names, values, color=color)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    save(fig, "proposal_summary.png")


def plot_category_taxonomy() -> None:
    cats = ["M1\nContext", "M2\nReasoning", "M3\nOutput", "M4\nPlanning", "M5\nTools", "M6\nCompilation", "M7\nRouting"]
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.set_axis_off()
    for i, cat in enumerate(cats):
        x = 0.04 + i * 0.135
        rect = plt.Rectangle((x, 0.35), 0.105, 0.28, facecolor=plt.cm.Set3(i), edgecolor="#333333")
        ax.add_patch(rect)
        ax.text(x + 0.0525, 0.49, cat, ha="center", va="center", fontsize=10, weight="bold")
        if i < len(cats) - 1:
            ax.annotate("", xy=(x + 0.13, 0.49), xytext=(x + 0.108, 0.49), arrowprops={"arrowstyle": "->", "lw": 1})
    ax.text(0.5, 0.82, "Weak efficiency taxonomy exposed only to group E", ha="center", fontsize=15, weight="bold")
    ax.text(0.5, 0.17, "Categories are labels for cost dimensions, not executable predefined strategies.", ha="center", fontsize=11)
    save(fig, "m1_m7_taxonomy.png")


def main() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    plot_pipeline()
    plot_design_matrix()
    plot_category_taxonomy()
    plot_overall()
    plot_task_matrix()
    plot_batch_trends()
    plot_behavior_and_proposals()


if __name__ == "__main__":
    main()
