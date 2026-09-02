"""Visualization helpers for EvoEfficient experiment reports."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def write_summary_table(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "group",
        "strategies",
        "tasks",
        "success",
        "success_rate",
        "avg_tokens",
        "avg_tokens_success",
        "avg_cost",
        "avg_cost_success",
        "cost_of_pass",
        "main_model_cost",
        "route_model_cost",
        "easy_model_cost",
        "hard_model_cost",
        "cache_read_input_tokens",
        "cache_write_input_tokens",
        "priced_usage_tasks",
        "tool_calls",
        "avg_tool_calls",
        "wall_time",
        "zero_token_tasks",
        "zero_cost_tasks",
        "estimated_usage_tasks",
        "token_record_quality",
    ]
    lines = [",".join(headers)]
    for row in rows:
        values = []
        for h in headers:
            value = row.get(h, "")
            if h == "strategies" and isinstance(value, list):
                value = ";".join(str(v) for v in value)
            values.append(str(value).replace(",", ";"))
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(rows: list[dict[str, Any]], output_dir: str | Path) -> dict[str, str]:
    """Write simple PNG plots. Matplotlib is already a project dependency."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))

    import matplotlib.pyplot as plt
    groups = [str(r["group"]) for r in rows]
    avg_tokens = [float(r.get("avg_tokens", 0) or 0) for r in rows]
    avg_cost = [
        float(r.get("avg_cost", r.get("avg_tokens", 0)) or 0)
        for r in rows
    ]
    success = [float(r.get("success_rate", 0) or 0) for r in rows]
    tool_calls = [float(r.get("avg_tool_calls", 0) or 0) for r in rows]

    paths: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 1.1), 4.8))
    ax.bar(groups, avg_tokens, color="#4C78A8")
    ax.set_ylabel("Average tokens per task")
    ax.set_title("EvoEfficient Strategy Cost Comparison")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    cost_path = out / "strategy_avg_tokens.png"
    fig.savefig(cost_path, dpi=160)
    plt.close(fig)
    paths["avg_tokens"] = str(cost_path)

    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 1.1), 4.8))
    ax.bar(groups, avg_cost, color="#59A14F")
    ax.set_ylabel("Average cost units per task")
    ax.set_title("EvoEfficient Strategy Full Cost Comparison")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    avg_cost_path = out / "strategy_avg_cost.png"
    fig.savefig(avg_cost_path, dpi=160)
    plt.close(fig)
    paths["avg_cost"] = str(avg_cost_path)

    fig, ax1 = plt.subplots(figsize=(max(8, len(groups) * 1.1), 4.8))
    ax1.bar(groups, avg_cost, color="#72B7B2", label="avg cost")
    ax1.set_ylabel("Average cost units")
    ax1.tick_params(axis="x", rotation=35)
    ax2 = ax1.twinx()
    ax2.plot(groups, success, color="#E45756", marker="o", label="success rate")
    ax2.set_ylabel("Success rate")
    ax2.set_ylim(0, 1.05)
    ax1.set_title("Cost vs Success")
    fig.tight_layout()
    pareto_path = out / "strategy_cost_success.png"
    fig.savefig(pareto_path, dpi=160)
    plt.close(fig)
    paths["cost_success"] = str(pareto_path)

    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 1.1), 4.8))
    ax.bar(groups, tool_calls, color="#F58518")
    ax.set_ylabel("Average tool calls per task")
    ax.set_title("Tool-Call Efficiency")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    tool_path = out / "strategy_tool_calls.png"
    fig.savefig(tool_path, dpi=160)
    plt.close(fig)
    paths["tool_calls"] = str(tool_path)

    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 1.1), 5.2))
    ax.scatter(avg_cost, success, s=80, color="#54A24B")
    for group, x, y in zip(groups, avg_cost, success):
        ax.annotate(group, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Average cost units per task")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Efficiency Frontier: Lower Cost, Higher Success")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    frontier_path = out / "strategy_efficiency_frontier.png"
    fig.savefig(frontier_path, dpi=160)
    plt.close(fig)
    paths["efficiency_frontier"] = str(frontier_path)

    (out / "plots.json").write_text(json.dumps(paths, indent=2), encoding="utf-8")
    return paths


def write_strategy_library_plot(rows: list[dict[str, Any]], output_dir: str | Path) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))

    import matplotlib.pyplot as plt
    names: list[str] = []
    hits: list[float] = []
    for row in rows:
        strategies = row.get("strategy_hits", {}) or {}
        for name, data in sorted(strategies.items()):
            if not isinstance(data, dict):
                continue
            names.append(f"{row['group']}:{name}")
            hits.append(float(data.get("hits", 0) or 0))
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.35), 4.8))
    ax.bar(names, hits, color="#9C755F")
    ax.set_ylabel("Strategy hits")
    ax.set_title("Strategy Hits by Group")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    path = out / "strategy_hits.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)
