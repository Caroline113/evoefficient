#!/usr/bin/env python3
"""Build a paired four-mode Terminal-Bench 2.1 comparison report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODE_ORDER = [
    "normal-evolve",
    "cost-aware",
    "category-seeded-cost-aware",
    "predefined-control",
]
SHORT = {
    "normal-evolve": "Normal",
    "cost-aware": "Cost-aware",
    "category-seeded-cost-aware": "Category-seeded",
    "predefined-control": "Predefined",
}
COLORS = {
    "normal-evolve": "#64748b",
    "cost-aware": "#2563eb",
    "category-seeded-cost-aware": "#16a34a",
    "predefined-control": "#f59e0b",
}


def pct(value: float, base: float) -> float:
    return (value / base - 1.0) * 100.0 if base else float("nan")


def fmt_delta(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:+.1f}%"


def load_run(path: Path) -> dict:
    manifest = json.loads((path / "experiment_manifest.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    results = json.loads((path / "results.json").read_text())
    behavior = json.loads((path / "behavior_stats.json").read_text())
    return {
        "path": path,
        "manifest": manifest,
        "overall": summary["overall"],
        "batches": summary["by_batch"],
        "results": results,
        "behavior": behavior,
    }


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal", required=True, type=Path)
    parser.add_argument("--cost-aware", required=True, type=Path)
    parser.add_argument("--category-seeded", required=True, type=Path)
    parser.add_argument("--predefined", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    paths = {
        "normal-evolve": args.normal,
        "cost-aware": args.cost_aware,
        "category-seeded-cost-aware": args.category_seeded,
        "predefined-control": args.predefined,
    }
    runs = {mode: load_run(path) for mode, path in paths.items()}
    out = args.output_dir
    figures = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    baseline_tasks = runs["normal-evolve"]["manifest"]["tasks"]
    for mode, run in runs.items():
        assert run["manifest"]["benchmark"] == "terminal_bench_2_1"
        assert run["manifest"]["tasks"] == baseline_tasks, f"task mismatch: {mode}"
        assert len(run["results"]) == len(baseline_tasks)

    rows = []
    for mode in MODE_ORDER:
        o = runs[mode]["overall"]
        tasks = int(o["tasks"])
        success = int(o["success"])
        rows.append({
            "mode": mode,
            "tasks": tasks,
            "success": success,
            "success_rate": float(o["success_rate"]),
            "tokens": int(o["tokens"]),
            "input_tokens": int(o["input_tokens"]),
            "output_tokens": int(o["output_tokens"]),
            "cache_read_tokens": int(o["cache_read_input_tokens"]),
            "cache_write_tokens": int(o["cache_write_input_tokens"]),
            "cost_usd": float(o["total_cost"]),
            "tool_calls": int(o["tool_calls"]),
            "turns": int(o["turns"]),
            "wall_time_s": float(o["wall_time"]),
            "cost_per_success": float(o["total_cost"]) / success if success else np.nan,
            "tokens_per_success": int(o["tokens"]) / success if success else np.nan,
            "cache_share": int(o["cache_read_input_tokens"]) /
                (int(o["input_tokens"]) + int(o["cache_read_input_tokens"]))
                if int(o["input_tokens"]) + int(o["cache_read_input_tokens"]) else 0,
            "repeated_exact_inputs": int(runs[mode]["behavior"]["overall"]["repeated_exact_inputs"]),
            "prompt_fragment_visible_tasks": int(runs[mode]["behavior"]["overall"]["prompt_fragment_visible_tasks"]),
            "zero_token_tasks": int(o["zero_token_tasks"]),
        })
    overall = pd.DataFrame(rows).set_index("mode")
    base = overall.loc["normal-evolve"]
    for col in ["tokens", "input_tokens", "output_tokens", "cache_read_tokens", "cost_usd",
                "tool_calls", "turns", "wall_time_s", "cost_per_success", "tokens_per_success"]:
        overall[f"{col}_delta_pct"] = [pct(v, base[col]) for v in overall[col]]
    overall["success_rate_delta_pp"] = (overall["success_rate"] - base["success_rate"]) * 100
    overall.to_csv(out / "overall_comparison.csv", float_format="%.6f")

    by_task = {}
    for mode in MODE_ORDER:
        by_task[mode] = {r["task_name"]: r for r in runs[mode]["results"]}
    task_rows = []
    for task in baseline_tasks:
        row = {"task": task}
        baseline_cost = float(by_task["normal-evolve"][task].get("cost_units", 0) or 0)
        for mode in MODE_ORDER:
            result = by_task[mode][task]
            cost = float(result.get("cost_units", 0) or 0)
            row[f"{mode}_success"] = bool(result.get("success"))
            row[f"{mode}_tokens"] = int(result.get("total_tokens", 0) or 0)
            row[f"{mode}_cache_read_tokens"] = int(result.get("cumulative_cache_read_input_tokens", 0) or 0)
            row[f"{mode}_cost_usd"] = cost
            row[f"{mode}_cost_delta_pct"] = pct(cost, baseline_cost)
        task_rows.append(row)
    task_df = pd.DataFrame(task_rows)
    task_df.to_csv(out / "per_task_comparison.csv", index=False, float_format="%.6f")

    paired_rows = []
    normal_success = task_df["normal-evolve_success"]
    for mode in MODE_ORDER:
        current = task_df[f"{mode}_success"]
        paired_rows.append({
            "mode": mode,
            "gained_vs_normal": int((~normal_success & current).sum()),
            "lost_vs_normal": int((normal_success & ~current).sum()),
            "both_pass": int((normal_success & current).sum()),
            "both_fail": int((~normal_success & ~current).sum()),
        })
    paired = pd.DataFrame(paired_rows).set_index("mode")
    paired.to_csv(out / "paired_success_changes.csv")

    batch_rows = []
    for mode in MODE_ORDER:
        for index, batch in enumerate(runs[mode]["batches"], 1):
            batch_rows.append({
                "mode": mode, "batch": index,
                "success_rate": batch["success_rate"], "tokens": batch["tokens"],
                "cost_usd": batch["total_cost"], "tool_calls": batch["tool_calls"],
                "wall_time_s": batch["wall_time"],
            })
    batches = pd.DataFrame(batch_rows)
    batches.to_csv(out / "batch_comparison.csv", index=False, float_format="%.6f")

    # Figure 1: all headline metrics in one normalized chart.  Normal=100
    # makes increases/decreases directly comparable despite different units.
    fig, ax = plt.subplots(figsize=(14, 7))
    configs = [
        ("success_rate", "Success rate", lambda v: f"{v*100:.0f}%", "success_rate_delta_pp", " pp"),
        ("cost_usd", "Total cost", lambda v: f"${v:.2f}", "cost_usd_delta_pct", "%"),
        ("tokens", "Tokens", lambda v: f"{v/1e6:.2f}M", "tokens_delta_pct", "%"),
        ("tool_calls", "Tool calls", lambda v: f"{int(v):,}", "tool_calls_delta_pct", "%"),
    ]
    x = np.arange(len(configs))
    width = 0.19
    for mode_index, mode in enumerate(MODE_ORDER):
        heights = [overall.loc[mode, metric] / base[metric] * 100 for metric, *_ in configs]
        positions = x + (mode_index - 1.5) * width
        bars = ax.bar(positions, heights, width, color=COLORS[mode], label=SHORT[mode])
        for bar, height, (metric, _, formatter, delta_col, unit) in zip(bars, heights, configs):
            if mode == "normal-evolve":
                delta_text = "base"
            else:
                delta_text = f"{overall.loc[mode, delta_col]:+.1f}{unit}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1.2,
                f"{formatter(overall.loc[mode, metric])}\n{delta_text}",
                ha="center", va="bottom", fontsize=8, rotation=0,
            )
    ax.axhline(100, color="#334155", linewidth=1.2, linestyle="--", label="Normal baseline")
    ax.set_xticks(x, [title for _, title, *_ in configs])
    ax.set_ylabel("Relative index (Normal = 100)")
    ax.set_title("Overall metrics and change relative to normal-evolve")
    ax.set_ylim(0, max(128, ax.get_ylim()[1] + 5))
    ax.grid(axis="y", alpha=.25)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    savefig(figures / "01_overall_metrics.png")

    # Figure 2: success/cost efficiency frontier.
    fig, ax = plt.subplots(figsize=(9, 6))
    for mode in MODE_ORDER:
        ax.scatter(overall.loc[mode, "cost_usd"], overall.loc[mode, "success_rate"] * 100,
                   s=170, color=COLORS[mode], label=SHORT[mode], zorder=3)
        ax.annotate(SHORT[mode], (overall.loc[mode, "cost_usd"], overall.loc[mode, "success_rate"] * 100),
                    xytext=(7, 7), textcoords="offset points")
    ax.set_xlabel("Total cost (USD, lower is better)")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Success–cost trade-off")
    ax.grid(alpha=.3)
    savefig(figures / "02_success_cost_frontier.png")

    # Figure 3: batch trajectories.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for mode in MODE_ORDER:
        data = batches[batches["mode"] == mode]
        axes[0].plot(data.batch, data.success_rate * 100, marker="o", color=COLORS[mode], label=SHORT[mode])
        axes[1].plot(data.batch, data.cost_usd, marker="o", color=COLORS[mode], label=SHORT[mode])
    axes[0].set(title="Success rate by batch", xlabel="Batch", ylabel="Success rate (%)", xticks=range(1, 6))
    axes[1].set(title="Cost by batch", xlabel="Batch", ylabel="USD", xticks=range(1, 6))
    for ax in axes: ax.grid(alpha=.3)
    axes[0].legend(ncol=2, fontsize=9)
    savefig(figures / "03_batch_trajectories.png")

    # Figure 4: task-level pass/fail matrix.
    matrix = np.array([[int(by_task[m][t].get("success", False)) for m in MODE_ORDER] for t in baseline_tasks])
    fig, ax = plt.subplots(figsize=(9, 10))
    ax.imshow(matrix, cmap=matplotlib.colors.ListedColormap(["#ef4444", "#22c55e"]), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(MODE_ORDER)), [SHORT[m] for m in MODE_ORDER], rotation=20, ha="right")
    ax.set_yticks(range(len(baseline_tasks)), baseline_tasks, fontsize=8)
    ax.set_title("Per-task outcomes (green=pass, red=fail)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]): ax.text(j, i, "P" if matrix[i,j] else "F", ha="center", va="center", color="white", fontsize=8)
    savefig(figures / "04_task_outcomes.png")

    # Figure 5: cost composition under the configured token prices.
    component_rows = []
    for mode in MODE_ORDER:
        rs = runs[mode]["results"]
        component_rows.append({
            "mode": mode,
            "input": sum(float(r.get("cumulative_input_tokens",0))*float(r.get("input_price_per_1m",0))/1e6 for r in rs),
            "output": sum(float(r.get("cumulative_output_tokens",0))*float(r.get("output_price_per_1m",0))/1e6 for r in rs),
            "cache_read": sum(float(r.get("cumulative_cache_read_input_tokens",0))*float(r.get("cache_read_price_per_1m",0))/1e6 for r in rs),
            "cache_write": sum(float(r.get("cumulative_cache_write_input_tokens",0))*float(r.get("cache_write_price_per_1m",0))/1e6 for r in rs),
        })
    components = pd.DataFrame(component_rows).set_index("mode")
    components.to_csv(out / "cost_components.csv", float_format="%.6f")
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(MODE_ORDER))
    for col, color in [("input","#60a5fa"),("output","#a78bfa"),("cache_read","#f97316"),("cache_write","#94a3b8")]:
        vals = components.loc[MODE_ORDER,col].to_numpy()
        ax.bar(x, vals, bottom=bottom, label=col.replace("_"," "), color=color)
        bottom += vals
    ax.set_xticks(x, [SHORT[m] for m in MODE_ORDER], rotation=15, ha="right")
    ax.set_ylabel("USD")
    ax.set_title("Cost composition")
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    savefig(figures / "05_cost_composition.png")

    # Figure 6: behavior and interaction volume, normalized to baseline.
    fig, ax = plt.subplots(figsize=(11, 6))
    metrics = ["tool_calls", "turns", "repeated_exact_inputs", "wall_time_s"]
    width = .19
    for idx, mode in enumerate(MODE_ORDER):
        vals = [overall.loc[mode,m] / base[m] * 100 if base[m] else 0 for m in metrics]
        ax.bar(np.arange(len(metrics)) + (idx-1.5)*width, vals, width, color=COLORS[mode], label=SHORT[mode])
    ax.axhline(100, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(range(len(metrics)), [m.replace("_"," ") for m in metrics])
    ax.set_ylabel("Baseline = 100")
    ax.set_title("Operational behavior relative to normal-evolve")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=.25)
    savefig(figures / "06_behavior_relative.png")

    # Figure 7: distinguish the shared normal-evolution layer from optional
    # cost-aware proposal modules.  Older runs may contain solver drafts that
    # were not parseable by GuidedSynthesis; keeping all three counts visible
    # prevents "proposal generated" from being mistaken for "proposal applied".
    evolution_rows = []
    for mode in MODE_ORDER:
        emergence = runs[mode]["path"] / "emergence"
        normal_records = []
        normal_path = emergence / "normal_evolution.jsonl"
        if normal_path.exists():
            normal_records = [
                json.loads(line) for line in normal_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        adoption_path = emergence / "proposal_adoption.json"
        adoptions = json.loads(adoption_path.read_text(encoding="utf-8")) if adoption_path.exists() else []
        evolution_rows.append({
            "mode": mode,
            "solver_proposals": sum(int(r.get("solver_proposals", 0) or 0) for r in normal_records),
            "normal_parsed": sum(int(r.get("result", {}).get("proposals", 0) or 0) for r in normal_records),
            "normal_applied": sum(len(r.get("result", {}).get("applied", []) or []) for r in normal_records),
            "module_applied": sum(1 for r in adoptions if str(r.get("status")) == "applied"),
        })
    evolution = pd.DataFrame(evolution_rows).set_index("mode")
    evolution.to_csv(out / "evolution_activity.csv")
    fig, ax = plt.subplots(figsize=(11, 6))
    evo_metrics = ["solver_proposals", "normal_parsed", "normal_applied", "module_applied"]
    labels = ["Solver drafts", "Normal parsed", "Normal applied", "Extra module applied"]
    width = .19
    for idx, mode in enumerate(MODE_ORDER):
        vals = evolution.loc[mode, evo_metrics].to_numpy(dtype=float)
        bars = ax.bar(np.arange(len(evo_metrics)) + (idx-1.5)*width, vals, width,
                      color=COLORS[mode], label=SHORT[mode])
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
    ax.set_xticks(range(len(evo_metrics)), labels)
    ax.set_ylabel("Count")
    ax.set_title("Evolution activity by layer")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=.25)
    savefig(figures / "07_evolution_activity.png")

    # Markdown report.
    metric_specs = [
        ("Success", "success", lambda x:f"{int(x)}/20", None),
        ("Success rate", "success_rate", lambda x:f"{x*100:.1f}%", "success_rate_delta_pp"),
        ("Tokens", "tokens", lambda x:f"{int(x):,}", "tokens_delta_pct"),
        ("Cache read", "cache_read_tokens", lambda x:f"{int(x):,}", "cache_read_tokens_delta_pct"),
        ("Cost", "cost_usd", lambda x:f"${x:.4f}", "cost_usd_delta_pct"),
        ("Cost / pass", "cost_per_success", lambda x:f"${x:.4f}", "cost_per_success_delta_pct"),
        ("Tool calls", "tool_calls", lambda x:f"{int(x):,}", "tool_calls_delta_pct"),
        ("Turns", "turns", lambda x:f"{int(x):,}", "turns_delta_pct"),
        ("Wall time", "wall_time_s", lambda x:f"{x/3600:.2f}h", "wall_time_s_delta_pct"),
    ]
    lines = ["# Terminal-Bench 2.1 四组涌现效率实验对比报告", "",
             "## 1. 实验与数据完整性", "",
             f"四组实验使用完全相同的 {len(baseline_tasks)} 个任务、batch size=4、parallel=2，主模型与 evolver 均为 `deepseek:deepseek-v4-flash`。每组均有 20/20 个任务使用 USD 价格表（`profile:route:hard`），因此 cost 可以直接比较。", "",
             "相对变化均以 **normal-evolve** 为基线。成功率使用百分点（pp）；其余指标使用相对百分比。负值表示数值下降，不自动代表更好（成功率越高越好，成本/token/tool calls 越低通常越好）。", "",
             "## 2. 核心结果", "",
             "| 指标 | Normal | Cost-aware（相对值） | Category-seeded（相对值） | Predefined（相对值） |", "|---|---:|---:|---:|---:|"]
    for label, metric, formatter, delta_col in metric_specs:
        cells = [label, formatter(overall.loc["normal-evolve",metric])]
        for mode in MODE_ORDER[1:]:
            value = formatter(overall.loc[mode,metric])
            if delta_col == "success_rate_delta_pp": delta = f"{overall.loc[mode,delta_col]:+.1f} pp"
            elif delta_col: delta = fmt_delta(overall.loc[mode,delta_col])
            else: delta = "—"
            cells.append(f"{value} ({delta})")
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "![总体指标](figures/01_overall_metrics.png)", "", "![成功率与成本](figures/02_success_cost_frontier.png)", "",
              "## 3. 主要发现", "",
              "- **最高成功率：category-seeded-cost-aware，18/20（90%）**，比 normal 高 10 pp；总成本 $80.6823，下降 26.1%；每成功任务成本约 $4.4824，下降 34.3%。",
              "- **最低总成本：cost-aware，$62.5781**，比 normal 下降 42.7%，但成功率降至 70%（-10 pp）。因此它更省钱，但可靠性明显较弱。",
              "- **最低非缓存 token：predefined-control，1,104,032**，比 normal 下降 9.3%；成功率为 85%（+5 pp），总成本下降 35.1%，是较均衡的方案。",
              "- normal-evolve 的总成本最高（$109.2166）。主要原因不是非缓存 token，而是 cache-read token 达到 70.63M；其 cache-read 成本占总成本的大部分。",
              "- 四组非缓存 token 总量很接近（最大差约 9%），但成本差异达到 43%。在当前价格表下，cache read 的规模是成本差异的核心驱动。", "",
              "## 4. 配对任务结果", "",
              "| 模式 | 相对 Normal 新增通过 | 相对 Normal 丢失通过 | 两者都通过 | 两者都失败 |", "|---|---:|---:|---:|---:|"]
    for mode in MODE_ORDER[1:]:
        p = paired.loc[mode]
        lines.append(f"| {mode} | {p.gained_vs_normal} | {p.lost_vs_normal} | {p.both_pass} | {p.both_fail} |")
    lines += ["", "![逐任务结果](figures/04_task_outcomes.png)", "",
              "配对比较说明，整体成功率变化并非所有任务一致变好或变差，而是不同模式在具体任务上发生成功/失败交换。完整逐任务状态、成本及相对成本变化见 `per_task_comparison.csv`。", "",
              "## 5. 分批趋势与进化过程", "", "![分批趋势](figures/03_batch_trajectories.png)", "",
              "- normal 在第 2 批出现明显成本尖峰（$82.91），主要由高 cache 使用和长轨迹任务推动；后两个 batch 均达到 100% 成功。",
              "- category-seeded 在第 2、3、5 批达到 100%，但第 3 批成本最高；第 4 批耗时较长但模型成本很低，说明 wall time 与 token cost 不完全同步。",
              "- cost-aware 五个 batch 的成功率为 75%、75%、75%、50%、75%，省成本效果稳定，但没有转化为更高成功率。",
              "- predefined-control 在第 2、3、5 批均为 100%，后期成本也较低，呈现较好的综合稳定性。", "",
              "## 6. 成本组成与 cache", "", "![成本组成](figures/05_cost_composition.png)", "",
              "当前价格表对 uncached input、output、cache read、cache write 分别计价。本次四组 cache write 均为 0；cache read 分别为 Normal 70.63M、Cost-aware 39.44M、Category-seeded 51.68M、Predefined 45.12M。由于 cache read 远大于非缓存输入，降低 cache-read 规模是总成本下降的主要来源。", "",
              "## 7. 操作行为", "", "![操作行为相对值](figures/06_behavior_relative.png)", "",
              f"- cost-aware 的工具调用为 {int(overall.loc['cost-aware','tool_calls'])}（{fmt_delta(overall.loc['cost-aware','tool_calls_delta_pct'])}），重复精确输入仅 {int(overall.loc['cost-aware','repeated_exact_inputs'])} 次，是四组最低。",
              f"- category-seeded 工具调用为 {int(overall.loc['category-seeded-cost-aware','tool_calls'])}，接近 normal，但成功率更高、成本更低。",
              f"- predefined-control 工具调用下降到 {int(overall.loc['predefined-control','tool_calls'])}，但重复精确输入增至 {int(overall.loc['predefined-control','repeated_exact_inputs'])} 次，说明减少调用总量并不等同于减少重复行为。", "",
              "## 8. 数据质量与解释限制", "",
              "- `compile-compcert` 在 category-seeded 和 cost-aware 中失败且 usage 为 0。对应运行目录存在 raw error，因此这两组总成本被轻微低估；不能把零成本理解为有效节省。",
              "- 每种模式只有单次 20-task 运行，没有多随机种子重复。成功率相差 1–2 个任务时，应视为描述性结果，而不是统计显著性结论。",
              "- wall time 是各任务耗时求和，并非 parallel=2 下的实际端到端时钟时间；适合衡量累计计算负担，不等于用户等待时间。",
              "- 四组运行时间不同，外部 API、网络和 Docker 下载波动可能影响 wall time，但不会改变 verifier pass/fail 与记录的 token。", "",
              "## 9. 结论与建议", "",
              "若目标是**最高成功率且兼顾成本**，本次结果支持 category-seeded-cost-aware；若目标是**最低 API 成本**，cost-aware 最低，但需要接受成功率下降；若需要**更稳健的成功率/成本折中**，predefined-control 在本次样本中最均衡。建议下一步对三种候选模式至少重复 3–5 个随机种子，并优先复查 `compile-compcert` 的零 usage 错误，再报告均值、标准差和置信区间。", "",
              "## 10. 产物索引", "",
              "- `overall_comparison.csv`：总体指标及相对 normal 的变化。",
              "- `per_task_comparison.csv`：逐任务成功、token、cache、cost 和相对变化。",
              "- `paired_success_changes.csv`：任务级新增/丢失通过统计。",
              "- `batch_comparison.csv`：五个 batch 的趋势数据。",
              "- `cost_components.csv`：输入、输出与 cache 成本拆分。", ""]
    lines += ["## 11. 进化活动核验", "",
              "| 模式 | Solver drafts | Normal 成功解析 | Normal 实际应用 | 附加模块实际应用 |",
              "|---|---:|---:|---:|---:|"]
    for mode in MODE_ORDER:
        row = evolution.loc[mode]
        lines.append(
            f"| {mode} | {int(row.solver_proposals)} | {int(row.normal_parsed)} | "
            f"{int(row.normal_applied)} | {int(row.module_applied)} |"
        )
    lines += ["", "![进化活动](figures/07_evolution_activity.png)", "",
              "这四组结果生成于 Terminal proposal 协议修复之前：虽然每组产生了 15–16 个 solver drafts，"
              "但旧 YAML/Markdown 格式未被 GuidedSynthesis 解析，因此 normal evolution 的解析和应用均为 0。"
              "Cost-aware 与 category-seeded 的附加模块分别实际应用 8 和 7 条提案。修复后的新运行应单独比较，"
              "不能把本报告中的 normal evolution 视为已经生效。", "",
              "此外，这些历史结果没有记录 reflection 调用的 token；报告中的 token 与 cost 是旧口径，"
              "会轻微低估所有模式的真实消耗。修复后生成的新结果将包含这部分 usage。", "",
              "新增产物：`evolution_activity.csv` 及 `figures/07_evolution_activity.png`。", ""]
    (out / "REPORT_CN.md").write_text("\n".join(lines), encoding="utf-8")
    print(out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
