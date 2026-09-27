#!/usr/bin/env python3
"""Generate a paired comparison report for the latest complete GAIA L3 runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODES = ["normal-evolve", "cost-aware", "category-seeded-cost-aware", "predefined-control"]
LABELS = {
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


def load_run(path: Path) -> dict:
    return {
        "path": path,
        "manifest": json.loads((path / "experiment_manifest.json").read_text()),
        "summary": json.loads((path / "summary.json").read_text()),
        "results": json.loads((path / "results.json").read_text()),
    }


def evolution_counts(path: Path) -> dict[str, int]:
    normal_path = path / "emergence" / "normal_evolution.jsonl"
    records = []
    if normal_path.exists():
        records = [json.loads(line) for line in normal_path.read_text().splitlines() if line.strip()]
    adoption_path = path / "emergence" / "proposal_adoption.json"
    adoptions = json.loads(adoption_path.read_text()) if adoption_path.exists() else []
    return {
        "solver_drafts": sum(int(r.get("solver_proposals", 0) or 0) for r in records),
        "normal_parsed": sum(int(r.get("result", {}).get("proposals", 0) or 0) for r in records),
        "normal_applied": sum(len(r.get("result", {}).get("applied", []) or []) for r in records),
        "module_applied": sum(str(r.get("status")) == "applied" for r in adoptions),
    }


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> int:
    p = argparse.ArgumentParser()
    for mode in MODES:
        p.add_argument("--" + mode, required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()
    paths = {mode: getattr(args, mode.replace("-", "_")) for mode in MODES}
    runs = {mode: load_run(path) for mode, path in paths.items()}
    out = args.output_dir
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    task_ids = runs[MODES[0]]["manifest"]["tasks"]
    for mode, run in runs.items():
        assert run["manifest"]["mode"] == mode
        assert run["manifest"]["tasks"] == task_ids, f"task mismatch: {mode}"
        assert len(run["results"]) == len(task_ids), f"incomplete results: {mode}"

    rows = []
    for mode in MODES:
        o = runs[mode]["summary"]["overall"]
        evo = evolution_counts(paths[mode])
        rows.append({
            "mode": mode, "tasks": int(o["tasks"]), "success": int(o["success"]),
            "success_rate": float(o["success_rate"]), "tokens": int(o["tokens"]),
            "input_tokens": int(o["input_tokens"]), "output_tokens": int(o["output_tokens"]),
            "cache_read_tokens": int(o["cache_read_input_tokens"]),
            "cost_usd": float(o["total_cost"]), "tool_calls": int(o["tool_calls"]),
            "turns": int(o["turns"]), "wall_time_s": float(o["wall_time"]),
            "inconsistent_usage": int(o.get("usage_quality_counts", {}).get("inconsistent", 0)),
            "errors": sum(bool(r.get("error")) for r in runs[mode]["results"]),
            "timeouts": sum(bool(r.get("timed_out")) for r in runs[mode]["results"]),
            "cost_per_pass": float(o["total_cost"]) / int(o["success"]) if o["success"] else np.nan,
            **evo,
        })
    overall = pd.DataFrame(rows).set_index("mode")
    base = overall.loc["normal-evolve"]
    overall["success_delta_pp"] = (overall.success_rate - base.success_rate) * 100
    for metric in ["tokens", "cache_read_tokens", "cost_usd", "tool_calls", "turns", "wall_time_s", "cost_per_pass"]:
        overall[metric + "_delta_pct"] = (overall[metric] / base[metric] - 1) * 100
    overall.to_csv(out / "overall_comparison.csv", float_format="%.6f")

    by_task = {m: {r["instance_id"]: r for r in runs[m]["results"]} for m in MODES}
    task_rows = []
    for task in task_ids:
        row = {"task": task}
        for mode in MODES:
            r = by_task[mode][task]
            row.update({
                f"{mode}_success": bool(r.get("success")),
                f"{mode}_answer": str(r.get("model_answer", "")),
                f"{mode}_tokens": int(r.get("total_tokens", 0) or 0),
                f"{mode}_cost_usd": float(r.get("cost_units", 0) or 0),
                f"{mode}_usage_quality": str(r.get("usage_record_quality", "")),
                f"{mode}_error": str(r.get("error", "")),
            })
        task_rows.append(row)
    task_df = pd.DataFrame(task_rows)
    task_df.to_csv(out / "per_task_comparison.csv", index=False)

    paired = []
    normal = task_df["normal-evolve_success"]
    for mode in MODES:
        current = task_df[f"{mode}_success"]
        paired.append({"mode": mode, "gained": int((~normal & current).sum()),
                       "lost": int((normal & ~current).sum()),
                       "both_pass": int((normal & current).sum()),
                       "both_fail": int((~normal & ~current).sum())})
    pd.DataFrame(paired).set_index("mode").to_csv(out / "paired_outcomes.csv")

    batch_rows = []
    for mode in MODES:
        for i, b in enumerate(runs[mode]["summary"]["by_batch"], 1):
            batch_rows.append({"mode": mode, "batch": i, "success_rate": b["success_rate"],
                               "cost_usd": b["total_cost"], "tokens": b["tokens"],
                               "tool_calls": b["tool_calls"], "wall_time_s": b["wall_time"]})
    batches = pd.DataFrame(batch_rows)
    batches.to_csv(out / "batch_comparison.csv", index=False)

    # 1. Headline metrics, normalized to Normal=100.
    metrics = [("success_rate", "Success rate"), ("cost_usd", "Total cost"),
               ("tokens", "Tokens"), ("tool_calls", "Tool calls")]
    x = np.arange(len(metrics)); width = .19
    fig, ax = plt.subplots(figsize=(14, 7))
    for j, mode in enumerate(MODES):
        vals = [overall.loc[mode, k] / base[k] * 100 for k, _ in metrics]
        bars = ax.bar(x + (j-1.5)*width, vals, width, color=COLORS[mode], label=LABELS[mode])
        labels = [f"{overall.loc[mode,'success_rate']*100:.1f}%",
                  f"${overall.loc[mode,'cost_usd']:.2f}", f"{overall.loc[mode,'tokens']/1e6:.1f}M",
                  f"{int(overall.loc[mode,'tool_calls'])}"]
        ax.bar_label(bars, labels=labels, fontsize=8, padding=2)
    ax.axhline(100, linestyle="--", color="#334155")
    ax.set_xticks(x, [label for _, label in metrics]); ax.set_ylabel("Relative index (Normal = 100)")
    ax.set_title("GAIA Level 3: overall comparison")
    ax.legend(ncol=4, loc="lower center"); ax.grid(axis="y", alpha=.25)
    savefig(figs / "01_overall_metrics.png")

    # 2. Success-cost frontier.
    fig, ax = plt.subplots(figsize=(9, 6))
    for mode in MODES:
        ax.scatter(overall.loc[mode, "cost_usd"], overall.loc[mode, "success_rate"]*100,
                   s=180, color=COLORS[mode])
        ax.annotate(LABELS[mode], (overall.loc[mode, "cost_usd"], overall.loc[mode, "success_rate"]*100),
                    xytext=(7, 7), textcoords="offset points")
    ax.set(xlabel="Total cost (USD)", ylabel="Success rate (%)", title="Success-cost trade-off")
    ax.grid(alpha=.3); savefig(figs / "02_success_cost_frontier.png")

    # 3. Batch trends.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for mode in MODES:
        d = batches[batches["mode"] == mode]
        axes[0].plot(d.batch, d.success_rate*100, marker="o", color=COLORS[mode], label=LABELS[mode])
        axes[1].plot(d.batch, d.cost_usd, marker="o", color=COLORS[mode], label=LABELS[mode])
    axes[0].set(title="Success rate by batch", xlabel="Batch", ylabel="Success rate (%)")
    axes[1].set(title="Cost by batch", xlabel="Batch", ylabel="USD")
    for ax in axes: ax.grid(alpha=.3)
    axes[0].legend(fontsize=8); savefig(figs / "03_batch_trends.png")

    # 4. Task outcome matrix.
    matrix = np.array([[int(by_task[m][t]["success"]) for m in MODES] for t in task_ids])
    fig, ax = plt.subplots(figsize=(10, 11))
    ax.imshow(matrix, cmap=matplotlib.colors.ListedColormap(["#ef4444", "#22c55e"]), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4), [LABELS[m] for m in MODES], rotation=20, ha="right")
    ax.set_yticks(range(len(task_ids)), [t[:12] for t in task_ids], fontsize=7)
    ax.set_title("Per-task outcomes (green=pass, red=fail)")
    for i in range(len(task_ids)):
        for j in range(4): ax.text(j, i, "P" if matrix[i,j] else "F", ha="center", va="center", color="white", fontsize=8)
    savefig(figs / "04_task_outcomes.png")

    # 5. Usage quality and runtime failures.
    fig, ax = plt.subplots(figsize=(10, 6)); idx = np.arange(4)
    ax.bar(idx-.2, overall.loc[MODES,"inconsistent_usage"], .2, label="Inconsistent usage", color="#dc2626")
    ax.bar(idx, overall.loc[MODES,"errors"], .2, label="Errors", color="#7c3aed")
    ax.bar(idx+.2, overall.loc[MODES,"timeouts"], .2, label="Timeouts", color="#0891b2")
    ax.set_xticks(idx, [LABELS[m] for m in MODES], rotation=15, ha="right")
    ax.set_ylabel("Tasks"); ax.set_title("Usage quality and runtime failures"); ax.legend(); ax.grid(axis="y", alpha=.25)
    savefig(figs / "05_data_quality.png")

    # 6. Evolution activity.
    fig, ax = plt.subplots(figsize=(11, 6)); evo_cols = ["solver_drafts","normal_parsed","normal_applied","module_applied"]
    for j, mode in enumerate(MODES):
        bars = ax.bar(np.arange(4)+(j-1.5)*width, overall.loc[mode,evo_cols], width,
                      color=COLORS[mode], label=LABELS[mode])
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
    ax.set_xticks(range(4), ["Solver drafts","Normal parsed","Normal applied","Extra module applied"])
    ax.set_ylabel("Count"); ax.set_title("Evolution activity by layer"); ax.legend(ncol=2); ax.grid(axis="y", alpha=.25)
    savefig(figs / "06_evolution_activity.png")

    def cell(mode: str, metric: str, formatter) -> str:
        return formatter(overall.loc[mode, metric])

    best_success = overall.success_rate.idxmax(); cheapest = overall.cost_usd.idxmin()
    lines = ["# GAIA Level 3 最新四模式实验分析报告", "", "## 1. 数据选择与完整性", "",
             f"本报告比较 4 个模式的最新完整运行；每组均覆盖相同的 {len(task_ids)} 个 GAIA 2023 validation Level 3 任务。",
             "较早的 `gaia_l3_normal-evolve_20260925_224209_862855` 因 26 个任务全部配置失败而排除。"
             "Category-seeded 存在两次完整运行，主分析采用时间更新的 `20260926_174726_727112`。", "",
             "| 模式 | 成功 | 成功率 | Tokens | Cache read | 成本 | Cost/pass | 工具调用 | 错误/超时 | Usage不一致 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in MODES:
        r = overall.loc[mode]
        lines.append(f"| {LABELS[mode]} | {int(r.success)}/26 | {r.success_rate*100:.1f}% | {int(r.tokens):,} | "
                     f"{int(r.cache_read_tokens):,} | ${r.cost_usd:.4f} | ${r.cost_per_pass:.4f} | "
                     f"{int(r.tool_calls)} | {int(r.errors)}/{int(r.timeouts)} | {int(r.inconsistent_usage)} |")
    lines += ["", "![总体指标](figures/01_overall_metrics.png)", "", "![成功成本前沿](figures/02_success_cost_frontier.png)", "",
              "## 2. 核心发现", "",
              f"- **最高成功率：{LABELS[best_success]}，{int(overall.loc[best_success,'success'])}/26（{overall.loc[best_success,'success_rate']*100:.1f}%）**。",
              f"- **最低总成本：{LABELS[cheapest]}，${overall.loc[cheapest,'cost_usd']:.4f}**。",
              f"- 相对 Normal，Category-seeded 成功率变化 {overall.loc['category-seeded-cost-aware','success_delta_pp']:+.1f} pp，"
              f"成本变化 {overall.loc['category-seeded-cost-aware','cost_usd_delta_pct']:+.1f}%。",
              f"- Cost-aware 相对 Normal 成功率变化 {overall.loc['cost-aware','success_delta_pp']:+.1f} pp，"
              f"成本变化 {overall.loc['cost-aware','cost_usd_delta_pct']:+.1f}%。",
              f"- Category-seeded 相对 Normal 的运行时间仅变化 {overall.loc['category-seeded-cost-aware','wall_time_s_delta_pct']:+.1f}%，"
              f"但每成功任务成本增加 {overall.loc['category-seeded-cost-aware','cost_per_pass_delta_pct']:+.1f}%；"
              f"Cost-aware 的运行时间增加 {overall.loc['cost-aware','wall_time_s_delta_pct']:+.1f}%。", "",
              "## 3. 分批与逐任务结果", "", "![分批趋势](figures/03_batch_trends.png)", "",
              "![逐任务结果](figures/04_task_outcomes.png)", "",
              "相对 Normal 的配对结果如下；`gained` 表示 Normal 失败而该模式成功，`lost` 表示相反。", "",
              "| 模式 | Gained | Lost | Both pass | Both fail |", "|---|---:|---:|---:|---:|"]
    for row in paired:
        lines.append(f"| {LABELS[row['mode']]} | {row['gained']} | {row['lost']} | {row['both_pass']} | {row['both_fail']} |")
    lines += ["", "Category-seeded 相对 Normal 净增 3 题（赢 4、输 1）；Cost-aware 净少 1 题；Predefined 净少 3 题。 "
              "逐任务答案、成功状态、token、成本、usage 质量和错误信息见 `per_task_comparison.csv`。", "",
              "## 4. 进化活动", "", "| 模式 | Solver drafts | Normal parsed | Normal applied | Extra module applied |",
              "|---|---:|---:|---:|---:|"]
    for mode in MODES:
        r=overall.loc[mode]; lines.append(f"| {LABELS[mode]} | {int(r.solver_drafts)} | {int(r.normal_parsed)} | {int(r.normal_applied)} | {int(r.module_applied)} |")
    lines += ["", "![进化活动](figures/06_evolution_activity.png)", "",
              "## 5. 数据质量", "", "![数据质量](figures/05_data_quality.png)", "",
              "- 四组均无 zero-token 任务，但 9–12 个任务被标记为 `inconsistent`，表示 Strands accumulated metrics 与逐消息 usage 差异超过 20%；当前结果已自动采用逐消息 usage。",
              "- Cost-aware 有 2 个错误且均超时；Predefined 有 4 个错误，其中 1 个超时。失败任务详情见逐任务 CSV。",
              "- Category-seeded 的两次完整运行分别为 18/26 与 25/26，说明单次运行受模型、网络与动态网页状态影响很大；25/26 不应被解释为稳定期望值。",
              "- 每个模式只有一次主运行，没有置信区间；模式间 1–3 题差异属于描述性证据。", "",
              "## 6. 结论", "",
              "本次单次运行中，Category-seeded 取得最高成功率，但总成本和每成功任务成本均高于 Normal；Normal 的总成本与每成功任务成本最低，但成功数少 3 个。"
              "Cost-aware 没有在本次样本中降低成本，且出现两次超时。Predefined 成功率与成本均弱于 Normal。"
              "鉴于 Category-seeded 两次运行波动达到 7 题，建议四模式至少重复 3 次，并以均值、标准差和配对任务胜负作为最终结论。", "",
              "## 7. 产物", "", "- `overall_comparison.csv`：总体指标与相对 Normal 变化。",
              "- `per_task_comparison.csv`：逐任务对比。", "- `paired_outcomes.csv`：相对 Normal 的任务得失。",
              "- `batch_comparison.csv`：分批趋势。", "- `figures/`：6 张可视化。", ""]
    (out / "REPORT_CN.md").write_text("\n".join(lines), encoding="utf-8")
    print(out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
