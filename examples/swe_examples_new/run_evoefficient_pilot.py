#!/usr/bin/env python3
"""Small-batch pilot runner for EvoEfficient on SWE-bench mini.

Runs:
  1) Baseline: solve_all.py on a small subset
  2) EvoEfficient pilot: evolve_sequential.py on the same subset
Then compares success, tokens, steps, and tool calls.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SWE_DIR = REPO_ROOT / "examples" / "swe_examples_new"


def _run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def _load_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _summarize(results: list[dict]) -> dict:
    if not results:
        return {
            "tasks": 0,
            "success": 0,
            "success_rate": 0.0,
            "tokens": 0,
            "steps": 0,
            "tool_calls": 0,
            "llm_calls": 0,
            "wall_time": 0.0,
        }

    tasks = len(results)
    success = sum(1 for r in results if r.get("success"))
    tokens = sum(
        int(r.get("cumulative_input_tokens", 0)) + int(r.get("cumulative_output_tokens", 0))
        for r in results
    )
    steps = sum(int(r.get("num_turns", r.get("turns", 0))) for r in results)
    tool_calls = sum(int(r.get("num_tool_calls", 0)) for r in results)
    wall_time = sum(float(r.get("elapsed", 0.0)) for r in results)

    return {
        "tasks": tasks,
        "success": success,
        "success_rate": round(success / tasks if tasks else 0.0, 4),
        "tokens": tokens,
        "steps": steps,
        "tool_calls": tool_calls,
        "llm_calls": steps,
        "wall_time": round(wall_time, 1),
    }


def _print_table(rows: list[tuple[str, dict]]) -> None:
    print("\n=== Pilot Summary ===")
    print("| Group | Tasks | Success | SR | Tokens | Steps | Tool Calls | Wall Time |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, m in rows:
        print(
            f"| {name} | {m['tasks']} | {m['success']} | {m['success_rate']:.2%} | "
            f"{m['tokens']} | {m['steps']} | {m['tool_calls']} | {m['wall_time']}s |"
        )


def main() -> int:
    p = argparse.ArgumentParser(description="Run EvoEfficient small-batch pilot on SWE-bench mini")
    p.add_argument("--dataset", type=str, default="MariusHobbhahn/swe-bench-verified-mini")
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--pilot-limit", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--parallel", type=int, default=2)
    p.add_argument("--model-id", type=str, default="us.anthropic.claude-opus-4-6-v1")
    p.add_argument("--region", type=str, default="us-west-2")
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--window-size", type=int, default=40)
    p.add_argument("--seed-workspace", type=str, default=str(REPO_ROOT / "seed_workspaces" / "swe"))
    p.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "logs" / "evoefficient_pilot"))
    p.add_argument("--no-baseline", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = out_dir / "baseline"
    evo_dir = out_dir / "evolution"

    if not args.no_baseline:
        _run([
            sys.executable,
            str(SWE_DIR / "solve_all.py"),
            "--dataset", args.dataset,
            "--model-id", args.model_id,
            "--region", args.region,
            "--max-tokens", str(args.max_tokens),
            "--max-turns", str(args.max_steps),
            "--workers", str(args.parallel),
            "--output-dir", str(baseline_dir),
            "--limit", str(args.limit),
        ])

    _run([
        sys.executable,
        str(SWE_DIR / "evolve_sequential.py"),
        "--dataset", args.dataset,
        "--batch-size", str(args.batch_size),
        "--parallel", str(args.parallel),
        "--max-steps", str(args.max_steps),
        "--window-size", str(args.window_size),
        "--model-id", args.model_id,
        "--region", args.region,
        "--max-tokens", str(args.max_tokens),
        "--seed-workspace", args.seed_workspace,
        "--output-dir", str(evo_dir),
        "--limit", str(args.pilot_limit),
        "--feedback", "minimal",
        "--efficiency-prompt",
        "--solver-proposes",
        "--verification-focus",
    ])

    baseline_results = _load_json(baseline_dir / "results.json")
    evo_results = _load_json(evo_dir / "results.json")

    baseline_summary = _summarize(baseline_results)
    evo_summary = _summarize(evo_results)

    _print_table([
        ("Baseline", baseline_summary),
        ("EvoEfficient Pilot", evo_summary),
    ])

    delta_tokens = baseline_summary["tokens"] - evo_summary["tokens"]
    delta_steps = baseline_summary["steps"] - evo_summary["steps"]
    delta_sr = evo_summary["success_rate"] - baseline_summary["success_rate"]

    report = {
        "baseline": baseline_summary,
        "evoefficient_pilot": evo_summary,
        "delta": {
            "tokens": delta_tokens,
            "steps": delta_steps,
            "success_rate": delta_sr,
        },
        "paths": {
            "baseline_dir": str(baseline_dir),
            "evolution_dir": str(evo_dir),
        },
    }
    (out_dir / "pilot_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nReport written to:", out_dir / "pilot_report.json")
    print("Baseline results:", baseline_dir / "results.json")
    print("Pilot results:", evo_dir / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
