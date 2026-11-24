#!/usr/bin/env python3
"""Run Terminal-Bench emergent-efficiency experiments with Harbor + AdaptiveSkillEngine.

This is an additive variant of examples/tb_examples/emergent_efficiency.  It
keeps the Harbor evaluation backend, but replaces the normal-evolve step with
the same AdaptiveSkillEngine/MasAdaptiveSkillEngine loop used by
examples/tb_examples/batch_evolve_terminal.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

_ORIGINAL_PATH_RESOLVE = Path.resolve


def _safe_path_resolve(self: Path, strict: bool = False) -> Path:
    try:
        return _ORIGINAL_PATH_RESOLVE(self, strict=strict)
    except OSError:
        return Path(os.path.abspath(self))


Path.resolve = _safe_path_resolve  # type: ignore[method-assign]

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from examples.tb_examples.emergent_efficiency_adaptive import experiment_controls as controls

logger = logging.getLogger("tb_emergent_efficiency_adaptive_harbor")

DEFAULT_OUTPUT_ROOT = "logs/tb_emergent_efficiency_adaptive_harbor"
DEFAULT_MODEL_ID = "deepseek:deepseek-v4-flash"
DEFAULT_HARBOR_UV_CMD = "uv run harbor run"

_base: Any | None = None
_harbor: Any | None = None


def _load_shared_runners() -> tuple[Any, Any]:
    global _base, _harbor
    if _base is None or _harbor is None:
        from examples.tb_examples.emergent_efficiency import (
            run_emergent_efficiency_tb_harbor as harbor_runner,
        )

        _harbor = harbor_runner
        _base = harbor_runner._tb
    return _base, _harbor


def _json_default(value: Any) -> str:
    return str(value)


def _build_adaptive_config(args: argparse.Namespace) -> Any:
    from agent_evolve.config import EvolveConfig

    return EvolveConfig(
        evolver_model=args.evolver_model or args.model_id,
        evolver_max_tokens=args.evolver_max_tokens,
        evolver_temperature=args.evolver_temperature,
        trajectory_only=args.trajectory_only,
        evolve_prompts=not args.skills_only and not args.prompt_only,
        evolve_skills=not args.prompt_only,
        evolve_memory=not args.prompt_only and not args.skills_only,
        evolve_tools=False,
        extra={
            "region": args.region,
            "max_skills": args.max_skills,
            "solver_proposed": args.propose_skill,
            "prompt_only": args.prompt_only,
            "skills_only": args.skills_only,
            "protect_skills": args.protect_skills,
        },
    )


def _write_solver_drafts(workspace_dir: Path, batch_results: list[dict[str, Any]]) -> int:
    base, _ = _load_shared_runners()
    from agent_evolve.contract.workspace import AgentWorkspace

    workspace = AgentWorkspace(workspace_dir)
    workspace.clear_drafts()
    draft_count = 0
    for result in batch_results:
        draft = str(result.get("skill_draft") or result.get("skill_proposal") or "").strip()
        if not draft:
            continue
        task_name = str(
            result.get("task_name") or result.get("instance_id") or f"task_{draft_count + 1}"
        )
        workspace.write_draft(base._slug(task_name, default=f"task_{draft_count + 1}"), draft)
        draft_count += 1
    return draft_count


def _write_adaptive_record(output_dir: Path, record: dict[str, Any]) -> None:
    emergence_dir = output_dir / "emergence"
    emergence_dir.mkdir(parents=True, exist_ok=True)
    for name in ("adaptive_evolution.jsonl", "normal_evolution.jsonl"):
        with (emergence_dir / name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def _run_adaptive_evolution(
    *,
    agent: Any,
    observer: Any,
    workspace_dir: Path,
    output_dir: Path,
    batch_results: list[dict[str, Any]],
    observations: list[Any],
    batch_num: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from agent_evolve.algorithms.adaptive_skill import AdaptiveSkillEngine
    from agent_evolve.algorithms.mas_adaptive_skill import MasAdaptiveSkillEngine

    draft_count = _write_solver_drafts(workspace_dir, batch_results) if args.propose_skill else 0
    observer.collect(observations)
    agent.export_to_fs()
    recent_logs = observer.get_recent_logs(n_batches=1)

    config = _build_adaptive_config(args)
    evolver = (
        MasAdaptiveSkillEngine(config)
        if args.evolver == "mas_adaptive_skill"
        else AdaptiveSkillEngine(config)
    )
    started = time.time()
    result = evolver.evolve(
        workspace=agent.workspace,
        observation_logs=recent_logs,
        evo_number=batch_num,
    )
    record = {
        "batch": batch_num,
        "mode": args.mode,
        "engine": args.evolver,
        "elapsed": round(time.time() - started, 1),
        "trajectory_only": args.trajectory_only,
        "skills_only": args.skills_only,
        "protect_skills": args.protect_skills,
        "max_skills": args.max_skills,
        "solver_proposals": draft_count,
        "result": result,
    }
    _write_adaptive_record(output_dir, record)
    agent.reload_from_fs()
    return record


def _write_experiment_manifest(
    output_dir: Path, args: argparse.Namespace, tasks: list[Any]
) -> None:
    base, _ = _load_shared_runners()
    base._write_experiment_manifest(output_dir, args, tasks)
    path = output_dir / "experiment_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(
        {
            "runner": "tb_emergent_efficiency_adaptive_harbor",
            "execution_backend": "harbor",
            "normal_evolution_engine": args.evolver
            if args.mode not in {"static-control"}
            else None,
            "evolution_enabled": args.mode not in {"static-control"},
            "task_selection": getattr(args, "task_selection_audit", {}),
            "efficiency_additions": {
                "failure_token_focus": args.mode in controls.COST_MODES,
                "early_stop": "optional_evolved_efficiency_strategy"
                if args.mode in controls.COST_MODES
                else None,
                "forced_runtime_stop": False,
            },
            "adaptive_engine_config": {
                "trajectory_only": args.trajectory_only,
                "skills_only": args.skills_only,
                "protect_skills": args.protect_skills,
                "prompt_only": args.prompt_only,
                "max_skills": args.max_skills,
                "propose_skill": args.propose_skill,
            },
            "harbor_repo": args.harbor_repo,
            "harbor_uv_cmd": args.harbor_uv_cmd,
            "harbor_command_template": args.harbor_command_template,
            "harbor_config_template": args.harbor_config_template,
            "harbor_agent_import_path": args.harbor_agent_import_path,
            "harbor_model_name": args.harbor_model_name or args.model_id,
            "harbor_delegate_agent_name": args.harbor_delegate_agent_name,
            "harbor_delegate_agent_import_path": args.harbor_delegate_agent_import_path,
            "harbor_delegate_model_name": args.harbor_delegate_model_name,
            "harbor_delegate_kwargs": args.harbor_delegate_kwargs,
            "harbor_jobs_dir": args.harbor_jobs_dir,
            "harbor_timeout_sec": args.harbor_timeout_sec,
            "dry_run_harbor": args.dry_run_harbor,
        }
    )
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Terminal-Bench emergent-efficiency runner using Harbor evaluation and AdaptiveSkillEngine evolution"
    )
    parser.add_argument(
        "--mode",
        choices=[
            "static-control",
            "normal-evolve",
            "cost-aware",
            "predefined-control",
            "predefined-cost-aware",
            "category-seeded-cost-aware",
        ],
        default="cost-aware",
    )
    parser.add_argument(
        "--challenges-dir", default=None, help="Path to Terminal-Bench 2.1 tasks directory"
    )
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Only with --fixed-output-dir; removes existing output dir",
    )

    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--evolver-model", default="")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--evolver-max-tokens", type=int, default=8192)
    parser.add_argument("--evolver-temperature", type=float, default=0.0)
    parser.add_argument("--evolver-call-retries", type=int, default=4)
    parser.add_argument("--evolver-retry-initial-delay", type=float, default=2.0)
    parser.add_argument("--evolver-retry-max-delay", type=float, default=30.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Tool-call limit used for waste-signal thresholds",
    )
    parser.add_argument("--max-proposals-per-batch", type=int, default=2)

    parser.add_argument(
        "--evolver", choices=["adaptive_skill", "mas_adaptive_skill"], default="adaptive_skill"
    )
    parser.add_argument(
        "--trajectory-only", dest="trajectory_only", action="store_true", default=True
    )
    parser.add_argument("--include-feedback", dest="trajectory_only", action="store_false")
    parser.add_argument("--skills-only", dest="skills_only", action="store_true", default=False)
    parser.add_argument("--allow-prompt-memory-evolution", dest="skills_only", action="store_false")
    parser.add_argument(
        "--protect-skills", dest="protect_skills", action="store_true", default=True
    )
    parser.add_argument("--allow-skill-edits", dest="protect_skills", action="store_false")
    parser.add_argument("--prompt-only", action="store_true", default=False)
    parser.add_argument("--max-skills", type=int, default=10)
    parser.add_argument("--propose-skill", action="store_true", default=False)

    parser.add_argument(
        "--category-seed",
        choices=["none", "basic-efficiency"],
        default="basic-efficiency",
        help="Taxonomy exposed only in category-seeded-cost-aware mode",
    )
    parser.add_argument(
        "--dry-run-proposer",
        action="store_true",
        help="Use deterministic dummy cost-aware proposal",
    )
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

    parser.add_argument("--harbor-repo", default=os.environ.get("HARBOR_REPO", ""))
    parser.add_argument("--harbor-config-template", default=None)
    parser.add_argument(
        "--harbor-command-template",
        default=None,
        help="Optional full command template with placeholders such as {config_path}, {task_name}, {workspace_dir}.",
    )
    parser.add_argument("--harbor-result-json-template", default=None)
    parser.add_argument(
        "--harbor-agent-import-path", default=os.environ.get("TB_HARBOR_AGENT_IMPORT_PATH", "")
    )
    parser.add_argument("--harbor-agent-name", default="terminal-bench-agent")
    parser.add_argument("--harbor-model-name", default=os.environ.get("HARBOR_MODEL_NAME", ""))
    parser.add_argument(
        "--harbor-delegate-agent-name",
        default=os.environ.get("TB_HARBOR_DELEGATE_AGENT_NAME", "codex"),
        help="Harbor built-in agent used by the A-Evolve adapter",
    )
    parser.add_argument(
        "--harbor-delegate-agent-import-path",
        default=os.environ.get("TB_HARBOR_DELEGATE_AGENT_IMPORT_PATH", ""),
        help="Optional custom Harbor agent import path delegated to by the adapter",
    )
    parser.add_argument(
        "--harbor-delegate-model-name", default=os.environ.get("TB_HARBOR_DELEGATE_MODEL_NAME", "")
    )
    parser.add_argument(
        "--harbor-delegate-kwargs", default=os.environ.get("TB_HARBOR_DELEGATE_KWARGS", "")
    )
    parser.add_argument("--harbor-jobs-dir", default=None)
    parser.add_argument("--harbor-timeout-sec", type=int, default=3600)
    parser.add_argument(
        "--harbor-uv-cmd", default=os.environ.get("HARBOR_UV_CMD", DEFAULT_HARBOR_UV_CMD)
    )
    parser.add_argument(
        "--dry-run-harbor", action="store_true", help="Use deterministic synthetic Harbor results"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--allow-rerun",
        action="store_true",
        help="Allow historical real attempts; default selects unseen tasks",
    )
    parser.add_argument(
        "--history-root",
        action="append",
        default=None,
        help="History root or results.json; repeat for additional roots",
    )
    parser.add_argument(
        "--selection-file", help="Write/reuse frozen task selection shared by matrix modes"
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Select and exit without creating a run or calling models",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    base, harbor = _load_shared_runners()
    from agent_evolve.agents.terminal.agent import TerminalAgent
    from agent_evolve.agents.terminal.dataset import load_all_tasks
    from agent_evolve.efficiency.profiler import summarize_results
    from agent_evolve.engine.observer import Observer

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
    for name in (
        "botocore",
        "urllib3",
        "httpcore",
        "httpx",
        "strands.models",
        "strands.tools",
        "strands.telemetry",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.dry_run_harbor:
        args.dry_run_proposer = True
    tasks, selection = controls.select_tasks(
        load_all_tasks(args.challenges_dir), args, base._filter_tasks
    )
    if not tasks:
        raise SystemExit(
            "No unseen Terminal-Bench tasks remain after filtering; use --allow-rerun intentionally or select other tasks"
        )
    args.task_selection_audit = selection
    if args.plan_only:
        print(json.dumps(selection, ensure_ascii=False, indent=2))
        return 0
    output_dir = base._resolve_output_dir(args)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"Output directory exists and is not empty: {output_dir}. "
            "Use --fixed-output-dir --overwrite for an intentional replacement, or choose a different prefix."
        )
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    workspace_dir = output_dir / "workspace"
    hide_efficiency = args.mode in {
        "static-control",
        "normal-evolve",
        "cost-aware",
        "category-seeded-cost-aware",
    }
    base._copy_seed_workspace(
        Path(args.seed_workspace), workspace_dir, hide_efficiency=hide_efficiency
    )

    if args.mode in {"predefined-control", "predefined-cost-aware"}:
        base._configure_predefined_control(
            workspace_dir,
            args.predefined_strategies,
            experiment=f"tb_emergent_efficiency_adaptive_{args.mode.replace('-', '_')}",
        )

    leakage = (
        base._swe._strategy_leakage_scan(workspace_dir)
        if hide_efficiency
        else {"passed": True, "hits": []}
    )
    emergence_dir = output_dir / "emergence"
    emergence_dir.mkdir(parents=True, exist_ok=True)
    (emergence_dir / "leakage_audit.json").write_text(
        json.dumps(leakage, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if hide_efficiency and not leakage.get("passed"):
        raise SystemExit(f"Leakage audit failed. See {emergence_dir / 'leakage_audit.json'}")

    batches = [tasks[i : i + args.batch_size] for i in range(0, len(tasks), args.batch_size)]
    core_tasks = [base._task_to_core_task(task) for task in tasks]
    tasks_by_id = {task.id: task for task in core_tasks}
    _write_experiment_manifest(output_dir, args, tasks)
    (output_dir / "task_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2)
    )

    agent = TerminalAgent(
        workspace_dir=workspace_dir,
        model_id=args.model_id,
        region=args.region,
        max_tokens=args.max_tokens,
    )
    observer = Observer(workspace_dir / "evolution", trajectory_only=args.trajectory_only)

    print(f"Mode: {args.mode}")
    print("Benchmark: Terminal-Bench 2.1")
    print("Execution backend: Harbor")
    evolution_label = "disabled" if args.mode in {"static-control"} else args.evolver
    print(f"Normal evolution engine: {evolution_label}")
    print(
        "Adaptive flags: "
        f"trajectory_only={args.trajectory_only} "
        f"skills_only={args.skills_only} "
        f"protect_skills={args.protect_skills} "
        f"max_skills={args.max_skills}"
    )
    print(f"Tasks: {len(tasks)} in {len(batches)} batch(es)")
    print(f"Output: {output_dir.resolve()}")
    print(f"Workspace: {workspace_dir.resolve()}")
    print()

    all_results: list[dict[str, Any]] = []
    all_analysis_results: list[dict[str, Any]] = []
    for batch_idx, batch_tasks in enumerate(batches, start=1):
        print(f"=== Batch {batch_idx}/{len(batches)} ===")
        agent.reload_from_fs()
        system_prompt_text = base._build_system_prompt_for_terminal(agent, workspace_dir)
        batch_results = harbor._solve_terminal_batch_harbor(
            batch_tasks=batch_tasks,
            batch_num=batch_idx,
            workspace_dir=workspace_dir,
            output_dir=output_dir,
            system_prompt_text=system_prompt_text,
            agent=agent,
            args=args,
        )
        analysis_results = [
            controls.analyze_result(row) if args.mode in controls.COST_MODES else dict(row)
            for row in batch_results
        ]
        all_analysis_results.extend(analysis_results)
        all_results.extend(batch_results)
        feedback = controls.failure_feedback(analysis_results)
        (emergence_dir / f"batch_{batch_idx:02d}_failure_tokens.json").write_text(
            json.dumps(feedback, ensure_ascii=False, indent=2)
        )
        (output_dir / "failure_token_distribution.json").write_text(
            json.dumps(
                controls.failure_feedback(all_analysis_results), ensure_ascii=False, indent=2
            )
        )
        base._write_run_outputs(output_dir, all_results, mode=args.mode)

        observations = base._results_to_observations(batch_results, tasks_by_id)
        base._write_workspace_evolution_records(
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

        if (
            batch_idx < len(batches)
            and args.mode not in {"static-control"}
            and not args.dry_run_harbor
        ):
            normal_record = _run_adaptive_evolution(
                agent=agent,
                observer=observer,
                workspace_dir=workspace_dir,
                output_dir=output_dir,
                batch_results=batch_results,
                observations=observations,
                batch_num=batch_idx,
                args=args,
            )
            result = normal_record.get("result", {})
            print(
                "  Adaptive normal evolution: "
                f"drafts={normal_record['solver_proposals']} "
                f"new_skills={result.get('new_skills', 0)} "
                f"skills={result.get('skills_before', '?')}->{result.get('skills_after', '?')}"
            )

        if args.mode in {
            "cost-aware",
            "predefined-cost-aware",
            "category-seeded-cost-aware",
        } and batch_idx < len(batches):
            with controls.failure_aware_proposer(base._swe, feedback):
                report = base._swe.propose_emergent_efficiency(
                    batch_results=batch_results,
                    batch_num=batch_idx,
                    max_steps=args.max_steps,
                    args=args,
                )
            report["failure_token_distribution"] = feedback
            base._swe._write_proposals(output_dir, report)
            applied = base._apply_tb_proposals(
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
            base._swe._record_workspace_snapshot(workspace_dir, output_dir, batch_idx)

        base._write_run_outputs(output_dir, all_results, mode=args.mode)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
