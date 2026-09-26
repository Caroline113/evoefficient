#!/usr/bin/env python3
"""Run the SWE/TB emergent-efficiency matrix on GAIA validation Level 3."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from agent_evolve.agents.gaia import GaiaAgent
from agent_evolve.benchmarks.gaia import GaiaTask, load_level3_validation, question_scorer
from agent_evolve.contract.workspace import AgentWorkspace
from agent_evolve.efficiency.profiler import normalize_usage, result_strategy_hits, summarize_results
from agent_evolve.efficiency.strategies import load_active_config
from agent_evolve.efficiency.visualization import write_plots, write_strategy_library_plot, write_summary_table
from agent_evolve.types import Feedback, Observation, Task, Trajectory


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


_swe = _load(ROOT / "examples/swe_examples_new/emergent_efficiency/run_emergent_efficiency.py", "_gaia_swe_emergent")
_tb = _load(ROOT / "examples/tb_examples/emergent_efficiency/run_emergent_efficiency_tb.py", "_gaia_tb_emergent")
logger = logging.getLogger("gaia_emergent_efficiency")
DEFAULT_OUTPUT = "logs/gaia_emergent_efficiency"
DEFAULT_MODEL = "deepseek:deepseek-v4-flash"

GAIA_PROPOSER_PROMPT = _swe.EMERGENCE_SYSTEM_PROMPT.replace(
    "SWE-bench solving agent", "GAIA Level 3 research agent"
).replace("deterministic Strands tools", "reusable research skills")
GAIA_PREDEFINED_PROMPT = _swe.PREDEFINED_COST_AWARE_SYSTEM_PROMPT.replace(
    "SWE-bench solving agent", "GAIA Level 3 research agent"
).replace("deterministic Strands tools", "reusable research skills")
GAIA_CATEGORY_PROMPT = _swe.CATEGORY_SEEDED_SYSTEM_PROMPT.replace(
    "SWE-bench solving agent", "GAIA Level 3 research agent"
).replace("deterministic Strands tools", "reusable research skills")
_swe.EMERGENCE_SYSTEM_PROMPT = GAIA_PROPOSER_PROMPT
_swe.PREDEFINED_COST_AWARE_SYSTEM_PROMPT = GAIA_PREDEFINED_PROMPT
_swe.CATEGORY_SEEDED_SYSTEM_PROMPT = GAIA_CATEGORY_PROMPT


def _stamp() -> str: return datetime.now().strftime("%Y%m%d_%H%M%S_%f")
def _safe_int(v: Any) -> int:
    try: return max(0, int(v or 0))
    except (TypeError, ValueError): return 0


def _output_dir(args: argparse.Namespace) -> Path:
    path = Path(args.output_dir)
    if args.fixed_output_dir: return path
    mode = args.mode.replace("-", "_")
    return path / f"{mode}_{_stamp()}" if args.output_dir == DEFAULT_OUTPUT else path.with_name(f"{path.name}_{_stamp()}")


def _filter(tasks: list[GaiaTask], args: argparse.Namespace) -> list[GaiaTask]:
    if args.tasks:
        selected = {x.strip() for x in args.tasks.split(",") if x.strip()}; tasks = [t for t in tasks if t.task_id in selected]
    if args.exclude:
        excluded = {x.strip() for x in args.exclude.split(",") if x.strip()}; tasks = [t for t in tasks if t.task_id not in excluded]
    if args.shuffle: random.Random(args.shuffle_seed).shuffle(tasks)
    return tasks[:args.limit] if args.limit is not None else tasks


def _core(task: GaiaTask) -> Task:
    return Task(id=task.task_id, input=task.question, metadata={
        **task.metadata, "benchmark": "gaia_2023", "split": "validation", "level": 3,
        "file_name": task.file_name, "file_path": task.file_path,
    })


def _write_artifact(path: Path, value: Any, text: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(value) if text else json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return str(path.resolve())


def _conversation_steps(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Strands/Bedrock content blocks into profiler-compatible tool steps."""
    steps: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for message in conversation:
        role = str(message.get("role") or "")
        content = message.get("content", [])
        if isinstance(content, str):
            content = [{"text": content}]
        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict):
                continue
            tool_use = block.get("toolUse") or block.get("tool_use")
            if isinstance(tool_use, dict):
                name = str(tool_use.get("name") or "")
                arguments = tool_use.get("input", {})
                call_id = str(tool_use.get("toolUseId") or tool_use.get("id") or "")
                step = {
                    "tool": name,
                    "action": name,
                    "input": arguments,
                    "input_summary": _tb._arguments_summary(arguments),
                }
                steps.append(step)
                if call_id:
                    by_id[call_id] = step
            tool_result = block.get("toolResult") or block.get("tool_result")
            if isinstance(tool_result, dict):
                call_id = str(tool_result.get("toolUseId") or tool_result.get("tool_use_id") or "")
                step = by_id.get(call_id)
                if step is not None:
                    result_content = tool_result.get("content", [])
                    text = json.dumps(result_content, ensure_ascii=False, default=str)
                    step["result_snippet"] = text[:1600]
                    step["output_chars"] = len(text)
    if steps:
        return steps
    return _tb._conversation_to_steps(conversation)


def _normalize(raw: dict[str, Any], task: GaiaTask, agent: GaiaAgent, output: Path, batch: int, mode: str) -> dict[str, Any]:
    usage = normalize_usage(raw.get("usage", {}))
    conversation = raw.get("conversation", [])
    steps = _conversation_steps(conversation)
    active = list(load_active_config(agent.workspace.root).get("strategies", []) or [])
    summary_step = {"usage": usage, "active_efficiency_strategies": active, "route_model_calls": []}
    steps.append(summary_step)
    predicted = str(raw.get("answer") or "").strip()
    passed = question_scorer(predicted, task.final_answer)
    safe = task.task_id.replace("/", "__")
    conversation_path = _write_artifact(output / "conversations" / f"{safe}.json", conversation)
    result = {
        "instance_id": task.task_id, "task_name": task.task_id, "success": passed,
        "passed": passed, "score": 1.0 if passed else 0.0, "status": "passed" if passed else "failed",
        "model_answer": predicted, "ground_truth": task.final_answer,
        "feedback_detail": "correct" if passed else f"incorrect prediction={predicted!r}",
        "elapsed": float(raw.get("elapsed", 0)), "num_tool_calls": len([s for s in steps if "tool" in s]),
        "num_turns": len([m for m in conversation if m.get("role") == "assistant"]),
        "cumulative_input_tokens": usage.get("input_tokens", 0), "cumulative_output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "cumulative_cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
        "cumulative_cache_write_input_tokens": usage.get("cache_write_input_tokens", 0),
        "cost_units": usage.get("cost_units", 0), "main_cost_units": usage.get("cost_units", 0), "route_model_cost_units": 0,
        "main_model_id": usage.get("model_id", agent.model_id), "main_model_provider": usage.get("provider", ""),
        "main_route_level": usage.get("route_level", "hard"),
        "input_price_per_1m": usage.get("input_price_per_1m", 0), "output_price_per_1m": usage.get("output_price_per_1m", 0),
        "cache_read_price_per_1m": usage.get("cache_read_price_per_1m", 0), "cache_write_price_per_1m": usage.get("cache_write_price_per_1m", 0),
        "cost_currency": usage.get("cost_currency", ""), "cost_price_source": usage.get("cost_price_source", ""),
        "cost_priced": usage.get("cost_priced", False), "usage_record_quality": usage.get("usage_record_quality", "missing"),
        "usage_source": usage.get("usage_source", "gaia_strands_metrics"), "usage_estimated": False,
        "active_efficiency_strategies": active, "strategies": active, "route_model_calls": [],
        "conversation_path": conversation_path, "steps": steps, "batch": batch, "mode": mode,
        "error": raw.get("error", ""), "timed_out": raw.get("timed_out", False),
        "skill_proposal": str(raw.get("skill_proposal") or ""),
        "solver_metrics_usage": raw.get("solver_metrics_usage", {}),
        "solver_message_usage": raw.get("solver_message_usage", {}),
        "proposal_usage": raw.get("proposal_usage", {}),
        "usage_consistency": raw.get("usage_consistency", {}),
        "metadata": {**_core(task).metadata, "file_attached": bool(task.file_path)},
    }
    result = _swe._attach_usage_fields(result)
    result["behavior_stats"] = _swe._behavior_hit_stats(result)
    return result


def _solve_batch(tasks: list[GaiaTask], agent: GaiaAgent, output: Path, batch: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    results = []
    def solve(task: GaiaTask):
        solved = agent.solve_task(question=task.question, attachment=task.file_path, timeout_sec=args.task_timeout)
        return task, {"answer": solved.answer, "raw_output": solved.raw_output, "conversation": solved.conversation,
                      "usage": solved.usage, "elapsed": solved.elapsed, "timed_out": solved.timed_out,
                      "error": solved.error, "skill_proposal": solved.skill_proposal,
                      "solver_metrics_usage": solved.solver_metrics_usage,
                      "solver_message_usage": solved.solver_message_usage,
                      "proposal_usage": solved.proposal_usage,
                      "usage_consistency": solved.usage_consistency}
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(solve, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try: task, raw = future.result()
            except Exception as exc: raw = {"answer":"", "conversation":[], "usage":{}, "elapsed":0, "error":str(exc)}
            result = _normalize(raw, task, agent, output, batch, args.mode); results.append(result)
            print(f"  {'PASS' if result['success'] else 'FAIL'} {task.task_id} cost={result['cost_units']:.4f} tokens={result['total_tokens']:,}")
            with (output / "raw_gaia_results.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({k:v for k,v in result.items() if k not in {'steps'}}, ensure_ascii=False, default=str)+"\n")
    return results


def _observations(results: list[dict[str, Any]], tasks: dict[str, Task]) -> list[Observation]:
    output=[]
    for r in results:
        task=tasks[r["instance_id"]]
        trajectory=Trajectory(task_id=task.id, output=r["model_answer"], steps=r["steps"], conversation=[])
        try: trajectory.conversation=json.loads(Path(r["conversation_path"]).read_text())
        except Exception: pass
        trajectory._skill_proposal=str(r.get("skill_proposal") or "")
        feedback=Feedback(success=r["success"], score=r["score"], detail=r["feedback_detail"], raw={"benchmark":"gaia_2023","level":3})
        output.append(Observation(task=task,trajectory=trajectory,feedback=feedback))
    return output


def _rows(results: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    rows=[]
    for batch in sorted({_safe_int(r["batch"]) for r in results}):
        subset=[r for r in results if _safe_int(r["batch"])==batch]; row={"group":f"{mode}_batch_{batch:02d}","strategies":sorted({s for r in subset for s in r.get('strategies',[])})}; row.update(summarize_results(subset)); row.update(_swe._summarize_behavior_stats(subset)); rows.append(row)
    row={"group":f"{mode}_overall","strategies":sorted({s for r in results for s in r.get('strategies',[])})}; row.update(summarize_results(results)); row.update(_swe._summarize_behavior_stats(results)); rows.append(row); return rows


def _write_outputs(output: Path, results: list[dict[str, Any]], mode: str) -> None:
    _write_artifact(output/"results.json",results); rows=_rows(results,mode)
    _write_artifact(output/"summary.json",{"mode":mode,"benchmark":"gaia_2023_validation_level3","overall":rows[-1],"by_batch":rows[:-1]})
    _write_artifact(output/"behavior_stats.json",{"overall":_swe._summarize_behavior_stats(results),"by_task":[r["behavior_stats"] for r in results]})
    write_summary_table(rows,output/"summary.csv")
    try: write_plots(rows,output/"figures"); write_strategy_library_plot(rows,output/"figures")
    except Exception as exc: logger.warning("Plot generation failed: %s",exc)


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description="GAIA 2023 validation Level 3 emergent-efficiency runner")
    p.add_argument("--mode",choices=["normal-evolve","cost-aware","predefined-control","predefined-cost-aware","category-seeded-cost-aware"],default="cost-aware")
    p.add_argument("--dataset-dir",default=None); p.add_argument("--tasks",default=None); p.add_argument("--exclude",default=None)
    p.add_argument("--limit",type=int,default=None,help="Default None runs all 26 Level 3 validation tasks")
    p.add_argument("--shuffle",action="store_true"); p.add_argument("--shuffle-seed",type=int,default=0)
    p.add_argument("--batch-size",type=int,default=4); p.add_argument("--parallel",type=int,default=2); p.add_argument("--task-timeout",type=int,default=1800)
    p.add_argument("--seed-workspace",default="agent_evolve/agents/gaia/seed_workspace"); p.add_argument("--output-dir",default=DEFAULT_OUTPUT)
    p.add_argument("--fixed-output-dir",action="store_true"); p.add_argument("--overwrite",action="store_true")
    p.add_argument("--model-id",default=DEFAULT_MODEL); p.add_argument("--evolver-model",default=""); p.add_argument("--region",default="us-west-2")
    p.add_argument("--max-tokens",type=int,default=16384); p.add_argument("--evolver-max-tokens",type=int,default=8192); p.add_argument("--evolver-temperature",type=float,default=.2)
    p.add_argument("--evolver-call-retries",type=int,default=4); p.add_argument("--evolver-retry-initial-delay",type=float,default=2); p.add_argument("--evolver-retry-max-delay",type=float,default=30)
    p.add_argument("--max-steps",type=int,default=500); p.add_argument("--max-proposals-per-batch",type=int,default=2)
    p.add_argument("--category-seed",choices=["none","basic-efficiency"],default="basic-efficiency"); p.add_argument("--dry-run-proposer",action="store_true")
    p.add_argument("--predefined-strategies",nargs="*",default=["observation_masking","plan_template_cache","adaptive_tool_trimming","stagnation_rescue_stop","skill_compilation"])
    p.add_argument("-v","--verbose",action="store_true"); return p.parse_args()


def main() -> int:
    args=parse_args(); logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s")
    if args.batch_size<=0 or args.parallel<=0: raise SystemExit("batch-size and parallel must be positive")
    output=_output_dir(args)
    if output.exists() and any(output.iterdir()) and not args.overwrite: raise SystemExit(f"Output exists: {output}")
    if output.exists() and args.overwrite: shutil.rmtree(output)
    output.mkdir(parents=True,exist_ok=True); workspace=output/"workspace"
    hide=args.mode in {"normal-evolve","cost-aware","category-seeded-cost-aware"}; _swe._copy_seed_workspace(Path(args.seed_workspace),workspace,hide_efficiency=hide)
    if args.mode in {"predefined-control","predefined-cost-aware"}: _swe._configure_predefined_control(workspace,args.predefined_strategies,experiment=f"gaia_{args.mode.replace('-','_')}")
    leakage=_swe._strategy_leakage_scan(workspace) if hide else {"passed":True,"hits":[]}; _write_artifact(output/"emergence/leakage_audit.json",leakage)
    if hide and not leakage.get("passed"): raise SystemExit("Strategy leakage audit failed")
    tasks=_filter(load_level3_validation(args.dataset_dir),args)
    if not tasks: raise SystemExit("No GAIA tasks selected")
    batches=[tasks[i:i+args.batch_size] for i in range(0,len(tasks),args.batch_size)]
    manifest={"benchmark":"gaia_2023","split":"validation","level":3,"expected_full_task_count":26,"mode":args.mode,"tasks":[t.task_id for t in tasks],"task_count":len(tasks),"batch_size":args.batch_size,"parallel":args.parallel,"model_id":args.model_id,"evolver_model":args.evolver_model or args.model_id,"seed_workspace":args.seed_workspace,"created_at":datetime.now().isoformat(timespec="seconds")}; _write_artifact(output/"experiment_manifest.json",manifest)
    core={t.task_id:_core(t) for t in tasks}; agent=GaiaAgent(
        workspace,args.model_id,args.region,args.max_tokens,
        # Match SWE: ordinary solver-proposal evolution is shared by all
        # modes; cost-aware/predefined modes add their own modules afterward.
        enable_skill_proposal=True,
    ); all_results=[]
    print(f"GAIA validation Level 3: {len(tasks)} tasks / {len(batches)} batches\nOutput: {output.resolve()}")
    for index,batch_tasks in enumerate(batches,1):
        print(f"=== Batch {index}/{len(batches)} ==="); agent.reload_from_fs(); batch_results=_solve_batch(batch_tasks,agent,output,index,args); all_results.extend(batch_results)
        observations=_observations(batch_results,core)
        evolution_results=[{k:v for k,v in result.items() if k != "ground_truth"} for result in batch_results]
        _tb._write_workspace_evolution_records(workspace_dir=workspace,output_dir=output,batch_results=evolution_results,observations=observations,batch_num=index)
        if index<len(batches):
            _swe._run_normal_evolution(workspace_dir=workspace,output_dir=output,observations=observations,batch_num=index,args=args); agent.reload_from_fs()
        if args.mode in {"cost-aware","predefined-cost-aware","category-seeded-cost-aware"} and index<len(batches):
            report=_swe.propose_emergent_efficiency(batch_results=batch_results,batch_num=index,max_steps=args.max_steps,args=args); _swe._write_proposals(output,report)
            # GAIA evolves prompt/skill/memory; generated executable tools are rejected unless valid Strands tools.
            _swe._apply_proposals(workspace,report,output_dir=output,max_apply=args.max_proposals_per_batch); agent.reload_from_fs()
        if index<len(batches): _swe._record_workspace_snapshot(workspace,output,index)
        _write_outputs(output,all_results,args.mode); summary=summarize_results(batch_results); print(f"  Batch: {summary['success']}/{summary['tasks']} avg_cost={summary['avg_cost']}\n")
    summary=summarize_results(all_results); print(f"Final: {summary['success']}/{summary['tasks']} success_rate={summary['success_rate']} total_cost={summary['total_cost']}"); return 0


if __name__ == "__main__": raise SystemExit(main())
