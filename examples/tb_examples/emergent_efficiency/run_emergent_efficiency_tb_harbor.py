#!/usr/bin/env python3
"""Run Terminal-Bench emergent-efficiency experiments with Harbor evaluation.

This file intentionally does not modify the original Docker-based runner.  It
imports that runner, reuses its task loading, workspace mutation, normal
evolution, and cost-aware modes, and replaces only the batch evaluation backend
with Harbor.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import shlex
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from agent_evolve.agents.harbor_usage import extract_harbor_usage
from agent_evolve.agents.terminal.dataset import TB2Task

logger = logging.getLogger("tb_emergent_efficiency_harbor")

DEFAULT_OUTPUT_ROOT = "logs/tb_emergent_efficiency_harbor"
DEFAULT_HARBOR_UV_CMD = "uv run harbor run"


def _load_tb_runner_module():
    path = Path(__file__).with_name("run_emergent_efficiency_tb.py")
    spec = importlib.util.spec_from_file_location("_tb_emergent_efficiency_docker", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Terminal-Bench emergent runner from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_tb = _load_tb_runner_module()
_original_write_experiment_manifest = _tb._write_experiment_manifest


def _json_default(value: Any) -> str:
    return str(value)


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _write_artifact(path: Path, payload: Any, *, text: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if text:
        path.write_text(str(payload), encoding="utf-8")
    else:
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
    return str(path.resolve())


def _task_dir(task: TB2Task) -> Path:
    if task.tests_dir:
        return Path(task.tests_dir).resolve().parent
    if task.test_sh_path:
        return Path(task.test_sh_path).resolve().parents[1]
    return Path.cwd()


def _deep_format(value: Any, context: dict[str, str]) -> Any:
    if isinstance(value, str):
        try:
            return value.format(**context)
        except KeyError:
            return value
    if isinstance(value, list):
        return [_deep_format(item, context) for item in value]
    if isinstance(value, dict):
        return {key: _deep_format(item, context) for key, item in value.items()}
    return value


def _format_harbor_command(template: str, context: dict[str, str]) -> list[str]:
    formatted = template.format(**context)
    return shlex.split(formatted, posix=os.name != "nt")


def _resolve_job_dir(jobs_dir: Path, job_name: str) -> Path:
    direct = jobs_dir / job_name
    if direct.exists():
        return direct
    candidates = sorted(
        jobs_dir.glob(f"{job_name}*"),
        key=lambda item: item.stat().st_mtime if item.exists() else 0,
        reverse=True,
    )
    return candidates[0] if candidates else direct


def _find_result_json(job_dir: Path, task_name: str) -> Path | None:
    if not job_dir.exists():
        return None
    paths = sorted(job_dir.rglob("result.json"))
    if not paths:
        return None
    task_lower = task_name.lower()
    for path in paths:
        if task_lower in str(path.parent).lower():
            return path
    return paths[0]


def _find_trajectory_json(trial_dir: Path) -> Path | None:
    preferred = trial_dir / "agent" / "trajectory.json"
    if preferred.exists():
        return preferred
    paths = sorted(trial_dir.rglob("trajectory.json"))
    return paths[0] if paths else None


def _extract_nested_dict(data: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if isinstance(value, dict):
                return value
        for value in data.values():
            nested = _extract_nested_dict(value, keys)
            if nested:
                return nested
    if isinstance(data, list):
        for item in data:
            nested = _extract_nested_dict(item, keys)
            if nested:
                return nested
    return {}


def _extract_number(data: Any, keys: tuple[str, ...]) -> float | None:
    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    pass
        for value in data.values():
            nested = _extract_number(value, keys)
            if nested is not None:
                return nested
    if isinstance(data, list):
        for item in data:
            nested = _extract_number(item, keys)
            if nested is not None:
                return nested
    return None


def _extract_bool(data: Any, keys: tuple[str, ...]) -> bool | None:
    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.lower() in {"true", "false", "pass", "fail", "passed", "failed"}:
                return value.lower() in {"true", "pass", "passed"}
        for value in data.values():
            nested = _extract_bool(value, keys)
            if nested is not None:
                return nested
    if isinstance(data, list):
        for item in data:
            nested = _extract_bool(item, keys)
            if nested is not None:
                return nested
    return None


def _extract_text(data: Any, keys: tuple[str, ...]) -> str:
    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in data.values():
            nested = _extract_text(value, keys)
            if nested:
                return nested
    if isinstance(data, list):
        for item in data:
            nested = _extract_text(item, keys)
            if nested:
                return nested
    return ""


def _generic_trajectory_steps(trajectory: Any) -> list[dict[str, Any]]:
    data = trajectory if isinstance(trajectory, list) else []
    steps: list[dict[str, Any]] = []
    pending: dict[str, dict[str, Any]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        tool_name = item.get("tool") or item.get("tool_name") or item.get("action") or item.get("name")
        if tool_name:
            step = {
                "tool": str(tool_name),
                "action": str(tool_name),
                "input": item.get("input") or item.get("arguments") or item.get("args") or {},
                "input_summary": str(item.get("input") or item.get("arguments") or "")[:800],
                "result_snippet": str(item.get("result") or item.get("output") or item.get("observation") or "")[:1600],
            }
            steps.append(step)
            continue
        content = item.get("content", item.get("parts", []))
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            tool_use = block.get("tool_use") or block.get("toolUse")
            if isinstance(tool_use, dict):
                call_id = str(tool_use.get("id") or tool_use.get("toolUseId") or "")
                step = {
                    "tool": str(tool_use.get("name") or ""),
                    "action": str(tool_use.get("name") or ""),
                    "input": tool_use.get("input", {}),
                    "input_summary": str(tool_use.get("input", {}))[:800],
                }
                steps.append(step)
                if call_id:
                    pending[call_id] = step
            tool_result = block.get("tool_result") or block.get("toolResult")
            if isinstance(tool_result, dict):
                call_id = str(tool_result.get("tool_use_id") or tool_result.get("toolUseId") or "")
                step = pending.get(call_id)
                if step is not None:
                    step["result_snippet"] = str(tool_result.get("content", ""))[:1600]
    return steps


def _load_trajectory(trial_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    path = _find_trajectory_json(trial_dir)
    if path is None:
        return [], [], ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return [], [], str(path)
    if isinstance(data, dict):
        for key in ("conversation", "messages", "trajectory", "events"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    conversation = data if isinstance(data, list) else []
    steps = _tb._conversation_to_steps(conversation)
    if not steps:
        steps = _generic_trajectory_steps(conversation)
    return steps, conversation, str(path)


def _build_harbor_config(
    *,
    task: TB2Task,
    args: argparse.Namespace,
    job_name: str,
    jobs_dir: Path,
    workspace_dir: Path,
    system_prompt_path: Path,
    task_json_path: Path,
    context: dict[str, str],
) -> dict[str, Any]:
    if args.harbor_config_template:
        template_path = Path(args.harbor_config_template)
        base = yaml.safe_load(template_path.read_text(encoding="utf-8")) or {}
        if not isinstance(base, dict):
            raise ValueError(f"Harbor config template must be a YAML mapping: {template_path}")
        base = _deep_format(base, context)
    else:
        if not args.harbor_agent_import_path:
            raise ValueError("--harbor-agent-import-path or --harbor-config-template is required unless --dry-run-harbor is set")
        base = {}

    base["job_name"] = job_name
    base["jobs_dir"] = str(jobs_dir)
    base.setdefault("n_attempts", 1)
    base.setdefault("timeout_multiplier", 1.0)
    base.setdefault(
        "orchestrator",
        {
            "type": "local",
            "n_concurrent_trials": 1,
            "quiet": True,
            "kwargs": {},
        },
    )

    agent_kwargs = {
        "workspace_dir": str(workspace_dir.resolve()),
        "skills_dir": str((workspace_dir / "skills").resolve()),
        "system_prompt_path": str(system_prompt_path.resolve()),
        "task_json_path": str(task_json_path.resolve()),
        "task_name": task.name,
        "max_steps": args.max_steps,
        "aws_region_name": args.region,
        "delegate_agent_name": args.harbor_delegate_agent_name,
        "delegate_model_name": args.harbor_delegate_model_name or args.harbor_model_name or args.model_id,
    }
    if args.harbor_delegate_agent_import_path:
        agent_kwargs["delegate_agent_import_path"] = args.harbor_delegate_agent_import_path
    if args.harbor_delegate_kwargs:
        agent_kwargs["delegate_kwargs"] = args.harbor_delegate_kwargs
    if args.harbor_agent_import_path:
        base["agents"] = [
            {
                "name": args.harbor_agent_name,
                "import_path": args.harbor_agent_import_path,
                "model_name": args.harbor_model_name or args.model_id,
                "override_timeout_sec": args.harbor_timeout_sec,
                "kwargs": agent_kwargs,
            }
        ]
    else:
        for agent in base.get("agents", []) if isinstance(base.get("agents"), list) else []:
            if isinstance(agent, dict):
                kwargs = agent.setdefault("kwargs", {})
                if isinstance(kwargs, dict):
                    kwargs.update(agent_kwargs)
                agent.setdefault("model_name", args.harbor_model_name or args.model_id)

    task_dir = _task_dir(task)
    dataset_path = args.challenges_dir or str(task_dir.parent)
    base["datasets"] = [
        {
            "path": dataset_path,
            "task_names": [task.name],
            "exclude_task_names": [],
            "kwargs": {
                "task_name": task.name,
                "task_dir": str(task_dir),
                "task_json_path": str(task_json_path.resolve()),
                "docker_image": task.docker_image,
                "test_sh_path": task.test_sh_path,
                "tests_dir": task.tests_dir or "",
            },
        }
    ]
    base.setdefault("metadata", {})
    if isinstance(base["metadata"], dict):
        base["metadata"].update(
            {
                "benchmark": "terminal_bench_2_1",
                "execution_backend": "harbor",
                "task_name": task.name,
                "workspace_dir": str(workspace_dir.resolve()),
            }
        )
    return base


def _dry_run_harbor_result(task: TB2Task, args: argparse.Namespace) -> dict[str, Any]:
    digest = hashlib.sha256(task.name.encode("utf-8")).hexdigest()
    passed = int(digest[:2], 16) % 3 != 0
    tool_calls = 4 + (int(digest[2:4], 16) % 5)
    conversation: list[dict[str, Any]] = []
    for index in range(tool_calls):
        call_id = f"call-{index}"
        conversation.append(
            {
                "role": "assistant",
                "content": f"Dry-run step {index}",
                "tool_calls": [
                    {
                        "id": call_id,
                        "name": "bash",
                        "arguments": {"cmd": f"echo dry-run-{index}"},
                    }
                ],
            }
        )
        conversation.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": "dry-run command output",
            }
        )
    usage = {
        "input_tokens": 1200 + tool_calls * 90,
        "output_tokens": 220 + tool_calls * 25,
        "total_tokens": 1420 + tool_calls * 115,
        "cost_units": round(0.01 + tool_calls * 0.002, 4),
        "usage_record_quality": "estimated",
        "usage_source": "terminal_bench_harbor_dry_run",
        "usage_estimated": True,
    }
    return {
        "task_name": task.name,
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "status": "passed" if passed else "failed",
        "eval_output": "dry-run Harbor result",
        "usage": usage,
        "solve_time": 0.0,
        "total_time": 0.0,
        "conversation_turns": tool_calls,
        "conversation": conversation,
        "steps": _tb._conversation_to_steps(conversation),
        "skill_draft": "",
        "metadata": {"dry_run_harbor": True, "model_id": args.model_id},
        "raw_job_path": "",
        "result_json_path": "",
        "trajectory_path": "",
        "returncode": 0,
        "harbor_command": ["dry-run-harbor"],
    }


def _run_harbor_task(
    *,
    task: TB2Task,
    batch_num: int,
    workspace_dir: Path,
    output_dir: Path,
    system_prompt_text: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.dry_run_harbor:
        return _dry_run_harbor_result(task, args)

    if not args.harbor_repo:
        raise ValueError("--harbor-repo is required unless --dry-run-harbor is set")
    harbor_repo = Path(args.harbor_repo).resolve()
    if not harbor_repo.exists():
        raise FileNotFoundError(f"Harbor repo not found: {harbor_repo}")

    safe_task = _tb._slug(task.name, default=f"task-{batch_num:02d}")
    job_name = f"aev-tb21-{safe_task}-{int(time.time())}"
    jobs_dir = Path(args.harbor_jobs_dir or output_dir / "harbor_jobs").resolve()
    jobs_dir.mkdir(parents=True, exist_ok=True)

    task_artifact_dir = output_dir / "harbor_inputs" / safe_task
    task_json_path = task_artifact_dir / "task.json"
    system_prompt_path = task_artifact_dir / "system_prompt.md"
    task_payload = {
        "name": task.name,
        "prompt": task.prompt,
        "docker_image": task.docker_image,
        "test_sh_path": task.test_sh_path,
        "test_py_path": task.test_py_path,
        "tests_dir": task.tests_dir,
        "files": task.files,
        "metadata": task.metadata,
        "timeout": task.timeout,
    }
    _write_artifact(task_json_path, task_payload)
    _write_artifact(system_prompt_path, system_prompt_text, text=True)

    task_dir = _task_dir(task)
    context = {
        "repo_root": str(REPO_ROOT.resolve()),
        "harbor_repo": str(harbor_repo),
        "workspace_dir": str(workspace_dir.resolve()),
        "task_name": task.name,
        "task_id": task.name,
        "task_dir": str(task_dir),
        "task_json_path": str(task_json_path.resolve()),
        "system_prompt_path": str(system_prompt_path.resolve()),
        "job_name": job_name,
        "jobs_dir": str(jobs_dir),
        "model_id": args.model_id,
        "harbor_model_name": args.harbor_model_name or args.model_id,
        "region": args.region,
        "output_dir": str(output_dir.resolve()),
    }
    config_data = _build_harbor_config(
        task=task,
        args=args,
        job_name=job_name,
        jobs_dir=jobs_dir,
        workspace_dir=workspace_dir,
        system_prompt_path=system_prompt_path,
        task_json_path=task_json_path,
        context=context,
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="aev-tb21-harbor-",
        delete=False,
        encoding="utf-8",
    ) as handle:
        yaml.safe_dump(config_data, handle, sort_keys=False)
        config_path = Path(handle.name)
    context["config_path"] = str(config_path)

    if args.harbor_command_template:
        cmd = _format_harbor_command(args.harbor_command_template, context)
    else:
        cmd = [*shlex.split(args.harbor_uv_cmd), "-c", str(config_path)]

    started = time.time()
    stdout = ""
    stderr = ""
    returncode: int | None = None
    try:
        env = {**os.environ, "AWS_REGION": args.region}
        pythonpath_parts = [str(REPO_ROOT.resolve())]
        if env.get("PYTHONPATH"):
            pythonpath_parts.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
        completed = subprocess.run(
            cmd,
            cwd=str(harbor_repo),
            capture_output=True,
            check=False,
            text=True,
            timeout=int(args.harbor_timeout_sec),
            env=env,
        )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else str(exc)
        returncode = -1
    finally:
        try:
            config_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.debug("Could not remove temporary Harbor config %s: %s", config_path, exc)

    job_dir = _resolve_job_dir(jobs_dir, job_name)
    result_json_path = None
    if args.harbor_result_json_template:
        result_json_path = Path(args.harbor_result_json_template.format(**context))
    if result_json_path is None or not result_json_path.exists():
        result_json_path = _find_result_json(job_dir, task.name)

    result_data: dict[str, Any] = {}
    trial_dir = job_dir
    if result_json_path and result_json_path.exists():
        trial_dir = result_json_path.parent
        try:
            parsed = json.loads(result_json_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                result_data = parsed
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            result_data = {"parse_error": str(exc)}

    score = _extract_number(result_data, ("reward", "score", "final_score", "accuracy"))
    if score is None:
        score = 0.0
    passed = _extract_bool(result_data, ("passed", "success", "pass_binary"))
    if passed is None:
        passed = score >= 1.0
    eval_output = _extract_text(
        result_data,
        ("detail", "message", "eval_output", "stdout", "stderr", "failure_reason", "result_state"),
    )
    if not eval_output:
        eval_output = "\n".join([stdout[-1500:], stderr[-1500:]]).strip()
    if not eval_output:
        eval_output = json.dumps(result_data, ensure_ascii=False, default=str)[:2000]

    steps, conversation, trajectory_path = _load_trajectory(trial_dir)
    usage_model_id = args.harbor_delegate_model_name or args.harbor_model_name or args.model_id
    usage = extract_harbor_usage(
        result_data,
        trial_dir=trial_dir,
        model_id=usage_model_id,
        usage_source="terminal_bench_harbor",
        include_codex_artifacts=True,
    )
    skill_draft = _extract_text(result_data, ("skill_proposal", "skill_draft", "proposal"))

    return {
        "task_name": task.name,
        "passed": bool(passed),
        "score": float(score or 0.0),
        "status": "passed" if passed else "failed",
        "eval_output": eval_output,
        "usage": usage,
        "solve_time": round(time.time() - started, 1),
        "total_time": round(time.time() - started, 1),
        "conversation_turns": len([entry for entry in conversation if isinstance(entry, dict) and entry.get("role") == "assistant"]),
        "conversation": conversation,
        "steps": steps,
        "skill_draft": skill_draft,
        "metadata": {"harbor": result_data.get("metadata", {}) if isinstance(result_data.get("metadata"), dict) else {}},
        "raw_job_path": str(trial_dir),
        "result_json_path": str(result_json_path) if result_json_path else "",
        "trajectory_path": trajectory_path,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": returncode,
        "harbor_command": cmd,
    }


def _normalize_harbor_tb_result(
    *,
    raw: dict[str, Any],
    task: TB2Task,
    workspace_dir: Path,
    output_dir: Path,
    system_prompt_text: str,
    model_id: str,
    batch_num: int,
    mode: str,
) -> dict[str, Any]:
    result = _tb._normalize_tb_result(
        raw=raw,
        task=task,
        workspace_dir=workspace_dir,
        output_dir=output_dir,
        system_prompt_text=system_prompt_text,
        model_id=model_id,
        batch_num=batch_num,
        mode=mode,
    )
    raw_path = _write_artifact(
        output_dir / "raw_harbor_results" / f"{task.name.replace('/', '_')}.json",
        raw,
    )
    raw_steps = raw.get("steps") if isinstance(raw.get("steps"), list) else []
    if raw_steps:
        summary_step = result["steps"][-1] if result.get("steps") else {}
        if isinstance(summary_step, dict) and isinstance(summary_step.get("usage"), dict):
            summary_step["usage"]["usage_source"] = "terminal_bench_harbor"
        result["steps"] = raw_steps + ([summary_step] if summary_step else [])
        result["num_tool_calls"] = len([step for step in raw_steps if isinstance(step, dict) and step.get("tool")])
    result["success"] = bool(raw.get("passed"))
    result["passed"] = bool(raw.get("passed"))
    result["score"] = _safe_float(raw.get("score"))
    result["usage_source"] = "terminal_bench_harbor"
    result["raw_harbor_result_path"] = raw_path
    result["raw_job_path"] = raw.get("raw_job_path", "")
    result["result_json_path"] = raw.get("result_json_path", "")
    result["trajectory_path"] = raw.get("trajectory_path", "")
    result["harbor_command"] = raw.get("harbor_command", [])
    result["metadata"].update(
        {
            "execution_backend": "harbor",
            "benchmark": "terminal_bench_2_1",
            "docker_image": task.docker_image,
            "docker_image_not_pulled_by_runner": True,
            "harbor": raw.get("metadata", {}).get("harbor", {}),
            "returncode": raw.get("returncode"),
        }
    )
    result["behavior_stats"] = _tb._swe._behavior_hit_stats(result)
    return result


def _solve_terminal_batch_harbor(
    *,
    batch_tasks: list[TB2Task],
    batch_num: int,
    workspace_dir: Path,
    output_dir: Path,
    system_prompt_text: str,
    agent: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    _ = agent
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(
                _run_harbor_task,
                task=task,
                batch_num=batch_num,
                workspace_dir=workspace_dir,
                output_dir=output_dir,
                system_prompt_text=system_prompt_text,
                args=args,
            ): task
            for task in batch_tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                raw = future.result()
            except Exception as exc:  # noqa: BLE001
                raw = {
                    "task_name": task.name,
                    "passed": False,
                    "score": 0.0,
                    "eval_output": f"ERROR: {exc}",
                    "usage": {},
                    "solve_time": 0,
                    "total_time": 0,
                    "conversation_turns": 0,
                    "conversation": [],
                    "steps": [],
                    "metadata": {"exception": str(exc)},
                    "status": "error",
                }
            result = _normalize_harbor_tb_result(
                raw=raw,
                task=task,
                workspace_dir=workspace_dir,
                output_dir=output_dir,
                system_prompt_text=system_prompt_text,
                model_id=args.model_id,
                batch_num=batch_num,
                mode=args.mode,
            )
            results.append(result)
            status = "PASS" if result.get("success") else "FAIL"
            print(
                f"  {status} {task.name} "
                f"score={_safe_float(result.get('score')):.2f} "
                f"cost={_safe_float(result.get('cost_units')):,.4f} "
                f"tokens={_safe_int(result.get('total_tokens')):,d} "
                f"calls={_safe_int(result.get('num_tool_calls'))}"
            )
    return results


def _write_experiment_manifest_harbor(output_dir: Path, args: argparse.Namespace, tasks: list[TB2Task]) -> None:
    _original_write_experiment_manifest(output_dir, args, tasks)
    path = output_dir / "experiment_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(
        {
            "execution_backend": "harbor",
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
            "note": "Evaluation is delegated to Harbor; this runner does not pull or run task Docker images directly.",
        }
    )
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emergent efficiency Terminal-Bench 2.1 experiment runner using Harbor evaluation"
    )
    parser.add_argument(
        "--mode",
        choices=[
            "normal-evolve",
            "cost-aware",
            "predefined-control",
            "predefined-cost-aware",
            "category-seeded-cost-aware",
        ],
        default="cost-aware",
    )
    parser.add_argument(
        "--challenges-dir",
        default=None,
        help="Path to Terminal-Bench 2.1 tasks directory (auto-downloads when omitted)",
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
    parser.add_argument("--overwrite", action="store_true", help="Only with --fixed-output-dir; removes existing output dir")
    parser.add_argument("--model-id", default=_tb.DEFAULT_MODEL_ID)
    parser.add_argument("--evolver-model", default="")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--evolver-max-tokens", type=int, default=8192)
    parser.add_argument("--evolver-temperature", type=float, default=0.2)
    parser.add_argument("--evolver-call-retries", type=int, default=4)
    parser.add_argument("--evolver-retry-initial-delay", type=float, default=2.0)
    parser.add_argument("--evolver-retry-max-delay", type=float, default=30.0)
    parser.add_argument("--max-steps", type=int, default=500, help="Tool-call limit used for waste-signal thresholds")
    parser.add_argument("--max-proposals-per-batch", type=int, default=2)
    parser.add_argument(
        "--category-seed",
        choices=["none", "basic-efficiency"],
        default="basic-efficiency",
        help="Taxonomy exposed only in category-seeded-cost-aware mode",
    )
    parser.add_argument("--dry-run-proposer", action="store_true", help="Use deterministic dummy proposal")
    parser.add_argument("--propose-skill", action="store_true", default=True, help="Ask solver for skill drafts")
    parser.add_argument("--no-propose-skill", dest="propose_skill", action="store_false")
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
    parser.add_argument("--harbor-agent-import-path", default=os.environ.get("TB_HARBOR_AGENT_IMPORT_PATH", ""))
    parser.add_argument("--harbor-agent-name", default="terminal-bench-agent")
    parser.add_argument("--harbor-model-name", default=os.environ.get("HARBOR_MODEL_NAME", ""))
    parser.add_argument(
        "--harbor-delegate-agent-name",
        default=os.environ.get("TB_HARBOR_DELEGATE_AGENT_NAME", "codex"),
        help="Harbor built-in agent used by the A-Evolve adapter, default: codex",
    )
    parser.add_argument(
        "--harbor-delegate-agent-import-path",
        default=os.environ.get("TB_HARBOR_DELEGATE_AGENT_IMPORT_PATH", ""),
        help="Optional custom Harbor agent import path delegated to by the A-Evolve adapter",
    )
    parser.add_argument(
        "--harbor-delegate-model-name",
        default=os.environ.get("TB_HARBOR_DELEGATE_MODEL_NAME", ""),
        help="Optional model name for the delegated Harbor agent",
    )
    parser.add_argument(
        "--harbor-delegate-kwargs",
        default=os.environ.get("TB_HARBOR_DELEGATE_KWARGS", ""),
        help="Optional JSON/YAML kwargs passed through to the delegated Harbor agent",
    )
    parser.add_argument("--harbor-jobs-dir", default=None)
    parser.add_argument("--harbor-timeout-sec", type=int, default=3600)
    parser.add_argument("--harbor-uv-cmd", default=os.environ.get("HARBOR_UV_CMD", DEFAULT_HARBOR_UV_CMD))
    parser.add_argument("--dry-run-harbor", action="store_true", help="Use deterministic synthetic Harbor results for smoke tests")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    _tb.parse_args = parse_args
    _tb._solve_terminal_batch = _solve_terminal_batch_harbor
    _tb._write_experiment_manifest = _write_experiment_manifest_harbor
    return _tb.main()


if __name__ == "__main__":
    raise SystemExit(main())
