#!/usr/bin/env python3
"""Run emergent-efficiency experiments on OSWorld 2.x through Harbor.

This runner mirrors the SWE and Terminal-Bench emergent-efficiency experiment
matrix, but delegates task execution and evaluation to Harbor.  The A-Evolve
side owns batching, workspace mutation, cost-aware proposal generation, and
summary artifacts; Harbor owns sandboxed OSWorld task execution and scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from agent_evolve.agents.harbor_usage import extract_harbor_usage
from agent_evolve.contract.workspace import AgentWorkspace
from agent_evolve.efficiency.profiler import (
    normalize_usage,
    result_strategy_hits,
    summarize_results,
)
from agent_evolve.efficiency.strategies import load_active_config
from agent_evolve.efficiency.visualization import (
    write_plots,
    write_strategy_library_plot,
    write_summary_table,
)
from agent_evolve.llm.base import LLMMessage
from agent_evolve.llm.factory import create_llm_provider, resolve_model_selection
from agent_evolve.types import Feedback, Observation, Task, Trajectory

logger = logging.getLogger("osworld2_emergent_efficiency")

DEFAULT_OUTPUT_ROOT = "logs/osworld2_emergent_efficiency"
DEFAULT_MODEL_ID = "us.anthropic.claude-opus-4-6-v1"
DEFAULT_HARBOR_UV_CMD = "uv run harbor run"
DEFAULT_HARBOR_AGENT_IMPORT_PATH = (
    "agent_evolve.agents.osworld2.harbor_adapter:AEvolveOSWorld2HarborAgent"
)
DEFAULT_HARBOR_DELEGATE_AGENT_IMPORT_PATH = "osworld.custom_agent:OSWorldAgent"


def _load_swe_runner_module():
    path = REPO_ROOT / "examples" / "swe_examples_new" / "emergent_efficiency" / "run_emergent_efficiency.py"
    spec = importlib.util.spec_from_file_location("_osworld2_swe_emergent", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load SWE emergent runner from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_swe = _load_swe_runner_module()


OSWORLD2_EMERGENCE_SYSTEM_PROMPT = """\
You are an experiment assistant for an OSWorld 2.x GUI desktop agent evaluated
through Harbor.

Goal: discover reusable cost-saving behavior changes from evidence, without
using any named predefined strategy library.

You may propose reusable GUI skills, short prompt fragments, compact memory
notes, or Harbor-agent-compatible tools. Each proposal must preserve task
success first. Avoid shortcuts that merely stop early, skip visual
verification, ignore task requirements, or reduce screenshots until the agent
can no longer inspect the desktop.

For kind=tool, only propose code if it is compatible with the configured
Harbor OSWorld agent workspace. Prefer skill, prompt_fragment, or memory when
the execution contract is uncertain.

Output JSON only:
{
  "proposals": [
    {
      "name": "short_generic_name",
      "kind": "skill" | "prompt_fragment" | "tool" | "memory",
      "description": "one sentence with trigger and do-not-trigger conditions",
      "trigger": "when to use it",
      "do_not_trigger": "when not to use it",
      "content": "procedure under 350 words, or complete compatible tool source",
      "parameters": {"optional_arg": {"type": "string", "description": "optional for kind=tool"}},
      "required": ["optional_arg"],
      "expected_savings": "why this should reduce cost",
      "correctness_risk": "what could go wrong",
      "confidence": "HIGH" | "MEDIUM" | "LOW",
      "evidence": ["task_id: observed waste -> proposed fix"]
    }
  ]
}

Return {"proposals": []} when there is no cross-task efficiency pattern.
"""

OSWORLD2_PREDEFINED_PROMPT = OSWORLD2_EMERGENCE_SYSTEM_PROMPT.replace(
    "without\nusing any named predefined strategy library.",
    "while predefined runtime efficiency strategies are already active.",
).replace(
    "why this should reduce cost",
    "why this should reduce cost beyond active strategies",
).replace(
    "observed waste -> proposed fix",
    "observed waste -> proposed incremental fix",
)

OSWORLD2_CATEGORY_PROMPT = OSWORLD2_EMERGENCE_SYSTEM_PROMPT.replace(
    "without\nusing any named predefined strategy library.",
    "with only broad M1-M7 efficiency categories, not predefined strategy names or implementations.",
).replace(
    '"description": "one sentence with trigger and do-not-trigger conditions",',
    '"categories": ["M5 Tools"],\n      "description": "one sentence with trigger and do-not-trigger conditions",',
).replace(
    '"expected_savings": "why this should reduce cost",',
    '"expected_savings": "which category metric should improve and why",',
)

_swe.EMERGENCE_SYSTEM_PROMPT = OSWORLD2_EMERGENCE_SYSTEM_PROMPT
_swe.PREDEFINED_COST_AWARE_SYSTEM_PROMPT = OSWORLD2_PREDEFINED_PROMPT
_swe.CATEGORY_SEEDED_SYSTEM_PROMPT = OSWORLD2_CATEGORY_PROMPT


@dataclass(frozen=True)
class OSWorld2Task:
    task_id: str
    instruction: str
    domain: str
    config_path: str
    raw: dict[str, Any]


@dataclass
class OSWorld2SkillMeta:
    name: str
    description: str
    path: str
    body: str = ""


OSWORLD2_BASE_AGENT_PROMPT = """\
You are an OSWorld 2.x desktop automation agent evaluated through Harbor.
Complete the requested GUI task carefully and preserve correctness above speed.

Use available skills when they match the application or workflow. Verify the
final visible state before finishing. Avoid repeated identical actions and do
not stop early unless the task requirements are visibly satisfied.
"""

OSWORLD2_PROPOSE_SYSTEM_PROMPT = """\
You are a skill extraction agent for an OSWorld GUI desktop agent.
Analyze task attempts and distill reusable skills.

A good skill is specific, actionable, transferable, and short. It should
capture concrete application knowledge such as menu paths, shortcuts, dialog
sequences, accessibility names, or non-obvious workflows.

Do not write generic advice about being careful, looking at screenshots,
using the computer tool, retrying, or basic mouse/keyboard operations.
If nothing useful was learned, output ACTION: NONE.
"""

OSWORLD2_ANALYZE_AND_PROPOSE_PROMPT = """\
The evaluation result for this task:

{eval_result}

## Trajectory signals
{trajectory_signals}

## Compressed trajectory
{compressed_trajectory}

{existing_skills_section}

## Step 1: Analyze the result
For each distinct issue or missing technique, output:
ISSUE: <one-line summary>
DETAIL: <specific GUI action, UI element, or application technique>

## Step 2: Propose a skill
Based on the analysis, write one short skill for future OSWorld tasks.

TOPIC: <broad application-level topic, e.g. libreoffice-calc, chrome, gimp, vlc, os-desktop>
ACTION: NEW / ENHANCE / NONE
TARGET: existing_skill_name (only for ENHANCE)
NAME: short-kebab-name (only for NEW)
DESCRIPTION: one sentence saying when this skill applies
CONTENT:
## Key techniques
- (specific menu paths, shortcuts, dialog sequences, or UI element names)
## Gotchas
- (specific pitfalls and how to avoid them)

Rules:
- Bullet points only. CONTENT must be under 200 words.
- Prefer ENHANCE when an existing skill is related.
- TOPIC should be broad and application-level.
- Output ACTION: NONE if no transferable application-specific knowledge was learned.
"""

OSWORLD2_ANALYZE_AND_PROPOSE_PASS_PROMPT = """\
The evaluation result for this task:

PASSED (score={score:.1f})

## Trajectory signals
{trajectory_signals}

## Compressed trajectory
{compressed_trajectory}

{existing_skills_section}

Analyze what worked. If the trajectory contains reusable, non-obvious
application knowledge worth preserving, propose one skill:

TOPIC: <broad application-level topic>
ACTION: NEW / ENHANCE / NONE
TARGET: existing_skill_name (only for ENHANCE)
NAME: short-kebab-name (only for NEW)
DESCRIPTION: one sentence saying when this skill applies
CONTENT:
## Key techniques
- (specific steps that worked)
## Gotchas
- (obstacles encountered and how they were avoided)

Only propose if the technique is transferable and not already covered.
Otherwise output ACTION: NONE.
"""

OSWORLD2_CURATOR_PROMPT = """\
You are a skill curator for a GUI task-solving agent on OSWorld 2.x.
Review skill proposals and decide which to keep for topic: {topic}.

## Current Skill Library ({n_skills}/{max_skills} slots used)
{existing_skills_list}

## Proposals from this batch
{proposals_list}

For each proposal, output one decision:

ACCEPT: <proposal_name>

MERGE: <proposal_name> INTO <existing_skill_name>
NEW_CONTENT:
(merged content, under 200 words, bullet points only)

SKIP: <proposal_name>
REASON: <brief reason>

Rules:
- Prefer merging related proposals into broader skills.
- Skip generic advice, basic GUI operations, screenshot usage advice, and retry-only advice.
- Keep skills short, specific, and application-focused.
- If no proposals are useful, output NO_PROPOSALS.
"""

OSWORLD2_GENERAL_CURATOR_PROMPT = """\
You are a meta-learning curator for OSWorld 2.x desktop tasks.
Analyze repeated failure patterns across tasks and distill general skills.

## Failed Task Analysis ({n_failed} tasks)
{failed_summaries}

## Current General Skills ({n_general}/{max_general} slots)
{general_skills_list}

For repeated patterns across at least two tasks, output:

NEW_GENERAL: <kebab-name>
DESCRIPTION: <one sentence saying when this skill applies>
CONTENT:
## Pattern
- (what failure type)
## Strategy
- (3-5 concrete GUI techniques or checks)

UPDATE_GENERAL: <existing-name>
NEW_CONTENT:
(updated content, under 200 words)

DELETE_GENERAL: <existing-name>
REASON: <why>

If no repeated pattern is useful, output NO_PATTERNS.
"""


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


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


def _slug(value: str, *, default: str = "task", limit: int = 80) -> str:
    return _swe._slug(value, default=default, limit=limit)


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


def _read_text_tail(path: Path, *, limit: int = 12000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    if len(text) <= limit:
        return text
    return text[-limit:]


def _load_harbor_task_toml(task_dir: Path) -> dict[str, Any]:
    task_toml = task_dir / "task.toml"
    if not task_toml.exists():
        return {}
    try:
        with task_toml.open("rb") as handle:
            loaded = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _timeout_from_task_config(config: dict[str, Any], section: str, key: str) -> float:
    section_data = config.get(section)
    if not isinstance(section_data, dict):
        return 0.0
    return _safe_float(section_data.get(key))


def _harbor_task_timeout_budget(task: OSWorld2Task) -> int:
    config = task.raw.get("task_toml") if isinstance(task.raw, dict) else {}
    if not isinstance(config, dict) or not config:
        config_path = Path(task.config_path)
        if config_path.is_dir():
            config = _load_harbor_task_toml(config_path)
    if not isinstance(config, dict):
        config = {}

    environment_build = _timeout_from_task_config(config, "environment", "build_timeout_sec")
    agent = _timeout_from_task_config(config, "agent", "timeout_sec")
    verifier = _timeout_from_task_config(config, "verifier", "timeout_sec")
    setup = _timeout_from_task_config(config, "agent", "setup_timeout_sec")
    # Harbor can legitimately spend the full build budget before the agent
    # emits a single trajectory event. Add a small buffer for orchestration,
    # log flushing, and post-run result collection.
    total = environment_build + setup + agent + verifier
    if total <= 0:
        return 3600
    return int(total + 600)


def _effective_harbor_timeout_sec(task: OSWorld2Task, args: argparse.Namespace) -> int:
    configured = int(args.harbor_timeout_sec or 0)
    if configured > 0:
        return configured
    return max(3600, _harbor_task_timeout_budget(task))


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except Exception:  # noqa: BLE001
        try:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            logger.debug("Could not force-kill Harbor subprocess", exc_info=True)


def _run_command_to_logs(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_sec: int,
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[int, bool, float]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with stdout_path.open("w", encoding="utf-8", errors="ignore") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8", errors="ignore"
    ) as stderr_handle:
        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                start_new_session=os.name != "nt",
            )
        except OSError as exc:
            stderr_handle.write(str(exc))
            return 127, False, round(time.time() - started, 1)
        try:
            returncode = process.wait(timeout=timeout_sec)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process(process)
            returncode = -1
    return returncode, timed_out, round(time.time() - started, 1)


def _external_volume_names_from_compose(compose_path: Path) -> list[str]:
    if not compose_path.exists():
        return []
    try:
        loaded = yaml.safe_load(compose_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
        return []
    if not isinstance(loaded, dict):
        return []
    volumes = loaded.get("volumes", {})
    if not isinstance(volumes, dict):
        return []

    names: list[str] = []
    for alias, config in volumes.items():
        if isinstance(config, dict):
            external = config.get("external")
            if isinstance(external, dict):
                external_name = external.get("name") or config.get("name") or alias
                is_external = True
            else:
                external_name = config.get("name") or alias
                is_external = bool(external)
            if is_external and isinstance(external_name, str) and external_name.strip():
                names.append(external_name.strip())
        elif config is True and isinstance(alias, str) and alias.strip():
            names.append(alias.strip())
    return sorted(set(names))


def _task_external_docker_volumes(task: OSWorld2Task) -> list[str]:
    task_config_path = Path(task.config_path)
    if not task_config_path.is_dir():
        return []
    return _external_volume_names_from_compose(task_config_path / "environment" / "docker-compose.yaml")


def _ensure_docker_volume(name: str) -> dict[str, Any]:
    inspected = subprocess.run(
        ["docker", "volume", "inspect", name],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    if inspected.returncode == 0:
        return {"name": name, "status": "exists"}

    created = subprocess.run(
        ["docker", "volume", "create", name],
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    if created.returncode != 0:
        detail = (created.stderr or created.stdout or inspected.stderr or inspected.stdout or "").strip()
        raise RuntimeError(f"Could not create required Docker volume {name!r}: {detail}")
    return {"name": name, "status": "created", "output": (created.stdout or "").strip()}


def _ensure_harbor_external_volumes(task: OSWorld2Task) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for volume_name in _task_external_docker_volumes(task):
        statuses.append(_ensure_docker_volume(volume_name))
    return statuses


def _device_source_from_compose_entry(entry: Any) -> str:
    if isinstance(entry, str):
        return entry.split(":", 1)[0].strip()
    if isinstance(entry, dict):
        value = entry.get("path_on_host") or entry.get("source") or entry.get("host_path")
        return str(value).strip() if value else ""
    return ""


def _task_required_host_devices(task: OSWorld2Task) -> list[str]:
    task_config_path = Path(task.config_path)
    if not task_config_path.is_dir():
        return []
    compose_path = task_config_path / "environment" / "docker-compose.yaml"
    if not compose_path.exists():
        return []
    try:
        loaded = yaml.safe_load(compose_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
        return []
    services = loaded.get("services", {}) if isinstance(loaded, dict) else {}
    if not isinstance(services, dict):
        return []
    devices: list[str] = []
    for service in services.values():
        if not isinstance(service, dict):
            continue
        for entry in service.get("devices", []) or []:
            source = _device_source_from_compose_entry(entry)
            if source.startswith("/dev/"):
                devices.append(source)
    return sorted(set(devices))


def _check_harbor_host_devices(task: OSWorld2Task) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for device in _task_required_host_devices(task):
        statuses.append(
            {
                "path": device,
                "exists": Path(device).exists(),
            }
        )
    return statuses


def _missing_host_devices(device_statuses: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("path")) for item in device_statuses if not item.get("exists")]


def _harbor_venv_executable(harbor_repo: Path) -> Path | None:
    candidates = [
        harbor_repo / ".venv" / "bin" / "harbor",
        harbor_repo / ".venv" / "Scripts" / "harbor.exe",
        harbor_repo / ".venv" / "Scripts" / "harbor",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_harbor_run_prefix(raw_cmd: str, harbor_repo: Path) -> tuple[list[str], str]:
    parts = shlex.split(raw_cmd)
    if not parts:
        return parts, "empty"
    if parts[:3] == ["uv", "run", "harbor"] and shutil.which("uv") is None:
        harbor_executable = _harbor_venv_executable(harbor_repo)
        if harbor_executable is not None:
            harbor_args = parts[3:] or ["run"]
            return [str(harbor_executable), *harbor_args], "harbor_repo_venv_fallback"
    return parts, "configured"


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    raw_output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT
    output_dir = Path(raw_output_dir)
    if args.fixed_output_dir:
        return output_dir
    stamp = _timestamp()
    mode_slug = args.mode.replace("-", "_")
    if raw_output_dir == DEFAULT_OUTPUT_ROOT:
        return output_dir / f"{mode_slug}_{stamp}"
    return output_dir.with_name(f"{output_dir.name}_{stamp}")


def _candidate_config_paths(base_dir: Path, domain: str, task_id: str) -> list[Path]:
    safe_id = str(task_id)
    return [
        base_dir / "examples" / domain / f"{safe_id}.json",
        base_dir / "tasks" / domain / f"{safe_id}.json",
        base_dir / domain / f"{safe_id}.json",
        base_dir / "examples" / f"{safe_id}.json",
        base_dir / "tasks" / f"{safe_id}.json",
        base_dir / f"{safe_id}.json",
    ]


def _task_id_from_config(config: dict[str, Any], fallback: str) -> str:
    for key in ("id", "task_id", "name", "uuid", "example_id", "instance_id"):
        value = config.get(key)
        if value:
            return str(value)
    return fallback


def _instruction_from_config(config: dict[str, Any]) -> str:
    for key in ("instruction", "task", "question", "goal", "prompt", "description"):
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _domain_from_config(config: dict[str, Any], fallback: str = "unknown") -> str:
    for key in ("domain", "category", "app", "application", "suite"):
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    apps = config.get("related_apps")
    if isinstance(apps, list) and apps:
        return str(apps[0])
    return fallback


def _normalize_task(
    config: dict[str, Any],
    *,
    fallback_id: str,
    domain: str,
    config_path: Path | None,
) -> OSWorld2Task:
    task_id = _task_id_from_config(config, fallback_id)
    clean_domain = _domain_from_config(config, domain)
    instruction = _instruction_from_config(config)
    raw = dict(config)
    raw.setdefault("id", task_id)
    raw.setdefault("domain", clean_domain)
    if config_path:
        raw.setdefault("config_path", str(config_path))
    return OSWorld2Task(
        task_id=task_id,
        instruction=instruction,
        domain=clean_domain,
        config_path=str(config_path) if config_path else "",
        raw=raw,
    )


def _load_task_config(base_dir: Path, domain: str, task_id: str) -> tuple[dict[str, Any], Path | None]:
    for path in _candidate_config_paths(base_dir, domain, task_id):
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                return data, path
    return {"id": task_id, "domain": domain, "instruction": ""}, None


def _load_task_file(task_file: Path) -> list[OSWorld2Task]:
    if task_file.suffix.lower() == ".jsonl":
        tasks = []
        with task_file.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                data = json.loads(line)
                if isinstance(data, dict):
                    tasks.append(
                        _normalize_task(
                            data,
                            fallback_id=f"task-{index:04d}",
                            domain=_domain_from_config(data),
                            config_path=task_file,
                        )
                    )
        return tasks

    with task_file.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        tasks = []
        for index, item in enumerate(payload, start=1):
            if isinstance(item, dict):
                tasks.append(
                    _normalize_task(
                        item,
                        fallback_id=f"task-{index:04d}",
                        domain=_domain_from_config(item),
                        config_path=task_file,
                    )
                )
            else:
                tasks.append(
                    OSWorld2Task(
                        task_id=str(item),
                        instruction="",
                        domain="unknown",
                        config_path="",
                        raw={"id": str(item), "domain": "unknown"},
                    )
                )
        return tasks

    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported task file shape: {task_file}")

    if isinstance(payload.get("tasks"), list):
        return [
            _normalize_task(
                item,
                fallback_id=f"task-{index:04d}",
                domain=_domain_from_config(item),
                config_path=task_file,
            )
            for index, item in enumerate(payload["tasks"], start=1)
            if isinstance(item, dict)
        ]

    if _instruction_from_config(payload):
        return [
            _normalize_task(
                payload,
                fallback_id=task_file.stem,
                domain=_domain_from_config(payload),
                config_path=task_file,
            )
        ]

    base_dir = task_file.parent
    tasks: list[OSWorld2Task] = []
    for domain, items in payload.items():
        if not isinstance(items, list):
            continue
        for index, item in enumerate(items, start=1):
            if isinstance(item, dict):
                tasks.append(
                    _normalize_task(
                        item,
                        fallback_id=f"{domain}-{index:04d}",
                        domain=str(domain),
                        config_path=task_file,
                    )
                )
                continue
            task_id = str(item)
            config, path = _load_task_config(base_dir, str(domain), task_id)
            tasks.append(
                _normalize_task(
                    config,
                    fallback_id=task_id,
                    domain=str(domain),
                    config_path=path,
                )
            )
    return tasks


def _instruction_from_harbor_task_dir(task_dir: Path) -> str:
    instruction_path = task_dir / "instruction.md"
    if not instruction_path.exists():
        return ""
    try:
        return instruction_path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return ""


def _domain_from_harbor_task_name(task_name: str) -> str:
    if "__" in task_name:
        return task_name.split("__", 1)[0] or "unknown"
    if "/" in task_name:
        return task_name.split("/", 1)[0] or "unknown"
    if "-" in task_name:
        return task_name.split("-", 1)[0] or "unknown"
    return "unknown"


def _load_harbor_task_dirs(dataset_root: Path) -> list[OSWorld2Task]:
    if not dataset_root.exists():
        return []
    tasks: list[OSWorld2Task] = []
    for task_dir in sorted(item for item in dataset_root.iterdir() if item.is_dir()):
        if not (task_dir / "task.toml").exists():
            continue
        task_id = task_dir.name
        domain = _domain_from_harbor_task_name(task_id)
        instruction = _instruction_from_harbor_task_dir(task_dir)
        task_toml = _load_harbor_task_toml(task_dir)
        raw = {
            "id": task_id,
            "task_id": task_id,
            "domain": domain,
            "instruction": instruction,
            "config_path": str(task_dir),
            "harbor_task_dir": str(task_dir),
            "task_source": "harbor_task_dir",
            "task_toml": task_toml,
        }
        tasks.append(
            OSWorld2Task(
                task_id=task_id,
                instruction=instruction,
                domain=domain,
                config_path=str(task_dir),
                raw=raw,
            )
        )
    return tasks


def _resolve_task_file(args: argparse.Namespace) -> Path | None:
    if args.task_file:
        return Path(args.task_file)
    env_task_file = os.environ.get("OSWORLD2_TASK_FILE")
    if env_task_file:
        return Path(env_task_file)
    root_value = args.osworld2_path or os.environ.get("OSWORLD2_PATH") or os.environ.get("OSWORLD_V2_PATH")
    if root_value:
        root = Path(root_value)
        candidates = [
            root / "evaluation_examples" / "test_all.json",
            root / "evaluation_examples" / "test_all_v2.json",
            root / "data" / "test_all.json",
            root / "data" / "tasks.json",
            root / "tasks" / "test_all.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return None


def _load_tasks_from_args(args: argparse.Namespace) -> tuple[list[OSWorld2Task], str]:
    task_file = _resolve_task_file(args)
    if task_file is not None:
        return _load_task_file(task_file), str(task_file)

    root_value = args.osworld2_path or os.environ.get("OSWORLD2_PATH") or os.environ.get("OSWORLD_V2_PATH")
    if root_value:
        root = Path(root_value)
        harbor_tasks = _load_harbor_task_dirs(root)
        if harbor_tasks:
            return harbor_tasks, f"{root} [harbor_task_dirs]"

    raise SystemExit(
        "No OSWorld2 task file or Harbor task directory found. Pass --task-file, "
        "set OSWORLD2_TASK_FILE, or set OSWORLD2_PATH to a Harbor dataset root "
        "whose children contain task.toml."
    )


def _filter_tasks(tasks: list[OSWorld2Task], args: argparse.Namespace) -> list[OSWorld2Task]:
    if args.domain:
        wanted_domains = {item.strip() for item in args.domain.split(",") if item.strip()}
        tasks = [task for task in tasks if task.domain in wanted_domains]
    if args.tasks:
        selected = {item.strip() for item in args.tasks.split(",") if item.strip()}
        tasks = [task for task in tasks if task.task_id in selected]
    if args.exclude:
        excluded = {item.strip() for item in args.exclude.split(",") if item.strip()}
        tasks = [task for task in tasks if task.task_id not in excluded]
    if args.shuffle:
        random.Random(args.shuffle_seed).shuffle(tasks)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    return tasks


def _core_task(task: OSWorld2Task) -> Task:
    return Task(
        id=task.task_id,
        input=task.instruction,
        metadata={
            "benchmark": "osworld2",
            "execution_backend": "harbor",
            "domain": task.domain,
            "config_path": task.config_path,
            "raw_task": task.raw,
        },
    )


def _strip_skill_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    text = str(content or "").strip()
    if not text.startswith("---"):
        return {}, text
    end = text.find("---", 3)
    if end == -1:
        return {}, text
    raw_meta = text[3:end].strip()
    body = text[end + 3 :].strip()
    try:
        parsed = yaml.safe_load(raw_meta) or {}
    except yaml.YAMLError:
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}, body


def _load_osworld2_skills(workspace_dir: Path) -> list[OSWorld2SkillMeta]:
    skills_dir = workspace_dir / "skills"
    if not skills_dir.exists():
        return []
    skills: list[OSWorld2SkillMeta] = []
    for skill_file in sorted(skills_dir.rglob("SKILL.md")):
        try:
            content = skill_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = skill_file.read_text(errors="ignore")
        meta, body = _strip_skill_frontmatter(content)
        name = str(meta.get("name") or skill_file.parent.name).strip() or skill_file.parent.name
        description = str(meta.get("description") or "").strip()
        if not description:
            for line in content.splitlines():
                if line.strip().lower().startswith("description:"):
                    description = line.split(":", 1)[1].strip()
                    break
        rel_path = str(skill_file.parent.relative_to(workspace_dir))
        skills.append(OSWorld2SkillMeta(name=name, description=description, path=rel_path, body=body or content))
    return skills


def _init_osworld2_seed_skills(workspace_dir: Path, seed_workspace: Path) -> None:
    """Copy OSWorld seed skills into workspace/skills/seed/ like the OSWorld agent framework."""
    seed_skills_src = seed_workspace / "skills"
    seed_skills_dst = workspace_dir / "skills" / "seed"
    if seed_skills_dst.exists():
        return
    if not seed_skills_src.exists():
        logger.warning("Seed skills not found at %s", seed_skills_src)
        return
    seed_skills_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(seed_skills_src, seed_skills_dst)
    logger.info(
        "Copied %d OSWorld seed skills into %s",
        len(list(seed_skills_dst.rglob("SKILL.md"))),
        seed_skills_dst,
    )


def _copy_osworld2_seed_workspace(seed_workspace: Path, workspace_dir: Path, *, hide_efficiency: bool) -> None:
    """Initialize a workspace while preserving the OSWorld skills/seed layout."""
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    if seed_workspace.exists():
        for child in seed_workspace.iterdir():
            if child.name == "skills":
                continue
            destination = workspace_dir / child.name
            if child.is_dir():
                shutil.copytree(child, destination)
            else:
                shutil.copy2(child, destination)
        _init_osworld2_seed_skills(workspace_dir, seed_workspace)
    (workspace_dir / "skills").mkdir(parents=True, exist_ok=True)
    if hide_efficiency:
        _swe._redact_predefined_names(workspace_dir)
        active = workspace_dir / "efficiency" / "active_strategies.yaml"
        if active.exists():
            active.unlink()


def _topic_from_skill_path(path: str) -> str:
    parts = Path(path).parts
    if len(parts) >= 4 and parts[0] == "skills" and parts[1] == "topic":
        return parts[2]
    return "other"


def _append_skill_bodies(parts: list[str], title: str, skills: list[OSWorld2SkillMeta]) -> None:
    if not skills:
        return
    parts.append(f"\n\n## {title}")
    for skill in sorted(skills, key=lambda item: (item.path, item.name)):
        content = skill.body.strip() or skill.description
        parts.append(f"\n### {skill.name}\n{content}")


def _build_osworld2_system_prompt(
    workspace_dir: Path,
    *,
    base_prompt: str | None = None,
    selected_topics: list[str] | None = None,
) -> str:
    """Build the OSWorld-style prompt from seed, topic, general, and evolved skills."""
    base = (base_prompt if base_prompt is not None else AgentWorkspace(workspace_dir).read_prompt()).strip()
    parts = [base or OSWORLD2_BASE_AGENT_PROMPT.strip()]
    skills = _load_osworld2_skills(workspace_dir)
    if not skills:
        return "\n".join(parts)

    seed_skills = [skill for skill in skills if Path(skill.path).parts[:2] == ("skills", "seed")]
    topic_skills = [skill for skill in skills if Path(skill.path).parts[:2] == ("skills", "topic")]
    general_skills = [skill for skill in skills if Path(skill.path).parts[:2] == ("skills", "general")]
    evolved_skills = [skill for skill in skills if Path(skill.path).parts[:2] == ("skills", "evolved")]
    direct_skills = [
        skill
        for skill in skills
        if skill not in seed_skills
        and skill not in topic_skills
        and skill not in general_skills
        and skill not in evolved_skills
        and not Path(skill.path).name.startswith("_")
    ]

    _append_skill_bodies(parts, "Core skills", seed_skills)
    if topic_skills:
        by_topic: dict[str, list[OSWorld2SkillMeta]] = defaultdict(list)
        for skill in topic_skills:
            topic = _topic_from_skill_path(skill.path)
            if selected_topics is None or topic in selected_topics:
                by_topic[topic].append(skill)
        if by_topic:
            parts.append("\n\n## Domain skills")
            for topic in sorted(by_topic):
                parts.append(f"\n**[{topic}]**")
                for skill in sorted(by_topic[topic], key=lambda item: item.name):
                    parts.append(f"\n### {skill.name}\n{skill.body.strip() or skill.description}")

    _append_skill_bodies(parts, "General strategies", general_skills)
    _append_skill_bodies(parts, "Evolved skills", evolved_skills)
    _append_skill_bodies(parts, "Workspace skills", direct_skills)
    if any(skill.name == "self-verification" for skill in seed_skills + general_skills + evolved_skills + direct_skills):
        parts.append("\n\nBefore finishing, use the self-verification skill when it applies.")
    return "\n".join(part for part in parts if str(part).strip())


def _ensure_workspace_prompt(workspace_dir: Path) -> str:
    workspace = AgentWorkspace(workspace_dir)
    if workspace.read_prompt().strip():
        return workspace.read_prompt()

    prompt = OSWORLD2_BASE_AGENT_PROMPT.strip() + "\n"
    workspace.write_prompt(prompt)
    return prompt


def _workspace_system_prompt(workspace_dir: Path) -> str:
    workspace = AgentWorkspace(workspace_dir)
    parts = [_build_osworld2_system_prompt(workspace_dir, base_prompt=workspace.read_prompt())]
    memories = workspace.read_all_memories(limit=30)
    if memories:
        parts.append("\n## Memories")
        for memory in memories:
            parts.append(f"- {memory.get('content') or memory.get('insight') or memory}")
    fragments = workspace.list_fragments()
    if fragments:
        parts.append("\n## Prompt Fragments")
        for name in fragments:
            parts.append(workspace.read_fragment(name))
    return "\n".join(part for part in parts if str(part).strip())


def _cost_price_profile(workspace_dir: Path) -> dict[str, Any]:
    path = workspace_dir / "efficiency" / "cost_profile.yaml"
    if not path.exists():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


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


def _build_harbor_config(
    *,
    task: OSWorld2Task,
    args: argparse.Namespace,
    job_name: str,
    jobs_dir: Path,
    workspace_dir: Path,
    system_prompt_path: Path,
    task_json_path: Path,
    context: dict[str, str],
    effective_timeout_sec: int,
) -> dict[str, Any]:
    if args.harbor_config_template:
        template_path = Path(args.harbor_config_template)
        base = yaml.safe_load(template_path.read_text(encoding="utf-8")) or {}
        if not isinstance(base, dict):
            raise ValueError(f"Harbor config template must be a YAML mapping: {template_path}")
        base = _deep_format(base, context)
    else:
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
            "quiet": False,
            "kwargs": {},
        },
    )

    agent_kwargs = {
        "workspace_dir": str(workspace_dir.resolve()),
        "system_prompt_path": str(system_prompt_path.resolve()),
        "skills_dir": str((workspace_dir / "skills").resolve()),
        "task_json_path": str(task_json_path.resolve()),
        "task_config_path": task.config_path,
        "task_id": task.task_id,
        "domain": task.domain,
        "max_steps": args.max_steps,
        "aws_region_name": args.region,
        "delegate_agent_import_path": args.harbor_delegate_agent_import_path,
        "delegate_model_name": args.harbor_delegate_model_name or args.harbor_model_name or args.model_id,
    }
    if args.harbor_delegate_kwargs:
        agent_kwargs["delegate_kwargs"] = args.harbor_delegate_kwargs
    if args.harbor_agent_import_path:
        agent_config = {
            "name": args.harbor_agent_name,
            "import_path": args.harbor_agent_import_path,
            "model_name": args.harbor_model_name or args.model_id,
            "kwargs": agent_kwargs,
        }
        if int(args.harbor_timeout_sec or 0) > 0:
            agent_config["override_timeout_sec"] = effective_timeout_sec
        base["agents"] = [agent_config]
    else:
        for agent in base.get("agents", []) if isinstance(base.get("agents"), list) else []:
            if isinstance(agent, dict):
                kwargs = agent.setdefault("kwargs", {})
                if isinstance(kwargs, dict):
                    kwargs.update(agent_kwargs)
                agent.setdefault("model_name", args.harbor_model_name or args.model_id)

    dataset_path = args.osworld2_path or str(Path(task.config_path).parent if task.config_path else Path("."))
    base["datasets"] = [
        {
            "path": dataset_path,
            "task_names": [task.task_id],
            "exclude_task_names": [],
            "kwargs": {
                "task_id": task.task_id,
                "task_config_path": task.config_path,
                "task_json_path": str(task_json_path.resolve()),
                "domain": task.domain,
            },
        }
    ]
    base.setdefault("metadata", {})
    if isinstance(base["metadata"], dict):
        base["metadata"].update(
            {
                "benchmark": "osworld2",
                "execution_backend": "harbor",
                "task_id": task.task_id,
                "workspace_dir": str(workspace_dir.resolve()),
            }
        )
    return base


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


def _find_result_json(job_dir: Path, task_id: str) -> Path | None:
    if not job_dir.exists():
        return None
    result_paths = sorted(job_dir.rglob("result.json"))
    if not result_paths:
        return None
    task_lower = task_id.lower()
    for path in result_paths:
        if path.parent.name.lower().startswith(task_lower + "__"):
            return path
    for path in result_paths:
        if task_lower in str(path.parent).lower():
            return path
    return result_paths[0]


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
            if isinstance(value, str) and value.lower() in {"true", "false", "pass", "fail"}:
                return value.lower() in {"true", "pass"}
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


def _trajectory_to_steps(trajectory: Any) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    if not isinstance(trajectory, list):
        return steps
    pending_by_id: dict[str, dict[str, Any]] = {}

    for item in trajectory:
        if not isinstance(item, dict):
            continue
        tool_name = (
            item.get("tool")
            or item.get("tool_name")
            or item.get("action")
            or item.get("name")
        )
        if tool_name:
            step = {
                "tool": str(tool_name),
                "action": str(tool_name),
                "input": item.get("input") or item.get("arguments") or item.get("args") or {},
                "input_summary": str(item.get("input") or item.get("arguments") or "")[:800],
                "result_snippet": str(
                    item.get("result") or item.get("output") or item.get("observation") or ""
                )[:1600],
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
                name = str(tool_use.get("name") or "")
                call_id = str(tool_use.get("id") or tool_use.get("toolUseId") or "")
                step = {
                    "tool": name,
                    "action": name,
                    "input": tool_use.get("input", {}),
                    "input_summary": str(tool_use.get("input", {}))[:800],
                }
                steps.append(step)
                if call_id:
                    pending_by_id[call_id] = step
                continue
            tool_result = block.get("tool_result") or block.get("toolResult")
            if isinstance(tool_result, dict):
                call_id = str(tool_result.get("tool_use_id") or tool_result.get("toolUseId") or "")
                step = pending_by_id.get(call_id)
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
    conversation = data if isinstance(data, list) else []
    steps = _trajectory_to_steps(data)
    return steps, conversation, str(path)


def _classify_failure(eval_output: str, returncode: int | None = None) -> str:
    lowered = (eval_output or "").lower()
    if "missing required host device" in lowered or "/dev/kvm" in lowered:
        return "host_device_missing"
    if "external volume" in lowered and "not found" in lowered:
        return "harbor_external_volume_missing"
    if returncode == -1 or "timeout" in lowered or "timed out" in lowered:
        return "harbor_timeout" if returncode not in (None, 0) else "verifier_timeout"
    if returncode not in (None, 0):
        return "harbor_runtime_error"
    if "assertionerror" in lowered or re.search(r"\bassert\b", lowered):
        return "assertion"
    if "filenotfounderror" in lowered or "no such file or directory" in lowered:
        return "file_missing"
    if "failed" in lowered and "pytest" in lowered:
        return "test_fail"
    return "verifier_fail"


def _dry_run_harbor_result(task: OSWorld2Task, args: argparse.Namespace) -> dict[str, Any]:
    digest = hashlib.sha256(task.task_id.encode("utf-8")).hexdigest()
    passed = int(digest[:2], 16) % 3 != 0
    action_count = 3 + (int(digest[2:4], 16) % 4)
    steps = [
        {
            "tool": "computer",
            "action": "screenshot",
            "input": {"action": "screenshot"},
            "result_snippet": "dry-run screenshot observation",
        },
        {
            "tool": "computer",
            "action": "left_click",
            "input": {"action": "left_click", "coordinate": [100, 100]},
            "result_snippet": "dry-run click observation",
        },
    ]
    for index in range(action_count - 2):
        steps.append(
            {
                "tool": "computer",
                "action": "type",
                "input": {"action": "type", "text": f"dry-run-{index}"},
                "result_snippet": "dry-run type observation",
            }
        )
    usage = {
        "input_tokens": 1000 + action_count * 100,
        "output_tokens": 250 + action_count * 25,
        "total_tokens": 1250 + action_count * 125,
        "cost_units": round(0.01 + action_count * 0.002, 4),
        "usage_record_quality": "estimated",
        "usage_source": "osworld2_harbor_dry_run",
        "usage_estimated": True,
    }
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "status": "passed" if passed else "failed",
        "eval_output": "dry-run Harbor result",
        "llm_output": "dry-run OSWorld2 task output",
        "usage": usage,
        "steps": steps,
        "conversation": [],
        "raw_job_path": "",
        "trajectory_path": "",
        "stdout": "",
        "stderr": "",
        "returncode": 0,
        "elapsed": 0.0,
        "harbor_command": ["dry-run-harbor"],
        "harbor_config_path": "",
        "metadata": {"dry_run_harbor": True, "model_id": args.model_id},
    }


def _run_harbor_task(
    *,
    task: OSWorld2Task,
    batch_num: int,
    workspace_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.dry_run_harbor:
        return _dry_run_harbor_result(task, args)

    if not args.harbor_repo:
        raise ValueError("--harbor-repo is required unless --dry-run-harbor is set")
    harbor_repo = Path(args.harbor_repo).resolve()
    if not harbor_repo.exists():
        raise FileNotFoundError(f"Harbor repo not found: {harbor_repo}")

    safe_task = _slug(task.task_id, default=f"task-{batch_num:02d}")
    job_name = f"aev-osworld2-{safe_task}-{int(time.time())}"
    jobs_dir = Path(args.harbor_jobs_dir or output_dir / "harbor_jobs").resolve()
    jobs_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = jobs_dir / "_a_evolve_runner" / job_name
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    external_volume_statuses: list[dict[str, Any]] = []
    if args.harbor_ensure_external_volumes:
        external_volume_statuses = _ensure_harbor_external_volumes(task)
    _write_artifact(diagnostics_dir / "external_volumes.json", external_volume_statuses)
    host_device_statuses: list[dict[str, Any]] = []
    if args.harbor_check_host_devices:
        host_device_statuses = _check_harbor_host_devices(task)
    _write_artifact(diagnostics_dir / "host_devices.json", host_device_statuses)
    missing_host_devices = _missing_host_devices(host_device_statuses)
    if missing_host_devices:
        job_dir = jobs_dir / job_name
        job_diagnostics_dir = job_dir / "a_evolve_runner"
        job_diagnostics_dir.mkdir(parents=True, exist_ok=True)
        for source in (diagnostics_dir / "external_volumes.json", diagnostics_dir / "host_devices.json"):
            try:
                shutil.copy2(source, job_diagnostics_dir / source.name)
            except OSError as exc:
                logger.debug("Could not copy Harbor preflight diagnostic %s: %s", source, exc)
        detail = (
            "Missing required host device(s) for Harbor OSWorld2 Docker environment: "
            f"{', '.join(missing_host_devices)}. Enable nested virtualization/KVM on "
            "the GPU server, or run OSWorld2 on a KVM-capable host."
        )
        return {
            "passed": False,
            "score": 0.0,
            "status": "failed",
            "eval_output": detail,
            "llm_output": "",
            "usage": {},
            "steps": [],
            "conversation": [],
            "raw_job_path": str(job_dir),
            "trajectory_path": "",
            "result_json_path": "",
            "stdout": "",
            "stderr": detail,
            "returncode": 126,
            "elapsed": 0.0,
            "harbor_command": [],
            "harbor_config_path": "",
            "skill_proposal": "",
            "metadata": {
                "job_name": job_name,
                "jobs_dir": str(jobs_dir),
                "result_json_found": False,
                "runner_diagnostics_dir": str(job_diagnostics_dir),
                "external_volumes": external_volume_statuses,
                "host_devices": host_device_statuses,
                "missing_host_devices": missing_host_devices,
            },
        }

    task_artifact_dir = output_dir / "harbor_inputs" / safe_task
    task_json_path = task_artifact_dir / "task.json"
    system_prompt_path = task_artifact_dir / "system_prompt.md"
    _write_artifact(task_json_path, task.raw)
    _write_artifact(system_prompt_path, _workspace_system_prompt(workspace_dir), text=True)

    context = {
        "repo_root": str(REPO_ROOT.resolve()),
        "harbor_repo": str(harbor_repo),
        "osworld2_path": args.osworld2_path or "",
        "workspace_dir": str(workspace_dir.resolve()),
        "skills_dir": str((workspace_dir / "skills").resolve()),
        "task_id": task.task_id,
        "task_name": task.task_id,
        "domain": task.domain,
        "task_config_path": task.config_path,
        "task_json_path": str(task_json_path.resolve()),
        "system_prompt_path": str(system_prompt_path.resolve()),
        "job_name": job_name,
        "jobs_dir": str(jobs_dir),
        "model_id": args.model_id,
        "harbor_model_name": args.harbor_model_name or args.model_id,
        "region": args.region,
        "output_dir": str(output_dir.resolve()),
    }
    effective_timeout_sec = _effective_harbor_timeout_sec(task, args)
    context["effective_harbor_timeout_sec"] = str(effective_timeout_sec)

    config_data = _build_harbor_config(
        task=task,
        args=args,
        job_name=job_name,
        jobs_dir=jobs_dir,
        workspace_dir=workspace_dir,
        system_prompt_path=system_prompt_path,
        task_json_path=task_json_path,
        context=context,
        effective_timeout_sec=effective_timeout_sec,
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="aev-osworld2-harbor-",
        delete=False,
        encoding="utf-8",
    ) as handle:
        yaml.safe_dump(config_data, handle, sort_keys=False)
        config_path = Path(handle.name)
    context["config_path"] = str(config_path)
    retained_config_path = diagnostics_dir / "harbor_config.yaml"
    retained_config_path.write_text(
        yaml.safe_dump(config_data, sort_keys=False),
        encoding="utf-8",
    )

    if args.harbor_command_template:
        cmd = _format_harbor_command(args.harbor_command_template, context)
        harbor_command_source = "template"
    else:
        harbor_prefix, harbor_command_source = _resolve_harbor_run_prefix(args.harbor_uv_cmd, harbor_repo)
        cmd = [*harbor_prefix, "-c", str(config_path)]
        if args.harbor_auto_confirm:
            cmd.append("--yes")
        if args.verbose:
            cmd.append("--debug")
    _write_artifact(
        diagnostics_dir / "harbor_command.json",
        {
            "cmd": cmd,
            "cwd": str(harbor_repo),
            "effective_harbor_timeout_sec": effective_timeout_sec,
            "configured_harbor_timeout_sec": args.harbor_timeout_sec,
            "task_timeout_budget_sec": _harbor_task_timeout_budget(task),
            "harbor_auto_confirm": args.harbor_auto_confirm,
            "harbor_command_source": harbor_command_source,
            "configured_harbor_uv_cmd": args.harbor_uv_cmd,
            "external_volumes": external_volume_statuses,
            "host_devices": host_device_statuses,
        },
    )

    started = time.time()
    stdout = ""
    stderr = ""
    returncode: int | None = None
    timed_out = False
    stdout_path = diagnostics_dir / "harbor_stdout.log"
    stderr_path = diagnostics_dir / "harbor_stderr.log"
    try:
        env = {**os.environ, "AWS_REGION": args.region}
        pythonpath_parts = [str(REPO_ROOT.resolve())]
        osworld_adapter_src = harbor_repo / "adapters" / "osworld" / "src"
        if osworld_adapter_src.exists():
            pythonpath_parts.append(str(osworld_adapter_src))
        if env.get("PYTHONPATH"):
            pythonpath_parts.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
        returncode, timed_out, _elapsed = _run_command_to_logs(
            cmd,
            cwd=harbor_repo,
            env=env,
            timeout_sec=effective_timeout_sec,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        stdout = _read_text_tail(stdout_path)
        stderr = _read_text_tail(stderr_path)
    finally:
        try:
            config_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.debug("Could not remove temporary Harbor config %s: %s", config_path, exc)

    job_dir = _resolve_job_dir(jobs_dir, job_name)
    if not job_dir.exists():
        job_dir.mkdir(parents=True, exist_ok=True)
    job_diagnostics_dir = job_dir / "a_evolve_runner"
    job_diagnostics_dir.mkdir(parents=True, exist_ok=True)
    for source in (
        retained_config_path,
        diagnostics_dir / "harbor_command.json",
        diagnostics_dir / "external_volumes.json",
        diagnostics_dir / "host_devices.json",
        stdout_path,
        stderr_path,
    ):
        try:
            shutil.copy2(source, job_diagnostics_dir / source.name)
        except OSError as exc:
            logger.debug("Could not copy Harbor diagnostic %s: %s", source, exc)
    if timed_out:
        timeout_payload = {
            "timed_out": True,
            "effective_harbor_timeout_sec": effective_timeout_sec,
            "configured_harbor_timeout_sec": args.harbor_timeout_sec,
            "task_timeout_budget_sec": _harbor_task_timeout_budget(task),
            "elapsed": round(time.time() - started, 1),
        }
        _write_artifact(diagnostics_dir / "timeout.json", timeout_payload)
        _write_artifact(job_diagnostics_dir / "timeout.json", timeout_payload)
    result_json_path = None
    if args.harbor_result_json_template:
        result_json_path = Path(args.harbor_result_json_template.format(**context))
    if result_json_path is None or not result_json_path.exists():
        result_json_path = _find_result_json(job_dir, task.task_id)

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
    if not eval_output and timed_out:
        eval_output = f"Command {cmd!r} timed out after {effective_timeout_sec} seconds"
    if not eval_output:
        eval_output = "\n".join([stdout[-1500:], stderr[-1500:]]).strip()
    if not eval_output and not (result_json_path and result_json_path.exists()):
        eval_output = (
            f"Harbor command returned {returncode} but no result.json was found "
            f"under {job_dir}. See runner diagnostics in {job_diagnostics_dir}."
        )
    if not eval_output:
        eval_output = json.dumps(result_data, ensure_ascii=False, default=str)[:2000]

    steps, conversation, trajectory_path = _load_trajectory(trial_dir)
    usage_model_id = args.harbor_delegate_model_name or args.harbor_model_name or args.model_id
    raw_usage = extract_harbor_usage(
        result_data,
        trial_dir=trial_dir,
        model_id=usage_model_id,
        usage_source="osworld2_harbor",
        include_codex_artifacts=True,
    )
    llm_output = _extract_text(result_data, ("llm_output", "output", "answer", "response", "text"))
    skill_proposal = _extract_text(result_data, ("skill_proposal", "skill_draft", "proposal"))

    return {
        "passed": bool(passed),
        "score": float(score or 0.0),
        "status": "passed" if passed else "failed",
        "eval_output": eval_output,
        "llm_output": llm_output,
        "usage": raw_usage,
        "steps": steps,
        "conversation": conversation,
        "raw_job_path": str(trial_dir),
        "trajectory_path": trajectory_path,
        "result_json_path": str(result_json_path) if result_json_path else "",
        "stdout": stdout,
        "stderr": stderr,
        "returncode": returncode,
        "elapsed": round(time.time() - started, 1),
        "harbor_command": cmd,
        "harbor_config_path": str(job_diagnostics_dir / "harbor_config.yaml"),
        "skill_proposal": skill_proposal,
        "metadata": {
            "job_name": job_name,
            "jobs_dir": str(jobs_dir),
            "result_json_found": bool(result_json_path and result_json_path.exists()),
            "timed_out": timed_out,
            "effective_harbor_timeout_sec": effective_timeout_sec,
            "configured_harbor_timeout_sec": args.harbor_timeout_sec,
            "task_timeout_budget_sec": _harbor_task_timeout_budget(task),
            "runner_diagnostics_dir": str(job_diagnostics_dir),
            "stdout_path": str(job_diagnostics_dir / "harbor_stdout.log"),
            "stderr_path": str(job_diagnostics_dir / "harbor_stderr.log"),
            "external_volumes": external_volume_statuses,
            "host_devices": host_device_statuses,
        },
    }


def _usage_quality(usage: dict[str, Any]) -> str:
    if usage.get("usage_record_quality"):
        return str(usage["usage_record_quality"])
    if _safe_int(usage.get("input_tokens")) or _safe_int(usage.get("output_tokens")):
        return "ok"
    if _safe_float(usage.get("cost_units")):
        return "cost_only"
    return "missing"


def _normalize_harbor_result(
    *,
    raw: dict[str, Any],
    task: OSWorld2Task,
    workspace_dir: Path,
    output_dir: Path,
    model_id: str,
    batch_num: int,
    mode: str,
) -> dict[str, Any]:
    provider, selected_model = resolve_model_selection(model_id, route_level="hard")
    usage = normalize_usage(
        raw.get("usage", {}),
        price_profile=_cost_price_profile(workspace_dir),
        model_id=selected_model,
        provider=provider,
        route_level="hard",
        usage_record_quality=_usage_quality(raw.get("usage", {})),
        usage_source="osworld2_harbor",
        usage_estimated=False,
    )
    active_config = load_active_config(workspace_dir)
    active_strategies = list(active_config.get("strategies", []) or [])
    steps = raw.get("steps", []) if isinstance(raw.get("steps"), list) else []
    summary_step = {
        "usage": usage,
        "active_efficiency_strategies": active_strategies,
        "efficiency_config_id": active_config.get("metadata", {}).get("efficiency_config_id", ""),
        "route_model_calls": [],
    }
    steps = steps + [summary_step]
    strategy_hits = result_strategy_hits({"steps": steps, "strategies": active_strategies})
    skill_usage_counts = {
        key.removeprefix("read_skill:"): value
        for key, value in strategy_hits.items()
        if key.startswith("read_skill:")
    }
    emergent_skill_usage_counts = {
        key.removeprefix("emergent_skill:"): value
        for key, value in strategy_hits.items()
        if key.startswith("emergent_skill:")
    }

    safe_task = task.task_id.replace("/", "__")
    conversation = raw.get("conversation", []) if isinstance(raw.get("conversation"), list) else []
    conversation_path = _write_artifact(
        output_dir / "conversations" / f"{safe_task}.json",
        conversation,
    )
    raw_path = _write_artifact(
        output_dir / "raw_harbor_results" / f"{safe_task}.json",
        raw,
    )
    passed = bool(raw.get("passed"))
    result = {
        "instance_id": task.task_id,
        "task_name": task.task_id,
        "success": passed,
        "score": _safe_float(raw.get("score")),
        "passed": passed,
        "status": raw.get("status", "passed" if passed else "failed"),
        "feedback_detail": raw.get("eval_output", ""),
        "eval_output": raw.get("eval_output", ""),
        "elapsed": _safe_float(raw.get("elapsed")),
        "solve_time": _safe_float(raw.get("elapsed")),
        "patch_len": 0,
        "patch": "",
        "skill_proposal": raw.get("skill_proposal", ""),
        "skill_draft": raw.get("skill_proposal", ""),
        "num_tool_calls": len([step for step in steps if isinstance(step, dict) and step.get("tool")]),
        "num_turns": len([item for item in conversation if isinstance(item, dict)]),
        "cumulative_input_tokens": usage.get("input_tokens", 0),
        "cumulative_output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "cumulative_cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
        "cumulative_cache_write_input_tokens": usage.get("cache_write_input_tokens", 0),
        "cost_units": usage.get("cost_units", 0.0),
        "main_cost_units": usage.get("main_cost_units", usage.get("cost_units", 0.0)),
        "route_model_cost_units": usage.get("route_model_cost_units", 0.0),
        "main_model_id": usage.get("model_id", model_id),
        "main_model_provider": usage.get("provider", ""),
        "main_route_level": usage.get("route_level", ""),
        "input_price_per_1m": usage.get("input_price_per_1m", 0.0),
        "output_price_per_1m": usage.get("output_price_per_1m", 0.0),
        "cache_read_price_per_1m": usage.get("cache_read_price_per_1m", 0.0),
        "cache_write_price_per_1m": usage.get("cache_write_price_per_1m", 0.0),
        "cost_currency": usage.get("cost_currency", ""),
        "cost_price_source": usage.get("cost_price_source", ""),
        "cost_priced": bool(usage.get("cost_priced", False)),
        "usage_record_quality": usage.get("usage_record_quality", _usage_quality(usage)),
        "usage_source": usage.get("usage_source", "osworld2_harbor"),
        "usage_estimated": bool(usage.get("usage_estimated", False)),
        "active_efficiency_strategies": active_strategies,
        "strategies": active_strategies,
        "skill_usage_counts": skill_usage_counts,
        "emergent_skill_usage_counts": emergent_skill_usage_counts,
        "efficiency_config_id": summary_step["efficiency_config_id"],
        "route_model_calls": [],
        "conversation_path": conversation_path,
        "raw_harbor_result_path": raw_path,
        "raw_job_path": raw.get("raw_job_path", ""),
        "result_json_path": raw.get("result_json_path", ""),
        "trajectory_path": raw.get("trajectory_path", ""),
        "steps": steps,
        "metadata": {
            "benchmark": "osworld2",
            "execution_backend": "harbor",
            "domain": task.domain,
            "config_path": task.config_path,
            "harbor": raw.get("metadata", {}),
            "returncode": raw.get("returncode"),
        },
        "batch": batch_num,
        "mode": mode,
    }
    result = _swe._attach_usage_fields(result)
    result["behavior_stats"] = _gui_behavior_stats(result)
    return result


def _gui_behavior_stats(result: dict[str, Any]) -> dict[str, Any]:
    tool_steps = [
        step
        for step in result.get("steps", [])
        if isinstance(step, dict) and step.get("tool")
    ]
    seen: dict[str, int] = {}
    repeated_exact_inputs = 0
    duplicate_examples: list[dict[str, Any]] = []
    screenshots = 0
    clicks = 0
    typing_actions = 0
    scrolls = 0
    for index, step in enumerate(tool_steps, start=1):
        signature = f"{step.get('tool')}::{step.get('action')}::{step.get('input_summary') or step.get('input')}"
        previous = seen.get(signature, 0)
        if previous:
            repeated_exact_inputs += 1
            if len(duplicate_examples) < 5:
                duplicate_examples.append(
                    {"step": index, "signature": signature[:220], "previous_count": previous}
                )
        seen[signature] = previous + 1
        text = json.dumps(step, ensure_ascii=False, default=str).lower()
        if "screenshot" in text:
            screenshots += 1
        if "click" in text:
            clicks += 1
        if "type" in text or "key" in text:
            typing_actions += 1
        if "scroll" in text:
            scrolls += 1
    fragment_visible = _swe._prompt_fragment_visible(result)
    return {
        "task_id": result.get("instance_id", ""),
        "prompt_fragment_visible": fragment_visible,
        "repeated_exact_inputs": repeated_exact_inputs,
        "repeated_test_inputs": 0,
        "same_state_repeated_tests": 0,
        "post_edit_repeated_tests": 0,
        "prompt_fragment_violation_events": 0,
        "duplicate_examples": duplicate_examples,
        "gui_screenshots": screenshots,
        "gui_clicks": clicks,
        "gui_typing_actions": typing_actions,
        "gui_scrolls": scrolls,
    }


def _solve_batch(
    *,
    batch_tasks: list[OSWorld2Task],
    batch_num: int,
    workspace_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(
                _run_harbor_task,
                task=task,
                batch_num=batch_num,
                workspace_dir=workspace_dir,
                output_dir=output_dir,
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
                    "passed": False,
                    "score": 0.0,
                    "status": "error",
                    "eval_output": f"ERROR: {exc}",
                    "usage": {},
                    "steps": [],
                    "conversation": [],
                    "elapsed": 0.0,
                    "returncode": None,
                    "metadata": {"exception": str(exc)},
                }
            result = _normalize_harbor_result(
                raw=raw,
                task=task,
                workspace_dir=workspace_dir,
                output_dir=output_dir,
                model_id=args.model_id,
                batch_num=batch_num,
                mode=args.mode,
            )
            results.append(result)
            status = "PASS" if result.get("success") else "FAIL"
            print(
                f"  {status} {task.task_id} "
                f"score={_safe_float(result.get('score')):.2f} "
                f"cost={_safe_float(result.get('cost_units')):,.4f} "
                f"tokens={_safe_int(result.get('total_tokens')):,d} "
                f"calls={_safe_int(result.get('num_tool_calls'))}"
            )
    return results


def _results_to_observations(
    results: list[dict[str, Any]],
    tasks_by_id: dict[str, Task],
) -> list[Observation]:
    observations: list[Observation] = []
    for result in results:
        task = tasks_by_id.get(str(result.get("instance_id", "")))
        if task is None:
            continue
        feedback = Feedback(
            success=bool(result.get("success")),
            score=float(result.get("score", 0.0) or 0.0),
            detail=str(result.get("feedback_detail") or ""),
            raw={
                "benchmark": "osworld2",
                "execution_backend": "harbor",
                "result_json_path": result.get("result_json_path", ""),
                "raw_job_path": result.get("raw_job_path", ""),
            },
        )
        trajectory = Trajectory(
            task_id=task.id,
            output=str(result.get("eval_output") or ""),
            steps=result.get("steps", []) if isinstance(result.get("steps"), list) else [],
            conversation=[],
        )
        conversation_path = str(result.get("conversation_path") or "")
        if conversation_path:
            try:
                trajectory.conversation = json.loads(
                    Path(conversation_path).read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                trajectory.conversation = []
        trajectory._skill_proposal = str(result.get("skill_proposal") or "")
        observations.append(Observation(task=task, trajectory=trajectory, feedback=feedback))
    return observations


def _truncate_text(value: Any, limit: int = 800) -> str:
    text = str(value or "")
    return text[:limit] + "..." if len(text) > limit else text


def _format_osworld2_signals(signals: dict[str, Any]) -> str:
    repeated = signals.get("repeated_actions") or []
    errors = signals.get("error_snippets") or []
    lines = [
        (
            f"Actions: {_safe_int(signals.get('n_actions'))}, "
            f"Clicks: {_safe_int(signals.get('n_clicks'))}, "
            f"Typing/key actions: {_safe_int(signals.get('n_keystrokes'))}, "
            f"Scrolls: {_safe_int(signals.get('n_scrolls'))}"
        ),
        (
            f"Screenshots: {_safe_int(signals.get('n_screenshots'))}, "
            f"Errors: {_safe_int(signals.get('n_errors'))}, "
            f"Timeouts: {_safe_int(signals.get('n_timeouts'))}"
        ),
    ]
    if repeated:
        lines.append("Repeated actions: " + "; ".join(str(item)[:100] for item in repeated[:4]))
    if errors:
        lines.append("Error samples: " + "; ".join(str(item)[:100] for item in errors[:4]))
    return "\n".join(lines)


def _osworld2_trajectory_signals(result: dict[str, Any]) -> dict[str, Any]:
    steps = result.get("steps", []) if isinstance(result.get("steps"), list) else []
    action_counts: dict[str, int] = {}
    error_snippets: list[str] = []
    signals = {
        "n_actions": 0,
        "n_clicks": 0,
        "n_keystrokes": 0,
        "n_scrolls": 0,
        "n_screenshots": 0,
        "n_errors": 0,
        "n_timeouts": 0,
        "repeated_actions": [],
        "error_snippets": error_snippets,
    }
    for step in steps:
        if not isinstance(step, dict):
            continue
        action = str(step.get("action") or step.get("tool") or "").lower()
        if not action:
            continue
        signals["n_actions"] += 1
        if "click" in action:
            signals["n_clicks"] += 1
        if action in {"type", "key", "hold_key"} or "type" in action or "key" in action:
            signals["n_keystrokes"] += 1
        if "scroll" in action:
            signals["n_scrolls"] += 1
        if "screenshot" in action:
            signals["n_screenshots"] += 1
        signature = f"{action}:{str(step.get('input') or step.get('input_summary') or '')[:120]}"
        action_counts[signature] = action_counts.get(signature, 0) + 1
        result_text = str(step.get("result_snippet") or step.get("output") or step.get("observation") or "")
        lowered = result_text.lower()
        if any(marker in lowered for marker in ("error", "traceback", "exception", "failed")):
            signals["n_errors"] += 1
            if len(error_snippets) < 5:
                error_snippets.append(result_text[:200])
        if "timeout" in lowered or "timed out" in lowered:
            signals["n_timeouts"] += 1
    signals["repeated_actions"] = [
        f"{signature} (x{count})"
        for signature, count in action_counts.items()
        if count >= 3
    ][:5]
    return signals


def _compress_osworld2_trajectory(result: dict[str, Any]) -> str:
    steps = [step for step in result.get("steps", []) if isinstance(step, dict)]
    action_lines: list[str] = []
    error_lines: list[str] = []
    for index, step in enumerate(steps, start=1):
        action = str(step.get("action") or step.get("tool") or "")
        if not action:
            continue
        input_summary = str(step.get("input_summary") or step.get("input") or "")[:180]
        line = f"{index}. {action} {input_summary}".strip()
        action_lines.append(line)
        result_text = str(step.get("result_snippet") or step.get("output") or "")
        lowered = result_text.lower()
        if any(marker in lowered for marker in ("error", "traceback", "exception", "failed", "timeout")):
            error_lines.append(f"{line} -> {_truncate_text(result_text, 220)}")
    parts = [f"Actions: {len(action_lines)}, success={bool(result.get('success'))}, score={_safe_float(result.get('score')):.2f}"]
    if action_lines:
        parts.append("\nStart actions:")
        parts.extend(f"- {line}" for line in action_lines[:4])
        parts.append("\nFinal actions:")
        parts.extend(f"- {line}" for line in action_lines[-4:])
    if error_lines:
        parts.append("\nErrors:")
        parts.extend(f"- {line}" for line in error_lines[:5])
    return "\n".join(parts)


def _osworld2_eval_text(result: dict[str, Any]) -> str:
    score = _safe_float(result.get("score"))
    if result.get("success"):
        return f"PASSED (score={score:.1f})"
    detail = str(result.get("feedback_detail") or result.get("eval_output") or "").strip()
    parts = [
        f"FAILED (score={score:.1f})",
        f"Status: {result.get('status', 'failed')}",
    ]
    failure = _classify_failure(detail, result.get("metadata", {}).get("returncode"))
    if failure:
        parts.append(f"Failure reason: {failure}")
    if detail:
        parts.append(f"Evaluation detail:\n{_truncate_text(detail, 1200)}")
    return "\n".join(parts)


def _existing_osworld2_skills_section(workspace_dir: Path) -> str:
    skills = _load_osworld2_skills(workspace_dir)
    if not skills:
        return "No existing skills yet."
    lines = ["Current skills:"]
    for skill in skills:
        path_parts = Path(skill.path).parts
        topic_tag = "/".join(path_parts[1:-1]) if len(path_parts) > 2 else "root"
        lines.append(f"- **{skill.name}** [{topic_tag}]: {skill.description}")
    return "\n".join(lines)


def _complete_evolver_text(
    *,
    system: str,
    user: str,
    args: argparse.Namespace,
    max_tokens: int,
    temperature: float,
) -> tuple[str, str]:
    model_id = args.evolver_model or args.model_id
    last_error = ""
    attempts = max(1, int(args.evolver_call_retries))
    for attempt in range(attempts):
        try:
            provider = create_llm_provider(model_id, region=args.region)
            response = provider.complete(
                [
                    LLMMessage(role="system", content=system),
                    LLMMessage(role="user", content=user),
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.content.strip(), ""
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if attempt + 1 < attempts:
                delay = min(
                    float(args.evolver_retry_max_delay),
                    float(args.evolver_retry_initial_delay) * (2**attempt),
                )
                time.sleep(delay)
    return "", last_error


def _parse_osworld2_proposal(resp: str, task_name: str) -> dict[str, Any] | None:
    if not str(resp or "").strip() or "ACTION: NONE" in resp.upper():
        return None
    proposal: dict[str, Any] = {
        "source_task": task_name,
        "topic": "general",
        "raw": resp,
        "action": "NEW",
        "target": "",
        "name": "",
        "description": "",
        "content": "",
    }
    for line in resp.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("TOPIC:"):
            raw_topic = stripped.split(":", 1)[1].strip()
            proposal["topic"] = re.sub(r"[^a-z0-9-]", "-", raw_topic.lower()).strip("-") or "general"
        elif upper.startswith("ACTION:"):
            proposal["action"] = stripped.split(":", 1)[1].strip().upper()
        elif upper.startswith("TARGET:"):
            proposal["target"] = stripped.split(":", 1)[1].strip()
        elif upper.startswith("NAME:"):
            raw_name = stripped.split(":", 1)[1].strip()
            proposal["name"] = re.sub(r"[^a-z0-9-]", "-", raw_name.lower()).strip("-")
        elif upper.startswith("DESCRIPTION:"):
            proposal["description"] = stripped.split(":", 1)[1].strip()[:180]
    idx = resp.upper().find("CONTENT:")
    if idx >= 0:
        raw_content = resp[idx + len("CONTENT:") :].strip()
        stop_tags = ("TOPIC:", "ACTION:", "TARGET:", "NAME:", "DESCRIPTION:")
        lines = raw_content.splitlines()
        while lines and any(lines[-1].strip().upper().startswith(tag) for tag in stop_tags):
            lines.pop()
        proposal["content"] = "\n".join(lines).strip()
    if proposal["action"] == "ENHANCE" and proposal["target"] and not proposal["name"]:
        proposal["name"] = proposal["target"]
    if not proposal["name"] and proposal["action"] != "NONE":
        proposal["name"] = re.sub(r"[^a-z0-9-]", "-", f"skill-{task_name[:40]}".lower()).strip("-")
    if not proposal["content"]:
        return None
    return proposal


def _proposal_from_json_payload(text: str, task_name: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    proposals = payload.get("proposals") if isinstance(payload, dict) else None
    if not isinstance(proposals, list) or not proposals:
        return None
    raw = proposals[0]
    if not isinstance(raw, dict):
        return None
    name = re.sub(r"[^a-z0-9-]", "-", str(raw.get("name") or f"skill-{task_name}").lower()).strip("-")
    content = str(raw.get("content") or raw.get("description") or "").strip()
    if not content:
        return None
    topic = raw.get("topic") or raw.get("domain") or raw.get("category") or "general"
    return {
        "source_task": task_name,
        "source_passed": False,
        "topic": re.sub(r"[^a-z0-9-]", "-", str(topic).lower()).strip("-") or "general",
        "raw": text,
        "action": "NEW",
        "target": "",
        "name": name,
        "description": str(raw.get("description") or raw.get("trigger") or name).strip()[:180],
        "content": content,
    }


def _derive_osworld2_proposal(
    *,
    result: dict[str, Any],
    workspace_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, str, str]:
    task_name = str(result.get("task_name") or result.get("instance_id") or "task")
    raw_proposal = str(result.get("skill_proposal") or result.get("skill_draft") or "").strip()
    if raw_proposal:
        proposal = _parse_osworld2_proposal(raw_proposal, task_name) or _proposal_from_json_payload(raw_proposal, task_name)
        if proposal:
            proposal["source_passed"] = bool(result.get("success"))
            proposal["source_score"] = _safe_float(result.get("score"))
            return proposal, "harbor_skill_proposal", ""

    if args.dry_run_proposer:
        return None, "dry_run_proposer", ""
    if result.get("success") and not args.evolve_all:
        return None, "passed_without_evolve_all", ""

    signals = _osworld2_trajectory_signals(result)
    compressed = _compress_osworld2_trajectory(result)
    existing = _existing_osworld2_skills_section(workspace_dir)
    if result.get("success"):
        user_prompt = OSWORLD2_ANALYZE_AND_PROPOSE_PASS_PROMPT.format(
            score=_safe_float(result.get("score")),
            trajectory_signals=_format_osworld2_signals(signals),
            compressed_trajectory=_truncate_text(compressed, 1800),
            existing_skills_section=existing,
        )
    else:
        user_prompt = OSWORLD2_ANALYZE_AND_PROPOSE_PROMPT.format(
            eval_result=_osworld2_eval_text(result),
            trajectory_signals=_format_osworld2_signals(signals),
            compressed_trajectory=_truncate_text(compressed, 1800),
            existing_skills_section=existing,
        )
    response, error = _complete_evolver_text(
        system=OSWORLD2_PROPOSE_SYSTEM_PROMPT,
        user=user_prompt,
        args=args,
        max_tokens=min(2048, int(args.evolver_max_tokens)),
        temperature=float(args.evolver_temperature),
    )
    if error:
        logger.warning("OSWorld2 proposal failed for %s: %s", task_name, error[:300])
        return None, "", error
    proposal = _parse_osworld2_proposal(response, task_name)
    feedback_analysis = response
    action_idx = response.upper().find("ACTION:")
    if action_idx > 0:
        feedback_analysis = response[:action_idx].strip()
    if proposal:
        proposal["source_passed"] = bool(result.get("success"))
        proposal["source_score"] = _safe_float(result.get("score"))
    return proposal, feedback_analysis, ""


def _curate_osworld2_topic_proposals(
    *,
    topic: str,
    proposals: list[dict[str, Any]],
    workspace_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not proposals:
        return {"added": 0, "merged": 0, "skipped": 0}
    if args.dry_run_proposer:
        return {"added": 0, "merged": 0, "skipped": 0, "dry_run": True}

    topic_dir = workspace_dir / "skills" / "topic" / topic
    existing: list[tuple[str, str]] = []
    if topic_dir.exists():
        for skill_file in sorted(topic_dir.rglob("SKILL.md")):
            meta, _body = _strip_skill_frontmatter(skill_file.read_text(encoding="utf-8", errors="ignore"))
            existing.append((str(meta.get("name") or skill_file.parent.name), str(meta.get("description") or "")))
    existing_list = "\n".join(f"- **{name}**: {desc}" for name, desc in existing) if existing else "(empty)"
    proposal_lines = []
    for proposal in proposals:
        proposal_lines.append(
            f"### [{proposal.get('action', 'NEW')}] {proposal.get('name', '?')}\n"
            f"Source: {proposal.get('source_task', '')} pass={proposal.get('source_passed')}\n"
            f"Description: {proposal.get('description', '')[:180]}\n"
            f"Content: {_truncate_text(proposal.get('content', ''), 500)}"
        )
    system_prompt = OSWORLD2_CURATOR_PROMPT.format(
        topic=topic,
        n_skills=len(existing),
        max_skills=args.max_skills_per_topic,
        existing_skills_list=existing_list,
        proposals_list="\n\n".join(proposal_lines),
    )
    response, error = _complete_evolver_text(
        system=system_prompt,
        user="Review and decide.",
        args=args,
        max_tokens=min(2048, int(args.evolver_max_tokens)),
        temperature=0.0,
    )
    if error:
        logger.warning("OSWorld2 topic curator failed for %s: %s", topic, error[:300])
        return {"added": 0, "merged": 0, "skipped": 0, "error": error}
    return _execute_osworld2_topic_curation(
        response,
        proposals,
        existing,
        workspace_dir,
        topic,
        args.max_skills_per_topic,
    )


def _execute_osworld2_topic_curation(
    text: str,
    proposals: list[dict[str, Any]],
    existing: list[tuple[str, str]],
    workspace_dir: Path,
    topic: str,
    max_skills: int,
) -> dict[str, Any]:
    proposal_map = {str(proposal["name"]): proposal for proposal in proposals if proposal.get("name")}
    existing_names = {name for name, _desc in existing}
    count = len(existing)
    stats: dict[str, Any] = {"added": 0, "merged": 0, "skipped": 0}

    def _write_skill(name: str, desc: str, content: str) -> None:
        skill_dir = workspace_dir / "skills" / "topic" / topic / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {desc}\n---\n\n{content.strip()}\n",
            encoding="utf-8",
        )

    def _fuzzy(raw: str, names: set[str]) -> str | None:
        clean = re.sub(r"[^a-z0-9-]", "-", raw.lower()).strip("-")
        if clean in names:
            return clean
        for name in names:
            if clean and (clean in name or name in clean):
                return name
        return None

    for line in text.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("ACCEPT:"):
            proposal_name = _fuzzy(stripped.split(":", 1)[1].strip(), set(proposal_map))
            if proposal_name and proposal_name not in existing_names and count < max_skills:
                proposal = proposal_map[proposal_name]
                _write_skill(
                    proposal_name,
                    str(proposal.get("description") or ""),
                    str(proposal.get("content") or ""),
                )
                existing_names.add(proposal_name)
                count += 1
                stats["added"] += 1
        elif upper.startswith("MERGE:"):
            merge_body = stripped.split(":", 1)[1].strip()
            split_match = re.split(r"\s+INTO\s+", merge_body, maxsplit=1, flags=re.IGNORECASE)
            if len(split_match) != 2:
                continue
            proposal_name = _fuzzy(split_match[0].strip(), set(proposal_map))
            target_name = _fuzzy(split_match[1].strip(), existing_names)
            if not proposal_name or not target_name:
                continue
            merge_idx = text.find(line)
            after = text[merge_idx + len(line) :]
            new_content = ""
            if "NEW_CONTENT:" in after:
                new_content = after.split("NEW_CONTENT:", 1)[1]
                for marker in ("ACCEPT:", "MERGE:", "SKIP:", "NO_PROPOSALS"):
                    marker_idx = new_content.find(marker)
                    if marker_idx >= 0:
                        new_content = new_content[:marker_idx]
                new_content = new_content.strip()
            if new_content:
                old_desc = next((desc for name, desc in existing if name == target_name), "")
                proposal = proposal_map[proposal_name]
                _write_skill(target_name, old_desc or str(proposal.get("description") or ""), new_content)
                stats["merged"] += 1
        elif upper.startswith("SKIP:"):
            stats["skipped"] += 1
    return stats


def _curate_osworld2_general_skills(
    *,
    failed_summaries: list[dict[str, Any]],
    workspace_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not failed_summaries or args.max_general_skills <= 0:
        return {"added": 0, "updated": 0, "deleted": 0}
    if args.dry_run_proposer:
        return {"added": 0, "updated": 0, "deleted": 0, "dry_run": True}

    summary_lines = []
    for summary in failed_summaries[:30]:
        parts = [f"### {summary['task_name']} ({summary.get('domain', '')})"]
        parts.append(f"Eval: score={summary.get('score')}, reason={summary.get('failure_reason')}")
        if summary.get("trajectory_signals"):
            parts.append(f"Signals:\n{summary['trajectory_signals']}")
        if summary.get("compressed_trajectory"):
            limit = 1000 if args.feedback_level == "full" else 450
            parts.append(f"Trajectory:\n{_truncate_text(summary['compressed_trajectory'], limit)}")
        if summary.get("feedback_analysis") and args.feedback_level != "minimal":
            parts.append(f"Analysis:\n{_truncate_text(summary['feedback_analysis'], 500)}")
        if summary.get("proposal_summary"):
            parts.append(f"Proposal: {_truncate_text(summary['proposal_summary'], 250)}")
        summary_lines.append("\n".join(parts))

    general_dir = workspace_dir / "skills" / "general"
    existing: list[tuple[str, str, str]] = []
    if general_dir.exists():
        for skill_file in sorted(general_dir.rglob("SKILL.md")):
            content = skill_file.read_text(encoding="utf-8", errors="ignore")
            meta, body = _strip_skill_frontmatter(content)
            existing.append(
                (
                    str(meta.get("name") or skill_file.parent.name),
                    str(meta.get("description") or ""),
                    body[:300],
                )
            )
    existing_list = "\n".join(f"- **{name}**: {desc}" for name, desc, _ in existing) if existing else "(empty)"
    system_prompt = OSWORLD2_GENERAL_CURATOR_PROMPT.format(
        n_failed=len(failed_summaries),
        failed_summaries="\n\n".join(summary_lines),
        n_general=len(existing),
        max_general=args.max_general_skills,
        general_skills_list=existing_list,
    )
    response, error = _complete_evolver_text(
        system=system_prompt,
        user="Analyze and decide.",
        args=args,
        max_tokens=min(4096, int(args.evolver_max_tokens)),
        temperature=0.0,
    )
    if error:
        logger.warning("OSWorld2 general curator failed: %s", error[:300])
        return {"added": 0, "updated": 0, "deleted": 0, "error": error}
    return _execute_osworld2_general_curation(
        response,
        workspace_dir,
        existing,
        args.max_general_skills,
    )


def _execute_osworld2_general_curation(
    text: str,
    workspace_dir: Path,
    existing: list[tuple[str, str, str]],
    max_general: int,
) -> dict[str, Any]:
    existing_names = {name for name, _desc, _body in existing}
    count = len(existing)
    stats = {"added": 0, "updated": 0, "deleted": 0}

    def _write_general(name: str, desc: str, content: str) -> None:
        skill_dir = workspace_dir / "skills" / "general" / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {desc}\n---\n\n{content.strip()}\n",
            encoding="utf-8",
        )

    lines = text.splitlines()
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        upper = stripped.upper()
        if upper.startswith("NEW_GENERAL:"):
            name = re.sub(r"[^a-z0-9-]", "-", stripped.split(":", 1)[1].strip().lower()).strip("-")
            desc = ""
            content_lines: list[str] = []
            index += 1
            while index < len(lines):
                current = lines[index].strip()
                current_upper = current.upper()
                if current_upper.startswith(("NEW_GENERAL:", "UPDATE_GENERAL:", "DELETE_GENERAL:", "NO_PATTERNS")):
                    break
                if current_upper.startswith("DESCRIPTION:"):
                    desc = current.split(":", 1)[1].strip()[:180]
                elif current_upper.startswith("CONTENT:"):
                    index += 1
                    while index < len(lines):
                        next_upper = lines[index].strip().upper()
                        if next_upper.startswith(("NEW_GENERAL:", "UPDATE_GENERAL:", "DELETE_GENERAL:", "NO_PATTERNS")):
                            break
                        content_lines.append(lines[index])
                        index += 1
                    break
                index += 1
            if name and content_lines and count < max_general:
                _write_general(name, desc, "\n".join(content_lines))
                count += 1
                stats["added"] += 1
            continue
        if upper.startswith("UPDATE_GENERAL:"):
            raw_name = re.sub(r"[^a-z0-9-]", "-", stripped.split(":", 1)[1].strip().lower()).strip("-")
            matched = next((name for name in existing_names if raw_name == name or raw_name in name or name in raw_name), None)
            content_lines = []
            index += 1
            while index < len(lines):
                current_upper = lines[index].strip().upper()
                if current_upper.startswith(("NEW_GENERAL:", "UPDATE_GENERAL:", "DELETE_GENERAL:", "NO_PATTERNS")):
                    break
                if current_upper.startswith(("NEW_CONTENT:", "CONTENT:")):
                    index += 1
                    while index < len(lines):
                        next_upper = lines[index].strip().upper()
                        if next_upper.startswith(("NEW_GENERAL:", "UPDATE_GENERAL:", "DELETE_GENERAL:", "NO_PATTERNS")):
                            break
                        content_lines.append(lines[index])
                        index += 1
                    break
                index += 1
            if matched and content_lines:
                old_desc = next((desc for name, desc, _body in existing if name == matched), "")
                _write_general(matched, old_desc, "\n".join(content_lines))
                stats["updated"] += 1
            continue
        if upper.startswith("DELETE_GENERAL:"):
            raw_name = re.sub(r"[^a-z0-9-]", "-", stripped.split(":", 1)[1].strip().lower()).strip("-")
            matched = next((name for name in existing_names if raw_name == name or raw_name in name or name in raw_name), None)
            if matched:
                skill_dir = workspace_dir / "skills" / "general" / matched
                if skill_dir.exists():
                    shutil.rmtree(skill_dir)
                    existing_names.discard(matched)
                    count -= 1
                    stats["deleted"] += 1
        index += 1
    return stats


def update_osworld2_skill_tree(workspace_dir: Path) -> None:
    skills = _load_osworld2_skills(workspace_dir)
    skills_dir = workspace_dir / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    if not skills:
        content = "# Skill Tree\n\nNo skills yet.\n"
    else:
        lines = ["# Skill Tree", "", f"Total: {len(skills)}", ""]
        by_prefix: dict[str, list[OSWorld2SkillMeta]] = defaultdict(list)
        for skill in skills:
            path_parts = Path(skill.path).parts
            if len(path_parts) >= 3 and path_parts[0] == "skills":
                prefix = "/".join(path_parts[1:-1]) or "root"
            else:
                prefix = "root"
            by_prefix[prefix].append(skill)
        for prefix in sorted(by_prefix):
            lines.append(f"## {prefix}/")
            for skill in sorted(by_prefix[prefix], key=lambda item: item.name):
                lines.append(f"- **{skill.name}**: {skill.description}")
            lines.append("")
        content = "\n".join(lines)
    (skills_dir / "SKILL_TREE.md").write_text(content, encoding="utf-8")


def _run_normal_evolution(
    *,
    workspace_dir: Path,
    output_dir: Path,
    batch_results: list[dict[str, Any]],
    observations: list[Observation],
    batch_num: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.time()
    proposals: list[dict[str, Any]] = []
    failed_summaries: list[dict[str, Any]] = []
    proposal_records: list[dict[str, Any]] = []

    for result in batch_results:
        proposal, feedback_analysis, error = _derive_osworld2_proposal(
            result=result,
            workspace_dir=workspace_dir,
            args=args,
        )
        task_name = str(result.get("task_name") or result.get("instance_id") or "task")
        if proposal:
            proposals.append(proposal)
        proposal_records.append(
            {
                "task_name": task_name,
                "success": bool(result.get("success")),
                "proposal": proposal,
                "feedback_analysis": feedback_analysis,
                "error": error,
            }
        )
        if not result.get("success"):
            signals = _osworld2_trajectory_signals(result)
            summary = {
                "task_name": task_name,
                "domain": result.get("metadata", {}).get("domain", ""),
                "score": _safe_float(result.get("score")),
                "failure_reason": _classify_failure(
                    str(result.get("feedback_detail") or result.get("eval_output") or ""),
                    result.get("metadata", {}).get("returncode"),
                ),
                "trajectory_signals": _format_osworld2_signals(signals),
                "compressed_trajectory": _compress_osworld2_trajectory(result),
                "feedback_analysis": feedback_analysis,
                "proposal_summary": (
                    f"[{proposal.get('action', 'NEW')}] {proposal.get('name', '')}: "
                    f"{proposal.get('description', '')}"
                    if proposal
                    else ""
                ),
            }
            failed_summaries.append(summary)

    topic_stats: dict[str, Any] = {"added": 0, "merged": 0, "skipped": 0}
    if proposals:
        topic_proposals: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for proposal in proposals:
            topic_proposals[str(proposal.get("topic") or "general")].append(proposal)
        per_topic: dict[str, Any] = {}
        for topic, topic_items in sorted(topic_proposals.items()):
            stats = _curate_osworld2_topic_proposals(
                topic=topic,
                proposals=topic_items,
                workspace_dir=workspace_dir,
                args=args,
            )
            per_topic[topic] = stats
            for key in ("added", "merged", "skipped"):
                topic_stats[key] += _safe_int(stats.get(key))
        topic_stats["by_topic"] = per_topic

    general_stats = {"added": 0, "updated": 0, "deleted": 0}
    if len(failed_summaries) >= 2:
        general_stats = _curate_osworld2_general_skills(
            failed_summaries=failed_summaries,
            workspace_dir=workspace_dir,
            args=args,
        )

    update_osworld2_skill_tree(workspace_dir)
    skills_after = _load_osworld2_skills(workspace_dir)
    detail_path = _write_artifact(
        output_dir / "emergence" / "normal_evolution" / f"batch_{batch_num:02d}.json",
        {
            "batch": batch_num,
            "mode": args.mode,
            "proposals": proposal_records,
            "failed_summaries": failed_summaries,
            "topic_stats": topic_stats,
            "general_stats": general_stats,
            "skills_after": [
                {"name": skill.name, "description": skill.description, "path": skill.path}
                for skill in skills_after
            ],
        },
    )
    result_summary = {
        "framework": "osworld_agent_evolution",
        "proposals": len(proposals),
        "topic_stats": topic_stats,
        "general_stats": general_stats,
        "skills_total": len(skills_after),
        "detail_path": detail_path,
    }
    record = {
        "batch": batch_num,
        "mode": args.mode,
        "elapsed": round(time.time() - started, 1),
        "result": result_summary,
        "solver_proposals": sum(
            1
            for item in proposal_records
            if item.get("proposal")
        ),
        "observations": len(observations),
    }
    path = output_dir / "emergence" / "normal_evolution.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
    return record


def _write_workspace_evolution_records(
    *,
    workspace_dir: Path,
    output_dir: Path,
    batch_results: list[dict[str, Any]],
    observations: list[Observation],
    batch_num: int,
) -> None:
    _ = output_dir
    _swe._write_workspace_evolution_records(
        workspace_dir=workspace_dir,
        batch_results=batch_results,
        observations=observations,
        batch_num=batch_num,
    )


def _strategies_for(results: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for result in results:
        for key in ("strategies", "active_efficiency_strategies"):
            values = result.get(key, []) or []
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, list):
                continue
            for value in values:
                name = str(value or "").strip()
                if name and name not in seen:
                    seen.add(name)
                    output.append(name)
    return output


def _summarize_gui_behavior(results: list[dict[str, Any]]) -> dict[str, Any]:
    stats = [result.get("behavior_stats") or _gui_behavior_stats(result) for result in results]
    return {
        **_swe._summarize_behavior_stats(results),
        "gui_screenshots": sum(_safe_int(item.get("gui_screenshots")) for item in stats),
        "gui_clicks": sum(_safe_int(item.get("gui_clicks")) for item in stats),
        "gui_typing_actions": sum(_safe_int(item.get("gui_typing_actions")) for item in stats),
        "gui_scrolls": sum(_safe_int(item.get("gui_scrolls")) for item in stats),
    }


def _make_summary_rows(results: list[dict[str, Any]], *, mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch in sorted({_safe_int(result.get("batch")) for result in results}):
        subset = [result for result in results if _safe_int(result.get("batch")) == batch]
        row = {"group": f"{mode}_batch_{batch:02d}", "strategies": _strategies_for(subset)}
        row.update(summarize_results(subset))
        row.update(_summarize_gui_behavior(subset))
        rows.append(row)
    overall = {"group": f"{mode}_overall", "strategies": _strategies_for(results)}
    overall.update(summarize_results(results))
    overall.update(_summarize_gui_behavior(results))
    rows.append(overall)
    return rows


def _write_run_outputs(output_dir: Path, results: list[dict[str, Any]], *, mode: str) -> None:
    _write_artifact(output_dir / "results.json", results)
    rows = _make_summary_rows(results, mode=mode)
    _write_artifact(
        output_dir / "summary.json",
        {"mode": mode, "benchmark": "osworld2_harbor", "overall": rows[-1], "by_batch": rows[:-1]},
    )
    _write_artifact(
        output_dir / "behavior_stats.json",
        {
            "overall": _summarize_gui_behavior(results),
            "by_task": [
                result.get("behavior_stats") or _gui_behavior_stats(result)
                for result in results
            ],
        },
    )
    write_summary_table(rows, output_dir / "summary.csv")
    try:
        write_plots(rows, output_dir / "figures")
        write_strategy_library_plot(rows, output_dir / "figures")
        _swe._write_proposal_category_outputs(output_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to render figures: %s", exc)


def _write_experiment_manifest(output_dir: Path, args: argparse.Namespace, tasks: list[OSWorld2Task]) -> None:
    task_file = _resolve_task_file(args)
    task_source = str(task_file) if task_file is not None else f"{args.osworld2_path or os.environ.get('OSWORLD2_PATH', '')} [harbor_task_dirs]"
    payload = {
        "benchmark": "osworld2",
        "execution_backend": "harbor",
        "mode": args.mode,
        "tasks": [task.task_id for task in tasks],
        "task_count": len(tasks),
        "domains": sorted({task.domain for task in tasks}),
        "task_file": task_source,
        "osworld2_path": args.osworld2_path,
        "harbor_repo": args.harbor_repo,
        "harbor_uv_cmd": args.harbor_uv_cmd,
        "harbor_command_template": args.harbor_command_template,
        "harbor_config_template": args.harbor_config_template,
        "harbor_agent_import_path": args.harbor_agent_import_path,
        "harbor_delegate_agent_import_path": args.harbor_delegate_agent_import_path,
        "harbor_delegate_model_name": args.harbor_delegate_model_name,
        "harbor_delegate_kwargs": args.harbor_delegate_kwargs,
        "harbor_model_name": args.harbor_model_name or args.model_id,
        "harbor_auto_confirm": args.harbor_auto_confirm,
        "harbor_ensure_external_volumes": args.harbor_ensure_external_volumes,
        "harbor_check_host_devices": args.harbor_check_host_devices,
        "dry_run_harbor": args.dry_run_harbor,
        "batch_size": args.batch_size,
        "parallel": args.parallel,
        "model_id": args.model_id,
        "evolver_model": args.evolver_model or args.model_id,
        "seed_workspace": args.seed_workspace,
        "max_skills_per_topic": args.max_skills_per_topic,
        "max_general_skills": args.max_general_skills,
        "evolve_all": args.evolve_all,
        "feedback_level": args.feedback_level,
        "predefined_strategies": args.predefined_strategies,
        "category_seed": args.category_seed,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_artifact(output_dir / "experiment_manifest.json", payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emergent efficiency OSWorld 2.x experiment runner using Harbor evaluation"
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
    parser.add_argument("--task-file", default=None, help="OSWorld2 task JSON/JSONL file")
    parser.add_argument("--osworld2-path", default=os.environ.get("OSWORLD2_PATH", ""))
    parser.add_argument("--domain", default=None, help="Comma-separated domain filter")
    parser.add_argument("--tasks", default=None, help="Comma-separated task ids to run")
    parser.add_argument("--exclude", default=None, help="Comma-separated task ids to skip")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--seed-workspace", default="seed_workspaces/osworld")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fixed-output-dir", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Only with --fixed-output-dir; removes an existing output directory",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--evolver-model", default="")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--evolver-max-tokens", type=int, default=8192)
    parser.add_argument("--evolver-temperature", type=float, default=0.2)
    parser.add_argument("--evolver-call-retries", type=int, default=4)
    parser.add_argument("--evolver-retry-initial-delay", type=float, default=2.0)
    parser.add_argument("--evolver-retry-max-delay", type=float, default=30.0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-proposals-per-batch", type=int, default=2)
    parser.add_argument(
        "--max-skills-per-topic",
        type=int,
        default=5,
        help="Maximum curated OSWorld-style skills kept under each skills/topic/<topic>/ directory",
    )
    parser.add_argument(
        "--max-general-skills",
        type=int,
        default=10,
        help="Maximum curated cross-task OSWorld-style skills kept under skills/general/",
    )
    parser.add_argument(
        "--evolve-all",
        action="store_true",
        help="Let normal-evolve propose skills from passed tasks as well as failed tasks",
    )
    parser.add_argument(
        "--feedback-level",
        choices=["minimal", "standard", "full"],
        default="standard",
        help="How much Harbor trajectory and evaluation detail the OSWorld-style curator sees",
    )
    parser.add_argument(
        "--category-seed",
        choices=["none", "basic-efficiency"],
        default="basic-efficiency",
        help="Taxonomy exposed only in category-seeded-cost-aware mode",
    )
    parser.add_argument("--dry-run-proposer", action="store_true")
    parser.add_argument(
        "--dry-run-harbor",
        action="store_true",
        help="Use deterministic synthetic Harbor results for runner smoke tests",
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
    )

    parser.add_argument("--harbor-repo", default=os.environ.get("HARBOR_REPO", ""))
    parser.add_argument("--harbor-config-template", default=None)
    parser.add_argument(
        "--harbor-command-template",
        default=None,
        help=(
            "Optional full command template. Placeholders include {config_path}, "
            "{task_id}, {task_json_path}, {workspace_dir}, {jobs_dir}, {job_name}."
        ),
    )
    parser.add_argument("--harbor-result-json-template", default=None)
    parser.add_argument(
        "--harbor-agent-import-path",
        default=os.environ.get("OSWORLD2_HARBOR_AGENT_IMPORT_PATH", DEFAULT_HARBOR_AGENT_IMPORT_PATH),
    )
    parser.add_argument("--harbor-agent-name", default="osworld2-agent")
    parser.add_argument("--harbor-model-name", default=os.environ.get("HARBOR_MODEL_NAME", ""))
    parser.add_argument(
        "--harbor-delegate-agent-import-path",
        default=os.environ.get("OSWORLD2_HARBOR_DELEGATE_AGENT_IMPORT_PATH", DEFAULT_HARBOR_DELEGATE_AGENT_IMPORT_PATH),
        help="Harbor OSWorld agent wrapped by the A-Evolve adapter.",
    )
    parser.add_argument(
        "--harbor-delegate-model-name",
        default=os.environ.get("OSWORLD2_HARBOR_DELEGATE_MODEL_NAME", ""),
        help="Optional model name for the delegated Harbor OSWorld agent.",
    )
    parser.add_argument(
        "--harbor-delegate-kwargs",
        default=os.environ.get("OSWORLD2_HARBOR_DELEGATE_KWARGS", ""),
        help="Optional JSON/YAML kwargs passed through to the delegated Harbor OSWorld agent.",
    )
    parser.add_argument("--harbor-jobs-dir", default=None)
    parser.add_argument(
        "--harbor-timeout-sec",
        type=int,
        default=0,
        help=(
            "Outer Harbor subprocess timeout. Use 0 for auto: at least the "
            "task.toml environment build + agent + verifier budget plus buffer."
        ),
    )
    parser.add_argument(
        "--no-harbor-auto-confirm",
        dest="harbor_auto_confirm",
        action="store_false",
        help="Do not pass --yes to the default Harbor run command.",
    )
    parser.set_defaults(harbor_auto_confirm=True)
    parser.add_argument(
        "--no-harbor-ensure-external-volumes",
        dest="harbor_ensure_external_volumes",
        action="store_false",
        help="Do not auto-create Docker volumes declared as external in task docker-compose.yaml.",
    )
    parser.set_defaults(harbor_ensure_external_volumes=True)
    parser.add_argument(
        "--no-harbor-check-host-devices",
        dest="harbor_check_host_devices",
        action="store_false",
        help="Do not fail fast when task docker-compose.yaml requires missing host devices.",
    )
    parser.set_defaults(harbor_check_host_devices=True)
    parser.add_argument("--harbor-uv-cmd", default=os.environ.get("HARBOR_UV_CMD", DEFAULT_HARBOR_UV_CMD))
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.parallel <= 0:
        raise SystemExit("--parallel must be positive")
    if args.max_proposals_per_batch < 0:
        raise SystemExit("--max-proposals-per-batch must be non-negative")
    if args.max_skills_per_topic < 0:
        raise SystemExit("--max-skills-per-topic must be non-negative")
    if args.max_general_skills < 0:
        raise SystemExit("--max-general-skills must be non-negative")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    for name in ("botocore", "urllib3", "httpcore", "httpx", "strands.models", "strands.tools"):
        logging.getLogger(name).setLevel(logging.WARNING)

    output_dir = _resolve_output_dir(args)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"Output directory exists and is not empty: {output_dir}. "
            "Use --fixed-output-dir --overwrite for intentional replacement."
        )
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    workspace_dir = output_dir / "workspace"
    seed_workspace = Path(args.seed_workspace)
    hide_efficiency = args.mode in {"normal-evolve", "cost-aware", "category-seeded-cost-aware"}
    if seed_workspace.exists():
        _copy_osworld2_seed_workspace(seed_workspace, workspace_dir, hide_efficiency=hide_efficiency)
    else:
        workspace_dir.mkdir(parents=True, exist_ok=True)
        (workspace_dir / "skills").mkdir(parents=True, exist_ok=True)
    _ensure_workspace_prompt(workspace_dir)
    update_osworld2_skill_tree(workspace_dir)

    if args.mode in {"predefined-control", "predefined-cost-aware"}:
        _swe._configure_predefined_control(
            workspace_dir,
            args.predefined_strategies,
            experiment=f"osworld2_emergent_efficiency_{args.mode.replace('-', '_')}",
        )

    emergence_dir = output_dir / "emergence"
    emergence_dir.mkdir(parents=True, exist_ok=True)
    leakage = _swe._strategy_leakage_scan(workspace_dir) if hide_efficiency else {"passed": True, "hits": []}
    _write_artifact(emergence_dir / "leakage_audit.json", leakage)
    if hide_efficiency and not leakage.get("passed"):
        raise SystemExit(f"Leakage audit failed. See {emergence_dir / 'leakage_audit.json'}")

    loaded_tasks, _task_source = _load_tasks_from_args(args)
    tasks = _filter_tasks(loaded_tasks, args)
    if not tasks:
        raise SystemExit("No OSWorld2 tasks selected.")
    batches = [tasks[i : i + args.batch_size] for i in range(0, len(tasks), args.batch_size)]
    core_tasks = [_core_task(task) for task in tasks]
    tasks_by_id = {task.id: task for task in core_tasks}
    _write_experiment_manifest(output_dir, args, tasks)

    print(f"Mode: {args.mode}")
    if args.mode == "category-seeded-cost-aware":
        print(f"Category seed: {args.category_seed}")
    print("Benchmark: OSWorld 2.x")
    print("Evaluation backend: Harbor")
    print(f"Dry-run Harbor: {args.dry_run_harbor}")
    print(f"Tasks: {len(tasks)} in {len(batches)} batch(es)")
    print(f"Output: {output_dir.resolve()}")
    print(f"Workspace: {workspace_dir.resolve()}")
    print()

    all_results: list[dict[str, Any]] = []
    for batch_idx, batch_tasks in enumerate(batches, start=1):
        print(f"=== Batch {batch_idx}/{len(batches)} ===")
        batch_results = _solve_batch(
            batch_tasks=batch_tasks,
            batch_num=batch_idx,
            workspace_dir=workspace_dir,
            output_dir=output_dir,
            args=args,
        )
        all_results.extend(batch_results)

        observations = _results_to_observations(batch_results, tasks_by_id)
        _write_workspace_evolution_records(
            workspace_dir=workspace_dir,
            output_dir=output_dir,
            batch_results=batch_results,
            observations=observations,
            batch_num=batch_idx,
        )
        _write_artifact(
            output_dir / "emergence" / "observations" / f"batch_{batch_idx:02d}.json",
            [
                {
                    "task_id": obs.task.id,
                    "success": obs.feedback.success,
                    "score": obs.feedback.score,
                    "steps": len(obs.trajectory.steps or []),
                }
                for obs in observations
            ],
        )

        if batch_idx < len(batches):
            normal_record = _run_normal_evolution(
                workspace_dir=workspace_dir,
                output_dir=output_dir,
                batch_results=batch_results,
                observations=observations,
                batch_num=batch_idx,
                args=args,
            )
            print(
                "  Normal evolution: "
                f"solver_proposals={normal_record['solver_proposals']} "
                f"result={normal_record['result']}"
            )

        if args.mode in {"cost-aware", "predefined-cost-aware", "category-seeded-cost-aware"} and batch_idx < len(batches):
            report = _swe.propose_emergent_efficiency(
                batch_results=batch_results,
                batch_num=batch_idx,
                max_steps=args.max_steps,
                args=args,
            )
            _swe._write_proposals(output_dir, report)
            applied = _swe._apply_proposals(
                workspace_dir,
                report,
                output_dir=output_dir,
                max_apply=args.max_proposals_per_batch,
            )
            print(
                f"  Emergent proposals: {len(report.get('proposals', []))}, "
                f"applied: {sum(1 for item in applied if item.get('status') == 'applied')}"
            )

        if batch_idx < len(batches):
            _swe._record_workspace_snapshot(workspace_dir, output_dir, batch_idx)

        _write_run_outputs(output_dir, all_results, mode=args.mode)
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
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
