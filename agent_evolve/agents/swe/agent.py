"""SWE-bench coding agent -- uses strands-agents at runtime.

The framework layer (BaseAgent) loads prompts/skills/memory from the file
system contract.  This concrete agent then assembles those pieces into a
real ``strands.Agent`` and calls it, exactly like CodeDojo's original
``solve_instance``.  This keeps framework-level code strands-free while
ensuring *this* agent behaves identically to the CodeDojo version.
"""

from __future__ import annotations

import importlib
import logging
import math
import os
import re
from pathlib import Path
from typing import Any

from strands import Agent

from ...llm.base import LLMMessage
from ...llm.factory import create_llm_provider, create_strands_model, resolve_model_selection
from ...efficiency.profile import profile_task_text
from ...efficiency.profiler import normalize_usage
from ...efficiency.routing import normalize_route_level
from ...efficiency.library import load_strategy_library, select_library_config
from ...efficiency.selector import config_to_runtime_config
from ...protocol.base_agent import BaseAgent
from ...types import Task, Trajectory
from ...efficiency.strategies import (
    load_active_config,
    materialize_strategy_artifacts,
    strategy_prompt_block,
)
from .env import SWEBenchContainer, pull_image

logger = logging.getLogger(__name__)

os.environ.setdefault("BYPASS_TOOL_CONSENT", "true")


EFFICIENCY_MARKERS = {
    "adaptive_tool_trimming": (
        "EvoEfficient adaptive_tool_trimming",
    ),
    "verbatim_compaction": (
        "EvoEfficient verbatim_compaction",
    ),
    "observation_masking": (
        "EvoEfficient observation_masking",
    ),
    "observation_filter": (
        "EvoEfficient observation_filter",
    ),
}

STAGNATION_EXPLORATION_ACTIONS = {"navigate", "read_file", "search"}
STAGNATION_PROGRESS_ACTIONS = {"edit_file", "inspect_patch", "run_test", "submit"}


def detect_efficiency_markers(text: str) -> list[str]:
    """Return strategy ids with explicit runtime markers in tool output."""
    if not text:
        return []
    hits: list[str] = []
    for strategy, markers in EFFICIENCY_MARKERS.items():
        if any(marker in text for marker in markers):
            hits.append(strategy)
    return hits


class SweAgent(BaseAgent):
    """Reference agent for SWE-bench coding tasks.

    Reads system prompt, skills, and memories from the workspace via BaseAgent,
    then builds a strands ``Agent`` with those assets at solve-time -- the same
    pattern CodeDojo uses.
    """

    def __init__(
        self,
        workspace_dir: str | Path,
        model_id: str = "us.anthropic.claude-opus-4-6-v1",
        region: str = "us-west-2",
        max_tokens: int = 16384,
        max_steps: int = 0,
        window_size: int = 40,
        verification_focus: bool = False,
        efficiency_prompt: bool = False,
        enable_skill_proposal: bool = False,
        runtime_config: dict[str, Any] | None = None,
    ):
        # Explicit candidate configs are used during independent validation.
        # Materialize their workspace artifacts before BaseAgent loads skills
        # so validation sees the same executable surface as later reuse.
        if runtime_config:
            materialize_strategy_artifacts(
                workspace_dir,
                list(runtime_config.get("strategies", []) or []),
            )
        super().__init__(workspace_dir)
        self.model_id = model_id
        self.region = region
        self.max_tokens = max_tokens
        self.max_steps = max_steps  # 0 = unlimited
        self.window_size = window_size
        self._base_window_size = window_size
        self.efficiency_prompt = efficiency_prompt
        self.verification_focus = verification_focus
        self.enable_skill_proposal = enable_skill_proposal
        self._runtime_config_override = runtime_config
        self.efficiency_config = runtime_config or load_active_config(self.workspace.root)
        self._selected_config_id = (
            str(runtime_config.get("config_id", "")) if runtime_config else ""
        )
        self._selection_reason = "explicit_runtime_config" if runtime_config else "workspace_default"
        self._route_model_calls: list[dict[str, Any]] = []
        self._apply_runtime_parameters()

    def _cost_price_profile(self) -> dict[str, Any]:
        path = self.workspace.root / "efficiency" / "cost_profile.yaml"
        if not path.exists():
            return {}
        try:
            import yaml

            value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return value if isinstance(value, dict) else {}
        except Exception as exc:
            logger.warning("Failed to load cost profile %s: %s", path, exc)
            return {}

    def _apply_runtime_parameters(self) -> None:
        """Apply executable parameters after static or task-level config selection."""
        self.window_size = self._base_window_size
        if "context_reduction" not in self.efficiency_config.get("strategies", []):
            return
        configured_window = (
            self.efficiency_config
            .get("parameters", {})
            .get("context_reduction", {})
            .get("window_size")
        )
        if configured_window:
            self.window_size = min(self._base_window_size, int(configured_window))

    def _load_tools_from_workspace(self) -> tuple[list, list]:
        """Load tool functions from the workspace tools/registry.yaml.

        Supports two loading modes per entry:

        1. ``module`` — import from an installed Python package.
           e.g. ``{name: bash, module: agent_evolve.agents.swe.tools}``

        2. ``file`` — load from a ``.py`` file inside the workspace ``tools/``
           directory.  The evolver can drop new tool files here without
           touching framework source code.
           e.g. ``{name: smart_edit, file: smart_edit.py}``

        Every entry must have ``name`` plus either ``module`` or ``file``.

        Returns:
            A tuple of (tool_functions, tool_modules).  tool_modules is the
            deduplicated list of loaded modules so the agent can call
            ``reset()`` on each before a task.
        """
        registry = self.workspace.read_tool_registry()
        registry = self._filter_tools_for_efficiency(registry)
        tools: list = []
        modules: dict[str, Any] = {}  # deduplicate by id
        for entry in registry:
            name = entry.get("name")
            module_path = entry.get("module")
            file_path = entry.get("file")

            if not name or (not module_path and not file_path):
                logger.warning("Skipping tool entry with missing name/module/file: %s", entry)
                continue

            try:
                if module_path:
                    mod = importlib.import_module(module_path)
                else:
                    # Load from workspace tools/ directory
                    spec = importlib.util.spec_from_file_location(
                        f"workspace_tool_{name}",
                        self.workspace.tools_dir / file_path,
                    )
                    if spec is None or spec.loader is None:
                        raise ImportError(f"Cannot load spec for {file_path}")
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)

                tool_fn = getattr(mod, name)
                tools.append(tool_fn)
                modules[id(mod)] = mod
            except (ImportError, AttributeError, FileNotFoundError) as exc:
                logger.error("Failed to load tool %s: %s", name, exc)
        return tools, list(modules.values())

    def _filter_tools_for_efficiency(self, registry: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply low-risk tool-surface reduction when the strategy is active."""
        strategies = set(self.efficiency_config.get("strategies", []))
        if "tool_lazy_loading" not in strategies:
            return registry
        allowed = (
            self.efficiency_config
            .get("parameters", {})
            .get("tool_lazy_loading", {})
            .get("allowed_tools", ["bash", "text_editor", "submit"])
        )
        allowed_set = set(allowed)
        filtered = [entry for entry in registry if entry.get("name") in allowed_set]
        return filtered or registry

    @staticmethod
    def _reset_tool_modules(modules: list, **kwargs) -> None:
        """Call ``reset(**kwargs)`` on each tool module that exposes it."""
        for mod in modules:
            reset_fn = getattr(mod, "reset", None)
            if callable(reset_fn):
                try:
                    reset_fn(**kwargs)
                except Exception as exc:
                    logger.warning("reset() failed on %s: %s", getattr(mod, "__name__", mod), exc)

    @staticmethod
    def _get_submitted_patch(modules: list) -> str | None:
        """Extract submitted patch from whichever tool module provides it."""
        for mod in modules:
            fn = getattr(mod, "get_submitted_patch", None)
            if callable(fn):
                return fn()
        return None

    @staticmethod
    def _strategy_params(config: dict[str, Any], name: str) -> dict[str, Any]:
        params = config.get("parameters", {}) if isinstance(config, dict) else {}
        value = params.get(name, {}) if isinstance(params, dict) else {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _as_int_param(params: dict[str, Any], name: str, default: int) -> int:
        try:
            return int(params.get(name, default))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_float_param(params: dict[str, Any], name: str, default: float) -> float:
        try:
            return float(params.get(name, default))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _tool_input_command(tool_input: Any) -> str:
        if not isinstance(tool_input, dict):
            return ""
        for key in ("command", "cmd"):
            value = tool_input.get(key)
            if isinstance(value, str):
                return value
        return ""

    @classmethod
    def _classify_tool_action(cls, tool_name: str, tool_input: Any) -> str:
        """Classify a pending or completed tool call for budget guards."""
        name = str(tool_name or "")
        if name == "submit":
            return "submit"
        if name == "route_model":
            return "route_model"
        if name == "text_editor" and isinstance(tool_input, dict):
            command = str(tool_input.get("command", "")).strip().lower()
            if command in {"create", "str_replace", "insert", "undo_edit"}:
                return "edit_file"
            if command == "view":
                return "read_file"
            return "text_editor"
        if name == "python_exec":
            code = str(tool_input.get("code", "") if isinstance(tool_input, dict) else "").lower()
            if any(token in code for token in ("open(", "write(", ".write(", "with open")):
                return "edit_file"
            return "run_test"
        if name != "bash":
            return name

        cmd_lower = cls._tool_input_command(tool_input).strip().lower()
        command_parts = [
            part.strip()
            for part in re.split(r"\s*(?:&&|;)\s*", cmd_lower)
            if part.strip()
        ]
        effective_cmd = command_parts[-1] if command_parts else cmd_lower
        test_markers = (
            "runtests.py",
            "manage.py test",
            "pytest",
            "python -m pytest",
            "unittest",
            "tox ",
            " nose",
            " nosetests",
        )
        if any(marker in cmd_lower for marker in test_markers):
            return "run_test"
        if any(
            part.startswith(("pytest", "tox ", "python -m pytest"))
            for part in command_parts
        ):
            return "run_test"
        cmd_lower = effective_cmd
        if any(cmd_lower.startswith(p) for p in ("cat ", "head ", "tail ", "less ", "view ")):
            return "read_file"
        if any(cmd_lower.startswith(p) for p in ("grep ", "find ", "ag ", "rg ")):
            return "search"
        if (
            any(cmd_lower.startswith(p) for p in ("sed -i", "patch ", "tee "))
            or any(k in cmd_lower for k in ("cat >", "cat >>", "echo >", "echo >>"))
        ):
            return "edit_file"
        if any(cmd_lower.startswith(p) for p in ("python -m pytest", "pytest ")):
            return "run_test"
        if cmd_lower.startswith(("python ", "python3 ")):
            if any(k in cmd_lower for k in ("open(", "write(", ".write(", "with open")):
                return "edit_file"
            return "run_test"
        if any(cmd_lower.startswith(p) for p in ("ls ", "pwd", "tree ")):
            return "navigate"
        if cmd_lower.startswith(("git diff", "git status")):
            return "inspect_patch"
        if cmd_lower.startswith(("sed -n", "sed '")):
            return "read_file"
        return "navigate"

    @classmethod
    def _stagnation_rescue_decision(
        cls,
        *,
        trace: list[dict[str, Any]],
        max_steps: int,
        candidate_tool: str,
        candidate_input: Any,
        params: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Return a conservative rescue/hard-stop decision for stagnant tasks."""
        if max_steps <= 0:
            return {"action": "none", "signals": []}

        completed = len(trace)
        rescue_at = max(
            cls._as_int_param(params, "min_steps_before_rescue", 45),
            int(math.ceil(max_steps * cls._as_float_param(params, "rescue_at_step_ratio", 0.90))),
        )
        hard_at = max(
            cls._as_int_param(params, "min_steps_before_hard_stop", 55),
            int(math.ceil(max_steps * cls._as_float_param(params, "hard_stop_at_step_ratio", 0.98))),
        )
        if completed < rescue_at:
            return {"action": "none", "signals": []}

        candidate_action = cls._classify_tool_action(candidate_tool, candidate_input)
        if candidate_action == "submit":
            return {"action": "none", "signals": []}

        window = max(4, cls._as_int_param(params, "stagnant_window", 12))
        recent = trace[-window:]
        recent_actions = [str(step.get("action", "")) for step in recent if isinstance(step, dict)]
        recent_inputs = [
            str(step.get("input_summary", ""))
            for step in recent
            if isinstance(step, dict) and step.get("input_summary")
        ]
        recent_files = [
            str(step.get("file", ""))
            for step in recent
            if isinstance(step, dict) and step.get("file")
        ]
        all_actions = [str(step.get("action", "")) for step in trace if isinstance(step, dict)]
        exploration_count = sum(action in STAGNATION_EXPLORATION_ACTIONS for action in recent_actions)
        progress_count = sum(action in STAGNATION_PROGRESS_ACTIONS for action in recent_actions)
        exploration_ratio = exploration_count / max(1, len(recent_actions))

        signals: list[str] = []
        if (
            completed >= cls._as_int_param(params, "no_edit_after_steps", 45)
            and "edit_file" not in all_actions
        ):
            signals.append("no_edit_after_high_step_count")
        if (
            exploration_ratio
            >= cls._as_float_param(params, "exploration_ratio", 0.70)
            and len(recent_actions) >= max(4, window // 2)
        ):
            signals.append("recent_exploration_dominant")
        if progress_count == 0 and len(recent_actions) >= max(4, window // 2):
            signals.append("no_recent_edit_test_or_submit")
        if (
            len(recent_inputs) >= max(4, window // 2)
            and len(set(recent_inputs)) <= cls._as_int_param(params, "max_unique_recent_inputs", 5)
        ):
            signals.append("repeated_recent_tool_inputs")
        if (
            len(recent_files) >= max(4, window // 2)
            and len(set(recent_files)) <= cls._as_int_param(params, "max_unique_recent_files", 2)
        ):
            signals.append("narrow_recent_file_loop")
        if "edit_file" in all_actions:
            last_edit = max(i for i, action in enumerate(all_actions, start=1) if action == "edit_file")
            tests_after_edit = any(
                action == "run_test"
                for action in all_actions[last_edit:]
            )
            if (
                not tests_after_edit
                and completed - last_edit >= cls._as_int_param(params, "unvalidated_edit_steps", 10)
            ):
                signals.append("edited_but_not_recently_validated")

        min_signals = cls._as_int_param(params, "min_signals", 3)
        if len(signals) < min_signals:
            return {
                "action": "none",
                "signals": signals,
                "candidate_action": candidate_action,
            }

        rescue_started_at = state.get("rescue_started_at")
        grace_steps = cls._as_int_param(params, "rescue_grace_steps", 4)
        hard_ready = (
            rescue_started_at is not None
            and completed >= hard_at
            and completed - int(rescue_started_at) >= grace_steps
        )
        if hard_ready and candidate_action not in STAGNATION_PROGRESS_ACTIONS:
            return {
                "action": "hard_stop",
                "signals": signals,
                "candidate_action": candidate_action,
                "completed_steps": completed,
            }

        rescue_count = int(state.get("rescue_count", 0) or 0)
        max_rescues = cls._as_int_param(params, "max_rescue_messages", 1)
        if (
            rescue_count < max_rescues
            and candidate_action in STAGNATION_EXPLORATION_ACTIONS
        ):
            return {
                "action": "rescue",
                "signals": signals,
                "candidate_action": candidate_action,
                "completed_steps": completed,
            }

        return {
            "action": "none",
            "signals": signals,
            "candidate_action": candidate_action,
        }

    @staticmethod
    def _usage_has_cost(usage: dict[str, Any]) -> bool:
        return any(
            int(usage.get(name, 0) or 0) > 0
            for name in (
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "cache_read_input_tokens",
                "cache_write_input_tokens",
            )
        )

    @staticmethod
    def _sum_usage(
        usages: list[dict[str, Any]],
        *,
        price_profile: dict[str, Any] | None = None,
        model_id: str | None = None,
        provider: str | None = None,
        route_level: str | None = None,
    ) -> dict[str, Any]:
        total = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_write_input_tokens": 0,
        }
        for usage in usages:
            normalized = normalize_usage(
                usage,
                price_profile=price_profile,
                model_id=model_id,
                provider=provider,
                route_level=route_level,
            )
            for key in total:
                total[key] += int(normalized.get(key, 0) or 0)
        return normalize_usage(
            total,
            price_profile=price_profile,
            model_id=model_id,
            provider=provider,
            route_level=route_level,
        )

    @staticmethod
    def _message_metadata(message: Any) -> dict[str, Any]:
        if isinstance(message, dict):
            metadata = message.get("metadata", {})
            return metadata if isinstance(metadata, dict) else {}
        metadata = getattr(message, "metadata", {})
        return metadata if isinstance(metadata, dict) else {}

    @classmethod
    def _usage_from_messages(
        cls,
        messages: list[Any],
        *,
        price_profile: dict[str, Any] | None = None,
        model_id: str | None = None,
        provider: str | None = None,
        route_level: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        per_turn: list[dict[str, Any]] = []
        for message in messages or []:
            metadata = cls._message_metadata(message)
            raw_usage = metadata.get("usage", {})
            if not raw_usage:
                continue
            usage = normalize_usage(
                raw_usage,
                price_profile=price_profile,
                model_id=model_id,
                provider=provider,
                route_level=route_level,
            )
            if cls._usage_has_cost(usage):
                per_turn.append(usage)
        if not per_turn:
            return normalize_usage(
                {},
                price_profile=price_profile,
                model_id=model_id,
                provider=provider,
                route_level=route_level,
            ), []
        usage = cls._sum_usage(
            per_turn,
            price_profile=price_profile,
            model_id=model_id,
            provider=provider,
            route_level=route_level,
        )
        usage.update(
            {
                "usage_record_quality": "ok",
                "usage_source": "agent_messages_metadata",
                "usage_estimated": False,
            }
        )
        return usage, per_turn

    @classmethod
    def _usage_from_metrics(
        cls,
        metrics: Any,
        *,
        source: str,
        price_profile: dict[str, Any] | None = None,
        model_id: str | None = None,
        provider: str | None = None,
        route_level: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if metrics is None:
            return normalize_usage(
                {},
                price_profile=price_profile,
                model_id=model_id,
                provider=provider,
                route_level=route_level,
            ), []
        raw_usage = getattr(metrics, "accumulated_usage", {}) or {}
        usage = normalize_usage(
            raw_usage,
            price_profile=price_profile,
            model_id=model_id,
            provider=provider,
            route_level=route_level,
            usage_record_quality="ok",
            usage_source=source,
            usage_estimated=False,
        )
        per_turn: list[dict[str, Any]] = []
        invocations = getattr(metrics, "agent_invocations", []) or []
        invocation = invocations[-1] if invocations else None
        for cycle in getattr(invocation, "cycles", []) or []:
            cycle_usage = getattr(cycle, "usage", {}) or {}
            normalized = normalize_usage(
                cycle_usage,
                price_profile=price_profile,
                model_id=model_id,
                provider=provider,
                route_level=route_level,
            )
            if cls._usage_has_cost(normalized):
                per_turn.append(normalized)
        if not cls._usage_has_cost(usage) and per_turn:
            usage = cls._sum_usage(
                per_turn,
                price_profile=price_profile,
                model_id=model_id,
                provider=provider,
                route_level=route_level,
            )
            usage.update(
                {
                    "usage_record_quality": "ok",
                    "usage_source": f"{source}_cycles",
                    "usage_estimated": False,
                }
            )
        return usage, per_turn

    @staticmethod
    def _text_from_message_part(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "\n".join(SweAgent._text_from_message_part(item) for item in value)
        if not isinstance(value, dict):
            return str(value)
        parts: list[str] = []
        text = value.get("text")
        if isinstance(text, str):
            parts.append(text)
        reasoning = value.get("reasoningContent")
        if isinstance(reasoning, dict):
            parts.append(SweAgent._text_from_message_part(reasoning))
        reasoning_text = value.get("reasoningText")
        if isinstance(reasoning_text, dict):
            parts.append(SweAgent._text_from_message_part(reasoning_text))
        tool_use = value.get("toolUse")
        if isinstance(tool_use, dict):
            parts.append(str(tool_use.get("name", "")))
            parts.append(str(tool_use.get("input", "")))
        tool_result = value.get("toolResult")
        if isinstance(tool_result, dict):
            parts.append(SweAgent._text_from_message_part(tool_result.get("content", [])))
        content = value.get("content")
        if content is not None:
            parts.append(SweAgent._text_from_message_part(content))
        return "\n".join(part for part in parts if part)

    @classmethod
    def _estimate_usage_from_runtime(
        cls,
        *,
        system_prompt: str,
        user_prompt: str,
        conversation: list[Any],
        tool_call_trace: list[dict[str, Any]],
        response: Any,
        price_profile: dict[str, Any] | None = None,
        model_id: str | None = None,
        provider: str | None = None,
        route_level: str | None = None,
    ) -> dict[str, Any]:
        input_chars = len(system_prompt or "") + len(user_prompt or "")
        output_chars = len(str(response or ""))
        if conversation:
            input_chars = len(system_prompt or "")
            output_chars = 0
            for message in conversation:
                role = ""
                content: Any = message
                if isinstance(message, dict):
                    role = str(message.get("role", ""))
                    content = message.get("content", message)
                text = cls._text_from_message_part(content)
                if role == "assistant":
                    output_chars += len(text)
                else:
                    input_chars += len(text)
        for step in tool_call_trace:
            if not isinstance(step, dict):
                continue
            input_chars += len(str(step.get("input_summary", "")))
            input_chars += len(str(step.get("result_snippet", "")))

        estimated = {
            "input_tokens": max(1, math.ceil(input_chars / 4)),
            "output_tokens": max(0, math.ceil(output_chars / 4)),
        }
        usage = normalize_usage(
            estimated,
            price_profile=price_profile,
            model_id=model_id,
            provider=provider,
            route_level=route_level,
            usage_record_quality="estimated",
            usage_source="text_heuristic_after_missing_provider_usage",
            usage_estimated=True,
        )
        return usage

    @classmethod
    def _extract_usage_telemetry(
        cls,
        *,
        response: Any,
        agent: Agent,
        conversation: list[Any],
        system_prompt: str,
        user_prompt: str,
        tool_call_trace: list[dict[str, Any]],
        price_profile: dict[str, Any] | None = None,
        model_id: str | None = None,
        provider: str | None = None,
        route_level: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        usage, per_turn = cls._usage_from_metrics(
            getattr(response, "metrics", None),
            source="response_metrics",
            price_profile=price_profile,
            model_id=model_id,
            provider=provider,
            route_level=route_level,
        )
        if not cls._usage_has_cost(usage):
            usage, per_turn = cls._usage_from_metrics(
                getattr(agent, "metrics", None),
                source="agent_metrics",
                price_profile=price_profile,
                model_id=model_id,
                provider=provider,
                route_level=route_level,
            )
        if not cls._usage_has_cost(usage):
            usage, per_turn = cls._usage_from_messages(
                conversation,
                price_profile=price_profile,
                model_id=model_id,
                provider=provider,
                route_level=route_level,
            )
        if not cls._usage_has_cost(usage):
            usage = cls._estimate_usage_from_runtime(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                conversation=conversation,
                tool_call_trace=tool_call_trace,
                response=response,
                price_profile=price_profile,
                model_id=model_id,
                provider=provider,
                route_level=route_level,
            )
            per_turn = [usage]
        return usage, per_turn

    def _build_strands_agent(self, task_profile: dict[str, Any] | None = None) -> tuple[Agent, list]:
        """Create a strands Agent wired with the workspace's current state.

        Returns:
            A tuple of (Agent, tool_modules) so the caller can reset modules.
        """
        model = create_strands_model(
            model_id=self.model_id,
            region=self.region,
            max_tokens=self.max_tokens,
            task_profile=task_profile,
            route_level="hard",
        )

        system_prompt = self._build_system_prompt(task_profile=task_profile)
        tools, modules = self._load_tools_from_workspace()
        if "difficulty_model_routing" in self.efficiency_config.get("strategies", []):
            tools.append(self._build_route_model_tool(task_profile=task_profile))

        # Only register read_skill tool if skills actually exist
        if self.skills:
            skill_data = {}
            for skill in self.skills:
                content = self.get_skill_content(skill.name)
                if content:
                    body = content.split("---", 2)[-1].strip() if "---" in content else content
                    skill_data[skill.name] = body

            from strands import tool

            @tool
            def read_skill(skill_name: str) -> str:
                """Read the full procedure for a skill. Call this when a skill's description matches your current task.

                Args:
                    skill_name: Name of the skill to read
                """
                if skill_name in skill_data:
                    return skill_data[skill_name]
                available = ", ".join(skill_data.keys())
                return f"Skill '{skill_name}' not found. Available: {available}"

            tools.append(read_skill)

        from .conversation_manager import PinnedFirstMessageManager

        agent = Agent(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            conversation_manager=PinnedFirstMessageManager(
                window_size=self.window_size,
                should_truncate_results=True,
                per_turn=True,
            ),
            )
        return agent, modules

    def _build_route_model_tool(self, task_profile: dict[str, Any] | None = None):
        """Expose explicit step-level model routing to the hard main agent."""
        from strands import tool

        @tool
        def route_model(
            route_level: str,
            task: str,
            context: str = "",
            expected_output: str = "",
            reason: str = "",
            max_tokens: int = 512,
        ) -> str:
            """Delegate a bounded substep to the easy or hard route.

            Use route_level="easy" for simple extraction, short summarization,
            formatting, or mechanical checks over already collected evidence.
            route_level="hard" records that the step should stay in the main
            hard-model reasoning loop; it does not make a second hard-model call.

            Args:
                route_level: "easy" or "hard"; only easy invokes a delegated model call.
                task: The precise subtask for the delegated model.
                context: Small, already collected evidence needed for the subtask.
                expected_output: The expected shape of the answer.
                reason: Why this step is safe to route.
                max_tokens: Output budget for the delegated model.
            """
            level = normalize_route_level(route_level)
            if level not in {"easy", "hard"}:
                return "ERROR: route_level must be 'easy' or 'hard'. Continue with the main hard model."

            task_text = str(task or "").strip()
            context_text = str(context or "").strip()
            output_contract = str(expected_output or "").strip()
            route_reason = str(reason or "").strip()
            capped_tokens = max(64, min(int(max_tokens or 512), 2048))
            if level == "easy":
                capped_tokens = min(capped_tokens, 768)
            selected_provider, selected_model = resolve_model_selection(
                self.model_id,
                task_profile=task_profile,
                route_level=level,
            )

            record: dict[str, Any] = {
                "route_level": level,
                "model_id": selected_model,
                "provider": selected_provider,
                "task": task_text[:240],
                "reason": route_reason[:240],
                "expected_output": output_contract[:160],
                "context_chars": len(context_text),
                "max_tokens": capped_tokens,
                "status": "started",
            }
            self._route_model_calls.append(record)

            if not task_text:
                record["status"] = "error"
                record["error"] = "empty_task"
                return "ERROR: route_model requires a non-empty subtask. Continue with the main hard model."

            if level == "hard":
                record.update({
                    "status": "kept_in_main",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_write_input_tokens": 0,
                    "cost_units": 0.0,
                })
                return (
                    "HARD_ROUTE_SELECTED: keep this substep in the main hard-model "
                    "reasoning loop and solve it directly."
                )

            if len(context_text) > 6000:
                record.update({
                    "status": "rejected_large_context",
                    "error": "easy_context_over_6000_chars",
                })
                return (
                    "ESCALATE_TO_HARD: easy route rejected because the provided "
                    "context is too large. Use a narrower context or continue in "
                    "the main hard-model reasoning loop."
                )

            prompt = (
                "You are a delegated step model inside a SWE-bench agent.\n"
                f"Route level: {level}\n\n"
                "Answer only the bounded subtask below. Do not edit files, do not run tools, "
                "and do not invent facts beyond the provided context. If the step requires "
                "root-cause diagnosis, patch design, or verification judgment, say "
                "'ESCALATE_TO_HARD' with a one-sentence reason.\n\n"
                f"Subtask:\n{task_text}\n\n"
                f"Why this route was chosen:\n{route_reason or 'not provided'}\n\n"
                f"Expected output:\n{output_contract or 'concise answer'}\n\n"
                f"Context:\n{context_text or '(no additional context)'}"
            )

            try:
                delegated_llm = create_llm_provider(
                    self.model_id,
                    region=self.region,
                    task_profile=task_profile,
                    route_level=level,
                )
                response = delegated_llm.complete(
                    [LLMMessage(role="user", content=prompt)],
                    max_tokens=capped_tokens,
                    temperature=0.0,
                )
                content = str(response.content or "").strip()
                usage = normalize_usage(
                    response.usage or {},
                    price_profile=self._cost_price_profile(),
                    model_id=selected_model,
                    provider=selected_provider,
                    route_level=level,
                )
                record.update({
                    "status": "success",
                    "response_chars": len(content),
                    "input_tokens": int(usage.get("input_tokens", 0) or 0),
                    "output_tokens": int(usage.get("output_tokens", 0) or 0),
                    "total_tokens": int(usage.get("total_tokens", 0) or 0),
                    "cache_read_input_tokens": int(
                        usage.get("cache_read_input_tokens", 0) or 0
                    ),
                    "cache_write_input_tokens": int(
                        usage.get("cache_write_input_tokens", 0) or 0
                    ),
                    "cost_units": float(usage.get("cost_units", 0.0) or 0.0),
                    "input_price_per_1m": float(usage.get("input_price_per_1m", 0.0) or 0.0),
                    "output_price_per_1m": float(usage.get("output_price_per_1m", 0.0) or 0.0),
                    "cache_read_price_per_1m": float(
                        usage.get("cache_read_price_per_1m", 0.0) or 0.0
                    ),
                    "cache_write_price_per_1m": float(
                        usage.get("cache_write_price_per_1m", 0.0) or 0.0
                    ),
                    "cost_currency": usage.get("cost_currency", ""),
                    "cost_price_source": usage.get("cost_price_source", ""),
                    "cost_priced": bool(usage.get("cost_priced", False)),
                })
                return content or "EMPTY_ROUTE_MODEL_RESPONSE"
            except Exception as exc:  # noqa: BLE001
                record.update({
                    "status": "error",
                    "error": str(exc)[:500],
                })
                logger.warning("route_model(%s) failed: %s", level, exc)
                return (
                    f"ERROR: route_model {level} call failed: {exc}. "
                    "Continue with the main hard model."
                )

        return route_model

    def solve(self, task: Task) -> Trajectory:
        """Solve a SWE-bench instance.

        Expects task.metadata to contain:
          - docker_image: str (SWE-bench Docker image name)
          - instance_id: str (optional, defaults to task.id)
        """
        docker_image = task.metadata.get("docker_image", "")
        instance_id = task.metadata.get("instance_id", task.id)
        problem_statement = task.input
        task_profile = profile_task_text(problem_statement, metadata=task.metadata, history=self.memories)
        self._route_model_calls = []
        if self._runtime_config_override is None:
            library = load_strategy_library(self.workspace.root)
            selected = select_library_config(library, task_profile.as_dict())
            if selected:
                self.efficiency_config = config_to_runtime_config(selected)
                self._selected_config_id = str(selected.get("id", ""))
                self._selection_reason = "matched_strategy_library_config"
            else:
                self.efficiency_config = load_active_config(self.workspace.root)
                self._selected_config_id = str(
                    self.efficiency_config.get("metadata", {}).get("efficiency_config_id", "")
                )
                self._selection_reason = "workspace_default_or_baseline_fallback"
            self._apply_runtime_parameters()
            materialize_strategy_artifacts(
                self.workspace.root,
                list(self.efficiency_config.get("strategies", []) or []),
            )
            # A library-selected config can add compiled skills after
            # BaseAgent initialization. Reload the filesystem contract so the
            # selected task sees those newly materialized skills.
            self.reload_from_fs()

        if not docker_image:
            raise ValueError(
                f"Task {task.id} missing 'docker_image' in metadata. "
                "SweAgent requires a SWE-bench Docker image."
            )

        pull_image(docker_image)
        container = SWEBenchContainer(docker_image)
        tool_call_trace: list[dict] = []
        early_stop_state: dict[str, Any] = {
            "rescue_started_at": None,
            "rescue_count": 0,
            "events": [],
        }

        with container:
            agent, tool_modules = self._build_strands_agent(task_profile=task_profile.as_dict())
            self._reset_tool_modules(
                tool_modules,
                container_name=container.container_name,
                workspace_dir=str(self.workspace.root),
                efficiency_config=self.efficiency_config,
            )

            # Register hook to capture tool-call-level trace
            self._register_trace_hook(agent, tool_call_trace)

            # Register turn limiter if max_steps is set
            if self.max_steps > 0:
                from strands.hooks.events import BeforeToolCallEvent
                _step_count = [0]
                _submitted = [False]
                early_stop_enabled = (
                    "stagnation_rescue_stop"
                    in self.efficiency_config.get("strategies", [])
                )
                early_stop_params = self._strategy_params(
                    self.efficiency_config,
                    "stagnation_rescue_stop",
                )

                def _turn_limiter(event: BeforeToolCallEvent):
                    if _submitted[0]:
                        event.cancel_tool = "Already submitted."
                        return
                    tool_name = event.tool_use.get("name", "")
                    if tool_name == "submit":
                        _submitted[0] = True
                        return
                    _step_count[0] += 1
                    if _step_count[0] > self.max_steps:
                        event.cancel_tool = f"Step limit reached ({self.max_steps}). Call submit now."
                        return
                    if early_stop_enabled:
                        decision = self._stagnation_rescue_decision(
                            trace=tool_call_trace,
                            max_steps=self.max_steps,
                            candidate_tool=tool_name,
                            candidate_input=event.tool_use.get("input", {}),
                            params=early_stop_params,
                            state=early_stop_state,
                        )
                        if decision.get("action") == "rescue":
                            early_stop_state["rescue_started_at"] = int(
                                decision.get("completed_steps", len(tool_call_trace))
                            )
                            early_stop_state["rescue_count"] = int(
                                early_stop_state.get("rescue_count", 0) or 0
                            ) + 1
                            event_record = {
                                "type": "rescue",
                                "step": int(decision.get("completed_steps", len(tool_call_trace))),
                                "candidate_tool": tool_name,
                                "candidate_action": decision.get("candidate_action", ""),
                                "signals": list(decision.get("signals", []) or []),
                            }
                            early_stop_state["events"].append(event_record)
                            event.cancel_tool = (
                                "EvoEfficient stagnation_rescue_stop: rescue checkpoint. "
                                "You are close to the step limit and repeated exploration appears stagnant. "
                                "Stop broad search now. Use the remaining budget only to make the smallest "
                                "safe patch, run one focused verification, or submit the current best patch."
                            )
                            return
                        if decision.get("action") == "hard_stop":
                            event_record = {
                                "type": "hard_stop",
                                "step": int(decision.get("completed_steps", len(tool_call_trace))),
                                "candidate_tool": tool_name,
                                "candidate_action": decision.get("candidate_action", ""),
                                "signals": list(decision.get("signals", []) or []),
                            }
                            early_stop_state["events"].append(event_record)
                            event.cancel_tool = (
                                "EvoEfficient stagnation_rescue_stop: hard stop for non-submit tools. "
                                "Submit the best available patch now. If no safe patch exists, submit an empty patch."
                            )
                            return

                agent.hooks.add_callback(BeforeToolCallEvent, _turn_limiter)
                logger.info("Turn limiter: max_steps=%d", self.max_steps)

            user_prompt = self._build_user_prompt(instance_id, problem_statement)

            logger.info("Solving %s with image %s", instance_id, docker_image)
            response = None
            agent_error = ""
            try:
                response = agent(user_prompt)
            except Exception as exc:
                agent_error = str(exc)
                logger.warning(
                    "Agent solve loop ended with an exception for %s; "
                    "will evaluate any patch produced so far: %s",
                    instance_id,
                    agent_error,
                )

            try:
                conversation = list(getattr(agent, "messages", []) or [])
            except Exception:
                conversation = []
            system_prompt_for_usage = self._build_system_prompt(
                task_profile=task_profile.as_dict()
            )
            price_profile = self._cost_price_profile()
            hard_provider, hard_model = resolve_model_selection(
                self.model_id,
                task_profile=task_profile.as_dict(),
                route_level="hard",
            )
            usage, per_turn_usage = self._extract_usage_telemetry(
                response=response,
                agent=agent,
                conversation=conversation,
                system_prompt=system_prompt_for_usage,
                user_prompt=user_prompt,
                tool_call_trace=tool_call_trace,
                price_profile=price_profile,
                model_id=hard_model,
                provider=hard_provider,
                route_level="hard",
            )
            main_cost_units = float(usage.get("cost_units", 0.0) or 0.0)
            route_model_cost_units = sum(
                float(call.get("cost_units", 0.0) or 0.0)
                for call in self._route_model_calls
                if isinstance(call, dict)
            )
            usage["main_cost_units"] = main_cost_units
            usage["route_model_cost_units"] = route_model_cost_units
            if route_model_cost_units:
                route_cost_by_level: dict[str, float] = {}
                for call in self._route_model_calls:
                    if not isinstance(call, dict):
                        continue
                    level = str(call.get("route_level") or "unknown")
                    route_cost_by_level[level] = route_cost_by_level.get(level, 0.0) + float(
                        call.get("cost_units", 0.0) or 0.0
                    )
                usage["route_model_cost_by_level"] = route_cost_by_level
                usage["cost_units"] = main_cost_units + route_model_cost_units

            patch = self._get_submitted_patch(tool_modules) or container.get_diff()

            # Build steps: tool-call trace + summary with usage
            steps = list(tool_call_trace)
            steps.append({
                "llm_output": str(response)[:2000] if response is not None else "",
                "agent_error": agent_error,
                "usage": usage,
                "per_turn_usage": per_turn_usage,
                "num_turns": len(per_turn_usage),
                "max_input_tokens_per_turn": max((t.get("input_tokens", 0) for t in per_turn_usage), default=0),
                "usage_record_quality": usage.get("usage_record_quality", ""),
                "usage_source": usage.get("usage_source", ""),
                "usage_estimated": bool(usage.get("usage_estimated", False)),
                "active_efficiency_strategies": self.efficiency_config.get("strategies", []),
                "task_profile": task_profile.as_dict(),
                "efficiency_config_id": self._selected_config_id,
                "efficiency_selection_reason": self._selection_reason,
                "efficiency_config": self.efficiency_config.get(
                    "metadata", {}
                ).get("efficiency_config", {}),
                "route_model_calls": list(self._route_model_calls),
                "early_stop_events": list(early_stop_state.get("events", [])),
            })

            if not patch.strip():
                logger.warning("No changes detected for %s", instance_id)

            skill_proposal = ""
            # Skill proposal is useful for in-situ evolution, but the MVP
            # strategy experiments should not pay this extra model-call cost.
            if self.enable_skill_proposal:
                try:
                    # Build skill context for the proposal prompt
                    skill_context = ""
                    if self.skills:
                        skill_list = "\n".join(f"- {s.name}: {s.description}" for s in self.skills)
                        skill_context = (
                            f"You had these skills available (you may have read some via read_skill):\n"
                            f"{skill_list}\n\n"
                        )

                    if self.verification_focus:
                        proposal_prompt = (
                        f"{skill_context}"
                        "Reflect on your VERIFICATION process — how you tested and validated your fix.\n\n"
                        "Think about:\n"
                        "- How did you find the right test file? Was it easy or did you guess?\n"
                        "- Did you run tests before AND after your edit?\n"
                        "- Did your repro script catch the real issue, or was it too simple?\n"
                        "- Were there edge cases in the issue that you didn't test?\n"
                        "- Did you verify your fix doesn't break adjacent functionality?\n\n"
                        "Propose a VERIFICATION skill that helps future solvers test more thoroughly.\n"
                        "Focus ONLY on how to verify/test — not on how to find or write the fix.\n\n"
                        "CRITICAL: The skill must be GENERALIZABLE — applicable to many different tasks, "
                        "not just this one. Abstract away the specific framework/module details.\n"
                        "  GOOD names: verify_falsy_edge_cases, verify_before_after_edit, verify_inheritance_chain, "
                        "verify_roundtrip_integrity, verify_mutable_state_isolation\n"
                        "  BAD names: verify_admin_kwargs_passthrough, verify_django_enum_str, verify_sphinx_toctree\n"
                        "  GOOD content: general methodology (e.g., 'test all falsy-but-valid values: 0, False, \"\", [], {}')\n"
                        "  BAD content: specific code paths (e.g., 'check django/contrib/admin/options.py line 200')\n\n"
                        "OPTION A — ENHANCE an existing verification skill:\n"
                        "CONFIDENCE: HIGH/MEDIUM/LOW\n"
                        "ACTION: ENHANCE\n"
                        "TARGET: existing_skill_name\n"
                        "ANALYSIS: what was missing in the verification approach\n"
                        "TYPE: skill\n"
                        "NAME: same_skill_name\n"
                        "DESCRIPTION: TRIGGER when / DO NOT TRIGGER when (one sentence)\n"
                        "CONTENT:\n"
                        "(verification methodology, under 500 words)\n\n"
                        "OPTION B — NEW verification skill:\n"
                        "CONFIDENCE: HIGH/MEDIUM/LOW\n"
                        "ACTION: NEW\n"
                        "TYPE: skill\n"
                        "NAME: verify_<general_pattern>\n"
                        "DESCRIPTION: TRIGGER when / DO NOT TRIGGER when (one sentence)\n"
                        "CONTENT:\n"
                        "(verification methodology, under 500 words)\n\n"
                        "OPTION C — No proposal:\n"
                        "CONFIDENCE: HIGH/MEDIUM/LOW\n"
                        "ACTION: NONE"
                        )
                    else:
                        proposal_prompt = (
                        f"{skill_context}"
                        "Help FUTURE solvers by proposing or enhancing a skill.\n\n"
                        "IMPORTANT RULES:\n"
                        "- NAME must be GENERIC and reusable.\n"
                        "  GOOD: fix_shallow_copy_mutation, align_parallel_code_paths\n"
                        "  BAD:  django_mti_pk_fix, sphinx_autodoc_classmethod\n"
                        "- DESCRIPTION must include TRIGGER and DO NOT TRIGGER conditions so the solver knows when to load it:\n"
                        "  GOOD: 'TRIGGER when: copied objects share mutable state, or __deepcopy__ is defined. "
                        "DO NOT TRIGGER when: simple value assignment or immutable types.'\n"
                        "  BAD:  'Fix for deep copy issues' (too vague, no trigger conditions)\n"
                        "- Prefer OPTION A (enhance existing) over OPTION B (new).\n\n"
                        "OPTION A — ENHANCE an existing skill:\n"
                        "CONFIDENCE: HIGH/MEDIUM/LOW\n"
                        "ACTION: ENHANCE\n"
                        "TARGET: existing_skill_name\n"
                        "ANALYSIS: one sentence on what to improve\n"
                        "TYPE: skill\n"
                        "NAME: same_skill_name\n"
                        "DESCRIPTION: one sentence, max 15 words\n"
                        "CONTENT:\n"
                        "(enhanced methodology, under 500 words)\n\n"
                        "OPTION B — NEW skill (only if nothing existing covers this):\n"
                        "CONFIDENCE: HIGH/MEDIUM/LOW\n"
                        "ACTION: NEW\n"
                        "TYPE: skill\n"
                        "NAME: pattern_name\n"
                        "DESCRIPTION: one sentence, max 15 words\n"
                        "CONTENT:\n"
                        "(methodology, under 500 words)\n\n"
                        "OPTION C — No proposal:\n"
                        "CONFIDENCE: HIGH/MEDIUM/LOW\n"
                        "ACTION: NONE"
                        )

                    proposal_llm = create_llm_provider(
                        self.model_id,
                        region=self.region,
                        task_profile=task_profile.as_dict(),
                    )
                    proposal_response = proposal_llm.complete(
                        [LLMMessage(role="user", content=proposal_prompt)],
                        max_tokens=1024,
                        temperature=0.2,
                    )
                    skill_proposal = str(proposal_response.content or "").strip()[:2500]
                    if "ACTION: NONE" not in skill_proposal.upper():
                        action = "ENHANCE" if "ACTION: ENHANCE" in skill_proposal.upper() else "NEW"
                        logger.info("Skill proposal (%s) for %s: %s", action, instance_id, skill_proposal[:100])
                    else:
                        skill_proposal = skill_proposal if "CONFIDENCE:" in skill_proposal else ""
                except Exception as e:
                    logger.warning("Skill proposal failed for %s: %s", instance_id, e)

        traj = Trajectory(task_id=task.id, output=patch, steps=steps)
        traj._conversation = conversation
        traj._system_prompt = system_prompt_for_usage
        traj._skill_proposal = skill_proposal
        return traj
    @staticmethod
    def _register_trace_hook(agent: Agent, trace: list[dict]) -> None:
        """Register a strands hook that captures every tool call into trace.

        Each entry records the tool name, input args, status, and a truncated
        snippet of the result — giving the evolver fine-grained visibility
        into what the agent actually did.
        """
        from strands.hooks import AfterToolCallEvent

        def _on_tool_call(event: AfterToolCallEvent) -> None:
            tool_name = event.tool_use.get("name", "unknown")
            tool_input = event.tool_use.get("input", {})
            status = event.result.get("status", "unknown") if event.result else "error"

            # Extract file path from common tool input patterns
            file_path = ""
            if isinstance(tool_input, dict):
                file_path = tool_input.get("file", tool_input.get("path", ""))
                # For bash tool, try to extract file from command
                if not file_path and tool_name == "bash":
                    cmd = tool_input.get("command", "")
                    if isinstance(cmd, str):
                        # Detect common file-reading commands
                        for prefix in ("cat ", "head ", "tail ", "less ", "view "):
                            if prefix in cmd:
                                file_path = cmd.split(prefix, 1)[-1].strip().split()[0] if prefix in cmd else ""
                                break

            # Classify the action type.
            action = SweAgent._classify_tool_action(tool_name, tool_input)
            skill_name = ""
            if tool_name == "read_skill" and isinstance(tool_input, dict):
                skill_name = str(
                    tool_input.get("skill_name")
                    or tool_input.get("name")
                    or tool_input.get("skill")
                    or ""
                ).strip()

            # Truncate result content for storage
            result_snippet = ""
            result_full = ""
            if event.result:
                content = event.result.get("content", [])
                if content and isinstance(content, list):
                    result_parts: list[str] = []
                    for item in content:
                        if isinstance(item, dict):
                            result_parts.append(str(item.get("text", "")))
                        else:
                            result_parts.append(str(item))
                    result_full = "\n".join(part for part in result_parts if part)
                    result_snippet = result_full[:500]

            efficiency_markers = detect_efficiency_markers(result_full)
            if skill_name:
                efficiency_markers.append(f"read_skill:{skill_name}")
                if skill_name.startswith("emergent_gen"):
                    efficiency_markers.append(f"emergent_skill:{skill_name}")

            trace.append({
                "tool": tool_name,
                "action": action,
                "input_summary": str(tool_input)[:300] if tool_input else "",
                "skill_name": skill_name,
                "file": file_path,
                "status": status,
                "result_snippet": result_snippet,
                "efficiency_markers": efficiency_markers,
            })

        agent.hooks.add_callback(AfterToolCallEvent, _on_tool_call)

    def _build_system_prompt(self, task_profile: dict[str, Any] | None = None) -> str:
        """Assemble the full system prompt from workspace files.

        Includes the base prompt, any evolved prompt fragments, and skills.
        Memories are injected into the user prompt via ``_build_user_prompt``
        so they appear once, in the task-specific context.
        """
        parts = [self.system_prompt]

        if task_profile:
            parts.append(
                "\n\n## Task Profile\n"
                f"- task type: {task_profile.get('task_type', 'general')}\n"
                f"- domain: {task_profile.get('domain', 'generic')}\n"
                f"- difficulty: {task_profile.get('difficulty', 'medium')}\n"
                f"- context pressure: {task_profile.get('context_pressure', 'medium')}\n"
                f"- tool pressure: {task_profile.get('tool_pressure', 'medium')}\n"
                f"- verification pressure: {task_profile.get('verification_pressure', 'medium')}\n"
                f"- output pressure: {task_profile.get('output_pressure', 'medium')}\n"
                "Use this profile to choose how much context, reasoning, tool use, and verification to spend."
            )

        # Verify-fix loop instruction
        parts.append(
            "\n\n## Verify Your Fix\n"
            "Before editing, find the existing test file for the module you're changing. Use "
            "`grep -r 'def test_' tests/ --include='*.py' -l` or check `tests/test_<module>.py`. "
            "Run these tests BEFORE your edit to see the baseline, then AFTER to confirm your fix "
            "passes and doesn't break other tests.\n\n"
            "Also write a small reproduction script from the issue description and run it. "
            "If it still fails after your edit, read the traceback — there may be another "
            "file that needs changing. Repeat until both the repro and existing tests pass."
        )

        if self.efficiency_prompt:
            parts.append(
                "\n\n## Efficiency Rules (CRITICAL)\n"
                "You have a LIMITED tool call budget. Work hypothesis-first, not exploration-first.\n\n"
                "**Phase 1 — Locate (max 10 tool calls):**\n"
                "- Read the issue. Identify the key class/function/error mentioned.\n"
                "- Use `grep -rn` to find the relevant source file(s). Limit to 3 candidate files.\n"
                "- Read only the relevant sections of those files (use `sed -n 'START,ENDp'`).\n\n"
                "**Phase 2 — Hypothesize (0 tool calls):**\n"
                "- State your hypothesis: what's wrong and what change will fix it.\n"
                "- If you can't form a hypothesis after Phase 1, re-read the issue — don't explore more files.\n\n"
                "**Phase 3 — Fix & Verify (max 15 tool calls):**\n"
                "- Make your edit. Run the relevant test(s). Check the result.\n"
                "- If tests fail, read the traceback and adjust your edit — don't go back to exploring.\n"
                "- Submit when tests pass.\n\n"
                "**ANTI-PATTERNS — Do NOT:**\n"
                "- Read more than 5 source files before making your first edit\n"
                "- Run `find` or `ls -R` on the entire repository\n"
                "- Read entire files when you only need a specific function\n"
                "- Explore 'just to understand the codebase' — you understand enough after Phase 1\n"
                "- Start over with a different approach more than once"
            )

        active_efficiency_prompt = strategy_prompt_block(self.efficiency_config, task_profile=task_profile)
        if active_efficiency_prompt:
            parts.append(active_efficiency_prompt)

        # Include evolved prompt fragments
        fragments = self.workspace.list_fragments()
        if fragments:
            for frag_name in fragments:
                content = self.workspace.read_fragment(frag_name)
                if content and content.strip():
                    # Fragments are already injected into system.md via evolve,
                    # but if they weren't (e.g. loaded from fragments/ dir),
                    # include them here as a fallback.
                    marker = f"<!-- evolve:{frag_name.removesuffix('.md')} -->"
                    if marker not in self.system_prompt:
                        parts.append(f"\n\n## {frag_name.removesuffix('.md').replace('_', ' ').title()}")
                        parts.append(content)

        parts.append("\n\n## Skills\n")
        if self.skills:
            parts.append(
                "You have skills learned from previous tasks. At the start of the task, scan "
                "the skill list below and decide whether any skill description matches the "
                "current problem, expected workflow, or likely efficiency bottleneck.\n\n"
                "Skill-use rule:\n"
                "- If one or more skills match, call `read_skill(skill_name)` before broad "
                "exploration so you can apply the full procedure.\n"
                # "- Prioritize skills whose names start with `emergent_gen` or whose descriptions "
                # "mention avoiding repeated commands, repeated file reads, focused testing, "
                # "temporary repro scripts, caching, or other cost-saving behavior.\n"
                # "- Usually read 1 matching skill; read up to 3 only when their triggers are "
                # "clearly relevant. Do not read unrelated skills just to browse.\n"
                "- Before repeating a command, re-reading a large file, running a broad test, "
                "or approaching the step limit, re-check whether a listed efficiency skill "
                "matches and call `read_skill(skill_name)` if you have not already loaded it.\n"
            )
            for skill in self.skills:
                parts.append(f"- **{skill.name}**: {skill.description}")
        else:
            parts.append(
                "No skills available yet. Skills are reusable strategies learned from solving tasks. "
                "When skills become available, they will be listed here with descriptions. "
                "Use `read_skill(skill_name)` to load a skill's full procedure.\n"
            )

        return "\n".join(parts)

    def _build_user_prompt(self, instance_id: str, problem_statement: str) -> str:
        # Build memory context from previous attempts
        memory_section = ""
        if self.memories:
            relevant = [m for m in self.memories if m.get("task_id") == instance_id]
            if relevant:
                max_memory_items = 5
                if "simple_memory" in self.efficiency_config.get("strategies", []):
                    max_memory_items = int(
                        self.efficiency_config
                        .get("parameters", {})
                        .get("simple_memory", {})
                        .get("max_memory_items", 3)
                        or 3
                    )
                memory_section = "\n\n## Previous Attempts on This Task\n"
                memory_section += (
                    "You have tried this task before. Use this compact memory and do not replay the full old trace:\n\n"
                )
                for mem in relevant[-max_memory_items:]:
                    memory_section += f"- {mem.get('approach_summary', '')}\n"

                    # Show specific failing tests so the agent knows what to target
                    f2p_failing = mem.get("fail_to_pass_failing", [])
                    p2p_broken = mem.get("pass_to_pass_broken", [])
                    if f2p_failing:
                        memory_section += (
                            f"  Target tests still failing: {', '.join(f2p_failing[:5])}\n"
                        )
                    if p2p_broken:
                        memory_section += (
                            f"  Tests you broke (regression): {', '.join(p2p_broken[:5])}\n"
                        )
                    # Show patch snippet if available (full feedback level)
                    patch_snippet = mem.get("patch_snippet", "")
                    if patch_snippet:
                        memory_section += f"  Your previous patch:\n```\n{patch_snippet[:500]}\n```\n"

                memory_section += (
                    "\nDo NOT repeat the same approach. "
                    "If the same tests keep failing, try a fundamentally different fix. "
                    "If you caused regressions, make a more targeted change.\n"
                )

        return f"""\
    ## Task
    Resolve the following GitHub issue by modifying the code in /testbed.

    ## Instance ID
    {instance_id}

    ## Problem Statement
    {problem_statement}
    {memory_section}
    ## Instructions
    1. Explore the repository structure at /testbed
    2. Understand the issue
    3. Find the relevant source files
    4. Implement a fix
    5. Test your fix if possible
    6. Once you are done, use your submit tool
    """
