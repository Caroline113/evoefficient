"""Harbor adapter for A-Evolve Terminal-Bench experiments.

The emergent-efficiency runner owns the evolving workspace: prompts, skills,
memories, and proposal artifacts. Harbor owns the benchmark environment and
verification. This adapter is the bridge between the two: it reads the prompt
artifacts produced by the runner, prepends them to the task instruction, then
delegates actual terminal execution to one of Harbor's built-in agents.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from harbor.agents.base import BaseAgent
from harbor.agents.factory import AgentFactory
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.agent.name import AgentName


def _read_text(path: str | Path | None, *, limit: int | None = None) -> str:
    if not path:
        return ""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return ""
    if limit is not None and len(text) > limit:
        return text[:limit] + "\n\n[truncated]"
    return text


def _load_mapping(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    task_path = Path(path)
    try:
        if task_path.suffix.lower() in {".yaml", ".yml"}:
            loaded = yaml.safe_load(task_path.read_text(encoding="utf-8")) or {}
        else:
            loaded = json.loads(task_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError, UnicodeDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            try:
                loaded = yaml.safe_load(value)
            except yaml.YAMLError:
                loaded = {}
        return loaded if isinstance(loaded, dict) else {}
    return {}


class AEvolveTerminalHarborAgent(BaseAgent):
    """Inject A-Evolve workspace context, then delegate to a Harbor agent.

    Constructor kwargs expected from the A-Evolve runner:

    - ``workspace_dir``: path to the evolved workspace.
    - ``system_prompt_path``: generated prompt containing skills/fragments.
    - ``task_json_path``: JSON task artifact written by the runner.
    - ``task_name``: Terminal-Bench task name.
    - ``delegate_agent_name``: Harbor built-in agent to run, default ``codex``.
    - ``delegate_agent_import_path``: optional custom Harbor agent import path.
    - ``delegate_model_name``: optional model override for the delegate.
    - ``delegate_kwargs``: optional dict/JSON/YAML of kwargs for the delegate.
    """

    @staticmethod
    def name() -> str:
        return "a-evolve-terminal-harbor-adapter"

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        workspace_dir: str | None = None,
        system_prompt_path: str | None = None,
        task_json_path: str | None = None,
        task_name: str | None = None,
        delegate_agent_name: str = AgentName.CODEX.value,
        delegate_agent_import_path: str | None = None,
        delegate_model_name: str | None = None,
        delegate_kwargs: dict[str, Any] | str | None = None,
        max_steps: int | None = None,
        aws_region_name: str | None = None,
        max_system_prompt_chars: int = 24000,
        max_task_json_prompt_chars: int = 12000,
        include_task_json_prompt: bool = True,
        **kwargs: Any,
    ):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self.workspace_dir = workspace_dir
        self.system_prompt_path = system_prompt_path
        self.task_json_path = task_json_path
        self.task_name = task_name
        self.delegate_agent_name = delegate_agent_name
        self.delegate_agent_import_path = delegate_agent_import_path
        self.delegate_model_name = delegate_model_name or model_name
        self.max_steps = max_steps
        self.aws_region_name = aws_region_name
        self.max_system_prompt_chars = max_system_prompt_chars
        self.max_task_json_prompt_chars = max_task_json_prompt_chars
        self.include_task_json_prompt = include_task_json_prompt

        delegate_extra = _coerce_mapping(delegate_kwargs)
        delegate_init_kwargs = dict(kwargs)
        delegate_init_kwargs.update(delegate_extra)
        if delegate_agent_import_path:
            self._delegate = AgentFactory.create_agent_from_import_path(
                delegate_agent_import_path,
                logs_dir=logs_dir,
                model_name=self.delegate_model_name,
                **delegate_init_kwargs,
            )
        else:
            self._delegate = AgentFactory.create_agent_from_name(
                AgentName(delegate_agent_name),
                logs_dir=logs_dir,
                model_name=self.delegate_model_name,
                **delegate_init_kwargs,
            )

    def version(self) -> str:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        await self._delegate.setup(environment)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        prompt = self._compose_instruction(instruction)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "a_evolve_terminal_adapter_prompt.md").write_text(
            prompt,
            encoding="utf-8",
        )
        await self._delegate.run(prompt, environment, context)
        metadata = dict(context.metadata or {})
        metadata["a_evolve_terminal_adapter"] = {
            "workspace_dir": self.workspace_dir,
            "system_prompt_path": self.system_prompt_path,
            "task_json_path": self.task_json_path,
            "task_name": self.task_name,
            "delegate_agent": (self.delegate_agent_import_path or self.delegate_agent_name),
            "delegate_model_name": self.delegate_model_name,
            "max_steps": self.max_steps,
            "aws_region_name": self.aws_region_name,
        }
        context.metadata = metadata

    def _compose_instruction(self, instruction: str) -> str:
        system_prompt = _read_text(
            self.system_prompt_path,
            limit=self.max_system_prompt_chars,
        )
        task_data = _load_mapping(self.task_json_path)
        task_prompt = ""
        if self.include_task_json_prompt:
            for key in ("prompt", "instruction", "description", "task"):
                value = task_data.get(key)
                if isinstance(value, str) and value.strip():
                    task_prompt = value.strip()
                    break
            if task_prompt and len(task_prompt) > self.max_task_json_prompt_chars:
                task_prompt = task_prompt[: self.max_task_json_prompt_chars] + "\n\n[truncated]"

        parts = [
            "# Terminal-Bench Task",
            instruction.strip() or task_prompt,
        ]
        if system_prompt:
            parts.extend(
                [
                    "",
                    "# A-Evolve Workspace Instructions",
                    system_prompt,
                ]
            )
        if task_prompt and task_prompt.strip() != instruction.strip():
            parts.extend(
                [
                    "",
                    "# Task Prompt Artifact",
                    task_prompt,
                ]
            )
        if self.workspace_dir:
            parts.extend(
                [
                    "",
                    "# Workspace Context",
                    f"A-Evolve workspace directory on the host: {self.workspace_dir}",
                    (
                        "Use the skills and instructions above when they apply. "
                        "Preserve task correctness before optimizing cost."
                    ),
                ]
            )
        return "\n".join(part for part in parts if part is not None).strip() + "\n"
