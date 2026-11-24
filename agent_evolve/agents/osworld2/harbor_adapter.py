"""Harbor adapter for A-Evolve OSWorld2 experiments.

The emergent-efficiency runner owns the evolving workspace: prompts, skills,
memories, and proposal artifacts. Harbor owns the desktop VM, Docker/QEMU
execution, and scoring. This adapter is the bridge between the two: it reads the
prompt artifacts produced by the runner, appends them to Harbor's OSWorld prompt
template, then delegates actual GUI execution to Harbor's OSWorld agent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
from harbor.agents.factory import AgentFactory
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

DEFAULT_DELEGATE_AGENT_IMPORT_PATH = "osworld.custom_agent:OSWorldAgent"

# Copied from Harbor's OSWorld adapter prompt so this adapter can be imported
# before ``adapters/osworld/src`` is on PYTHONPATH. The runner still prepends
# Harbor's OSWorld adapter path before launching Harbor.
DEFAULT_OSWORLD_PROMPT_TEMPLATE = """\
You are an agent which follow my instruction and perform desktop computer tasks as instructed.
You have good knowledge of computer and good internet connection and assume your code will run on a computer for controlling the mouse and keyboard.
For each step, you will get an observation of the desktop by 1) a screenshot; and 2) accessibility tree, which is based on AT-SPI library.
And you will predict the action of the computer based on the screenshot and accessibility tree.

You are required to use `pyautogui` to perform the action grounded to the observation, but DONOT use the `pyautogui.locateCenterOnScreen` function to locate the element you want to operate with since we have no image of the element you want to operate with. DONOT USE `pyautogui.screenshot()` to make screenshot.
Return one line or multiple lines of python code to perform the action each time, be time efficient. When predicting multiple lines of code, make some small sleep like `time.sleep(0.5);` interval so that the machine could take; Each time you need to predict a complete code, no variables or function can be shared from history
You need to to specify the coordinates of by yourself based on your observation of current observation, but you should be careful to ensure that the coordinates are correct.
You ONLY need to return the code inside a code block, like this:
```python
# your code here
```
Specially, it is also allowed to return the following special code:
When you think you have to wait for some time, return ```WAIT```;
When you think the task can not be done, return ```FAIL```, don't easily say ```FAIL```, try your best to do the task;
When you think the task is done, return ```DONE```.

My computer's password is '{CLIENT_PASSWORD}', feel free to use it when you need sudo rights.
First give the current screenshot and previous things we did a short reflection, then RETURN ME THE CODE OR SPECIAL CODE I ASKED FOR. NEVER EVER RETURN ME ANYTHING ELSE.
"""


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


def _escape_format_braces(text: str) -> str:
    """Escape text appended to OSWorld prompt_template.format(...)."""

    return text.replace("{", "{{").replace("}", "}}")


def _task_instruction(task_data: dict[str, Any]) -> str:
    for key in ("instruction", "task", "question", "goal", "prompt", "description"):
        value = task_data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


class AEvolveOSWorld2HarborAgent(BaseAgent):
    """Inject A-Evolve workspace context, then delegate to Harbor OSWorldAgent.

    Constructor kwargs expected from the A-Evolve OSWorld2 runner:

    - ``workspace_dir``: path to the evolved workspace.
    - ``system_prompt_path``: generated prompt containing skills/fragments.
    - ``skills_dir``: evolved skill library directory.
    - ``task_json_path``: JSON task artifact written by the runner.
    - ``task_config_path``: original OSWorld2 task config path, when available.
    - ``task_id`` and ``domain``: task metadata for logs and prompt context.
    - ``delegate_agent_import_path``: Harbor OSWorld agent import path; defaults
      to ``osworld.custom_agent:OSWorldAgent``.
    - ``delegate_model_name``: optional model override for the delegate.
    - ``delegate_kwargs``: optional dict/JSON/YAML kwargs for the delegate.
    """

    capabilities = AgentCapabilities(atif=True)

    @staticmethod
    def name() -> str:
        return "a-evolve-osworld2-harbor-adapter"

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        workspace_dir: str | None = None,
        system_prompt_path: str | None = None,
        skills_dir: str | None = None,
        task_json_path: str | None = None,
        task_config_path: str | None = None,
        task_id: str | None = None,
        domain: str | None = None,
        delegate_agent_import_path: str | None = None,
        delegate_model_name: str | None = None,
        delegate_kwargs: dict[str, Any] | str | None = None,
        prompt_template: str | None = None,
        max_steps: int | None = None,
        max_system_prompt_chars: int = 32000,
        max_task_json_prompt_chars: int = 12000,
        include_task_json_instruction: bool = True,
        prefer_task_json_instruction: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            logs_dir=logs_dir,
            model_name=model_name,
            skills_dir=skills_dir,
            **kwargs,
        )
        self.workspace_dir = workspace_dir
        self.system_prompt_path = system_prompt_path
        self.skills_dir = skills_dir
        self.task_json_path = task_json_path
        self.task_config_path = task_config_path
        self.task_id = task_id
        self.domain = domain
        self.delegate_agent_import_path = (
            delegate_agent_import_path or DEFAULT_DELEGATE_AGENT_IMPORT_PATH
        )
        self.delegate_model_name = delegate_model_name or model_name
        self.max_system_prompt_chars = max_system_prompt_chars
        self.max_task_json_prompt_chars = max_task_json_prompt_chars
        self.include_task_json_instruction = include_task_json_instruction
        self.prefer_task_json_instruction = prefer_task_json_instruction
        self.task_data = _load_mapping(task_json_path)

        delegate_init_kwargs = dict(kwargs)
        delegate_init_kwargs.update(_coerce_mapping(delegate_kwargs))
        if max_steps is not None:
            delegate_init_kwargs["max_steps"] = max_steps
        delegate_init_kwargs["prompt_template"] = self._compose_prompt_template(
            prompt_template
            or str(delegate_init_kwargs.pop("prompt_template", ""))
            or DEFAULT_OSWORLD_PROMPT_TEMPLATE
        )

        self._delegate = AgentFactory.create_agent_from_import_path(
            self.delegate_agent_import_path,
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
        effective_instruction = self._effective_instruction(instruction)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "a_evolve_osworld2_prompt_template.md").write_text(
            self._delegate.prompt_template
            if hasattr(self._delegate, "prompt_template")
            else "",
            encoding="utf-8",
        )
        (self.logs_dir / "a_evolve_osworld2_instruction.md").write_text(
            effective_instruction,
            encoding="utf-8",
        )
        await self._delegate.run(effective_instruction, environment, context)
        metadata = (
            dict(context.metadata)
            if isinstance(context.metadata, dict)
            else {}
        )
        metadata["a_evolve_osworld2_adapter"] = {
            "workspace_dir": self.workspace_dir,
            "system_prompt_path": self.system_prompt_path,
            "skills_dir": self.skills_dir,
            "task_json_path": self.task_json_path,
            "task_config_path": self.task_config_path,
            "task_id": self.task_id,
            "domain": self.domain,
            "delegate_agent_import_path": self.delegate_agent_import_path,
            "delegate_model_name": self.delegate_model_name,
        }
        context.metadata = metadata
        (self.logs_dir / "a_evolve_osworld2_adapter_metadata.json").write_text(
            json.dumps(metadata["a_evolve_osworld2_adapter"], indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

    def _effective_instruction(self, instruction: str) -> str:
        task_prompt = ""
        if self.include_task_json_instruction:
            task_prompt = _task_instruction(self.task_data)
            if len(task_prompt) > self.max_task_json_prompt_chars:
                task_prompt = task_prompt[: self.max_task_json_prompt_chars] + "\n\n[truncated]"
        current = instruction.strip()
        if self.prefer_task_json_instruction and task_prompt:
            return task_prompt
        return current or task_prompt

    def _compose_prompt_template(self, base_template: str) -> str:
        system_prompt = _read_text(
            self.system_prompt_path,
            limit=self.max_system_prompt_chars,
        )
        metadata = {
            "task_id": self.task_id,
            "domain": self.domain,
            "task_config_path": self.task_config_path,
            "task_json_path": self.task_json_path,
            "workspace_dir": self.workspace_dir,
            "skills_dir": self.skills_dir,
        }
        metadata_lines = [
            f"- {key}: {value}"
            for key, value in metadata.items()
            if value not in (None, "")
        ]

        parts = [base_template.strip()]
        a_evolve_sections: list[str] = []
        if system_prompt:
            a_evolve_sections.extend(
                [
                    "# A-Evolve Workspace Instructions",
                    system_prompt,
                ]
            )
        if metadata_lines:
            a_evolve_sections.extend(
                [
                    "# A-Evolve Task Metadata",
                    "\n".join(metadata_lines),
                ]
            )
        if a_evolve_sections:
            a_evolve_sections.extend(
                [
                    "# A-Evolve Operating Rule",
                    (
                        "Use the A-Evolve instructions and skills only when they "
                        "match the current OSWorld task and visible desktop state. "
                        "Preserve correctness above cost. Do not stop early unless "
                        "the requested final state is visibly satisfied."
                    ),
                ]
            )
            parts.append(_escape_format_braces("\n\n".join(a_evolve_sections)))
        return "\n\n".join(part for part in parts if part).strip() + "\n"
