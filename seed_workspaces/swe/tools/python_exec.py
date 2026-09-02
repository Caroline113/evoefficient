"""Python execution tool — run Python code inside the SWE-bench Docker container.

Executes Python code via stdin (python3 -). Each execution is independent —
no state persists between calls. Use print() to see output.
"""

from __future__ import annotations

import subprocess
from typing import Any

from strands import tool

_container_name: str | None = None
_efficiency_config: dict[str, Any] = {}


def reset(**kwargs) -> None:
    global _container_name, _efficiency_config
    _container_name = kwargs.get("container_name")
    _efficiency_config = kwargs.get("efficiency_config") or {}


def _strategy_enabled(name: str) -> bool:
    return name in set(_efficiency_config.get("strategies", []))


def _strategy_params(name: str) -> dict[str, Any]:
    params = _efficiency_config.get("parameters", {})
    val = params.get(name, {}) if isinstance(params, dict) else {}
    return val if isinstance(val, dict) else {}


def _trim_output(output: str) -> str:
    max_chars = 15000
    label = "truncated"
    if _strategy_enabled("adaptive_tool_trimming"):
        max_chars = int(_strategy_params("adaptive_tool_trimming").get("max_chars", 8000) or 8000)
        label = "EvoEfficient adaptive_tool_trimming truncated python output"
    elif _strategy_enabled("verbatim_compaction"):
        max_chars = int(_strategy_params("verbatim_compaction").get("max_chars", 6000) or 6000)
        label = "EvoEfficient verbatim_compaction truncated python output"
    elif _strategy_enabled("observation_filter"):
        max_chars = int(_strategy_params("observation_filter").get("max_chars", 6000) or 6000)
        label = "EvoEfficient observation_filter truncated python output"
    if len(output) <= max_chars:
        return output
    head = max_chars // 2
    tail = max_chars - head
    return output[:head] + f"\n\n... [{label}] ...\n\n" + output[-tail:]


@tool
def python_exec(code: str) -> str:
    """Execute Python code inside the Docker container.

    Runs the code with python3. Each execution is independent — no state persists
    between calls. Use print() statements to see output.

    Args:
        code: Python code to execute.
    """
    if not _container_name:
        return "Error: container not initialized"
    try:
        result = subprocess.run(
            ["docker", "exec", "-w", "/testbed", _container_name, "python3", "-"],
            input=code,
            capture_output=True,
            text=True,
            timeout=180,
        )
        output = ""
        if result.stderr:
            output += result.stderr
        if result.stdout:
            if output:
                output += "\n"
            output += result.stdout
        if not output.strip():
            output = "(no output — use print() to see results)"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return _trim_output(output)
    except subprocess.TimeoutExpired:
        return "ERROR: Python execution timed out after 180 seconds."
    except Exception as e:
        return f"ERROR: {e}"
