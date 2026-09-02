"""Bash tool -- execute commands inside the SWE-bench Docker container."""
from __future__ import annotations

import re
import subprocess
from typing import Any

from strands import tool

_container_name: str | None = None
_efficiency_config: dict[str, Any] = {}
_tool_call_index = 0


def reset(**kwargs) -> None:
    global _container_name, _efficiency_config, _tool_call_index
    _container_name = kwargs.get("container_name")
    _efficiency_config = kwargs.get("efficiency_config") or {}
    _tool_call_index = 0


def _get_container() -> str:
    if _container_name is None:
        raise RuntimeError("Container name not set. Call reset(container_name=...) first.")
    return _container_name


def _strategy_enabled(name: str) -> bool:
    return name in set(_efficiency_config.get("strategies", []))


def _strategy_params(name: str) -> dict[str, Any]:
    params = _efficiency_config.get("parameters", {})
    val = params.get(name, {}) if isinstance(params, dict) else {}
    return val if isinstance(val, dict) else {}


def _looks_like_validation_command(command: str) -> bool:
    cmd = command.lower()
    return any(
        token in cmd
        for token in (
            "pytest",
            "python -m pytest",
            "unittest",
            "tox ",
            "go test",
            "cargo test",
            "npm test",
            "pnpm test",
            "yarn test",
            "gradle test",
            "mvn test",
            "nose",
        )
    )


def _looks_like_file_dump(command: str) -> bool:
    cmd = command.strip().lower()
    return cmd.startswith(("cat ", "nl ", "head ", "tail ", "sed -n", "grep ", "rg "))


def _keep_test_lines(output: str, *, max_lines: int) -> str:
    lines = output.splitlines()
    if len(lines) <= max_lines:
        return output

    keep: list[str] = []
    markers = (
        "failed",
        "error",
        "traceback",
        "assert",
        "short test summary",
        "====",
        "failures",
        "errors",
        "[exit code:",
    )
    for i, line in enumerate(lines):
        low = line.lower()
        if any(marker in low for marker in markers):
            start = max(0, i - 3)
            end = min(len(lines), i + 8)
            keep.extend(lines[start:end])

    if not keep:
        keep = lines[: max_lines // 2] + ["... [middle omitted by EvoEfficient] ..."] + lines[-max_lines // 2 :]

    return "\n".join(_dedupe_preserve_order(keep, max_lines=max_lines))


def _verbatim_compact(output: str, *, max_lines: int) -> str:
    lines = output.splitlines()
    if len(lines) <= max_lines:
        return output

    keep: list[str] = []
    important = (
        "traceback",
        "error",
        "failed",
        "assert",
        "exception",
        "warning",
        "diff --git",
        "@@",
        "+++",
        "---",
        "[exit code:",
    )
    low_value = (
        "collecting ",
        "collected ",
        "cached",
        "__pycache__",
        "site-packages",
        "warnings summary",
        "generated xml file",
    )

    for line in lines:
        stripped = line.strip()
        low = stripped.lower()
        if not stripped:
            continue
        if any(token in low for token in important):
            keep.append(line)
        elif any(token in low for token in low_value):
            continue
        elif re.match(r"^[\.\s]+$", stripped):
            continue
        elif len(stripped) <= 240:
            keep.append(line)
        if len(keep) >= max_lines:
            break

    if not keep:
        keep = lines[: max_lines // 2] + ["... [middle omitted by EvoEfficient verbatim_compaction] ..."] + lines[-max_lines // 2 :]

    return "\n".join(_dedupe_preserve_order(keep, max_lines=max_lines))


def _dedupe_preserve_order(lines: list[str], *, max_lines: int) -> list[str]:
    result: list[str] = []
    seen = set()
    for line in lines:
        key = line.strip()
        if key in seen:
            continue
        seen.add(key)
        result.append(line)
        if len(result) >= max_lines:
            break
    return result


def _truncate_middle(output: str, *, max_chars: int, label: str) -> str:
    if len(output) <= max_chars:
        return output
    head = max_chars // 2
    tail = max_chars - head
    return output[:head] + f"\n\n... [{label}] ...\n\n" + output[-tail:]


def _mask_stale_observation(command: str, output: str) -> str:
    if not _strategy_enabled("observation_masking"):
        return output
    params = _strategy_params("observation_masking")
    stale_after = int(params.get("stale_after_calls", 5) or 5)
    large_chars = int(params.get("large_output_chars", 4000) or 4000)
    placeholder_chars = int(params.get("placeholder_chars", 360) or 360)

    if _tool_call_index <= stale_after or len(output) <= large_chars:
        return output

    if _looks_like_validation_command(command):
        return output

    preview = output[:placeholder_chars].replace("\n", " ")
    return (
        "[EvoEfficient observation_masking: stale large output masked. "
        "Rerun a targeted command if exact content is needed.]\n"
        f"command: {command[:180]}\n"
        f"preview: {preview}"
    )


def _filter_output(command: str, output: str) -> str:
    if _strategy_enabled("observation_filter") or _strategy_enabled("adaptive_tool_trimming"):
        params = _strategy_params("adaptive_tool_trimming") or _strategy_params("observation_filter")
        test_max_lines = int(params.get("test_max_lines", params.get("pytest_max_lines", 140)) or 140)
        if _looks_like_validation_command(command):
            output = _keep_test_lines(output, max_lines=test_max_lines)

    if _strategy_enabled("verbatim_compaction"):
        params = _strategy_params("verbatim_compaction")
        max_lines = int(params.get("max_lines", 160) or 160)
        output = _verbatim_compact(output, max_lines=max_lines)

    output = _mask_stale_observation(command, output)

    max_chars = 15000
    label = "truncated"
    if _strategy_enabled("verbatim_compaction"):
        max_chars = int(_strategy_params("verbatim_compaction").get("max_chars", 6000) or 6000)
        label = "EvoEfficient verbatim_compaction truncated long output"
    elif _strategy_enabled("adaptive_tool_trimming"):
        max_chars = int(_strategy_params("adaptive_tool_trimming").get("max_chars", 8000) or 8000)
        label = "EvoEfficient adaptive_tool_trimming truncated long output"
    elif _strategy_enabled("observation_filter"):
        max_chars = int(_strategy_params("observation_filter").get("max_chars", 6000) or 6000)
        label = "EvoEfficient observation_filter truncated long output"

    return _truncate_middle(output, max_chars=max_chars, label=label)


@tool
def bash(command: str, workdir: str = "/testbed") -> str:
    """Execute a bash command inside the SWE-bench Docker container.

    Use this to explore the codebase, edit files, run tests, and generate patches.
    The repository is located at /testbed.

    Args:
        command: The bash command to execute.
        workdir: Working directory for the command. Defaults to /testbed.
    """
    global _tool_call_index
    _tool_call_index += 1
    try:
        result = subprocess.run(
            ["docker", "exec", "-w", workdir, _get_container(), "bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = (result.stdout or "") + (result.stderr or "")
        if not output.strip():
            output = "(no output)"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return _filter_output(command, output)
    except subprocess.TimeoutExpired:
        return "ERROR: Command timed out after 300 seconds."
    except Exception as e:
        return f"ERROR: {e}"
