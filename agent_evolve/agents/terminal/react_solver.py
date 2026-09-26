"""Standalone ReAct solver for Terminal-Bench 2.0.

Implements the same ReAct loop as inspect_ai's react() solver.  It supports
AWS Bedrock Converse and OpenAI-compatible providers such as OpenRouter.

The loop:
  1. Send messages + tool definitions to the LLM
  2. If the LLM returns tool_calls → execute them
     - If submit() was called → stop
  3. If the LLM returns text only (no tool_calls) → send continue prompt
  4. Repeat until submit, timeout, or max turns

This matches inspect_ai's behavior exactly:
  - on_continue: nudge the model when it stops without calling tools
  - bash --login -c: login shell for proper PATH
  - python3 - via stdin: reliable multi-line code execution
  - stderr before stdout: errors shown first
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from typing import Any

logger = logging.getLogger(__name__)

# ── Prompts (matching inspect_ai/terminal_bench_2) ──────────────────

SYSTEM_PROMPT = """\
You are an AI assistant tasked with solving command-line tasks in a Linux environment.
You will be given a task description and the output from previously executed commands.
You have several tools available to help with finding the solution. You are running as root inside a Docker container. Do not use sudo - it is not installed and not needed since you already have root privileges.

Your goal is to solve the task by providing batches of shell commands. If you need to perform multiple actions, you can always send more messages with additional tool calls.

Before taking action, you should:
  1. ANALYZE the current state based on previous tool outputs
  2. PLAN your next steps - what commands will you run and why

Format your reasoning as follows before calling tools:

  **Analysis:** [Analyze the current state. What do you see? What has been accomplished? What still needs to be done?]

  **Plan:** [Describe your plan for the next steps. What commands will you run and why? Be specific about what you expect each command to accomplish.]

Then call the appropriate tools to execute your plan.

Important: Each bash() call is independent - no state is preserved between calls. If you need to run commands in sequence, chain them with &&

When you have completed the task, call the `submit()` tool with "DONE" as argument to report that the task is complete."""

CONTINUE_PROMPT = (
    "Please proceed to the next step using your best judgement.\n"
    "If you believe you have completed the task, please call the "
    '`submit()` tool with "DONE" as argument to report that the task is complete.'
)

SKILL_REFLECTION_PROMPT = """\
Now that you've finished this task, reflect on what you learned. \
If there is domain-specific knowledge that would help a future agent \
solve SIMILAR tasks (not this exact task), propose a reusable intervention.

A useful skill contains:
- Specific tools, libraries, or commands needed for this category of task
- Common pitfalls and how to avoid them
- Verification steps to confirm the task is solved

Do NOT include generic advice (timeout handling, package installation, debugging). \
Only include knowledge specific to this task's domain.

If this task was straightforward and no special knowledge was needed, \
respond with exactly:
CONFIDENCE: LOW
ACTION: NONE

Otherwise, respond with EXACTLY this format (no other text before or after):

CONFIDENCE: HIGH/MEDIUM/LOW
ACTION: NEW
TYPE: skill
NAME: <generic-snake-case-name>
DESCRIPTION: TRIGGER when ...; DO NOT TRIGGER when ...
CONTENT:
<domain-specific checklist, commands, pitfalls, and verification steps; under 500 words>
"""


# ── Tool definitions (Bedrock Converse format) ──────────────────────

TOOL_SPECS = [
    {
        "toolSpec": {
            "name": "bash",
            "description": "Use this function to execute bash commands.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "cmd": {
                            "type": "string",
                            "description": "The bash command to execute.",
                        }
                    },
                    "required": ["cmd"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "python",
            "description": (
                "Use the python function to execute Python code. "
                "Each execution is independent - no state is preserved between runs. "
                "You must explicitly use print() statements to see any output."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The python code to execute.",
                        }
                    },
                    "required": ["code"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "submit",
            "description": "Submit an answer for evaluation.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "Submitted answer.",
                        }
                    },
                    "required": ["answer"],
                }
            },
        }
    },
]

READ_SKILL_SPEC = {
    "toolSpec": {
        "name": "read_skill",
        "description": (
            "Read the full content of a skill by name. "
            "Use this to load detailed guidance for a skill listed in your system prompt."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The skill name to read.",
                    }
                },
                "required": ["name"],
            }
        },
    }
}


# ── Tool executors ──────────────────────────────────────────────────

def _exec_bash(container_name: str, cmd: str, log: logging.Logger) -> str:
    """Execute bash command in the container. Matches inspect_ai's bash tool."""
    cmd_preview = cmd[:200] + ("..." if len(cmd) > 200 else "")
    log.info("[bash] $ %s", cmd_preview)
    t0 = time.time()
    try:
        docker_cmd = ["docker", "exec", container_name, "bash", "--login", "-c", cmd]
        result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=60)
        output = ""
        if result.stderr:
            output = f"{result.stderr}\n"
        output = f"{output}{result.stdout}"
        if not output.strip():
            output = "(no output)"
        if len(output) > 15000:
            output = output[:7000] + "\n\n... [truncated] ...\n\n" + output[-7000:]
        elapsed = time.time() - t0
        log.info("[bash] done (%.1fs, %d chars)", elapsed, len(output))
        return output
    except subprocess.TimeoutExpired:
        log.warning("[bash] TIMEOUT after 60s")
        return "ERROR: Command timed out after 60 seconds."
    except Exception as e:
        log.error("[bash] ERROR: %s", e)
        return f"ERROR: {e}"


def _exec_python(container_name: str, code: str, log: logging.Logger) -> str:
    """Execute Python code in the container. Matches inspect_ai's python tool."""
    code_preview = code[:200] + ("..." if len(code) > 200 else "")
    log.info("[python] >>> %s", code_preview.replace("\n", "\\n"))
    t0 = time.time()
    try:
        docker_cmd = [
            "docker", "exec", "-i", container_name,
            "bash", "--login", "-c", "python3 -",
        ]
        result = subprocess.run(
            docker_cmd, capture_output=True, text=True, timeout=60, input=code,
        )
        output = ""
        if result.stderr:
            output = f"{result.stderr}\n"
        output = f"{output}{result.stdout}"
        if not output.strip():
            output = "(no output)"
        if len(output) > 15000:
            output = output[:7000] + "\n\n... [truncated] ...\n\n" + output[-7000:]
        elapsed = time.time() - t0
        log.info("[python] done (%.1fs, %d chars)", elapsed, len(output))
        return output
    except subprocess.TimeoutExpired:
        log.warning("[python] TIMEOUT after 60s")
        return "ERROR: Command timed out after 60 seconds."
    except Exception as e:
        log.error("[python] ERROR: %s", e)
        return f"ERROR: {e}"


def _exec_submit(answer: str, log: logging.Logger) -> str:
    """Handle submit tool call."""
    log.info("[submit] %s", answer)
    return answer


# ── The ReAct loop ──────────────────────────────────────────────────

class ReactSolverResult:
    """Result from the ReAct solver."""

    def __init__(self):
        self.messages: list[dict] = []
        self.submitted: bool = False
        self.submit_answer: str = ""
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cache_read_input_tokens: int = 0
        self.total_cache_write_input_tokens: int = 0
        self.tool_call_count: int = 0
        self.timed_out: bool = False
        self.skill_draft: str | None = None  # Solver-proposed skill (v15)


def react_solve(
    task_prompt: str,
    container_name: str,
    model_id: str = "us.anthropic.claude-opus-4-5-20251101-v1:0",
    region: str = "us-west-2",
    max_tokens: int = 16384,
    timeout_sec: int = 900,
    max_turns: int = 500,
    log: logging.Logger | None = None,
    system_prompt: str | None = None,
    propose_skill: bool = False,
    skills: dict[str, str] | None = None,
    tool_specs: list[dict] | None = None,
    tool_executors: dict[str, callable] | None = None,
) -> ReactSolverResult:
    """Run the ReAct loop to solve a task.

    This implements the same logic as inspect_ai's react() solver:
      1. Send messages + tools to LLM
      2. If tool_calls → execute them; if submit → stop
      3. If no tool_calls → send continue prompt
      4. Repeat until submit, timeout, or max_turns

    Args:
        task_prompt: The task description (user message).
        container_name: Docker container name for tool execution.
        model_id: Bedrock model ID, or OpenAI/OpenRouter-compatible model ID.
        region: AWS region for Bedrock.
        max_tokens: Max tokens per LLM call.
        timeout_sec: Wall-clock timeout for the entire solve.
        max_turns: Safety limit on LLM calls.
        log: Logger instance.
        system_prompt: Override the default system prompt (e.g. with
            evolved workspace prompt including skills/memories).

    Returns:
        ReactSolverResult with messages, submission status, and usage.
    """
    if log is None:
        log = logger

    from ...llm.factory import is_openai_compatible_model

    if is_openai_compatible_model(model_id):
        return _react_solve_openai(
            task_prompt=task_prompt,
            container_name=container_name,
            model_id=model_id,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
            max_turns=max_turns,
            log=log,
            system_prompt=system_prompt,
            propose_skill=propose_skill,
            skills=skills,
            tool_specs=tool_specs,
            tool_executors=tool_executors,
        )

    import boto3

    from botocore.config import Config as BotoConfig
    client = boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=BotoConfig(read_timeout=300, retries={"max_attempts": 0}),
    )
    result = ReactSolverResult()

    # Build initial messages
    system_blocks = [{"text": system_prompt or SYSTEM_PROMPT}]
    messages = [{"role": "user", "content": [{"text": task_prompt}]}]
    result.messages = messages

    base_specs = tool_specs if tool_specs is not None else TOOL_SPECS
    all_specs = base_specs + ([READ_SKILL_SPEC] if skills else [])
    tool_config = {"tools": all_specs}
    t0 = time.time()
    consecutive_errors = 0
    retry_lost_time = 0  # Time lost to failed API calls + retry waits (not counted toward timeout)

    for turn in range(max_turns):
        # Check timeout before LLM call (subtract time lost to API retries)
        effective_elapsed = time.time() - t0 - retry_lost_time
        if effective_elapsed >= timeout_sec:
            log.warning("Timeout reached before LLM call (%.0fs effective >= %ds)", effective_elapsed, timeout_sec)
            result.timed_out = True
            break

        # Call LLM
        log.debug("[turn %d] Calling LLM (%.0fs elapsed, %.0fs effective)...",
                  turn + 1, time.time() - t0, effective_elapsed)
        call_start = time.time()
        try:
            response = client.converse(
                modelId=model_id,
                messages=messages,
                system=system_blocks,
                inferenceConfig={"maxTokens": max_tokens},
                toolConfig=tool_config,
            )
            consecutive_errors = 0
        except Exception as e:
            err_str = str(e)
            # Retry on transient errors (API throttling, timeouts, server errors)
            if any(kw in err_str for kw in [
                "ThrottlingException", "internalServerException",
                "ServiceUnavailableException", "ModelTimeoutException",
                "Read timeout", "ConnectTimeoutError", "EndpointConnectionError",
                "content filtering policy",
            ]):
                consecutive_errors += 1
                wait = min(120, 15 * consecutive_errors)
                # Don't count failed call time + retry wait toward task timeout
                retry_lost_time += time.time() - call_start + wait
                # Give up if: too many consecutive errors OR wall clock exceeds 2x timeout
                wall_elapsed = time.time() - t0 + wait
                if consecutive_errors >= 5 or wall_elapsed >= timeout_sec * 2.5:
                    log.error("Giving up after %d retries (%.0fs wall, %.0fs paused): %s",
                              consecutive_errors, wall_elapsed, retry_lost_time, err_str[:200])
                    break
                log.warning("Transient API error (%d/5): %s. Retrying in %ds... (%.0fs paused total)",
                            consecutive_errors, err_str[:150], wait, retry_lost_time)
                time.sleep(wait)
                continue
            else:
                log.error("LLM error: %s", err_str[:300])
                break

        # Track usage before the timeout check: a completed provider call is
        # billable even when it returns just after the task deadline.
        usage = response.get("usage", {})
        result.total_input_tokens += usage.get("inputTokens", 0)
        result.total_output_tokens += usage.get("outputTokens", 0)
        result.total_cache_read_input_tokens += usage.get("cacheReadInputTokens", 0)
        result.total_cache_write_input_tokens += usage.get("cacheWriteInputTokens", 0)

        # Check timeout after recording the completed call.
        effective_elapsed = time.time() - t0 - retry_lost_time
        if effective_elapsed >= timeout_sec:
            log.warning("Timeout reached after LLM call (%.0fs effective >= %ds)", effective_elapsed, timeout_sec)
            result.timed_out = True
            break

        # Parse response
        output_msg = response.get("output", {}).get("message", {})
        content_blocks = output_msg.get("content", [])
        stop_reason = response.get("stopReason", "end_turn")

        # Handle context window overflow (matches inspect_ai's model_length check)
        if stop_reason == "max_tokens":
            log.warning("[turn %d] Model hit max_tokens — response may be truncated", turn + 1)
            # Still append the partial response and continue
            # The model may have partial tool calls that won't parse

        # Append assistant message
        messages.append({"role": "assistant", "content": content_blocks})

        # Separate text and tool_use blocks
        text_blocks = [b for b in content_blocks if "text" in b]
        tool_use_blocks = [b for b in content_blocks if "toolUse" in b]

        if text_blocks:
            text_preview = text_blocks[0]["text"][:200]
            log.debug("[turn %d] Assistant text: %s", turn + 1, text_preview)

        # ── Handle tool calls ────────────────────────────────────────
        if tool_use_blocks:
            tool_results = []
            submitted = False

            for tu_block in tool_use_blocks:
                # Check timeout before each tool execution (subtract retry lost time)
                if time.time() - t0 - retry_lost_time >= timeout_sec:
                    log.warning("Timeout reached during tool execution")
                    result.timed_out = True
                    # Return error for remaining tool calls
                    tool_results.append({
                        "toolResult": {
                            "toolUseId": tu_block["toolUse"]["toolUseId"],
                            "content": [{"text": "ERROR: Agent timeout reached."}],
                            "status": "error",
                        }
                    })
                    continue

                tu = tu_block["toolUse"]
                tool_name = tu["name"]
                tool_input = tu.get("input", {})
                tool_use_id = tu["toolUseId"]
                result.tool_call_count += 1

                # Execute the tool
                if tool_name == "read_skill":
                    # read_skill is handled inline (dynamic, tied to skill system)
                    skill_name = tool_input.get("name", "")
                    if skills and skill_name in skills:
                        tool_output = skills[skill_name]
                        log.info("[read_skill] %s (%d chars)", skill_name, len(tool_output))
                    else:
                        available = ", ".join(skills.keys()) if skills else "none"
                        tool_output = f"Skill '{skill_name}' not found. Available: {available}"
                        log.warning("[read_skill] not found: %s", skill_name)
                elif tool_executors and tool_name in tool_executors:
                    # Use workspace-loaded executor
                    tool_output = tool_executors[tool_name](container_name, tool_input, log)
                elif tool_name == "bash":
                    tool_output = _exec_bash(
                        container_name, tool_input.get("cmd", ""), log
                    )
                elif tool_name == "python":
                    tool_output = _exec_python(
                        container_name, tool_input.get("code", ""), log
                    )
                elif tool_name == "submit":
                    tool_output = _exec_submit(
                        tool_input.get("answer", ""), log
                    )
                else:
                    tool_output = f"ERROR: Unknown tool '{tool_name}'"
                    log.warning("Unknown tool: %s", tool_name)

                # Track submit state regardless of which executor handled it
                if tool_name == "submit":
                    submitted = True
                    result.submitted = True
                    result.submit_answer = tool_input.get("answer", "")

                tool_results.append({
                    "toolResult": {
                        "toolUseId": tool_use_id,
                        "content": [{"text": str(tool_output)}],
                    }
                })

            # Append tool results as user message
            messages.append({"role": "user", "content": tool_results})

            # If submitted, stop the loop
            if submitted:
                log.info("Agent submitted after %d turns, %.0fs",
                         turn + 1, time.time() - t0)
                break

            # If timed out during tool execution, stop
            if result.timed_out:
                break

        # ── No tool calls: send continue prompt ──────────────────────
        else:
            # Check timeout before sending continue (subtract retry lost time)
            if time.time() - t0 - retry_lost_time >= timeout_sec:
                log.warning("Timeout reached, not sending continue prompt")
                result.timed_out = True
                break

            log.info("[turn %d] No tool calls — sending continue prompt", turn + 1)
            messages.append({
                "role": "user",
                "content": [{"text": CONTINUE_PROMPT}],
            })

    # Final stats
    elapsed = time.time() - t0
    log.info(
        "ReAct loop done: %d turns, %d tool calls, %.0fs, "
        "tokens=%d in + %d out + %d cache-read + %d cache-write, submitted=%s",
        turn + 1, result.tool_call_count, elapsed,
        result.total_input_tokens, result.total_output_tokens,
        result.total_cache_read_input_tokens, result.total_cache_write_input_tokens,
        result.submitted,
    )

    result.messages = messages

    # ── Skill reflection (v15) ──────────────────────────────────────
    if propose_skill:
        result.skill_draft, reflection_usage = _reflect_for_skill(
            client, model_id, system_blocks, messages, max_tokens, log,
        )
        result.total_input_tokens += reflection_usage[0]
        result.total_output_tokens += reflection_usage[1]
        result.total_cache_read_input_tokens += reflection_usage[2]
        result.total_cache_write_input_tokens += reflection_usage[3]

    return result


def _react_solve_openai(
    task_prompt: str,
    container_name: str,
    model_id: str,
    max_tokens: int,
    timeout_sec: int,
    max_turns: int,
    log: logging.Logger,
    system_prompt: str | None = None,
    propose_skill: bool = False,
    skills: dict[str, str] | None = None,
    tool_specs: list[dict] | None = None,
    tool_executors: dict[str, callable] | None = None,
) -> ReactSolverResult:
    """OpenAI/OpenRouter-compatible ReAct loop for Terminal-Bench."""
    from ...llm.factory import create_openai_client_for_model

    client, selected_model, provider = create_openai_client_for_model(model_id)
    result = ReactSolverResult()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
        {"role": "user", "content": task_prompt},
    ]
    result.messages = messages

    base_specs = tool_specs if tool_specs is not None else TOOL_SPECS
    all_specs = base_specs + ([READ_SKILL_SPEC] if skills else [])
    openai_tools = _to_openai_tools(all_specs)

    t0 = time.time()
    consecutive_errors = 0
    retry_lost_time = 0.0

    for turn in range(max_turns):
        effective_elapsed = time.time() - t0 - retry_lost_time
        if effective_elapsed >= timeout_sec:
            log.warning("Timeout reached before LLM call (%.0fs effective >= %ds)", effective_elapsed, timeout_sec)
            result.timed_out = True
            break

        log.debug("[turn %d] Calling %s LLM (%.0fs elapsed, %.0fs effective)...",
                  turn + 1, provider, time.time() - t0, effective_elapsed)
        call_start = time.time()
        try:
            params: dict[str, Any] = {
                "model": selected_model,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            if openai_tools:
                params["tools"] = openai_tools
                params["tool_choice"] = "auto"
            response = client.chat.completions.create(**params)
            consecutive_errors = 0
        except Exception as e:
            err_str = str(e)
            if any(kw in err_str.lower() for kw in [
                "rate limit", "timeout", "temporarily unavailable", "503", "502",
                "too many requests", "throttl", "connection",
            ]):
                consecutive_errors += 1
                wait = min(120, 15 * consecutive_errors)
                retry_lost_time += time.time() - call_start + wait
                wall_elapsed = time.time() - t0 + wait
                if consecutive_errors >= 5 or wall_elapsed >= timeout_sec * 2.5:
                    log.error("Giving up after %d retries (%.0fs wall, %.0fs paused): %s",
                              consecutive_errors, wall_elapsed, retry_lost_time, err_str[:200])
                    break
                log.warning("Transient API error (%d/5): %s. Retrying in %ds...",
                            consecutive_errors, err_str[:150], wait)
                time.sleep(wait)
                continue
            log.error("LLM error: %s", err_str[:300])
            break

        usage = getattr(response, "usage", None)
        input_tokens, output_tokens, cache_read, cache_write = _openai_usage(usage)
        result.total_input_tokens += input_tokens
        result.total_output_tokens += output_tokens
        result.total_cache_read_input_tokens += cache_read
        result.total_cache_write_input_tokens += cache_write

        effective_elapsed = time.time() - t0 - retry_lost_time
        if effective_elapsed >= timeout_sec:
            log.warning("Timeout reached after LLM call (%.0fs effective >= %ds)", effective_elapsed, timeout_sec)
            result.timed_out = True
            break

        message = response.choices[0].message
        content = getattr(message, "content", None) or ""
        tool_calls = list(getattr(message, "tool_calls", None) or [])
        assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_message["tool_calls"] = [
                _openai_tool_call_to_dict(call) for call in tool_calls
            ]
        messages.append(assistant_message)

        if content:
            log.debug("[turn %d] Assistant text: %s", turn + 1, content[:200])

        if tool_calls:
            submitted = False
            for call in tool_calls:
                if time.time() - t0 - retry_lost_time >= timeout_sec:
                    log.warning("Timeout reached during tool execution")
                    result.timed_out = True
                    break

                tool_name = call.function.name
                raw_args = call.function.arguments or "{}"
                try:
                    tool_input = json.loads(raw_args)
                    if not isinstance(tool_input, dict):
                        tool_input = {"value": tool_input}
                except json.JSONDecodeError:
                    tool_input = {}
                result.tool_call_count += 1

                if tool_name == "read_skill":
                    skill_name = tool_input.get("name", "")
                    if skills and skill_name in skills:
                        tool_output = skills[skill_name]
                        log.info("[read_skill] %s (%d chars)", skill_name, len(tool_output))
                    else:
                        available = ", ".join(skills.keys()) if skills else "none"
                        tool_output = f"Skill '{skill_name}' not found. Available: {available}"
                        log.warning("[read_skill] not found: %s", skill_name)
                elif tool_executors and tool_name in tool_executors:
                    tool_output = tool_executors[tool_name](container_name, tool_input, log)
                elif tool_name == "bash":
                    tool_output = _exec_bash(container_name, tool_input.get("cmd", ""), log)
                elif tool_name == "python":
                    tool_output = _exec_python(container_name, tool_input.get("code", ""), log)
                elif tool_name == "submit":
                    tool_output = _exec_submit(tool_input.get("answer", ""), log)
                else:
                    tool_output = f"ERROR: Unknown tool '{tool_name}'"
                    log.warning("Unknown tool: %s", tool_name)

                if tool_name == "submit":
                    submitted = True
                    result.submitted = True
                    result.submit_answer = tool_input.get("answer", "")

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": tool_name,
                    "content": str(tool_output),
                })

            if submitted:
                log.info("Agent submitted after %d turns, %.0fs", turn + 1, time.time() - t0)
                break
            if result.timed_out:
                break
        else:
            if time.time() - t0 - retry_lost_time >= timeout_sec:
                log.warning("Timeout reached, not sending continue prompt")
                result.timed_out = True
                break
            log.info("[turn %d] No tool calls - sending continue prompt", turn + 1)
            messages.append({"role": "user", "content": CONTINUE_PROMPT})

    elapsed = time.time() - t0
    log.info(
        "ReAct loop done: %d turns, %d tool calls, %.0fs, "
        "tokens=%d in + %d out + %d cache-read + %d cache-write, submitted=%s",
        turn + 1, result.tool_call_count, elapsed,
        result.total_input_tokens, result.total_output_tokens,
        result.total_cache_read_input_tokens, result.total_cache_write_input_tokens,
        result.submitted,
    )

    result.messages = messages
    if propose_skill:
        result.skill_draft, reflection_usage = _reflect_for_skill_openai(
            client, selected_model, messages, openai_tools, max_tokens, log
        )
        result.total_input_tokens += reflection_usage[0]
        result.total_output_tokens += reflection_usage[1]
        result.total_cache_read_input_tokens += reflection_usage[2]
        result.total_cache_write_input_tokens += reflection_usage[3]
    return result


# ── Skill reflection (v15) ──────────────────────────────────────────

def _reflect_for_skill(
    client: Any,
    model_id: str,
    system_blocks: list[dict],
    messages: list[dict],
    max_tokens: int,
    log: logging.Logger,
) -> tuple[str | None, tuple[int, int, int, int]]:
    """Ask the solver to propose a skill draft based on what it learned.

    Uses the existing conversation context (solver has full task history).
    Returns ``(draft, usage_tuple)`` so the reflection call is included in
    experiment token and cost accounting even when no proposal is produced.
    """
    reflect_messages = messages + [
        {"role": "user", "content": [{"text": SKILL_REFLECTION_PROMPT}]}
    ]
    try:
        log.info("[reflection] Asking solver to propose a skill draft...")
        # Must include toolConfig because messages contain toolUse/toolResult blocks
        response = client.converse(
            modelId=model_id,
            messages=reflect_messages,
            system=system_blocks,
            toolConfig={"tools": TOOL_SPECS},
            inferenceConfig={"maxTokens": max_tokens},
        )
        # Extract text from response
        content_blocks = response.get("output", {}).get("message", {}).get("content", [])
        text = ""
        for block in content_blocks:
            if "text" in block:
                text += block["text"]

        raw_usage = response.get("usage", {})
        reflection_usage = (
            int(raw_usage.get("inputTokens", 0) or 0),
            int(raw_usage.get("outputTokens", 0) or 0),
            int(raw_usage.get("cacheReadInputTokens", 0) or 0),
            int(raw_usage.get("cacheWriteInputTokens", 0) or 0),
        )
        if not text or "ACTION: NONE" in text.upper():
            log.info("[reflection] Solver declined: no skill needed")
            return None, reflection_usage

        if "CONFIDENCE:" in text.upper() and "ACTION:" in text.upper() and "CONTENT:" in text.upper():
            start = text.upper().index("CONFIDENCE:")
            draft = text[start:].strip().strip("`").strip()
            log.info("[reflection] Got skill draft (%d chars)", len(draft))
            return draft, reflection_usage

        log.info("[reflection] Response didn't contain a valid skill draft")
        return None, reflection_usage

    except Exception as e:
        log.warning("[reflection] Skill reflection failed: %s", str(e)[:200])
        return None, (0, 0, 0, 0)


def _reflect_for_skill_openai(
    client: Any,
    model_id: str,
    messages: list[dict],
    tools: list[dict],
    max_tokens: int,
    log: logging.Logger,
) -> tuple[str | None, tuple[int, int, int, int]]:
    """OpenAI-compatible skill reflection."""
    reflect_messages = messages + [
        {"role": "user", "content": SKILL_REFLECTION_PROMPT}
    ]
    try:
        log.info("[reflection] Asking solver to propose a skill draft...")
        params: dict[str, Any] = {
            "model": model_id,
            "messages": reflect_messages,
            "max_tokens": max_tokens,
        }
        if tools:
            params["tools"] = tools
        response = client.chat.completions.create(**params)
        text = response.choices[0].message.content or ""
        reflection_usage = _openai_usage(getattr(response, "usage", None))

        if not text or "ACTION: NONE" in text.upper():
            log.info("[reflection] Solver declined: no skill needed")
            return None, reflection_usage

        if "CONFIDENCE:" in text.upper() and "ACTION:" in text.upper() and "CONTENT:" in text.upper():
            start = text.upper().index("CONFIDENCE:")
            draft = text[start:].strip().strip("`").strip()
            log.info("[reflection] Got skill draft (%d chars)", len(draft))
            return draft, reflection_usage

        log.info("[reflection] Response didn't contain a valid skill draft")
        return None, reflection_usage

    except Exception as e:
        log.warning("[reflection] Skill reflection failed: %s", str(e)[:200])
        return None, (0, 0, 0, 0)


def _to_openai_tools(tools: list[dict]) -> list[dict]:
    """Convert Bedrock Converse tool specs to OpenAI function tools."""
    converted: list[dict] = []
    for spec in tools:
        if spec.get("type") == "function":
            converted.append(spec)
            continue
        tool_spec = spec.get("toolSpec", spec)
        schema = tool_spec.get("inputSchema", {}).get("json")
        if schema is None:
            schema = tool_spec.get("input_schema", {"type": "object"})
        converted.append({
            "type": "function",
            "function": {
                "name": tool_spec.get("name", ""),
                "description": tool_spec.get("description", ""),
                "parameters": schema,
            },
        })
    return converted


def _openai_tool_call_to_dict(tool_call: Any) -> dict[str, Any]:
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments or "{}",
        },
    }


def _usage_value(usage: Any, key: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get(key, 0) or 0)
    return int(getattr(usage, key, 0) or 0)


def _nested_usage_value(usage: Any, parent: str, key: str) -> int:
    details = usage.get(parent) if isinstance(usage, dict) else getattr(usage, parent, None)
    return _usage_value(details, key)


def _openai_usage(usage: Any) -> tuple[int, int, int, int]:
    """Return uncached input, output, cache-read and cache-write tokens.

    OpenAI-compatible providers expose cache telemetry in different places.
    DeepSeek reports prompt_cache_hit/miss_tokens, while OpenAI reports
    prompt_tokens_details.cached_tokens.  Since prompt_tokens includes cache
    hits for those APIs, subtract hits to avoid charging them twice.
    """
    prompt_total = _usage_value(usage, "prompt_tokens")
    output_tokens = _usage_value(usage, "completion_tokens")

    cache_read = _usage_value(usage, "prompt_cache_hit_tokens")
    cache_miss = _usage_value(usage, "prompt_cache_miss_tokens")
    if cache_read <= 0:
        cache_read = _nested_usage_value(usage, "prompt_tokens_details", "cached_tokens")
    if cache_read <= 0:
        cache_read = _usage_value(usage, "cache_read_input_tokens")

    cache_write = _usage_value(usage, "cache_write_input_tokens")
    if cache_write <= 0:
        cache_write = _usage_value(usage, "cache_creation_input_tokens")

    # DeepSeek's miss counter is the authoritative billable uncached input.
    # Otherwise prompt_tokens is inclusive, so remove cache hits.
    input_tokens = cache_miss if cache_miss > 0 else max(0, prompt_total - cache_read)
    return input_tokens, output_tokens, cache_read, cache_write


# ── Conversation extraction ─────────────────────────────────────────

def extract_conversation(messages: list[dict]) -> list[dict]:
    """Convert Bedrock/OpenAI messages to standard assistant/tool format."""
    conv = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content_blocks = msg.get("content", [])

        if isinstance(content_blocks, str):
            if role == "assistant" and msg.get("tool_calls"):
                entry: dict = {
                    "role": "assistant",
                    "content": content_blocks,
                    "tool_calls": [],
                }
                for tc in msg.get("tool_calls", []):
                    raw_args = tc.get("function", {}).get("arguments", "{}")
                    try:
                        args = json.loads(raw_args)
                    except Exception:
                        args = raw_args
                    entry["tool_calls"].append({
                        "id": tc.get("id", ""),
                        "function": tc.get("function", {}).get("name", ""),
                        "arguments": args,
                        "type": "function",
                    })
                conv.append(entry)
            elif role == "tool":
                conv.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "function": msg.get("name", ""),
                    "content": content_blocks[:3000]
                    + ("\n...[truncated]" if len(content_blocks) > 3000 else ""),
                })
            elif content_blocks:
                conv.append({"role": role, "content": content_blocks})
            continue

        text_parts = []
        tool_uses = []
        tool_results = []

        for b in content_blocks:
            if "text" in b:
                text_parts.append(b["text"])
            elif "toolUse" in b:
                tool_uses.append(b["toolUse"])
            elif "toolResult" in b:
                tool_results.append(b["toolResult"])

        if role == "assistant" and tool_uses:
            entry: dict = {
                "role": "assistant",
                "content": "\n".join(text_parts) if text_parts else "",
                "tool_calls": [],
            }
            for tu in tool_uses:
                inp = tu.get("input", {})
                inp_str = json.dumps(inp)
                if len(inp_str) > 2000:
                    inp = {"_truncated": inp_str[:2000] + "..."}
                entry["tool_calls"].append({
                    "id": tu.get("toolUseId", ""),
                    "function": tu.get("name", ""),
                    "arguments": inp,
                    "type": "function",
                })
            conv.append(entry)

        elif tool_results:
            for tr in tool_results:
                tool_use_id = tr.get("toolUseId", "")
                parts = []
                for c in tr.get("content", []):
                    if isinstance(c, dict) and "text" in c:
                        txt = c["text"]
                        if len(txt) > 3000:
                            txt = txt[:3000] + "\n...[truncated]"
                        parts.append(txt)
                func_name = ""
                for e in reversed(conv):
                    for tc in e.get("tool_calls", []):
                        if tc.get("id") == tool_use_id:
                            func_name = tc.get("function", "")
                            break
                    if func_name:
                        break
                conv.append({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "function": func_name,
                    "content": "\n".join(parts),
                })

        elif text_parts:
            conv.append({
                "role": role,
                "content": "\n".join(text_parts),
            })

    return conv
