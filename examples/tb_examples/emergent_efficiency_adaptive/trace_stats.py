"""Read-only Codex session statistics; does not alter execution or stop processes."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def new_state():
    return {
        "usage": {},
        "timeline": [],
        "calls": {},
        "steps": [],
        "seen_calls": [],
        "failed_signature": "",
        "repeated_failures": 0,
        "declared_stop": False,
    }


def consume(state, event):
    payload = event.get("payload", {})
    if not isinstance(payload, dict):
        return
    usage = None
    if event.get("type") == "token_usage_record":
        usage = payload.get("thread_token_usage")
    elif event.get("type") == "event_msg" and payload.get("type") == "token_count":
        usage = (payload.get("info") or {}).get("total_token_usage")
    if isinstance(usage, dict) and "input_tokens" in usage and "output_tokens" in usage:
        incoming, outgoing = int(usage["input_tokens"]), int(usage["output_tokens"])
        cached = int(usage.get("cached_input_tokens", 0))
        total = int(usage.get("total_tokens", incoming + outgoing))
        if 0 <= cached <= incoming and total == incoming + outgoing:
            normalized = {
                "input_tokens": incoming - cached,
                "cache_read_input_tokens": cached,
                "output_tokens": outgoing,
                "total_tokens": total,
            }
            previous = state["usage"]
            if not previous or total > previous.get("total_tokens", 0):
                state["usage"] = normalized
                state["timeline"].append(normalized.copy())
    if event.get("type") != "response_item":
        return
    kind = payload.get("type")
    cid = payload.get("call_id") or payload.get("id")
    if kind in ("function_call", "custom_tool_call") and cid:
        if cid in state["calls"]:
            return
        args = payload.get("arguments", payload.get("input", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (ValueError, TypeError):
                args = {"input": args}
        if not isinstance(args, dict):
            args = {}
        name = payload.get("name", "unknown")
        command = str(args.get("cmd", args.get("command", args.get("input", ""))))
        signature = hashlib.sha256(json.dumps([name, args], sort_keys=True).encode()).hexdigest()
        state["calls"][cid] = {"tool": name, "command": command[:6000], "signature": signature}
        # An attempted edit invalidates assumptions about an unchanged failure state.
        if name in ("apply_patch", "functions.apply_patch") or "apply_patch" in command:
            state["failed_signature"] = ""
            state["repeated_failures"] = 0
    elif kind in ("function_call_output", "custom_tool_call_output") and cid in state["calls"]:
        if cid in state["seen_calls"]:
            return
        state["seen_calls"].append(cid)
        call = state["calls"][cid]
        output = str(payload.get("output", ""))
        match = re.search(r"(?:Process exited with code|Exit code:)\s*(-?\d+)", output)
        code = int(match.group(1)) if match else None
        state["steps"].append(
            {
                "tool": call["tool"],
                "action": call["tool"],
                "arguments": call["command"],
                "output": output[:6000],
                "exit_code": code,
                "call_id": cid,
            }
        )
        if code is not None:
            if code != 0:
                state["repeated_failures"] = (
                    state["repeated_failures"] + 1
                    if state["failed_signature"] == call["signature"]
                    else 1
                )
                state["failed_signature"] = call["signature"]
            else:
                state["failed_signature"] = ""
                state["repeated_failures"] = 0
    elif (
        kind == "message"
        and payload.get("role") == "assistant"
        and payload.get("channel") == "final"
    ):
        text = str(payload.get("content", ""))
        if "A_EVOLVE_EARLY_STOP:" in text:
            state["declared_stop"] = True


def parse_session(path):
    state = new_state()
    with Path(path).open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict):
                consume(state, event)
    return state
