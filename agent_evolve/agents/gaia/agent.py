"""Workspace-backed tool-using agent for GAIA."""

from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import logging
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...efficiency.profiler import normalize_usage
from ...efficiency.strategies import load_active_config, strategy_prompt_block
from ...llm.factory import create_llm_provider, create_strands_model, resolve_model_selection
from ...llm.base import LLMMessage
from ...protocol.base_agent import BaseAgent
from .tools import inspect_file_impl, python_impl, web_fetch_impl, web_search_impl

logger = logging.getLogger(__name__)


@dataclass
class GaiaSolveResult:
    answer: str = ""
    raw_output: str = ""
    conversation: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    elapsed: float = 0.0
    timed_out: bool = False
    error: str = ""
    skill_proposal: str = ""
    solver_metrics_usage: dict[str, Any] = field(default_factory=dict)
    solver_message_usage: dict[str, Any] = field(default_factory=dict)
    proposal_usage: dict[str, Any] = field(default_factory=dict)
    usage_consistency: dict[str, Any] = field(default_factory=dict)


class GaiaAgent(BaseAgent):
    def __init__(
        self,
        workspace_dir: str | Path,
        model_id: str,
        region: str = "us-west-2",
        max_tokens: int = 16384,
        *,
        enable_skill_proposal: bool = False,
    ):
        super().__init__(workspace_dir)
        self.model_id = model_id
        self.region = region
        self.max_tokens = max_tokens
        self.enable_skill_proposal = enable_skill_proposal

    @staticmethod
    def _message_metadata(message: Any) -> dict[str, Any]:
        if isinstance(message, dict):
            metadata = message.get("metadata", {})
        else:
            metadata = getattr(message, "metadata", {})
        return metadata if isinstance(metadata, dict) else {}

    @classmethod
    def _raw_usage_from_messages(cls, messages: list[Any]) -> dict[str, int]:
        """Recover per-turn usage when Strands raises before returning a result."""
        totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_write_input_tokens": 0,
        }
        found = False
        for message in messages or []:
            raw = cls._message_metadata(message).get("usage", {})
            if not isinstance(raw, dict) or not raw:
                continue
            usage = normalize_usage(raw)
            if not any(int(usage.get(key, 0) or 0) for key in totals):
                continue
            found = True
            for key in totals:
                totals[key] += int(usage.get(key, 0) or 0)
        return totals if found else {}

    @staticmethod
    def _select_solver_usage(
        metrics_usage: dict[str, Any], message_usage: dict[str, Any], *, tolerance: float = 0.20
    ) -> tuple[dict[str, Any], str, dict[str, Any]]:
        """Select usage while detecting duplicated/incomplete Strands aggregates."""
        metrics = normalize_usage(metrics_usage) if metrics_usage else {}
        messages = normalize_usage(message_usage) if message_usage else {}
        metrics_total = int(metrics.get("total_tokens", 0) or 0)
        message_total = int(messages.get("total_tokens", 0) or 0)
        details = {
            "metrics_total_tokens": metrics_total,
            "message_total_tokens": message_total,
            "tolerance": tolerance,
            "status": "missing",
            "selected": "none",
        }
        if metrics_total and message_total:
            ratio = metrics_total / message_total
            details["metrics_to_messages_ratio"] = ratio
            if not (1.0 - tolerance <= ratio <= 1.0 + tolerance):
                details.update(status="inconsistent", selected="messages")
                return message_usage, "gaia_message_metadata_inconsistent_metrics", details
            details.update(status="consistent", selected="metrics")
            return metrics_usage, "gaia_strands_metrics", details
        if metrics_total:
            details.update(status="metrics_only", selected="metrics")
            return metrics_usage, "gaia_strands_metrics", details
        if message_total:
            details.update(status="messages_only", selected="messages")
            return message_usage, "gaia_message_metadata", details
        return {}, "gaia_usage_missing", details

    def _propose_intervention(
        self,
        *,
        question: str,
        answer: str,
        error: str,
        conversation: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any]]:
        """Produce the structured solver proposal consumed by GuidedSynthesisEngine."""
        if not self.enable_skill_proposal:
            return "", {}
        tool_names: list[str] = []
        for message in conversation:
            for block in message.get("content", []) if isinstance(message.get("content"), list) else []:
                tool_use = block.get("toolUse") if isinstance(block, dict) else None
                name = str(tool_use.get("name") or "") if isinstance(tool_use, dict) else ""
                if name and name not in tool_names:
                    tool_names.append(name)
        prompt = f"""Reflect on this completed GAIA research attempt and propose one reusable intervention for future solvers.

Question: {question[:4000]}
Submitted answer: {answer or '(none)'}
Execution error: {error or '(none)'}
Tools used: {', '.join(tool_names) or '(none)'}

The proposal must generalize across many GAIA questions, not encode this question or its answer. Prefer a research/verification skill or prompt fragment. Return exactly one of:

CONFIDENCE: HIGH/MEDIUM/LOW
ACTION: NEW
TYPE: skill | fragment | memory
NAME: generic_snake_case_name
DESCRIPTION: TRIGGER when ...; DO NOT TRIGGER when ...
CONTENT:
(concise reusable methodology, under 500 words)

or, if no useful general lesson exists:
CONFIDENCE: LOW
ACTION: NONE
"""
        try:
            provider = create_llm_provider(self.model_id, region=self.region)
            response = provider.complete(
                [LLMMessage(role="user", content=prompt)], max_tokens=2048, temperature=0.2
            )
            proposal = str(response.content or "").strip()
            valid = "ACTION:" in proposal.upper() and "CONFIDENCE:" in proposal.upper()
            return (proposal if valid else ""), dict(response.usage or {})
        except Exception as exc:
            logger.warning("GAIA intervention proposal failed: %s", exc)
            return "", {}

    def _price_profile(self) -> dict[str, Any]:
        path = self.workspace.root / "efficiency" / "cost_profile.yaml"
        if not path.exists(): return {}
        try:
            import yaml
            value = yaml.safe_load(path.read_text()) or {}
            return value if isinstance(value, dict) else {}
        except Exception as exc:
            logger.warning("Failed to load cost profile: %s", exc)
            return {}

    def _system_prompt(self) -> str:
        parts = [self.system_prompt.strip()]
        active = load_active_config(self.workspace.root)
        strategy_prompt = strategy_prompt_block(
            active,
            task_profile={"task_type": "general-research", "benchmark": "gaia", "difficulty": "level3"},
        )
        if strategy_prompt.strip():
            parts.append(strategy_prompt.strip())
        fragment_names = self.workspace.list_fragments()
        if fragment_names:
            parts.append("## Evolved Prompt Fragments\n" + "\n\n".join(self.workspace.read_fragment(n) for n in fragment_names))
        if self.skills:
            parts.append("## Available Skills\nUse read_skill(name) before applying a relevant skill.\n" + "\n".join(
                f"- {skill.name}: {skill.description}" for skill in self.skills
            ))
        strategy_memories = self.workspace.read_memories("strategy", limit=20)
        if strategy_memories:
            parts.append("## Efficiency Memory\n" + "\n".join(f"- {m.get('content','')}" for m in strategy_memories))
        return "\n\n".join(p for p in parts if p)

    def _workspace_tools(self) -> list[Any]:
        """Load evolved Strands tools without exposing them to other benchmarks."""
        loaded: list[Any] = []
        for entry in self.workspace.read_tool_registry():
            name = str(entry.get("name") or "")
            filename = str(entry.get("file") or f"{name}.py")
            path = self.workspace.tools_dir / filename
            if not name or not path.exists():
                continue
            try:
                spec = importlib.util.spec_from_file_location(f"gaia_workspace_tool_{name}", path)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                candidate = getattr(module, name, None)
                if candidate is not None:
                    loaded.append(candidate)
            except Exception as exc:
                logger.warning("Failed to load evolved GAIA tool %s: %s", name, exc)
        return loaded

    @staticmethod
    def _conversation(messages: list[Any]) -> list[dict[str, Any]]:
        output = []
        for message in messages or []:
            if isinstance(message, dict):
                output.append(json.loads(json.dumps(message, default=str)))
            else:
                output.append({"role": str(getattr(message, "role", "")), "content": str(getattr(message, "content", message))})
        return output

    @staticmethod
    def _extract_answer(raw: str, submitted: str) -> str:
        if submitted.strip(): return submitted.strip()
        patterns = [r"<final_answer>\s*(.*?)\s*</final_answer>", r"FINAL ANSWER\s*:\s*(.+)"]
        for pattern in patterns:
            found = re.findall(pattern, raw, flags=re.I | re.S)
            if found: return found[-1].strip()
        return raw.strip()

    def solve_task(self, *, question: str, attachment: str = "", timeout_sec: int = 1800) -> GaiaSolveResult:
        try:
            from strands import Agent, tool
        except ImportError as exc:
            raise RuntimeError("GAIA solving requires strands-agents; install the mcp or all extra.") from exc
        started = time.time()
        submitted = {"answer": ""}
        workdir = tempfile.mkdtemp(prefix="aevolve-gaia-")

        @tool
        def web_search(query: str, max_results: int = 8) -> str:
            """Search the live web. Args: query; max_results."""
            return web_search_impl(query, max_results)

        @tool
        def web_fetch(url: str, max_chars: int = 20000) -> str:
            """Fetch a web page as readable text. Args: url; max_chars."""
            return web_fetch_impl(url, max_chars)

        @tool
        def inspect_attachment(max_chars: int = 30000) -> str:
            """Inspect the attachment supplied with the current GAIA question."""
            return inspect_file_impl(attachment, max_chars) if attachment else "No attachment for this task."

        @tool
        def python(code: str, timeout: int = 120) -> str:
            """Run Python for calculations or local file analysis. Args: code; timeout."""
            return python_impl(code, workdir, timeout)

        @tool
        def read_skill(name: str) -> str:
            """Read one available evolved skill by name."""
            return self.get_skill_content(name) or f"Unknown skill: {name}"

        @tool
        def final_answer(answer: str) -> str:
            """Submit the concise final GAIA answer, with no explanation."""
            submitted["answer"] = answer.strip()
            return "Final answer recorded."

        model = create_strands_model(model_id=self.model_id, region=self.region, max_tokens=self.max_tokens)
        tools = [web_search, web_fetch, inspect_attachment, python, read_skill, final_answer]
        tools.extend(self._workspace_tools())
        agent = Agent(model=model, system_prompt=self._system_prompt(), tools=tools)
        attachment_note = (
            f"\nAn attachment is available via inspect_attachment. Its local path for Python is: {attachment}"
            if attachment else ""
        )
        prompt = question + attachment_note + "\nResearch carefully, then call final_answer with only the exact answer."
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(agent, prompt)
        response = None
        error = ""
        timed_out = False
        try:
            response = future.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError:
            timed_out = True
            error = f"Timed out after {timeout_sec}s"
            future.cancel()
        except Exception as exc:
            error = str(exc)[:1000]
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        raw_output = str(response or "")
        messages = getattr(agent, "messages", [])
        conversation = self._conversation(messages)
        metrics_usage: dict[str, Any] = {}
        if response is not None:
            try: metrics_usage = dict(response.metrics.accumulated_usage or {})
            except Exception: pass
        message_usage = self._raw_usage_from_messages(messages)
        raw_usage, usage_source, usage_consistency = self._select_solver_usage(
            metrics_usage, message_usage
        )
        solver_usage_found = bool(raw_usage)
        provider, selected_model = resolve_model_selection(self.model_id, route_level="hard")
        answer = self._extract_answer(raw_output, submitted["answer"])
        skill_proposal, proposal_usage = self._propose_intervention(
            question=question, answer=answer, error=error, conversation=conversation
        )
        if proposal_usage:
            solver_usage = normalize_usage(raw_usage)
            reflection_usage = normalize_usage(proposal_usage)
            raw_usage = {
                key: int(solver_usage.get(key, 0) or 0) + int(reflection_usage.get(key, 0) or 0)
                for key in (
                    "input_tokens", "output_tokens", "cache_read_input_tokens",
                    "cache_write_input_tokens",
                )
            }
            usage_source += "+gaia_skill_proposal"
        usage = normalize_usage(
            raw_usage, price_profile=self._price_profile(), model_id=selected_model,
            provider=provider, route_level="hard",
            usage_record_quality=(
                "inconsistent" if usage_consistency.get("status") == "inconsistent"
                else "ok" if solver_usage_found else "partial" if proposal_usage else "missing"
            ),
            usage_source=usage_source, usage_estimated=False,
        )
        return GaiaSolveResult(
            answer=answer, raw_output=raw_output, conversation=conversation, usage=usage,
            elapsed=time.time()-started, timed_out=timed_out, error=error,
            skill_proposal=skill_proposal,
            solver_metrics_usage=normalize_usage(metrics_usage) if metrics_usage else {},
            solver_message_usage=normalize_usage(message_usage) if message_usage else {},
            proposal_usage=normalize_usage(proposal_usage) if proposal_usage else {},
            usage_consistency=usage_consistency,
        )

    def solve(self, task):
        raise NotImplementedError("Use solve_task() through the isolated GAIA experiment runner.")
