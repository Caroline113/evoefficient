"""Task profiling helpers for EvoEfficient strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskProfile:
    """Compact task profile used by strategy selection and routing."""

    task_type: str = "general"
    domain: str = "generic"
    difficulty: str = "hard"
    context_pressure: str = "medium"
    tool_pressure: str = "medium"
    verification_pressure: str = "medium"
    output_pressure: str = "medium"
    route_level: str = "hard"
    estimated_tokens: int = 0
    notes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "domain": self.domain,
            "difficulty": self.difficulty,
            "context_pressure": self.context_pressure,
            "tool_pressure": self.tool_pressure,
            "verification_pressure": self.verification_pressure,
            "output_pressure": self.output_pressure,
            "route_level": self.route_level,
            "estimated_tokens": self.estimated_tokens,
            "notes": dict(self.notes),
        }


def profile_task_text(
    text: str,
    *,
    metadata: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> TaskProfile:
    """Build a coarse profile from task text, metadata, and recent history."""
    text = text or ""
    meta = metadata or {}
    lower = text.lower()
    words = len(text.split())
    lines = len(text.splitlines()) or 1
    history = history or []

    task_type = "general"
    if any(k in lower for k in ("test", "traceback", "assert", "exception", "fix", "bug", "regression", "failure")):
        task_type = "debug"
    elif any(k in lower for k in ("summar", "condense", "overview", "report", "brief")):
        task_type = "summarization"
    elif any(k in lower for k in ("translate", "translation", "localize", "rewrite", "paraphrase")):
        task_type = "translation"
    elif any(k in lower for k in ("classify", "label", "predict", "tag", "categor")):
        task_type = "classification"
    elif any(k in lower for k in ("route", "routing", "model", "cost", "profile", "select", "choose")):
        task_type = "routing"
    elif any(k in lower for k in ("extract", "parse", "transform", "convert", "normalize", "reshape")):
        task_type = "transformation"
    elif any(k in lower for k in ("compare", "ablation", "evaluate", "benchmark", "analyze", "analysis")):
        task_type = "analysis"
    elif any(k in lower for k in ("generate", "draft", "compose", "create", "write", "produce")):
        task_type = "generation"
    elif any(k in lower for k in ("plan", "schedule", "sequence", "workflow", "strategy")):
        task_type = "planning"
    elif any(k in lower for k in ("qa", "question", "answer", "lookup", "search")):
        task_type = "qa"

    domain = str(
        meta.get("domain")
        or meta.get("benchmark")
        or meta.get("task_type")
        or meta.get("dataset")
        or meta.get("app")
        or "generic"
    )

    context_score = 0
    if words > 500:
        context_score += 1
    if lines > 50:
        context_score += 1
    if any(k in lower for k in ("long", "large", "many files", "multi-file", "multi step")):
        context_score += 1

    tool_score = 0
    if any(k in lower for k in ("file", "json", "yaml", "csv", "log", "test", "trace", "stack")):
        tool_score += 1
    if any(k in lower for k in ("shell", "cli", "terminal", "docker", "command")):
        tool_score += 1

    verification_score = 0
    if any(k in lower for k in ("verify", "test", "reproduce", "benchmark", "validate")):
        verification_score += 1
    if any(k in lower for k in ("regression", "edge case", "before", "after")):
        verification_score += 1

    output_score = 0
    if words > 800:
        output_score += 1
    if any(k in lower for k in ("verbose", "long output", "large output", "log")):
        output_score += 1

    difficulty_score = context_score + tool_score + verification_score
    if len(history) >= 3:
        difficulty_score += 1
    if any(k in lower for k in ("ambiguous", "cross-file", "multiple", "interacting", "tradeoff")):
        difficulty_score += 1

    difficulty = _score_to_binary_difficulty(difficulty_score, easy_cut=2)
    context_pressure = _score_to_level(context_score, easy_cut=1, hard_cut=3)
    tool_pressure = _score_to_level(tool_score, easy_cut=1, hard_cut=3)
    verification_pressure = _score_to_level(verification_score, easy_cut=1, hard_cut=3)
    output_pressure = _score_to_level(output_score, easy_cut=1, hard_cut=3)

    estimated_tokens = max(words * 3, lines * 40)

    task_reason = {
        "task_type": task_type,
        "keywords": [
            key
            for key in (
                "test",
                "traceback",
                "summar",
                "translate",
                "classify",
                "route",
                "extract",
                "compare",
                "generate",
                "plan",
                "qa",
            )
            if key in lower
        ],
    }

    return TaskProfile(
        task_type=task_type,
        domain=domain,
        difficulty=difficulty,
        context_pressure=context_pressure,
        tool_pressure=tool_pressure,
        verification_pressure=verification_pressure,
        output_pressure=output_pressure,
        route_level="hard",
        estimated_tokens=estimated_tokens,
        notes={
            "word_count": words,
            "line_count": lines,
            "history_items": len(history),
            "metadata_keys": sorted(str(k) for k in meta.keys()),
            "task_reason": task_reason,
        },
    )


def _score_to_level(score: int, *, easy_cut: int, hard_cut: int) -> str:
    if score <= easy_cut:
        return "easy"
    if score >= hard_cut:
        return "hard"
    return "medium"


def _score_to_binary_difficulty(score: int, *, easy_cut: int) -> str:
    return "easy" if score <= easy_cut else "hard"
