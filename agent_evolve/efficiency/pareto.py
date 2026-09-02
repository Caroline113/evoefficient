"""Success-first Pareto selection for efficiency configurations."""

from __future__ import annotations

from typing import Any


def _summary(value: dict[str, Any]) -> dict[str, Any]:
    return value.get("summary", value) if isinstance(value, dict) else {}


def is_dominated(candidate: dict[str, Any], other: dict[str, Any]) -> bool:
    """Return whether ``other`` is at least as good in all tracked dimensions."""
    c = _summary(candidate)
    o = _summary(other)
    c_success = float(c.get("success_rate", 0.0) or 0.0)
    o_success = float(o.get("success_rate", 0.0) or 0.0)
    c_cost = float(c.get("cost_of_pass", c.get("avg_tokens", 0.0)) or 0.0)
    o_cost = float(o.get("cost_of_pass", o.get("avg_tokens", 0.0)) or 0.0)
    c_calls = float(c.get("avg_tool_calls", 0.0) or 0.0)
    o_calls = float(o.get("avg_tool_calls", 0.0) or 0.0)

    no_worse = o_success >= c_success and o_cost <= c_cost and o_calls <= c_calls
    strictly_better = o_success > c_success or o_cost < c_cost or o_calls < c_calls
    return no_worse and strictly_better


def pareto_frontier(
    candidates: list[dict[str, Any]],
    *,
    epsilon_success: float = 0.03,
) -> list[dict[str, Any]]:
    """Return success-first non-dominated candidate records."""
    frontier: list[dict[str, Any]] = []
    for candidate in candidates:
        c = _summary(candidate)
        c_success = float(c.get("success_rate", 0.0) or 0.0)
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            o = _summary(other)
            o_success = float(o.get("success_rate", 0.0) or 0.0)
            if o_success < c_success - epsilon_success:
                continue
            if is_dominated(candidate, other):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return frontier


def choose_best_candidate(
    candidates: list[dict[str, Any]],
    *,
    baseline: dict[str, Any],
    epsilon_success: float = 0.03,
) -> dict[str, Any] | None:
    """Choose the cheapest candidate among those preserving baseline success."""
    b = _summary(baseline)
    baseline_success = float(b.get("success_rate", 0.0) or 0.0)
    eligible = [
        item for item in candidates
        if float(_summary(item).get("success_rate", 0.0) or 0.0)
        >= baseline_success - epsilon_success
    ]
    if not eligible:
        return None
    eligible.sort(
        key=lambda item: (
            float(_summary(item).get("cost_of_pass", float("inf")) or float("inf")),
            float(_summary(item).get("avg_tokens", float("inf")) or float("inf")),
        )
    )
    return eligible[0]

