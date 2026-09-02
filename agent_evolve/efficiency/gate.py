"""Success-first gate for EvoEfficient candidate strategies."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass
class GateConfig:
    epsilon_success: float = 0.03
    delta_cost: float = 0.10
    min_trials: int = 2
    discovery_overhead_tokens: float = 0.0
    expected_future_hits: int = 10
    max_break_even_hits: int | None = None


def gate_candidate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    config: GateConfig | None = None,
) -> dict[str, Any]:
    cfg = config or GateConfig()
    b = baseline.get("summary", baseline)
    c = candidate.get("summary", candidate)

    b_success = float(b.get("success_rate", 0.0) or 0.0)
    c_success = float(c.get("success_rate", 0.0) or 0.0)
    # b_cost = float(b.get("avg_tokens", 0.0) or 0.0)
    # c_cost = float(c.get("avg_tokens", 0.0) or 0.0)
    b_cost = float(b.get("avg_cost", b.get("avg_tokens", 0.0)) or 0.0)
    c_cost = float(c.get("avg_cost", c.get("avg_tokens", 0.0)) or 0.0)
    trials = int(c.get("tasks", 0) or 0)
    expected_hits = max(1, int(cfg.expected_future_hits or 1))
    overhead = max(0.0, float(cfg.discovery_overhead_tokens or 0.0))
    gross_saving_per_task = b_cost - c_cost
    amortized_overhead_per_task = overhead / expected_hits
    net_saving_per_task = gross_saving_per_task - amortized_overhead_per_task

    if gross_saving_per_task > 0:
        break_even_hits = int(math.ceil(overhead / gross_saving_per_task)) if overhead else 0
    else:
        break_even_hits = math.inf
    max_break_even = (
        int(cfg.max_break_even_hits)
        if cfg.max_break_even_hits is not None
        else expected_hits
    )

    success_ok = c_success >= b_success - cfg.epsilon_success
    cost_ok = c_cost <= b_cost * (1.0 - cfg.delta_cost) if b_cost > 0 else False
    trials_ok = trials >= cfg.min_trials
    net_saving_ok = net_saving_per_task > 0
    break_even_ok = break_even_hits <= max_break_even
    # accepted = success_ok and cost_ok and trials_ok and net_saving_ok and break_even_ok
    accepted = success_ok and cost_ok

    return {
        "accepted": accepted,
        "success_ok": success_ok,
        "cost_ok": cost_ok,
        "trials_ok": trials_ok,
        "net_saving_ok": net_saving_ok,
        "break_even_ok": break_even_ok,
        "success_delta": round(c_success - b_success, 6),
        "cost_delta_pct": round((c_cost - b_cost) / b_cost, 6) if b_cost else 0.0,
        "baseline_avg_cost": round(b_cost, 4),
        "candidate_avg_cost": round(c_cost, 4),
        "gross_saving_per_task": round(gross_saving_per_task, 4),
        "discovery_overhead_tokens": round(overhead, 4),
        "expected_future_hits": expected_hits,
        "amortized_overhead_per_task": round(amortized_overhead_per_task, 4),
        "net_saving_per_task": round(net_saving_per_task, 4),
        "break_even_hits": break_even_hits if math.isfinite(break_even_hits) else "inf",
        "max_break_even_hits": max_break_even,
        "reason": (
            "accepted"
            if accepted
            else "; ".join(
                label
                for label, ok in [
                    ("success gate failed", success_ok),
                    ("cost gate failed", cost_ok),
                    ("min trials gate failed", trials_ok),
                    ("net saving gate failed", net_saving_ok),
                    ("break-even gate failed", break_even_ok),
                ]
                if not ok
            )
        ),
    }
