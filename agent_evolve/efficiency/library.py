"""Structured strategy library helpers for EvoEfficient experiments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .profiler import result_cost, result_tokens

from .selector import config_to_runtime_config, select_runtime_config
from .strategies import DEFAULT_STRATEGIES, materialize_strategy_artifacts, write_active_config


@dataclass
class StrategyLibraryEntry:
    id: str
    strategy: str
    status: str
    scope: dict[str, Any]
    gate: dict[str, Any]
    stats: dict[str, Any]
    rationale: str = ""
    source: str = "evoefficient"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "strategy": self.strategy,
            "status": self.status,
            "scope": self.scope,
            "rationale": self.rationale,
            "gate": self.gate,
            "stats": self.stats,
            "source": self.source,
            "created_at": datetime.utcnow().isoformat(),
        }


def strategy_library_path(workspace_or_dir: str | Path) -> Path:
    return Path(workspace_or_dir) / "efficiency" / "strategy_library.yaml"


def load_strategy_library(workspace_or_dir: str | Path) -> dict[str, Any]:
    path = strategy_library_path(workspace_or_dir)
    if not path.exists():
        return {"strategies": []}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    strategies = data.get("strategies", [])
    if not isinstance(strategies, list):
        strategies = []
    data["strategies"] = strategies
    return data


def write_strategy_library(workspace_or_dir: str | Path, library: dict[str, Any]) -> Path:
    path = strategy_library_path(workspace_or_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(library)
    payload.setdefault("strategies", [])
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def active_strategy_ids(library: dict[str, Any]) -> list[str]:
    active: list[str] = []
    for entry in library.get("strategies", []) or []:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("status", "")).lower() not in {"active", "accepted"}:
            continue
        strategy = str(entry.get("strategy") or entry.get("name") or "")
        if strategy in DEFAULT_STRATEGIES and strategy not in active:
            active.append(strategy)
    return active


def active_config_entries(library: dict[str, Any]) -> list[dict[str, Any]]:
    """Return validated structured configs that may be used at runtime."""
    configs = library.get("configs", []) or []
    if not isinstance(configs, list):
        return []
    has_active_config_filter = "active_configs" in library
    active_ids = {
        str(config_id)
        for config_id in (library.get("active_configs", []) or [])
        if config_id
    }
    return [
        config for config in configs
        if isinstance(config, dict)
        and str(config.get("status", "")).lower() in {"active", "accepted"}
        and (
            not has_active_config_filter
            or str(config.get("id", "")) in active_ids
        )
    ]


def active_config_strategy_ids(library: dict[str, Any]) -> list[str]:
    strategies: list[str] = []
    for config in active_config_entries(library):
        for name in config.get("strategies", []) or []:
            if name in DEFAULT_STRATEGIES and name not in strategies:
                strategies.append(name)
    return strategies


def select_library_config(
    library: dict[str, Any],
    task_profile: dict[str, Any],
    cost_profile: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Choose a matching active config for one task profile."""
    return select_runtime_config(library, task_profile, cost_profile)


def record_config_outcome(
    library: dict[str, Any],
    *,
    config_id: str | None,
    result: dict[str, Any],
) -> None:
    """Update lightweight online statistics for a deployed config."""
    if not config_id:
        return
    for config in library.get("configs", []) or []:
        if not isinstance(config, dict) or config.get("id") != config_id:
            continue
        stats = config.setdefault("stats", {})
        stats["trials"] = int(stats.get("trials", 0) or 0) + 1
        if result.get("success"):
            stats["successes"] = int(stats.get("successes", 0) or 0) + 1
        stats["success_rate"] = round(
            float(stats.get("successes", 0) or 0) / max(1, int(stats["trials"])),
            6,
        )
        # cost = int(result.get("total_tokens", 0) or 0)
        # stats["tokens"] = int(stats.get("tokens", 0) or 0) + cost
        # stats["avg_tokens"] = round(stats["tokens"] / max(1, int(stats["trials"])), 2)
        tokens = int(result_tokens(result))
        cost = float(result_cost(result))
        stats["tokens"] = int(stats.get("tokens", 0) or 0) + tokens
        stats["cost_units"] = round(float(stats.get("cost_units", 0.0) or 0.0) + cost, 4)
        stats["avg_tokens"] = round(stats["tokens"] / max(1, int(stats["trials"])), 2)
        stats["avg_cost"] = round(stats["cost_units"] / max(1, int(stats["trials"])), 2)
        hits = result.get("strategy_hits", {}) or {}
        if isinstance(hits, dict):
            stats["hit_count"] = int(stats.get("hit_count", 0) or 0) + sum(
                int(value.get("hits", 0) if isinstance(value, dict) else value or 0)
                for value in hits.values()
            )
        return


def apply_library_to_workspace(
    workspace_or_dir: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
    task_profile: dict[str, Any] | None = None,
    cost_profile: dict[str, Any] | None = None,
) -> list[str]:
    library = load_strategy_library(workspace_or_dir)
    selected_config = (
        select_library_config(library, task_profile, cost_profile)
        if task_profile
        else None
    )
    if selected_config:
        strategies = list(selected_config.get("strategies", []) or [])
    elif library.get("configs"):
        # Structured configs are selected per task by SweAgent. Do not expose
        # every learned config globally when no task profile is available.
        # A missing profile must therefore fall back to baseline.
        strategies = []
    else:
        strategies = active_strategy_ids(library)
    write_active_config(
        workspace_or_dir,
        strategies,
        metadata={
            "source": "strategy_library",
            "efficiency_config_id": selected_config.get("id", "") if selected_config else "",
            **(metadata or {}),
        },
    )
    materialize_strategy_artifacts(workspace_or_dir, strategies)
    return strategies


def build_library_from_gate(
    *,
    candidate_strategies: list[str],
    accepted_strategies: list[str],
    gate: dict[str, Any],
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    reuse_summary: dict[str, Any] | None = None,
    selector_profile: dict[str, Any] | None = None,
    source: str = "evoefficient_mvp",
    status_reason: str = "",
    candidate_configs: list[dict[str, Any]] | None = None,
    selected_config_id: str | None = None,
    pareto_frontier: list[str] | None = None,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    accepted = set(accepted_strategies)
    bottlenecks = selector_profile.get("bottlenecks", []) if selector_profile else []
    train_gate_accepted = bool(gate.get("accepted"))

    for strategy in candidate_strategies:
        if strategy in accepted:
            status = "active"
        elif train_gate_accepted and status_reason:
            status = "quarantined"
        else:
            status = "rejected"
        rationale_parts = [
            f"Selected from train bottlenecks: {', '.join(bottlenecks) or 'none'}",
            f"train_gate={gate.get('reason', 'unknown')}",
        ]
        if status_reason:
            rationale_parts.append(status_reason)
        entry = StrategyLibraryEntry(
            id=f"swe.{strategy}.v1",
            strategy=strategy,
            status=status,
            scope={
                "benchmark": "swebench_verified",
                "task_type": "coding",
                "bottlenecks": list(bottlenecks),
            },
            rationale="; ".join(rationale_parts),
            gate=gate,
            stats={
                "train_baseline": baseline_summary,
                "train_candidate": candidate_summary,
                "heldout_reuse": reuse_summary or {},
            },
            source=source,
        )
        entries.append(entry.as_dict())

    config_entries: list[dict[str, Any]] = []
    for candidate in candidate_configs or []:
        config = dict(candidate)
        config_id = str(config.get("id", ""))
        gate_info = config.get("gate", {}) if isinstance(config.get("gate", {}), dict) else {}
        if config_id == selected_config_id and not status_reason:
            status = "active"
        elif gate_info.get("accepted"):
            status = "validated"
        else:
            status = "rejected"
        if status_reason and config_id == selected_config_id:
            status = "quarantined"
        config["status"] = status
        config["gate"] = gate_info
        config.setdefault("stats", {})
        config_entries.append(config)

    return {
        "version": 1,
        "created_at": datetime.utcnow().isoformat(),
        "active_strategies": list(accepted_strategies),
        "strategies": entries,
        "configs": config_entries,
        "active_configs": [selected_config_id] if selected_config_id and not status_reason else [],
        "pareto_frontier": pareto_frontier or [],
        "meta": {
            "selection_layer": "meta_strategy_selector",
            "status_reason": status_reason,
        },
    }
