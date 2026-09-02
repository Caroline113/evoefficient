"""Meta efficiency strategy selection and runtime configuration matching.

The selector is deliberately lightweight for the MVP.  It turns a task/cost
profile into executable configuration candidates, while the library matcher
chooses an already validated configuration for future tasks.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from .strategies import DEFAULT_STRATEGIES, STRATEGY_ORDER


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value in DEFAULT_STRATEGIES))


def _config_id(strategies: list[str], suffix: str = "v1") -> str:
    names = "-".join(_unique(strategies))
    return f"swe.meta.{names or 'baseline'}.{suffix}"


def make_efficiency_config(
    *,
    config_id: str,
    strategies: list[str],
    trigger: dict[str, Any],
    modules: dict[str, Any],
    rationale: str,
    fallback: dict[str, Any] | None = None,
    risk: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the structured config stored in the strategy library."""
    clean = _unique(strategies)
    return {
        "id": config_id,
        "type": "efficiency_config",
        "status": "candidate",
        "scope": {
            "benchmark": "swebench_verified",
            "task_type": trigger.get("task_type", "coding"),
        },
        "trigger": deepcopy(trigger),
        "modules": deepcopy(modules),
        "strategies": clean,
        "fallback": deepcopy(fallback or {
            "on_failure": "disable_config",
            "on_uncertainty": "keep_hard_model_and_full_verification",
        }),
        "risk": deepcopy(risk or {
            "level": "medium",
            "failure_mode": "efficiency shortcut may hide evidence",
        }),
        "rationale": rationale,
        "selector": "rule_prior_v1",
    }


def _profile_bottlenecks(
    task_profile: dict[str, Any],
    cost_profile: dict[str, Any],
) -> set[str]:
    tags = set(str(tag) for tag in cost_profile.get("bottlenecks", []) or [])
    tags.update(str(tag) for tag in task_profile.get("bottlenecks", []) or [])

    if task_profile.get("context_pressure") == "hard":
        tags.add("context-heavy")
    if task_profile.get("tool_pressure") == "hard":
        tags.add("tool-heavy")
    if task_profile.get("verification_pressure") == "hard":
        tags.add("verification-heavy")
    if task_profile.get("difficulty") == "hard":
        tags.add("reasoning-heavy")

    return tags


def candidate_configs_for_profile(
    task_profile: dict[str, Any],
    cost_profile: dict[str, Any] | None = None,
    *,
    mode: str = "meta",
    explicit_strategies: list[str] | None = None,
    max_configs: int | None = 20,
) -> list[dict[str, Any]]:
    """Generate candidate configurations for one profiled task.

    ``meta`` is the documented selector mode.  The legacy modes are retained
    as compatibility shims for existing experiments.
    """
    cost = cost_profile or {}
    task_type = str(task_profile.get("task_type", "coding"))
    bottlenecks = _profile_bottlenecks(task_profile, cost)

    if mode == "explicit":
        strategies = _unique(explicit_strategies or [])
        return [
            make_efficiency_config(
                config_id=_config_id(strategies, "explicit-v1"),
                strategies=strategies,
                trigger={"task_type": task_type, "bottlenecks": sorted(bottlenecks)},
                modules={"explicit": {"strategies": strategies}},
                rationale="Explicitly requested strategy configuration.",
                risk={"level": "unknown", "failure_mode": "manual configuration risk"},
            )
        ] if strategies else []

    if mode == "all":
        configs = [
            make_efficiency_config(
                config_id=f"swe.{task_type}.{strategy}.all-v1",
                strategies=[strategy],
                trigger={"task_type": task_type},
                modules={"all_mode": {"strategy": strategy}},
                rationale=f"Independent validation of registered strategy {strategy}.",
                risk={
                    "level": "unknown",
                    "failure_mode": "single-strategy behavior may not generalize",
                },
            )
            for strategy in STRATEGY_ORDER
        ]
        return configs if max_configs is None else configs[:max(1, int(max_configs))]

    if mode == "safe":
        strategies = [
            "observation_masking",
            "verbatim_compaction",
            "plan_template_cache",
            # "cognitive_router",
            "difficulty_model_routing",
            "skill_compilation",
        ]
        configs = [
            make_efficiency_config(
                config_id=f"swe.{task_type}.{strategy}.safe-v1",
                strategies=[strategy],
                trigger={"task_type": task_type, "bottlenecks": sorted(bottlenecks)},
                modules={"safe_mode": {"strategy": strategy}},
                rationale=f"Independent low-risk validation of {strategy}.",
                risk={"level": "low", "failure_mode": "strategy may not fire on this task slice"},
            )
            for strategy in strategies
        ]
        configs.append(
            make_efficiency_config(
                config_id=_config_id(strategies, "safe-combo-v1"),
                strategies=strategies,
                trigger={"task_type": task_type, "bottlenecks": sorted(bottlenecks)},
                modules={"safe_mode": {"composition": "low_risk_combo"}},
                rationale="Composition candidate for interaction effects among low-risk modules.",
                risk={"level": "medium", "failure_mode": "module interaction may erase individual gains"},
            )
        )
        return configs if max_configs is None else configs[:max(1, int(max_configs))]

    candidates: list[dict[str, Any]] = []

    if {"context-heavy", "long-observation"} & bottlenecks:
        candidates.append(
            make_efficiency_config(
                config_id="swe.coding.observation_compact.v1",
                strategies=["observation_masking", "verbatim_compaction"],
                trigger={
                    "task_type": task_type,
                    "bottlenecks": ["context-heavy", "long-observation"],
                },
                modules={
                    "context": {"mode": "trajectory_reduction"},
                    "observation": {
                        "mode": "shell_test_output_filter",
                        "preserve": ["traceback", "assertion", "failed_test", "summary"],
                    },
                },
                rationale="Reduce stale and low-value tool observations while preserving failure evidence.",
                risk={"level": "low", "failure_mode": "important context may be masked"},
            )
        )

    if "tool-heavy" in bottlenecks:
        candidates.append(
            make_efficiency_config(
                config_id="swe.coding.tool_trim.v1",
                strategies=["adaptive_tool_trimming"],
                trigger={"task_type": task_type, "bottlenecks": ["tool-heavy"]},
                modules={
                    "tools": {
                        "mode": "task_aware_output_trimming",
                        "preserve": ["diff", "traceback", "assertion", "summary"],
                    }
                },
                rationale="Reduce tool-output and repeated navigation overhead without hiding errors.",
                risk={"level": "low", "failure_mode": "trimming may require targeted reread"},
            )
        )

    if "reasoning-heavy" in bottlenecks or task_profile.get("difficulty") == "hard":
        candidates.append(
            make_efficiency_config(
                config_id="swe.coding.step_route.v1",
                strategies=["cognitive_router", "difficulty_model_routing"],
                trigger={
                    "task_type": task_type,
                    "bottlenecks": ["reasoning-heavy"],
                    "difficulty": "hard",
                },
                modules={
                    "reasoning": {"mode": "adaptive_effort"},
                    "routing": {
                        "main_route": "hard",
                        "delegate_only": ["extraction", "short_summary", "mechanical_check"],
                    },
                },
                rationale="Keep diagnosis, edits, recovery, and verification on hard; delegate only bounded simple steps.",
                risk={"level": "medium", "failure_mode": "easy delegation may be overused"},
            )
        )

    if cost.get("repeated_tool_sequence") or cost.get("repeated_plan"):
        candidates.append(
            make_efficiency_config(
                config_id="swe.coding.plan_reuse.v1",
                strategies=["plan_template_cache"],
                trigger={
                    "task_type": task_type,
                    "bottlenecks": ["repeated-plan"],
                    "repeated_plan": True,
                },
                modules={"planning": {"mode": "intent_template_cache"}},
                rationale="Reuse task shape and verification order, never task-specific answers.",
                risk={"level": "medium", "failure_mode": "template may be mismatched"},
            )
        )

    if not candidates and task_type in {"coding", "debug"}:
        candidates.extend(
            [
                make_efficiency_config(
                    config_id="swe.coding.safe_context.v1",
                    strategies=["simple_memory", "plan_template_cache"],
                    trigger={"task_type": task_type, "bottlenecks": []},
                    modules={
                        "memory": {"max_memory_items": 3},
                        "planning": {"mode": "intent_template_cache"},
                    },
                    rationale="Conservative task-shape reuse with compact working memory.",
                    risk={
                        "level": "low",
                        "failure_mode": "template may be ignored when irrelevant",
                    },
                ),
                make_efficiency_config(
                    config_id="swe.coding.safe_routing.v1",
                    strategies=["difficulty_model_routing"],
                    trigger={"task_type": task_type, "bottlenecks": []},
                    modules={
                        "routing": {
                            "main_route": "hard",
                            "delegate_only": [
                                "extraction",
                                "short_summary",
                                "mechanical_check",
                            ],
                        }
                    },
                    rationale="Keep the main coding task on hard and expose only explicit easy-step delegation.",
                    risk={
                        "level": "medium",
                        "failure_mode": "easy delegation may be unused or overused",
                    },
                ),
            ]
        )

    if len(candidates) > 1:
        combined = []
        for candidate in candidates:
            combined.extend(candidate["strategies"])
        candidates.insert(
            0,
            make_efficiency_config(
                config_id="swe.coding.profiled_combo.v1",
                strategies=_unique(combined),
                trigger={"task_type": task_type, "bottlenecks": sorted(bottlenecks)},
                modules={
                    "meta": {
                        "selected_from": [candidate["id"] for candidate in candidates],
                        "compose_only_low_risk": True,
                    }
                },
                rationale="Profile-conditioned composition of independently motivated low-risk modules.",
                risk={"level": "medium", "failure_mode": "module interaction may erase individual gains"},
            ),
        )

    if max_configs is None:
        return candidates
    return candidates[: max(1, int(max_configs or 1))]


def config_matches_profile(
    config: dict[str, Any],
    task_profile: dict[str, Any],
    cost_profile: dict[str, Any] | None = None,
) -> tuple[bool, float]:
    """Return whether a validated config applies and a matching score."""
    trigger = config.get("trigger", {}) if isinstance(config, dict) else {}
    cost = cost_profile or {}
    task_type = str(task_profile.get("task_type", "coding"))
    if trigger.get("task_type") and trigger["task_type"] != task_type:
        # SWE issue text is often profiled as ``debug`` while the benchmark
        # selector scope is ``coding``.  Treat those as one task family.
        if not (
            trigger["task_type"] == "coding"
            and task_type in {"coding", "debug"}
        ):
            return False, 0.0

    bottlenecks = _profile_bottlenecks(task_profile, cost)
    required = set(str(x) for x in trigger.get("bottlenecks", []) or [])
    overlap = required & bottlenecks
    if required and not overlap:
        pressure_by_bottleneck = {
            "context-heavy": task_profile.get("context_pressure"),
            "long-observation": task_profile.get("context_pressure"),
            "tool-heavy": task_profile.get("tool_pressure"),
            "reasoning-heavy": task_profile.get("difficulty"),
            "verification-heavy": task_profile.get("verification_pressure"),
        }
        hard_negative = any(
            pressure_by_bottleneck.get(tag) == "easy"
            for tag in required
            if tag in pressure_by_bottleneck
        )
        if hard_negative:
            return False, 0.0

    if trigger.get("difficulty") and trigger["difficulty"] != task_profile.get("difficulty"):
        return False, 0.0
    repeated_plan_unknown = bool(
        trigger.get("repeated_plan") and not cost.get("repeated_plan")
    )

    score = 1.0
    score += 2.0 if trigger.get("task_type") == task_type else 1.0
    score += float(len(overlap))
    if required and not overlap:
        score -= 0.25
    if repeated_plan_unknown:
        score -= 0.25
    stats = config.get("stats", {}) if isinstance(config, dict) else {}
    success_rate = stats.get("success_rate")
    cost_delta = stats.get("cost_delta_pct")
    if isinstance(success_rate, (int, float)):
        score += float(success_rate)
    if isinstance(cost_delta, (int, float)) and cost_delta < 0:
        score += min(2.0, abs(float(cost_delta)) * 2.0)
    return True, score


def select_runtime_config(
    library: dict[str, Any] | None,
    task_profile: dict[str, Any],
    cost_profile: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Choose the best active config for one future task."""
    if not library:
        return None
    configs = library.get("configs", []) or []
    has_active_config_filter = "active_configs" in library
    active_ids = {
        str(config_id)
        for config_id in (library.get("active_configs", []) or [])
        if config_id
    }
    ranked: list[tuple[float, dict[str, Any]]] = []
    for config in configs:
        if not isinstance(config, dict):
            continue
        if str(config.get("status", "")).lower() not in {"active", "accepted"}:
            continue
        if has_active_config_filter and str(config.get("id", "")) not in active_ids:
            continue
        matches, score = config_matches_profile(config, task_profile, cost_profile)
        if matches:
            ranked.append((score, config))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0], reverse=True)
    return deepcopy(ranked[0][1])


def select_primary_config(
    task_profile: dict[str, Any],
    cost_profile: dict[str, Any] | None = None,
    *,
    mode: str = "meta",
) -> dict[str, Any] | None:
    """Return the first selector candidate, useful for task-level routing."""
    candidates = candidate_configs_for_profile(task_profile, cost_profile, mode=mode)
    return candidates[0] if candidates else None


def config_to_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    """Convert a structured library config into the workspace runtime shape."""
    strategies = _unique(config.get("strategies", []) or [])
    parameters: dict[str, Any] = {}
    modules = config.get("modules", {}) if isinstance(config.get("modules", {}), dict) else {}
    for name in strategies:
        defaults = {
            key: deepcopy(value)
            for key, value in DEFAULT_STRATEGIES.get(name, {}).items()
            if key not in {"module", "description"}
        }
        parameters[name] = defaults

    # A config may optionally carry explicit runtime parameters.  Module-level
    # metadata remains available to the prompt/runtime adapter, while these
    # values are the executable knobs understood by existing strategies.
    explicit_parameters = config.get("parameters", {})
    if isinstance(explicit_parameters, dict):
        for name, values in explicit_parameters.items():
            if name in strategies and isinstance(values, dict):
                parameters.setdefault(name, {}).update(deepcopy(values))

    context_module = modules.get("context", {})
    if isinstance(context_module, dict) and "context_reduction" in strategies:
        if context_module.get("window_size"):
            parameters.setdefault("context_reduction", {})["window_size"] = int(
                context_module["window_size"]
            )

    memory_module = modules.get("memory", {})
    if isinstance(memory_module, dict) and "simple_memory" in strategies:
        if memory_module.get("max_memory_items"):
            parameters.setdefault("simple_memory", {})["max_memory_items"] = int(
                memory_module["max_memory_items"]
            )

    tools_module = modules.get("tools", {})
    if isinstance(tools_module, dict) and "adaptive_tool_trimming" in strategies:
        if tools_module.get("max_chars"):
            parameters.setdefault("adaptive_tool_trimming")["max_chars"] = int(
                tools_module["max_chars"]
            )

    return {
        "strategies": strategies,
        "parameters": parameters,
        "metadata": {
            "source": "meta_strategy_library",
            "efficiency_config_id": config.get("id", ""),
            "efficiency_config": deepcopy(config),
        },
        "config_id": config.get("id", ""),
        "config": deepcopy(config),
        "modules": deepcopy(modules),
    }


def config_strategy_ids(config: dict[str, Any] | None) -> list[str]:
    if not isinstance(config, dict):
        return []
    return _unique(config.get("strategies", []) or [])
