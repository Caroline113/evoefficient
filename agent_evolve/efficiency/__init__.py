"""Lightweight EvoEfficient utilities.

This package is intentionally small: it provides just enough strategy
configuration, profiling, gating, and plotting support for the SWE-bench MVP
experiments without changing the core A-Evolve algorithms.
"""

from .gate import GateConfig, gate_candidate
from .library import (
    active_strategy_ids,
    apply_library_to_workspace,
    build_library_from_gate,
    load_strategy_library,
    write_strategy_library,
)
from .profile import TaskProfile, profile_task_text
from .pareto import choose_best_candidate, pareto_frontier
from .profiler import (
    # build_cost_profile,
    # result_cost,
    # result_strategy_hits,
    # summarize_results,
    # summarize_strategy_hits,
    build_cost_profile,
    normalize_usage,
    result_cost,
    result_route_model_cost,
    result_tokens,
    result_strategy_hits,
    resolve_token_prices,
    summarize_results,
    summarize_strategy_hits,
    usage_cost_units,
)
from .routing import resolve_routing_overrides
from .selector import (
    candidate_configs_for_profile,
    config_to_runtime_config,
    config_strategy_ids,
    config_matches_profile,
    select_primary_config,
    select_runtime_config,
)
from .strategies import (
    DEFAULT_STRATEGIES,
    STRATEGY_ORDER,
    apply_strategy_set_to_workspace,
    load_active_config,
    write_active_config,
)

__all__ = [
    "DEFAULT_STRATEGIES",
    "STRATEGY_ORDER",
    "GateConfig",
    "TaskProfile",
    "active_strategy_ids",
    "apply_strategy_set_to_workspace",
    "apply_library_to_workspace",
    "build_library_from_gate",
    "build_cost_profile",
    "candidate_configs_for_profile",
    "choose_best_candidate",
    "config_matches_profile",
    "config_strategy_ids",
    "config_to_runtime_config",
    "gate_candidate",
    "load_strategy_library",
    "load_active_config",
    # "profile_task_text",
    # "pareto_frontier",
    # "result_cost",
    # "result_strategy_hits",
    # "resolve_routing_overrides",
    # "summarize_results",
    # "summarize_strategy_hits",
    # "select_primary_config",
    "pareto_frontier",
    "normalize_usage",
    "result_cost",
    "result_route_model_cost",
    "result_tokens",
    "result_strategy_hits",
    "resolve_token_prices",
    "resolve_routing_overrides",
    "summarize_results",
    "summarize_strategy_hits",
    "usage_cost_units",
    "select_primary_config",
    "select_runtime_config",
    "write_strategy_library",
    "write_active_config",
]
