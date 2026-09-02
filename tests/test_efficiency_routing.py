from __future__ import annotations

import pytest

from agent_evolve.efficiency.profile import profile_task_text
from agent_evolve.efficiency.routing import (
    ROUTE_LEVELS,
    normalize_route_level,
    resolve_routing_overrides,
    route_client_env_suffix,
    route_level_from_profile,
)
from agent_evolve.efficiency.library import (
    active_strategy_ids,
    active_config_entries,
    build_library_from_gate,
    load_strategy_library,
    record_config_outcome,
    select_library_config,
    write_strategy_library,
)
from agent_evolve.efficiency.pareto import choose_best_candidate, pareto_frontier
from agent_evolve.efficiency.profiler import (
    normalize_usage,
    result_cost,
    result_strategy_hits,
    summarize_results,
)
from agent_evolve.efficiency.selector import (
    candidate_configs_for_profile,
    config_matches_profile,
    config_to_runtime_config,
    select_runtime_config,
)
from agent_evolve.efficiency.strategies import strategy_prompt_block
from examples.swe_examples_new.run_evoefficient_mvp import _select_strategies


_ROUTE_ENV_KEYS = (
    "AEVOLVE_ROUTE_FORCE_LEVEL",
    "AEVOLVE_ROUTE_MODEL_MAP",
    "AEVOLVE_MODEL_EASY",
    "AEVOLVE_MODEL_HARD",
    "AEVOLVE_MODEL_ID_EASY",
    "AEVOLVE_MODEL_ID_HARD",
    "AEVOLVE_MODEL_PROVIDER_EASY",
    "AEVOLVE_MODEL_PROVIDER_HARD",
)


def _clear_route_env(monkeypatch) -> None:
    for key in _ROUTE_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_whole_task_profile_defaults_to_hard_route(monkeypatch) -> None:
    _clear_route_env(monkeypatch)
    profile = profile_task_text("Summarize this short note.")

    assert profile.difficulty == "easy"
    assert profile.route_level == "hard"

    route = resolve_routing_overrides(model_id="base-model", task_profile=profile.as_dict())

    assert route["route_level"] == "hard"
    assert route["model_id"] == "base-model"
    assert tuple(ROUTE_LEVELS) == ("easy", "hard")


def test_easy_model_is_used_only_for_explicit_easy_step(monkeypatch) -> None:
    _clear_route_env(monkeypatch)
    monkeypatch.setenv("AEVOLVE_MODEL_HARD", "hard-model")
    monkeypatch.setenv("AEVOLVE_MODEL_EASY", "easy-model")

    default_route = resolve_routing_overrides(model_id="base-model")
    easy_step_route = resolve_routing_overrides(model_id="base-model", route_level="easy")

    assert default_route["route_level"] == "hard"
    assert default_route["model_id"] == "hard-model"
    assert easy_step_route["route_level"] == "easy"
    assert easy_step_route["model_id"] == "easy-model"


def test_legacy_medium_route_values_fold_into_hard(monkeypatch) -> None:
    _clear_route_env(monkeypatch)
    monkeypatch.setenv("AEVOLVE_ROUTE_FORCE_LEVEL", "medium")

    assert normalize_route_level("medium") == "hard"
    assert route_client_env_suffix("medium") == "_HARD"
    assert route_level_from_profile({"difficulty": "medium"}) == "hard"

    route = resolve_routing_overrides(model_id="base-model", route_level="easy")

    assert route["route_level"] == "hard"


def test_difficulty_model_routing_prompt_is_step_level() -> None:
    block = strategy_prompt_block(
        {"strategies": ["difficulty_model_routing"], "parameters": {}},
        task_profile={"difficulty": "easy", "route_level": "hard"},
    )

    assert "main agent" in block
    assert "hard model by default" in block
    assert "`route_model` tool" in block
    assert 'route_model(route_level="easy", ...)' in block
    assert "Do not route an entire task to easy" in block
    assert "root-cause diagnosis" in block


def test_stagnation_rescue_stop_prompt_is_high_threshold() -> None:
    block = strategy_prompt_block(
        {"strategies": ["stagnation_rescue_stop"], "parameters": {}},
        task_profile={"difficulty": "hard", "route_level": "hard"},
    )

    assert "Stagnation Rescue Stop" in block
    assert "high-threshold" in block
    assert "rescue at about 90%" in block
    assert "hard stop at about 98%" in block


def test_stagnation_rescue_stop_decision_requires_late_multi_signal() -> None:
    pytest.importorskip("strands")
    from agent_evolve.agents.swe.agent import SweAgent

    params = {
        "rescue_at_step_ratio": 0.90,
        "hard_stop_at_step_ratio": 0.98,
        "min_steps_before_rescue": 45,
        "min_steps_before_hard_stop": 55,
        "stagnant_window": 12,
        "min_signals": 3,
        "exploration_ratio": 0.70,
        "max_unique_recent_inputs": 3,
        "max_unique_recent_files": 2,
        "no_edit_after_steps": 45,
        "rescue_grace_steps": 4,
    }
    early_trace = [
        {"action": "search", "input_summary": "{'command': 'rg foo'}", "file": "a.py"}
        for _ in range(20)
    ]
    late_trace = [
        {
            "action": "search" if i % 2 else "read_file",
            "input_summary": "{'command': 'rg foo'}",
            "file": "a.py",
        }
        for i in range(54)
    ]

    assert SweAgent._stagnation_rescue_decision(
        trace=early_trace,
        max_steps=60,
        candidate_tool="bash",
        candidate_input={"command": "rg foo"},
        params=params,
        state={"rescue_started_at": None, "rescue_count": 0},
    )["action"] == "none"

    rescue = SweAgent._stagnation_rescue_decision(
        trace=late_trace,
        max_steps=60,
        candidate_tool="bash",
        candidate_input={"command": "rg foo"},
        params=params,
        state={"rescue_started_at": None, "rescue_count": 0},
    )

    assert rescue["action"] == "rescue"
    assert len(rescue["signals"]) >= 3


def test_stagnation_rescue_stop_hard_stop_only_after_rescue() -> None:
    pytest.importorskip("strands")
    from agent_evolve.agents.swe.agent import SweAgent

    params = {
        "rescue_at_step_ratio": 0.90,
        "hard_stop_at_step_ratio": 0.98,
        "min_steps_before_rescue": 45,
        "min_steps_before_hard_stop": 55,
        "stagnant_window": 12,
        "min_signals": 3,
        "exploration_ratio": 0.70,
        "max_unique_recent_inputs": 3,
        "max_unique_recent_files": 2,
        "no_edit_after_steps": 45,
        "rescue_grace_steps": 4,
        "max_rescue_messages": 1,
    }
    trace = [
        {"action": "search", "input_summary": "{'command': 'rg foo'}", "file": "a.py"}
        for _ in range(59)
    ]

    hard = SweAgent._stagnation_rescue_decision(
        trace=trace,
        max_steps=60,
        candidate_tool="bash",
        candidate_input={"command": "ls"},
        params=params,
        state={"rescue_started_at": 54, "rescue_count": 1},
    )
    constructive = SweAgent._stagnation_rescue_decision(
        trace=trace,
        max_steps=60,
        candidate_tool="text_editor",
        candidate_input={"command": "str_replace", "path": "a.py"},
        params=params,
        state={"rescue_started_at": 54, "rescue_count": 1},
    )
    submit = SweAgent._stagnation_rescue_decision(
        trace=trace,
        max_steps=60,
        candidate_tool="submit",
        candidate_input={},
        params=params,
        state={"rescue_started_at": 54, "rescue_count": 1},
    )

    assert hard["action"] == "hard_stop"
    assert constructive["action"] == "none"
    assert submit["action"] == "none"


def test_strategy_library_tracks_active_gate_entries(tmp_path) -> None:
    library = build_library_from_gate(
        candidate_strategies=["adaptive_tool_trimming", "difficulty_model_routing"],
        accepted_strategies=["adaptive_tool_trimming"],
        gate={"accepted": True, "reason": "accepted"},
        baseline_summary={"success_rate": 1.0, "avg_tokens": 100},
        candidate_summary={"success_rate": 1.0, "avg_tokens": 80},
        selector_profile={"bottlenecks": ["tool-heavy"]},
    )
    write_strategy_library(tmp_path, library)
    loaded = load_strategy_library(tmp_path)

    assert active_strategy_ids(loaded) == ["adaptive_tool_trimming"]
    statuses = {entry["strategy"]: entry["status"] for entry in loaded["strategies"]}
    assert statuses["difficulty_model_routing"] == "rejected"


def test_strategy_library_quarantines_train_accepted_heldout_regressions(tmp_path) -> None:
    library = build_library_from_gate(
        candidate_strategies=["observation_masking", "adaptive_tool_trimming"],
        accepted_strategies=[],
        gate={"accepted": True, "reason": "accepted"},
        baseline_summary={"success_rate": 1.0, "avg_tokens": 100},
        candidate_summary={"success_rate": 1.0, "avg_tokens": 70},
        reuse_summary={"success_rate": 0.5, "avg_tokens": 50},
        selector_profile={"bottlenecks": ["context-heavy"]},
        status_reason="heldout_verdict=success_regressed",
    )
    write_strategy_library(tmp_path, library)
    loaded = load_strategy_library(tmp_path)

    assert active_strategy_ids(loaded) == []
    assert loaded["active_strategies"] == []
    statuses = {entry["strategy"]: entry["status"] for entry in loaded["strategies"]}
    assert statuses == {
        "observation_masking": "quarantined",
        "adaptive_tool_trimming": "quarantined",
    }


def test_mvp_candidate_strategy_modes_are_explicit() -> None:
    profile = {"bottlenecks": ["context-heavy", "reasoning-heavy", "tool-heavy"]}

    assert _select_strategies(profile, mode="bottleneck") == [
        "observation_masking",
        "verbatim_compaction",
        "simple_memory",
        "cognitive_router",
        "adaptive_tool_trimming",
    ]
    assert "difficulty_model_routing" in _select_strategies(profile, mode="safe")
    assert "skill_compilation" in _select_strategies(profile, mode="all")
    assert _select_strategies(
        profile,
        mode="explicit",
        explicit_strategies=["difficulty_model_routing", "unknown", "difficulty_model_routing"],
    ) == ["difficulty_model_routing"]


def test_profiler_records_strategy_hits_and_token_quality() -> None:
    result = {
        "success": True,
        "score": 1.0,
        "cumulative_input_tokens": 0,
        "cumulative_output_tokens": 0,
        "strategies": ["adaptive_tool_trimming", "difficulty_model_routing"],
        "steps": [
            {
                "tool": "bash",
                "result_snippet": "... [EvoEfficient adaptive_tool_trimming truncated long output] ...",
            },
            {
                "tool": "text_editor",
                "result_snippet": "marker was outside the stored snippet",
                "efficiency_markers": ["verbatim_compaction"],
            },
            {
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "route_model_calls": [
                    {"route_level": "easy", "status": "success"},
                    {"route_level": "hard", "status": "kept_in_main"},
                ],
                "early_stop_events": [
                    {"type": "rescue", "step": 54},
                    {"type": "hard_stop", "step": 59},
                ],
            },
        ],
    }

    hits = result_strategy_hits(result)
    summary = summarize_results([result])

    assert hits["adaptive_tool_trimming"] == 1
    assert hits["verbatim_compaction"] == 1
    assert hits["difficulty_model_routing"] == 1
    assert hits["stagnation_rescue_stop"] == 2
    assert summary["token_record_quality"] == "missing"
    assert summary["strategy_hits"]["adaptive_tool_trimming"]["hits"] == 1


def test_profiler_splits_token_average_and_full_cost() -> None:
    results = [
        {
            "success": True,
            "cumulative_input_tokens": 100,
            "cumulative_output_tokens": 20,
            "cumulative_cache_read_input_tokens": 50,
        },
        {
            "success": False,
            "cumulative_input_tokens": 80,
            "cumulative_output_tokens": 20,
            "cumulative_cache_write_input_tokens": 40,
        },
        {
            "success": True,
            "total_tokens": 60,
        },
    ]

    summary = summarize_results(results)

    assert result_cost(results[0]) == 170
    assert summary["tokens"] == 280
    assert summary["cache_read_input_tokens"] == 50
    assert summary["cache_write_input_tokens"] == 40
    assert summary["avg_tokens"] == 93.33
    assert summary["avg_tokens_success"] == 90.0
    assert summary["avg_cost"] == 123.33
    assert summary["avg_cost_success"] == 115.0
    assert summary["cost_of_pass"] == 185.0


def test_profiler_uses_route_specific_model_prices() -> None:
    price_profile = {
        "model_prices": {
            "currency": "USD",
            "routes": {
                "hard": {
                    "input_per_1m": 10,
                    "output_per_1m": 20,
                    "cache_read_per_1m": 1,
                    "cache_write_per_1m": 12,
                },
                "easy": {
                    "input_per_1m": 1,
                    "output_per_1m": 2,
                    "cache_read_per_1m": 0.1,
                    "cache_write_per_1m": 1.2,
                },
            },
        }
    }

    hard = normalize_usage(
        {
            "input_tokens": 1_000_000,
            "output_tokens": 500_000,
            "cache_read_input_tokens": 100_000,
        },
        price_profile=price_profile,
        model_id="hard-model",
        route_level="hard",
    )
    easy = normalize_usage(
        {
            "input_tokens": 1_000_000,
            "output_tokens": 500_000,
            "cache_read_input_tokens": 100_000,
        },
        price_profile=price_profile,
        model_id="easy-model",
        route_level="easy",
    )

    assert hard["cost_units"] == pytest.approx(20.1)
    assert easy["cost_units"] == pytest.approx(2.01)
    assert hard["cost_currency"] == "USD"
    assert hard["cost_priced"] is True
    assert "route:hard" in hard["cost_price_source"]
    assert "route:easy" in easy["cost_price_source"]


def test_profiler_adds_delegated_route_model_cost_once() -> None:
    result = {
        "success": True,
        "steps": [
            {
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "cost_units": 12.0,
                    "model_id": "hard-model",
                    "route_level": "hard",
                    "cost_priced": True,
                },
                "route_model_calls": [
                    {
                        "route_level": "easy",
                        "status": "success",
                        "model_id": "easy-model",
                        "cost_units": 0.5,
                    }
                ],
            }
        ],
    }

    summary = summarize_results([result])

    assert result_cost(result) == pytest.approx(12.5)
    assert summary["total_cost"] == pytest.approx(12.5)
    assert summary["main_model_cost"] == pytest.approx(12.0)
    assert summary["route_model_cost"] == pytest.approx(0.5)
    assert summary["easy_model_cost"] == pytest.approx(0.5)

    result["steps"][-1]["usage"]["route_model_cost_units"] = 0.5
    result["steps"][-1]["usage"]["cost_units"] = 12.5
    assert result_cost(result) == pytest.approx(12.5)


def test_profiler_does_not_double_count_top_level_route_cost_without_steps() -> None:
    result = {
        "success": True,
        "cumulative_input_tokens": 100,
        "cumulative_output_tokens": 20,
        "cost_units": 12.5,
        "main_cost_units": 12.0,
        "route_model_cost_units": 0.5,
        "route_model_calls": [
            {
                "route_level": "easy",
                "status": "success",
                "cost_units": 0.5,
            }
        ],
    }

    summary = summarize_results([result])

    assert result_cost(result) == pytest.approx(12.5)
    assert summary["total_cost"] == pytest.approx(12.5)
    assert summary["main_model_cost"] == pytest.approx(12.0)
    assert summary["route_model_cost"] == pytest.approx(0.5)


def test_profiler_marks_estimated_usage_without_zero_token_penalty() -> None:
    summary = summarize_results([
        {
            "success": False,
            "cumulative_input_tokens": 120,
            "cumulative_output_tokens": 10,
            "usage_record_quality": "estimated",
            "usage_estimated": True,
        }
    ])

    assert summary["zero_token_tasks"] == 0
    assert summary["estimated_usage_tasks"] == 1
    assert summary["token_record_quality"] == "estimated"


def test_meta_selector_generates_independent_configs_and_runtime_parameters() -> None:
    profile = {
        "task_type": "coding",
        "difficulty": "hard",
        "context_pressure": "hard",
        "tool_pressure": "hard",
        "bottlenecks": ["context-heavy", "tool-heavy"],
    }
    configs = candidate_configs_for_profile(profile, profile, mode="meta")

    assert configs
    assert all(config["type"] == "efficiency_config" for config in configs)
    assert any(config["id"] == "swe.coding.observation_compact.v1" for config in configs)
    assert all(isinstance(config.get("modules"), dict) for config in configs)

    runtime = config_to_runtime_config(configs[0])
    assert runtime["config_id"] == configs[0]["id"]
    assert runtime["strategies"]
    assert all(name in runtime["parameters"] for name in runtime["strategies"])


def test_selector_matches_swe_debug_profile_to_coding_config() -> None:
    config = {
        "id": "coding-context",
        "status": "active",
        "strategies": ["observation_masking"],
        "trigger": {
            "task_type": "coding",
            "bottlenecks": ["context-heavy"],
        },
        "stats": {"success_rate": 1.0, "cost_delta_pct": -0.2},
    }
    task_profile = {
        "task_type": "debug",
        "difficulty": "hard",
        "context_pressure": "hard",
    }

    matches, score = config_matches_profile(config, task_profile)

    assert matches
    assert score > 0


def test_structured_library_only_deploys_explicit_active_config(tmp_path) -> None:
    configs = [
        {
            "id": "active-config",
            "status": "active",
            "strategies": ["simple_memory"],
            "trigger": {"task_type": "coding"},
        },
        {
            "id": "other-config",
            "status": "active",
            "strategies": ["difficulty_model_routing"],
            "trigger": {"task_type": "coding"},
        },
    ]
    library = {
        "configs": configs,
        "active_configs": ["active-config"],
        "strategies": [],
    }

    assert [c["id"] for c in active_config_entries(library)] == ["active-config"]
    selected = select_runtime_config(
        library,
        {"task_type": "coding", "difficulty": "hard"},
    )
    assert selected["id"] == "active-config"

    quarantined = {
        "configs": configs,
        "active_configs": [],
        "strategies": [],
    }
    assert select_library_config(
        quarantined,
        {"task_type": "coding", "difficulty": "hard"},
    ) is None


def test_record_config_outcome_updates_online_statistics() -> None:
    library = {
        "configs": [
            {
                "id": "cfg",
                "stats": {"trials": 1, "successes": 1, "tokens": 100},
            }
        ]
    }
    record_config_outcome(
        library,
        config_id="cfg",
        result={
            "success": False,
            "total_tokens": 60,
            "strategy_hits": {"simple_memory": {"hits": 2}},
        },
    )

    stats = library["configs"][0]["stats"]
    assert stats["trials"] == 2
    assert stats["successes"] == 1
    assert stats["success_rate"] == 0.5
    assert stats["tokens"] == 160
    assert stats["avg_tokens"] == 80.0
    assert stats["hit_count"] == 2


def test_success_first_pareto_selection_prefers_lowest_cost_of_pass() -> None:
    candidates = [
        {
            "config_id": "expensive",
            "summary": {"success_rate": 1.0, "cost_of_pass": 100, "avg_tool_calls": 10},
        },
        {
            "config_id": "cheap",
            "summary": {"success_rate": 1.0, "cost_of_pass": 80, "avg_tool_calls": 9},
        },
        {
            "config_id": "unsafe",
            "summary": {"success_rate": 0.5, "cost_of_pass": 20, "avg_tool_calls": 2},
        },
    ]

    frontier = pareto_frontier(candidates)
    selected = choose_best_candidate(
        frontier,
        baseline={"success_rate": 1.0, "cost_of_pass": 120},
    )

    # A cheaper but unsafe candidate remains on the multi-objective frontier;
    # the success-first chooser removes it against the baseline success floor.
    assert {item["config_id"] for item in frontier} == {"cheap", "unsafe"}
    assert selected["config_id"] == "cheap"
