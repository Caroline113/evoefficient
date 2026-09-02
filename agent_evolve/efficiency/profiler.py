"""Profiling and summary helpers for EvoEfficient experiments."""

from __future__ import annotations

import ast
import math
import os
import json
from typing import Any


PRICE_SCALE = 1_000_000.0
TOKEN_EQUIVALENT_PRICE_PER_1M = 1_000_000.0

_PRICE_ALIASES = {
    "input_price_per_1m": (
        "input_price_per_1m",
        "input_per_1m",
        "prompt_price_per_1m",
        "prompt_per_1m",
    ),
    "output_price_per_1m": (
        "output_price_per_1m",
        "output_per_1m",
        "completion_price_per_1m",
        "completion_per_1m",
    ),
    "cache_read_price_per_1m": (
        "cache_read_price_per_1m",
        "cache_read_per_1m",
        "cache_read_input_price_per_1m",
        "cache_read_input_per_1m",
        "cached_input_price_per_1m",
        "cached_input_per_1m",
    ),
    "cache_write_price_per_1m": (
        "cache_write_price_per_1m",
        "cache_write_per_1m",
        "cache_creation_price_per_1m",
        "cache_creation_per_1m",
        "cache_creation_input_price_per_1m",
        "cache_creation_input_per_1m",
    ),
}


def _as_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _usage_value(raw: Any, *names: str) -> int:
    for name in names:
        if isinstance(raw, dict):
            if name in raw and raw.get(name) is not None:
                return _as_int(raw.get(name))
        elif hasattr(raw, name):
            value = getattr(raw, name)
            if value is not None:
                return _as_int(value)
    return 0


def _usage_float_value(raw: Any, *names: str) -> float:
    for name in names:
        if isinstance(raw, dict):
            if name in raw and raw.get(name) is not None:
                return _as_float(raw.get(name))
        elif hasattr(raw, name):
            value = getattr(raw, name)
            if value is not None:
                return _as_float(value)
    return 0.0


def _cost_weight(env_name: str, default: float = 1.0) -> float:
    return _as_float(os.environ.get(env_name, default))


def _json_env(name: str) -> dict[str, Any] | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _profile_price_root(price_profile: dict[str, Any] | None) -> dict[str, Any]:
    profile = price_profile or {}
    if not isinstance(profile, dict):
        return {}
    nested = profile.get("model_prices")
    return nested if isinstance(nested, dict) else profile


def _price_value(data: dict[str, Any], canonical_key: str) -> float | None:
    for key in _PRICE_ALIASES[canonical_key]:
        if key in data and data.get(key) is not None:
            return _as_float(data.get(key))
    return None


def _price_from_mapping(
    mapping: dict[str, Any],
    *,
    model_id: str,
    route_level: str,
    provider: str,
) -> tuple[dict[str, float], str, str] | None:
    profile = _profile_price_root(mapping)
    candidates: list[tuple[str, Any]] = []
    models = profile.get("models", {})
    if isinstance(models, dict):
        if provider and model_id:
            candidates.append((f"model:{provider}:{model_id}", models.get(f"{provider}:{model_id}")))
        if model_id:
            candidates.append((f"model:{model_id}", models.get(model_id)))
    routes = profile.get("routes", {})
    if isinstance(routes, dict) and route_level:
        candidates.append((f"route:{route_level}", routes.get(route_level)))
    if route_level:
        candidates.append((f"route:{route_level}", profile.get(route_level)))
    candidates.append(("default", profile.get("default")))

    for source, value in candidates:
        if not isinstance(value, dict):
            continue
        input_price = _price_value(value, "input_price_per_1m")
        output_price = _price_value(value, "output_price_per_1m")
        if input_price is None and output_price is None:
            continue
        input_price = float(input_price or 0.0)
        output_price = float(output_price or 0.0)
        cache_read = _price_value(value, "cache_read_price_per_1m")
        cache_write = _price_value(value, "cache_write_price_per_1m")
        if (
            input_price
            + output_price
            + float(cache_read or 0.0)
            + float(cache_write or 0.0)
        ) <= 0:
            continue
        prices = {
            "input_price_per_1m": input_price,
            "output_price_per_1m": output_price,
            "cache_read_price_per_1m": float(cache_read if cache_read is not None else input_price),
            "cache_write_price_per_1m": float(cache_write if cache_write is not None else input_price),
        }
        currency = str(value.get("currency") or profile.get("currency") or "USD")
        return prices, source, currency
    return None


def _price_from_env(*, route_level: str) -> tuple[dict[str, float], str, str] | None:
    def pick(kind: str) -> str | None:
        suffix = f"_{route_level.upper()}" if route_level else ""
        names = (
            f"AEVOLVE_{kind}_PRICE_PER_1M{suffix}",
            f"AEVOLVE_{kind}_COST_PER_1M{suffix}",
            f"AEVOLVE_{kind}_PRICE_PER_1M",
            f"AEVOLVE_{kind}_COST_PER_1M",
        )
        for name in names:
            value = os.environ.get(name)
            if value:
                return value
        return None

    input_raw = pick("INPUT")
    output_raw = pick("OUTPUT")
    if input_raw is None and output_raw is None:
        return None
    input_price = _as_float(input_raw)
    output_price = _as_float(output_raw)
    cache_read_raw = pick("CACHE_READ")
    cache_write_raw = pick("CACHE_WRITE")
    prices = {
        "input_price_per_1m": input_price,
        "output_price_per_1m": output_price,
        "cache_read_price_per_1m": _as_float(cache_read_raw) if cache_read_raw is not None else input_price,
        "cache_write_price_per_1m": _as_float(cache_write_raw) if cache_write_raw is not None else input_price,
    }
    currency = os.environ.get("AEVOLVE_COST_CURRENCY", "USD")
    source = f"env:route:{route_level or 'default'}"
    return prices, source, currency


def _preserved_price_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    """Return explicit price metadata already attached to a usage record."""
    if not isinstance(raw, dict):
        return {}
    preserved: dict[str, Any] = {}
    for key in (
        "input_price_per_1m",
        "output_price_per_1m",
        "cache_read_price_per_1m",
        "cache_write_price_per_1m",
    ):
        value = _price_value(raw, key)
        if value is not None:
            preserved[key] = float(value)
    for key in ("cost_currency", "cost_price_source", "cost_priced"):
        if key in raw and raw.get(key) is not None:
            preserved[key] = raw.get(key)
    has_price_signal = any(
        float(preserved.get(key, 0.0) or 0.0) > 0.0
        for key in (
            "input_price_per_1m",
            "output_price_per_1m",
            "cache_read_price_per_1m",
            "cache_write_price_per_1m",
        )
    )
    has_source_signal = bool(
        str(preserved.get("cost_currency") or "").strip()
        or str(preserved.get("cost_price_source") or "").strip()
        or preserved.get("cost_priced") is True
    )
    if not has_price_signal and not has_source_signal:
        return {}
    if preserved and "cost_priced" in preserved:
        preserved["cost_priced"] = bool(preserved["cost_priced"])
    return preserved


def resolve_token_prices(
    *,
    model_id: str | None = None,
    route_level: str | None = None,
    provider: str | None = None,
    price_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve per-1M-token prices for one model call."""
    model = str(model_id or "")
    route = str(route_level or "")
    provider_name = str(provider or "")

    for mapping, label in (
        (price_profile, "profile"),
        (_json_env("AEVOLVE_MODEL_PRICE_MAP"), "env:AEVOLVE_MODEL_PRICE_MAP"),
    ):
        if isinstance(mapping, dict):
            resolved = _price_from_mapping(
                mapping,
                model_id=model,
                route_level=route,
                provider=provider_name,
            )
            if resolved:
                prices, source, currency = resolved
                prices.update(
                    {
                        "cost_currency": currency,
                        "cost_price_source": f"{label}:{source}",
                        "cost_priced": True,
                    }
                )
                return prices

    env_resolved = _price_from_env(route_level=route)
    if env_resolved:
        prices, source, currency = env_resolved
        prices.update(
            {
                "cost_currency": currency,
                "cost_price_source": source,
                "cost_priced": True,
            }
        )
        return prices

    return {
        "input_price_per_1m": TOKEN_EQUIVALENT_PRICE_PER_1M,
        "output_price_per_1m": TOKEN_EQUIVALENT_PRICE_PER_1M,
        "cache_read_price_per_1m": TOKEN_EQUIVALENT_PRICE_PER_1M,
        "cache_write_price_per_1m": TOKEN_EQUIVALENT_PRICE_PER_1M,
        "cost_currency": "token_equivalent",
        "cost_price_source": "token_equivalent_fallback",
        "cost_priced": False,
    }


def usage_cost_units(
    usage: dict[str, Any],
    *,
    model_id: str | None = None,
    route_level: str | None = None,
    provider: str | None = None,
    price_profile: dict[str, Any] | None = None,
) -> float:
    """Return cost using model prices, falling back to token-equivalent units."""
    prices = resolve_token_prices(
        model_id=model_id or usage.get("model_id"),
        route_level=route_level or usage.get("route_level"),
        provider=provider or usage.get("provider"),
        price_profile=price_profile,
    )
    if prices.get("cost_priced"):
        return (
            _as_int(usage.get("input_tokens")) * float(prices["input_price_per_1m"])
            + _as_int(usage.get("output_tokens")) * float(prices["output_price_per_1m"])
            + _as_int(usage.get("cache_read_input_tokens")) * float(prices["cache_read_price_per_1m"])
            + _as_int(usage.get("cache_write_input_tokens")) * float(prices["cache_write_price_per_1m"])
        ) / PRICE_SCALE

    # Backward-compatible token-equivalent fallback.  These weights are useful
    # for old experiments without a price table.
    input_weight = _cost_weight("AEVOLVE_INPUT_COST_WEIGHT")
    output_weight = _cost_weight("AEVOLVE_OUTPUT_COST_WEIGHT")
    cache_read_weight = _cost_weight("AEVOLVE_CACHE_READ_COST_WEIGHT")
    cache_write_weight = _cost_weight("AEVOLVE_CACHE_WRITE_COST_WEIGHT")
    base_cost = (
        _as_int(usage.get("input_tokens")) * input_weight
        + _as_int(usage.get("output_tokens")) * output_weight
        + _as_int(usage.get("cache_read_input_tokens")) * cache_read_weight
        + _as_int(usage.get("cache_write_input_tokens")) * cache_write_weight
    )
    if base_cost == 0 and _as_int(usage.get("total_tokens")) > 0:
        return float(_as_int(usage.get("total_tokens")))
    return base_cost


def normalize_usage(
    raw: Any | None = None,
    *,
    price_profile: dict[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Normalize provider/Strands usage into one experiment schema."""
    raw = raw or {}
    raw_dict = raw if isinstance(raw, dict) else {}
    input_tokens = _usage_value(
        raw,
        "input_tokens",
        "inputTokens",
        "prompt_tokens",
        "promptTokens",
    )
    output_tokens = _usage_value(
        raw,
        "output_tokens",
        "outputTokens",
        "completion_tokens",
        "completionTokens",
    )
    total_tokens = _usage_value(raw, "total_tokens", "totalTokens")
    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens
    cache_read = _usage_value(
        raw,
        "cache_read_input_tokens",
        "cacheReadInputTokens",
        "cache_read_tokens",
        "cacheReadTokens",
        "cached_tokens",
        "cachedTokens",
    )
    cache_write = _usage_value(
        raw,
        "cache_write_input_tokens",
        "cacheWriteInputTokens",
        "cache_creation_input_tokens",
        "cacheCreationInputTokens",
        "cache_write_tokens",
        "cacheWriteTokens",
    )

    usage: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cache_read_input_tokens": cache_read,
        "cache_write_input_tokens": cache_write,
    }
    for key in ("model_id", "provider", "route_level"):
        value = overrides.get(key, raw_dict.get(key))
        if value:
            usage[key] = str(value)

    prices = resolve_token_prices(
        model_id=usage.get("model_id"),
        route_level=usage.get("route_level"),
        provider=usage.get("provider"),
        price_profile=price_profile,
    )
    usage.update(prices)
    usage.update(_preserved_price_metadata(raw_dict))

    raw_cost = _usage_float_value(raw, "cost_units", "costUnits", "total_cost", "totalCost")
    if not raw_cost and overrides.get("cost_units") is not None:
        raw_cost = _as_float(overrides.get("cost_units"))
    usage["cost_units"] = raw_cost if raw_cost else usage_cost_units(
        usage,
        price_profile=price_profile,
    )

    for key in (
        "usage_record_quality",
        "usage_source",
        "usage_estimated",
        "main_cost_units",
        "route_model_cost_units",
    ):
        if key in raw_dict:
            usage[key] = raw_dict[key]
    usage.update({key: value for key, value in overrides.items() if value is not None})
    return usage


def result_steps(result: dict[str, Any]) -> list[dict[str, Any]]:
    steps = result.get("steps", [])
    return steps if isinstance(steps, list) else []


def result_summary_step(result: dict[str, Any]) -> dict[str, Any]:
    for step in reversed(result_steps(result)):
        if isinstance(step, dict) and (
            "usage" in step
            or "active_efficiency_strategies" in step
            or "route_model_calls" in step
        ):
            return step
    return {}


def result_usage(result: dict[str, Any]) -> dict[str, Any]:
    """Return normalized task-level usage from either top-level or steps."""
    summary = result_summary_step(result)
    raw_usage: dict[str, Any] = {}
    if isinstance(summary, dict) and isinstance(summary.get("usage"), dict):
        raw_usage.update(summary["usage"])

    def _merged_value(result_key: str, usage_key: str) -> Any:
        result_value = result.get(result_key)
        usage_value = raw_usage.get(usage_key, 0)
        if _as_float(result_value) > 0 or _as_float(usage_value) == 0:
            return result_value
        return usage_value

    raw_usage.update(
        {
            "input_tokens": _merged_value("cumulative_input_tokens", "input_tokens"),
            "output_tokens": _merged_value("cumulative_output_tokens", "output_tokens"),
            "total_tokens": _merged_value("total_tokens", "total_tokens"),
            "cache_read_input_tokens": _merged_value(
                "cumulative_cache_read_input_tokens",
                "cache_read_input_tokens",
            ),
            "cache_write_input_tokens": _merged_value(
                "cumulative_cache_write_input_tokens",
                "cache_write_input_tokens",
            ),
            "cost_units": _merged_value("cost_units", "cost_units"),
            "main_cost_units": _merged_value("main_cost_units", "main_cost_units"),
            "route_model_cost_units": _merged_value(
                "route_model_cost_units",
                "route_model_cost_units",
            ),
        }
    )
    route_cost = result_route_model_cost(result)
    if route_cost > 0 and _as_float(raw_usage.get("route_model_cost_units")) <= 0:
        main_cost = _as_float(raw_usage.get("cost_units"))
        raw_usage["main_cost_units"] = main_cost
        raw_usage["route_model_cost_units"] = route_cost
        raw_usage["cost_units"] = main_cost + route_cost
    quality = result.get("usage_record_quality")
    source = result.get("usage_source")
    estimated = result.get("usage_estimated")
    if not quality and isinstance(summary, dict):
        quality = summary.get("usage_record_quality")
    if not source and isinstance(summary, dict):
        source = summary.get("usage_source")
    if estimated is None and isinstance(summary, dict):
        estimated = summary.get("usage_estimated")
    return normalize_usage(
        raw_usage,
        usage_record_quality=quality,
        usage_source=source,
        usage_estimated=estimated,
    )


def result_tokens(result: dict[str, Any]) -> int:
    """Input + output token count, excluding cache counters."""
    return _as_int(result_usage(result).get("total_tokens", 0))


def result_cost(result: dict[str, Any]) -> float:
    """Weighted token-equivalent cost including cache counters."""
    return float(result_usage(result).get("cost_units", 0.0) or 0.0)


def result_route_model_cost(result: dict[str, Any]) -> float:
    summary = result_summary_step(result)
    route_calls = summary.get("route_model_calls", []) if isinstance(summary, dict) else []
    if not isinstance(route_calls, list) or not route_calls:
        route_calls = result.get("route_model_calls", []) or []
    total = 0.0
    for call in route_calls if isinstance(route_calls, list) else []:
        if not isinstance(call, dict):
            continue
        total += _as_float(call.get("cost_units", 0.0))
    return total


def result_route_model_cost_by_level(result: dict[str, Any]) -> dict[str, float]:
    summary = result_summary_step(result)
    route_calls = summary.get("route_model_calls", []) if isinstance(summary, dict) else []
    if not isinstance(route_calls, list) or not route_calls:
        route_calls = result.get("route_model_calls", []) or []
    by_level: dict[str, float] = {}
    for call in route_calls if isinstance(route_calls, list) else []:
        if not isinstance(call, dict):
            continue
        level = str(call.get("route_level") or "unknown")
        by_level[level] = by_level.get(level, 0.0) + _as_float(call.get("cost_units", 0.0))
    return by_level


def result_usage_quality(result: dict[str, Any]) -> str:
    usage = result_usage(result)
    quality = str(usage.get("usage_record_quality") or "").strip().lower()
    if quality:
        return quality
    if result_tokens(result) > 0 or result_cost(result) > 0:
        return "ok"
    return "missing"


def result_strategy_hits(result: dict[str, Any]) -> dict[str, int]:
    """Return best-effort per-strategy hit counters from a task result."""
    hits: dict[str, int] = {}
    for strategy in result.get("strategies", []) or []:
        hits[str(strategy)] = 0

    steps = result_steps(result)
    summary = result_summary_step(result)
    route_calls = summary.get("route_model_calls", []) if isinstance(summary, dict) else []
    early_stop_events = summary.get("early_stop_events", []) if isinstance(summary, dict) else []
    if isinstance(route_calls, list):
        easy_calls = sum(
            1
            for call in route_calls
            if isinstance(call, dict)
            and call.get("route_level") == "easy"
            and call.get("status") == "success"
        )
        if easy_calls:
            hits["difficulty_model_routing"] = hits.get("difficulty_model_routing", 0) + easy_calls
    if isinstance(early_stop_events, list) and early_stop_events:
        hits["stagnation_rescue_stop"] = hits.get("stagnation_rescue_stop", 0) + len(
            [event for event in early_stop_events if isinstance(event, dict)]
        )

    def read_skill_name(step: dict[str, Any]) -> str:
        name = str(step.get("skill_name") or "").strip()
        if name:
            return name
        raw = step.get("input_summary", "")
        if not raw:
            return ""
        try:
            parsed = ast.literal_eval(str(raw))
        except (SyntaxError, ValueError):
            return ""
        if not isinstance(parsed, dict):
            return ""
        return str(
            parsed.get("skill_name")
            or parsed.get("name")
            or parsed.get("skill")
            or ""
        ).strip()

    for step in steps:
        if not isinstance(step, dict):
            continue
        tool = str(step.get("tool", ""))
        snippet = str(step.get("result_snippet", ""))
        markers = step.get("efficiency_markers", [])
        marker_names: set[str] = set()
        if isinstance(markers, list):
            for marker in markers:
                name = str(marker)
                if name:
                    marker_names.add(name)
                    hits[name] = hits.get(name, 0) + 1
        if "adaptive_tool_trimming" not in marker_names and "EvoEfficient adaptive_tool_trimming" in snippet:
            hits["adaptive_tool_trimming"] = hits.get("adaptive_tool_trimming", 0) + 1
        if "verbatim_compaction" not in marker_names and "EvoEfficient verbatim_compaction" in snippet:
            hits["verbatim_compaction"] = hits.get("verbatim_compaction", 0) + 1
        if "observation_masking" not in marker_names and "EvoEfficient observation_masking" in snippet:
            hits["observation_masking"] = hits.get("observation_masking", 0) + 1
        if "observation_filter" not in marker_names and "EvoEfficient observation_filter" in snippet:
            hits["observation_filter"] = hits.get("observation_filter", 0) + 1
        if tool == "read_skill":
            hits["skill_compilation"] = hits.get("skill_compilation", 0) + 1
            skill_name = read_skill_name(step)
            if skill_name:
                read_key = f"read_skill:{skill_name}"
                if read_key not in marker_names:
                    hits[read_key] = hits.get(read_key, 0) + 1
                if skill_name.startswith("emergent_gen"):
                    emergent_key = f"emergent_skill:{skill_name}"
                    if emergent_key not in marker_names:
                        hits[emergent_key] = hits.get(emergent_key, 0) + 1
        if tool == "route_model":
            hits["difficulty_model_routing"] = hits.get("difficulty_model_routing", 0) + 1

    if result.get("max_input_tokens_per_turn", 0):
        active = set(result.get("strategies", []) or [])
        for name in ("context_reduction", "simple_memory", "plan_template_cache", "cognitive_router"):
            if name in active:
                hits.setdefault(name, 0)

    return hits


def summarize_strategy_hits(results: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    tasks_with_hit: dict[str, int] = {}
    for result in results:
        task_hits = result_strategy_hits(result)
        for name, count in task_hits.items():
            counts[name] = counts.get(name, 0) + int(count)
            if count > 0:
                tasks_with_hit[name] = tasks_with_hit.get(name, 0) + 1

    tasks = len(results)
    return {
        name: {
            "hits": count,
            "tasks_with_hit": tasks_with_hit.get(name, 0),
            "task_hit_rate": round(tasks_with_hit.get(name, 0) / tasks, 6) if tasks else 0.0,
        }
        for name, count in sorted(counts.items())
    }


def summarize_skill_usage(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate read_skill/emergent-skill usage counters across task results."""
    read_counts: dict[str, int] = {}
    read_task_counts: dict[str, int] = {}
    emergent_counts: dict[str, int] = {}
    emergent_task_counts: dict[str, int] = {}
    for result in results:
        task_read: dict[str, int] = {}
        task_emergent: dict[str, int] = {}
        for key, count in result_strategy_hits(result).items():
            try:
                value = int(count)
            except (TypeError, ValueError):
                value = 0
            if value <= 0:
                continue
            if key.startswith("read_skill:"):
                name = key.removeprefix("read_skill:")
                task_read[name] = task_read.get(name, 0) + value
            elif key.startswith("emergent_skill:"):
                name = key.removeprefix("emergent_skill:")
                task_emergent[name] = task_emergent.get(name, 0) + value
        for name, value in task_read.items():
            read_counts[name] = read_counts.get(name, 0) + value
            read_task_counts[name] = read_task_counts.get(name, 0) + 1
        for name, value in task_emergent.items():
            emergent_counts[name] = emergent_counts.get(name, 0) + value
            emergent_task_counts[name] = emergent_task_counts.get(name, 0) + 1

    return {
        "total_read_skill_calls": sum(read_counts.values()),
        "tasks_with_read_skill": sum(
            1 for result in results
            if any(key.startswith("read_skill:") and count for key, count in result_strategy_hits(result).items())
        ),
        "read_skill_counts": dict(sorted(read_counts.items())),
        "read_skill_task_counts": dict(sorted(read_task_counts.items())),
        "total_emergent_skill_calls": sum(emergent_counts.values()),
        "tasks_with_emergent_skill": sum(
            1 for result in results
            if any(key.startswith("emergent_skill:") and count for key, count in result_strategy_hits(result).items())
        ),
        "emergent_skill_counts": dict(sorted(emergent_counts.items())),
        "emergent_skill_task_counts": dict(sorted(emergent_task_counts.items())),
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = len(results)
    success = sum(1 for r in results if r.get("success"))
    score = sum(float(r.get("score", 1.0 if r.get("success") else 0.0) or 0.0) for r in results)
    tokens = sum(result_tokens(r) for r in results)
    success_tokens = sum(result_tokens(r) for r in results if r.get("success"))
    total_cost = sum(result_cost(r) for r in results)
    success_cost = sum(result_cost(r) for r in results if r.get("success"))
    route_model_cost = sum(result_route_model_cost(r) for r in results)
    route_model_cost_success = sum(result_route_model_cost(r) for r in results if r.get("success"))
    main_model_cost = sum(
        _as_float(result_usage(r).get("main_cost_units"))
        if _as_float(result_usage(r).get("main_cost_units")) > 0
        else max(0.0, result_cost(r) - result_route_model_cost(r))
        for r in results
    )
    route_cost_by_level: dict[str, float] = {}
    for result in results:
        for level, value in result_route_model_cost_by_level(result).items():
            route_cost_by_level[level] = route_cost_by_level.get(level, 0.0) + float(value)
    input_tokens = sum(_as_int(result_usage(r).get("input_tokens")) for r in results)
    output_tokens = sum(_as_int(result_usage(r).get("output_tokens")) for r in results)
    cache_read_tokens = sum(_as_int(result_usage(r).get("cache_read_input_tokens")) for r in results)
    cache_write_tokens = sum(_as_int(result_usage(r).get("cache_write_input_tokens")) for r in results)
    tool_calls = sum(int(r.get("num_tool_calls", r.get("turns", 0)) or 0) for r in results)
    turns = sum(int(r.get("num_turns", r.get("turns", 0)) or 0) for r in results)
    elapsed = sum(float(r.get("elapsed", 0.0) or 0.0) for r in results)
    zero_token_tasks = sum(1 for r in results if result_tokens(r) == 0)
    zero_cost_tasks = sum(1 for r in results if result_cost(r) == 0)
    estimated_usage_tasks = sum(1 for r in results if result_usage(r).get("usage_estimated"))
    usage_quality_counts: dict[str, int] = {}
    cost_price_source_counts: dict[str, int] = {}
    cost_currency_counts: dict[str, int] = {}
    priced_usage_tasks = 0
    for result in results:
        usage = result_usage(result)
        quality = result_usage_quality(result)
        usage_quality_counts[quality] = usage_quality_counts.get(quality, 0) + 1
        source = str(usage.get("cost_price_source") or "unknown")
        currency = str(usage.get("cost_currency") or "unknown")
        cost_price_source_counts[source] = cost_price_source_counts.get(source, 0) + 1
        cost_currency_counts[currency] = cost_currency_counts.get(currency, 0) + 1
        if usage.get("cost_priced"):
            priced_usage_tasks += 1
    strategy_hits = summarize_strategy_hits(results)
    skill_usage = summarize_skill_usage(results)

    success_rate = success / tasks if tasks else 0.0
    avg_tokens = tokens / tasks if tasks else 0.0
    avg_tokens_success = success_tokens / success if success else 0.0
    avg_cost = total_cost / tasks if tasks else 0.0
    avg_cost_success = success_cost / success if success else 0.0
    cost_of_pass = total_cost / success if success else math.inf
    if tasks == 0:
        token_quality = "ok"
    elif zero_token_tasks == 0 and estimated_usage_tasks == 0:
        token_quality = "ok"
    elif zero_token_tasks == 0:
        token_quality = "estimated"
    elif zero_token_tasks < tasks:
        token_quality = "partial"
    else:
        token_quality = "missing"

    return {
        "tasks": tasks,
        "success": success,
        "score_sum": round(score, 4),
        "success_rate": round(success_rate, 6),
        "tokens": tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cache_read_tokens,
        "cache_write_input_tokens": cache_write_tokens,
        "total_cost": round(total_cost, 4),
        "cost_units": round(total_cost, 4),
        "main_model_cost": round(main_model_cost, 4),
        "route_model_cost": round(route_model_cost, 4),
        "route_model_cost_success": round(route_model_cost_success, 4),
        "route_model_cost_by_level": {
            key: round(value, 4) for key, value in sorted(route_cost_by_level.items())
        },
        "easy_model_cost": round(route_cost_by_level.get("easy", 0.0), 4),
        "hard_model_cost": round(
            main_model_cost + route_cost_by_level.get("hard", 0.0),
            4,
        ),
        "avg_tokens": round(avg_tokens, 2),
        "avg_tokens_success": round(avg_tokens_success, 2),
        "avg_cost": round(avg_cost, 2),
        "avg_cost_success": round(avg_cost_success, 2),
        "cost_of_pass": round(cost_of_pass, 2) if math.isfinite(cost_of_pass) else "inf",
        "tool_calls": tool_calls,
        "turns": turns,
        "wall_time": round(elapsed, 1),
        "avg_tool_calls": round(tool_calls / tasks, 2) if tasks else 0.0,
        "zero_token_tasks": zero_token_tasks,
        "zero_cost_tasks": zero_cost_tasks,
        "estimated_usage_tasks": estimated_usage_tasks,
        "priced_usage_tasks": priced_usage_tasks,
        "usage_quality_counts": usage_quality_counts,
        "cost_price_source_counts": cost_price_source_counts,
        "cost_currency_counts": cost_currency_counts,
        "token_record_quality": token_quality,
        "strategy_hits": strategy_hits,
        "skill_usage": skill_usage,
    }


def build_cost_profile(results: list[dict[str, Any]], *, group: str = "") -> dict[str, Any]:
    summary = summarize_results(results)
    bottlenecks: list[str] = []

    if summary["tasks"]:
        avg_input = summary["input_tokens"] / summary["tasks"]
        avg_output = summary["output_tokens"] / summary["tasks"]
        avg_calls = summary["tool_calls"] / summary["tasks"]
        if avg_input > max(avg_output * 2, 20000):
            bottlenecks.append("context-heavy")
        if avg_output > 10000:
            bottlenecks.append("reasoning-heavy")
        if avg_calls > 40:
            bottlenecks.append("tool-heavy")
        if any(int(r.get("max_input_tokens_per_turn", 0) or 0) > 30000 for r in results):
            bottlenecks.append("long-observation")

    if not bottlenecks:
        bottlenecks.append("balanced")

    return {
        "group": group,
        "summary": summary,
        "bottlenecks": bottlenecks,
        "strategy_hits": summary.get("strategy_hits", {}),
        "task_profile": {
            "task_type": "coding",
            "benchmark": "swebench_verified",
            "tool_dependency": "shell-heavy",
            "requires_code": True,
        },
    }
