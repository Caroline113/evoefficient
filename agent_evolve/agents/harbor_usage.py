"""Best-effort usage extraction for Harbor experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_INPUT_ALIASES = (
    "input_tokens",
    "inputTokens",
    "prompt_tokens",
    "promptTokens",
    "n_input_tokens",
    "total_input_tokens",
)
_OUTPUT_ALIASES = (
    "output_tokens",
    "outputTokens",
    "completion_tokens",
    "completionTokens",
    "n_output_tokens",
    "total_output_tokens",
)
_TOTAL_ALIASES = ("total_tokens", "totalTokens", "n_total_tokens")
_CACHE_READ_ALIASES = (
    "cache_read_input_tokens",
    "cacheReadInputTokens",
    "cache_read_tokens",
    "cacheReadTokens",
    "cached_input_tokens",
    "cachedInputTokens",
    "cached_tokens",
    "cachedTokens",
    "n_cache_tokens",
    "prompt_cache_hit_tokens",
)
_CACHE_WRITE_ALIASES = (
    "cache_write_input_tokens",
    "cacheWriteInputTokens",
    "cache_creation_input_tokens",
    "cacheCreationInputTokens",
    "cache_write_tokens",
    "cacheWriteTokens",
)
_CACHE_MISS_ALIASES = (
    "prompt_cache_miss_tokens",
    "cache_miss_input_tokens",
    "cacheMissInputTokens",
)
_COST_ALIASES = ("cost_units", "costUnits", "total_cost", "totalCost", "cost_usd")
_REASONING_ALIASES = ("reasoning_output_tokens", "reasoningOutputTokens")

_TOKEN_SIGNAL_KEYS = set(
    _INPUT_ALIASES
    + _OUTPUT_ALIASES
    + _TOTAL_ALIASES
    + _CACHE_READ_ALIASES
    + _CACHE_WRITE_ALIASES
    + _CACHE_MISS_ALIASES
    + _COST_ALIASES
)
_INCLUSIVE_CACHE_KEYS = {
    "cached_input_tokens",
    "cachedInputTokens",
    "cached_tokens",
    "cachedTokens",
    "prompt_cache_hit_tokens",
}
_INCLUSIVE_INPUT_KEYS = {"prompt_tokens", "promptTokens"}


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _value_with_key(data: dict[str, Any], aliases: tuple[str, ...]) -> tuple[int, str]:
    for key in aliases:
        if key in data and data.get(key) is not None:
            return _safe_int(data.get(key)), key
    return 0, ""


def _float_with_key(data: dict[str, Any], aliases: tuple[str, ...]) -> tuple[float, str]:
    for key in aliases:
        if key in data and data.get(key) is not None:
            return _safe_float(data.get(key)), key
    return 0.0, ""


def _nested_dict(data: dict[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        value = data.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _nested_usage_value(data: dict[str, Any], parent: str, aliases: tuple[str, ...]) -> tuple[int, str]:
    nested = data.get(parent)
    if not isinstance(nested, dict):
        return 0, ""
    value, key = _value_with_key(nested, aliases)
    return value, f"{parent}.{key}" if key else ""


def _has_usage_signal(data: Any) -> bool:
    return isinstance(data, dict) and any(key in data and data.get(key) is not None for key in _TOKEN_SIGNAL_KEYS)


def _usage_from_mapping(
    data: dict[str, Any],
    *,
    model_id: str = "",
    usage_source: str,
    artifact_path: str = "",
) -> dict[str, Any]:
    input_total, input_key = _value_with_key(data, _INPUT_ALIASES)
    output_tokens, _ = _value_with_key(data, _OUTPUT_ALIASES)
    total_tokens, _ = _value_with_key(data, _TOTAL_ALIASES)
    cache_read, cache_key = _value_with_key(data, _CACHE_READ_ALIASES)
    cache_write, _ = _value_with_key(data, _CACHE_WRITE_ALIASES)
    cache_miss, _ = _value_with_key(data, _CACHE_MISS_ALIASES)
    raw_cost, cost_key = _float_with_key(data, _COST_ALIASES)
    reasoning_tokens, _ = _value_with_key(data, _REASONING_ALIASES)

    if cache_read <= 0:
        cache_read, cache_key = _nested_usage_value(
            data,
            "prompt_tokens_details",
            ("cached_tokens", "cachedTokens"),
        )
    if cache_read <= 0:
        cache_read, cache_key = _nested_usage_value(
            data,
            "input_tokens_details",
            (
                "cached_tokens",
                "cachedTokens",
                "cache_read_input_tokens",
                "cacheReadInputTokens",
            ),
        )

    if not any((input_total, output_tokens, total_tokens, cache_read, cache_write, raw_cost)):
        return {}

    # OpenAI-compatible logs report prompt/input tokens including cache hits.
    # The experiment schema prices uncached input and cache-read tokens
    # separately, so split inclusive counters before later cost normalization.
    input_tokens = input_total
    input_is_inclusive = (
        input_key in _INCLUSIVE_INPUT_KEYS
        or cache_key in _INCLUSIVE_CACHE_KEYS
        or bool(_nested_dict(data, "prompt_tokens_details", "input_tokens_details"))
    )
    if cache_miss > 0:
        input_tokens = cache_miss
    elif input_is_inclusive and cache_read > 0 and input_total >= cache_read:
        input_tokens = input_total - cache_read

    if total_tokens <= 0:
        if input_is_inclusive and input_total > 0:
            total_tokens = input_total + output_tokens
        else:
            total_tokens = input_tokens + output_tokens

    source_model = (
        data.get("model_id")
        or data.get("model")
        or data.get("model_name")
        or data.get("modelName")
        or model_id
    )
    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cache_read_input_tokens": cache_read,
        "cache_write_input_tokens": cache_write,
        "usage_record_quality": "ok" if (input_total or output_tokens or total_tokens) else "cost_only",
        "usage_source": usage_source,
        "usage_estimated": False,
    }
    if source_model:
        usage["model_id"] = str(source_model)
    if reasoning_tokens:
        usage["reasoning_output_tokens"] = reasoning_tokens
    if raw_cost > 0:
        usage["cost_units"] = raw_cost
        if cost_key == "cost_usd":
            usage["cost_currency"] = "USD"
            usage["cost_price_source"] = f"{usage_source}:cost_usd"
            usage["cost_priced"] = True
    if input_is_inclusive and input_total:
        usage["raw_input_tokens_including_cache"] = input_total
    if artifact_path:
        usage["usage_artifact_path"] = artifact_path
    return usage


def _aggregate_usages(usages: list[dict[str, Any]], *, usage_source: str, model_id: str = "") -> dict[str, Any]:
    usable = [usage for usage in usages if usage]
    if not usable:
        return {}
    if len(usable) == 1:
        return usable[0]
    aggregate = {
        "input_tokens": sum(_safe_int(item.get("input_tokens")) for item in usable),
        "output_tokens": sum(_safe_int(item.get("output_tokens")) for item in usable),
        "total_tokens": sum(_safe_int(item.get("total_tokens")) for item in usable),
        "cache_read_input_tokens": sum(_safe_int(item.get("cache_read_input_tokens")) for item in usable),
        "cache_write_input_tokens": sum(_safe_int(item.get("cache_write_input_tokens")) for item in usable),
        "usage_record_quality": "ok",
        "usage_source": usage_source,
        "usage_estimated": False,
    }
    cost_units = sum(_safe_float(item.get("cost_units")) for item in usable)
    if cost_units > 0:
        aggregate["cost_units"] = cost_units
    if model_id:
        aggregate["model_id"] = model_id
    return aggregate


def _usage_from_any(
    value: Any,
    *,
    model_id: str = "",
    usage_source: str,
    artifact_path: str = "",
    _depth: int = 0,
) -> dict[str, Any]:
    if _depth > 4:
        return {}
    if isinstance(value, dict):
        if _has_usage_signal(value):
            usage = _usage_from_mapping(
                value,
                model_id=model_id,
                usage_source=usage_source,
                artifact_path=artifact_path,
            )
            if usage:
                return usage

        usages: list[dict[str, Any]] = []
        for nested in value.values():
            if isinstance(nested, (dict, list)):
                usage = _usage_from_any(
                    nested,
                    model_id=model_id,
                    usage_source=usage_source,
                    artifact_path=artifact_path,
                    _depth=_depth + 1,
                )
                if usage:
                    usages.append(usage)
        return _aggregate_usages(usages, usage_source=usage_source, model_id=model_id)

    if isinstance(value, list):
        usages = [
            _usage_from_any(
                item,
                model_id=model_id,
                usage_source=usage_source,
                artifact_path=artifact_path,
                _depth=_depth + 1,
            )
            for item in value
        ]
        return _aggregate_usages(usages, usage_source=usage_source, model_id=model_id)
    return {}


def _extract_path(data: Any, path: tuple[str, ...]) -> Any:
    current = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _usage_from_harbor_result(
    result_data: dict[str, Any],
    *,
    model_id: str = "",
    usage_source: str,
) -> dict[str, Any]:
    candidate_paths = (
        ("usage",),
        ("token_usage",),
        ("llm_usage",),
        ("cost",),
        ("metrics",),
        ("metrics", "usage"),
        ("agent_result", "model_usage"),
        ("agent_result", "usage"),
        ("agent_result",),
        ("agent_execution", "usage"),
        ("verifier_result", "usage"),
    )
    for path in candidate_paths:
        usage = _usage_from_any(
            _extract_path(result_data, path),
            model_id=model_id,
            usage_source=f"{usage_source}:{'.'.join(path)}",
        )
        if usage:
            return usage
    return {}


def _iter_json_lines(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return records
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _codex_txt_usage(trial_dir: Path, *, model_id: str = "") -> dict[str, Any]:
    path = trial_dir / "agent" / "codex.txt"
    usage: dict[str, Any] = {}
    for record in _iter_json_lines(path) if path.exists() else []:
        raw_usage = record.get("usage")
        if isinstance(raw_usage, dict):
            usage = _usage_from_mapping(
                raw_usage,
                model_id=model_id,
                usage_source="harbor_codex_artifact:codex.txt",
                artifact_path=str(path),
            )
    return usage


def _codex_session_usage(trial_dir: Path, *, model_id: str = "") -> dict[str, Any]:
    sessions_dir = trial_dir / "agent" / "sessions"
    if not sessions_dir.exists():
        return {}
    usage: dict[str, Any] = {}
    session_paths = sorted(
        sessions_dir.rglob("*.jsonl"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
    )
    for path in session_paths:
        for record in _iter_json_lines(path):
            record_type = record.get("type")
            payload = record.get("payload")
            if not isinstance(payload, dict):
                continue
            raw_usage: Any = None
            if record_type == "token_usage_record":
                raw_usage = (
                    payload.get("thread_token_usage")
                    or payload.get("turn_token_usage")
                    or payload.get("usage")
                )
            elif record_type == "event_msg" and payload.get("type") == "token_count":
                info = payload.get("info")
                if isinstance(info, dict):
                    raw_usage = info.get("total_token_usage") or info.get("last_token_usage")
            if isinstance(raw_usage, dict):
                usage = _usage_from_mapping(
                    raw_usage,
                    model_id=model_id,
                    usage_source="harbor_codex_artifact:session_jsonl",
                    artifact_path=str(path),
                )
    return usage


def extract_codex_artifact_usage(trial_dir: str | Path, *, model_id: str = "") -> dict[str, Any]:
    """Read cumulative Codex CLI token usage from a Harbor trial directory."""
    trial_path = Path(trial_dir)
    return _codex_session_usage(trial_path, model_id=model_id) or _codex_txt_usage(
        trial_path,
        model_id=model_id,
    )


def extract_harbor_usage(
    result_data: dict[str, Any],
    *,
    trial_dir: str | Path | None = None,
    model_id: str = "",
    usage_source: str = "harbor",
    include_codex_artifacts: bool = False,
) -> dict[str, Any]:
    """Extract task-level LLM usage from Harbor result JSON and artifacts.

    Harbor agents are not fully consistent about where they write usage.  This
    helper first checks standard result JSON locations such as
    ``agent_result.model_usage`` and, when requested, falls back to Codex CLI
    logs produced under ``agent/codex.txt`` or ``agent/sessions``.
    """

    usage = _usage_from_harbor_result(
        result_data if isinstance(result_data, dict) else {},
        model_id=model_id,
        usage_source=usage_source,
    )
    if usage:
        return usage
    if include_codex_artifacts and trial_dir:
        return extract_codex_artifact_usage(trial_dir, model_id=model_id)
    return {}
