"""Environment-driven model routing helpers."""

from __future__ import annotations

import json
import os
from typing import Any

from .profile import TaskProfile

ROUTE_LEVELS = ("easy", "hard")
DEFAULT_ROUTE_LEVEL = "hard"
_LEGACY_ROUTE_LEVEL_ALIASES = {
    "medium": DEFAULT_ROUTE_LEVEL,
}


def normalize_route_level(value: str | None) -> str | None:
    if not value:
        return None
    level = str(value).strip().lower()
    if level in ROUTE_LEVELS:
        return level
    return _LEGACY_ROUTE_LEVEL_ALIASES.get(level)


def route_level_from_profile(profile: Any | None) -> str | None:
    if profile is None:
        return None
    if isinstance(profile, TaskProfile):
        return normalize_route_level(profile.route_level or profile.difficulty)
    if isinstance(profile, dict):
        return normalize_route_level(
            profile.get("route_level")
            or profile.get("difficulty")
            or profile.get("task_difficulty")
        )
    return None


def resolve_routing_overrides(
    *,
    model_id: str,
    provider: str | None = None,
    task_profile: Any | None = None,
    route_level: str | None = None,
) -> dict[str, Any]:
    """Return route overrides derived from explicit step routes and env vars.

    Whole-task execution defaults to the hard model.  The easy model is reserved
    for callers that explicitly mark a simple step with ``route_level="easy"``
    or force it through the environment.
    """
    level = _select_route_level(task_profile=task_profile, route_level=route_level)
    result = {
        "route_level": level,
        "model_id": model_id,
        "provider": provider,
        "model_id_overridden": False,
        "provider_overridden": False,
    }
    if not level:
        return result

    env_model = _level_env(("AEVOLVE_MODEL_ID", "AEVOLVE_MODEL"), level)
    env_provider = _level_env(("AEVOLVE_MODEL_PROVIDER", "AEVOLVE_PROVIDER"), level)
    env_route = _level_env(("AEVOLVE_ROUTE_LEVEL",), level)

    if env_route:
        result["route_level"] = normalize_route_level(env_route) or level
    if env_model:
        result["model_id"] = env_model
        result["model_id_overridden"] = True
    if env_provider:
        result["provider"] = env_provider
        result["provider_overridden"] = True

    route_map = _json_env("AEVOLVE_ROUTE_MODEL_MAP")
    if isinstance(route_map, dict):
        mapped = route_map.get(level)
        if isinstance(mapped, dict):
            if mapped.get("model_id"):
                result["model_id"] = str(mapped["model_id"])
                result["model_id_overridden"] = True
            if mapped.get("provider"):
                result["provider"] = str(mapped["provider"])
                result["provider_overridden"] = True

    return result


def route_client_env_suffix(level: str | None) -> str:
    level = normalize_route_level(level)
    return f"_{level.upper()}" if level else ""


def _select_route_level(
    *,
    task_profile: Any | None = None,
    route_level: str | None = None,
) -> str | None:
    forced = normalize_route_level(os.environ.get("AEVOLVE_ROUTE_FORCE_LEVEL"))
    if forced:
        return forced
    explicit = normalize_route_level(route_level)
    if explicit:
        return explicit
    return DEFAULT_ROUTE_LEVEL


def _level_env(names: tuple[str, ...], level: str) -> str | None:
    suffix = route_client_env_suffix(level)
    for name in names:
        value = os.environ.get(f"{name}{suffix}")
        if value:
            return value
    return None


def _json_env(name: str) -> dict[str, Any] | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
