"""Model/provider selection helpers.

This module keeps provider routing in one place.  Existing Bedrock model IDs
continue to work, while OpenRouter/OpenAI-compatible APIs can be selected
either globally with ``AEVOLVE_MODEL_PROVIDER=...`` or per model with a
provider prefix such as ``openrouter:<model>`` or ``openai:<model>``.
"""

from __future__ import annotations

import os
from typing import Any

from ..efficiency.routing import resolve_routing_overrides, route_client_env_suffix

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_PROVIDER_PREFIXES = {"bedrock", "openrouter", "openai", "anthropic", "litellm"}
_PROVIDER_ENV_VARS = (
    "AEVOLVE_MODEL_PROVIDER",
    "AGENT_EVOLVE_MODEL_PROVIDER",
    "MODEL_PROVIDER",
    "LLM_PROVIDER",
)


def split_provider_prefix(model_id: str) -> tuple[str | None, str]:
    """Return ``(<provider>, <model>)`` for strings like ``openrouter:x/y``."""
    prefix, sep, rest = (model_id or "").partition(":")
    if sep and prefix.lower() in _PROVIDER_PREFIXES and rest:
        return _normalize_provider(prefix), rest
    return None, model_id


def resolve_model_selection(
    model_id: str,
    provider: str | None = None,
    *,
    task_profile: Any | None = None,
    route_level: str | None = None,
) -> tuple[str, str]:
    """Resolve the provider and final model ID for a user-supplied model.

    Selection order:
    1. Explicit ``provider`` argument.
    2. Model prefix such as ``openrouter:anthropic/claude-sonnet-4``.
    3. Environment provider, usually ``AEVOLVE_MODEL_PROVIDER=openrouter``.
    4. Conservative inference from the model name, defaulting to Bedrock.

    When OpenRouter or OpenAI-compatible routing is selected globally, provider
    model env vars can override legacy Bedrock CLI defaults.
    """
    route = resolve_routing_overrides(
        model_id=model_id,
        provider=provider,
        task_profile=task_profile,
        route_level=route_level,
    )
    prefixed_provider, clean_model = split_provider_prefix(str(route["model_id"]))
    resolved_provider = (
        _normalize_provider(route["provider"])
        if route.get("provider")
        else prefixed_provider or _provider_from_env() or _infer_provider(clean_model)
    )

    if resolved_provider == "openrouter" and prefixed_provider is None:
        override_model = _env_pick(
            "AEVOLVE_OPENROUTER_MODEL",
            "OPENROUTER_MODEL",
            route_level=route.get("route_level"),
        )
        if override_model and _looks_like_bedrock_model(clean_model):
            clean_model = override_model
    elif resolved_provider == "openai" and prefixed_provider is None:
        override_model = _env_pick(
            "AEVOLVE_OPENAI_MODEL",
            "OPENAI_MODEL",
            route_level=route.get("route_level"),
        )
        if override_model and _looks_like_bedrock_model(clean_model):
            clean_model = override_model

    return resolved_provider, clean_model


def create_strands_model(
    model_id: str,
    *,
    region: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    stop_sequences: list[str] | tuple[str, ...] | None = None,
    provider: str | None = None,
    params: dict[str, Any] | None = None,
    task_profile: Any | None = None,
    route_level: str | None = None,
    **model_kwargs: Any,
) -> Any:
    """Create a Strands model for the selected provider."""
    selected_provider, selected_model = resolve_model_selection(
        model_id,
        provider,
        task_profile=task_profile,
        route_level=route_level,
    )
    resolved_route_level = _selected_route_level(task_profile=task_profile, route_level=route_level)

    if selected_provider == "openrouter":
        from strands.models.openai import OpenAIModel

        return OpenAIModel(
            model_id=selected_model,
            client_args=_openrouter_client_args(route_level=resolved_route_level),
            params=_openai_params(max_tokens, temperature, stop_sequences, params),
        )

    if selected_provider == "openai":
        from strands.models.openai import OpenAIModel

        return OpenAIModel(
            model_id=selected_model,
            client_args=_openai_client_args(route_level=resolved_route_level),
            params=_openai_params(max_tokens, temperature, stop_sequences, params),
        )

    if selected_provider == "litellm":
        from strands.models.litellm import LiteLLMModel

        return LiteLLMModel(
            model_id=selected_model,
            params=_openai_params(max_tokens, temperature, stop_sequences, params),
        )

    from strands.models import BedrockModel

    bedrock_kwargs: dict[str, Any] = {
        "model_id": selected_model,
        "region_name": region,
    }
    if max_tokens is not None:
        bedrock_kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        bedrock_kwargs["temperature"] = temperature
    if stop_sequences is not None:
        bedrock_kwargs["stop_sequences"] = list(stop_sequences)
    bedrock_kwargs.update(model_kwargs)
    return BedrockModel(**bedrock_kwargs)


def create_llm_provider(
    model_id: str,
    *,
    region: str = "us-west-2",
    provider: str | None = None,
    task_profile: Any | None = None,
    route_level: str | None = None,
) -> Any:
    """Create an internal ``LLMProvider`` for evolver/judge code."""
    selected_provider, selected_model = resolve_model_selection(
        model_id,
        provider,
        task_profile=task_profile,
        route_level=route_level,
    )
    resolved_route_level = _selected_route_level(task_profile=task_profile, route_level=route_level)

    if selected_provider == "openrouter":
        from agent_evolve.algorithms.unified.openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(
            model=selected_model,
            api_key=_env_pick(
                "AEVOLVE_OPENROUTER_API_KEY",
                "OPENROUTER_API_KEY",
                route_level=resolved_route_level,
            ),
            base_url=(
                _env_pick("AEVOLVE_OPENROUTER_BASE_URL", "OPENROUTER_BASE_URL", route_level=resolved_route_level)
                or OPENROUTER_BASE_URL
            ),
        )

    if selected_provider == "openai":
        from agent_evolve.algorithms.unified.openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(
            model=selected_model,
            api_key=(
                _env_pick(
                    "AEVOLVE_OPENAI_API_KEY",
                    "EVOLVER_OPENAI_API_KEY",
                    "OPENAI_API_KEY",
                    route_level=resolved_route_level,
                )
            ),
            base_url=(
                _env_pick(
                    "AEVOLVE_OPENAI_BASE_URL",
                    "EVOLVER_OPENAI_BASE_URL",
                    "OPENAI_BASE_URL",
                    route_level=resolved_route_level,
                )
            ),
        )

    if selected_provider == "anthropic":
        from agent_evolve.llm.anthropic import AnthropicProvider

        return AnthropicProvider(model=selected_model)

    from agent_evolve.llm.bedrock import BedrockProvider

    return BedrockProvider(model_id=selected_model, region=region)


def create_openai_client_for_model(
    model_id: str,
    provider: str | None = None,
    *,
    task_profile: Any | None = None,
    route_level: str | None = None,
) -> tuple[Any, str, str]:
    """Return ``(client, clean_model_id, provider)`` for OpenAI-compatible calls."""
    selected_provider, selected_model = resolve_model_selection(
        model_id,
        provider,
        task_profile=task_profile,
        route_level=route_level,
    )
    if selected_provider == "openrouter":
        from openai import OpenAI

        return OpenAI(**_openrouter_client_args(route_level=_selected_route_level(task_profile=task_profile, route_level=route_level))), selected_model, selected_provider
    if selected_provider == "openai":
        from openai import OpenAI

        return OpenAI(**_openai_client_args(route_level=_selected_route_level(task_profile=task_profile, route_level=route_level))), selected_model, selected_provider
    raise ValueError(f"Provider {selected_provider!r} is not OpenAI-compatible")


def is_openai_compatible_model(model_id: str, provider: str | None = None) -> bool:
    selected_provider, _ = resolve_model_selection(model_id, provider)
    return selected_provider in {"openrouter", "openai"}


def _normalize_provider(provider: str) -> str:
    value = provider.strip().lower().replace("_", "-")
    aliases = {
        "aws": "bedrock",
        "aws-bedrock": "bedrock",
        "openrouter.ai": "openrouter",
        "or": "openrouter",
        "openai-compatible": "openai",
        "openai-compat": "openai",
    }
    value = aliases.get(value, value)
    if value not in _PROVIDER_PREFIXES:
        raise ValueError(
            f"Unsupported model provider {provider!r}. "
            f"Supported providers: {', '.join(sorted(_PROVIDER_PREFIXES))}"
        )
    return value


def _provider_from_env() -> str | None:
    for name in _PROVIDER_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return _normalize_provider(value)
    return None


def _infer_provider(model_id: str) -> str:
    if model_id.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if "/" in model_id and os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    if model_id.startswith("claude"):
        return "anthropic"
    return "bedrock"


def _looks_like_bedrock_model(model_id: str) -> bool:
    return "." in model_id and any(
        vendor in model_id for vendor in ("anthropic", "amazon", "meta")
    )


def _openai_params(
    max_tokens: int | None,
    temperature: float | None,
    stop_sequences: list[str] | tuple[str, ...] | None,
    params: dict[str, Any] | None,
) -> dict[str, Any] | None:
    merged: dict[str, Any] = dict(params or {})
    if max_tokens is not None:
        merged.setdefault("max_tokens", max_tokens)
    if temperature is not None:
        merged.setdefault("temperature", temperature)
    if stop_sequences:
        merged.setdefault("stop", list(stop_sequences))
    return merged or None


def _openrouter_client_args(*, route_level: str | None = None) -> dict[str, Any]:
    args: dict[str, Any] = {
        "api_key": _require_env("OPENROUTER_API_KEY", "OpenRouter", route_level=route_level)
        if not _env_pick("AEVOLVE_OPENROUTER_API_KEY", route_level=route_level)
        else _env_pick("AEVOLVE_OPENROUTER_API_KEY", route_level=route_level),
        "base_url": (
            _env_pick("AEVOLVE_OPENROUTER_BASE_URL", "OPENROUTER_BASE_URL", route_level=route_level)
            or OPENROUTER_BASE_URL
        ),
    }
    headers = _openrouter_headers()
    if headers:
        args["default_headers"] = headers
    return args


def _openai_client_args(*, route_level: str | None = None) -> dict[str, Any]:
    args: dict[str, Any] = {}
    api_key = _env_pick(
        "AEVOLVE_OPENAI_API_KEY",
        "EVOLVER_OPENAI_API_KEY",
        "OPENAI_API_KEY",
        route_level=route_level,
    )
    base_url = _env_pick(
        "AEVOLVE_OPENAI_BASE_URL",
        "EVOLVER_OPENAI_BASE_URL",
        "OPENAI_BASE_URL",
        route_level=route_level,
    )
    if api_key:
        args["api_key"] = api_key
    if base_url:
        args["base_url"] = base_url
    return args


def _openrouter_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    referer = os.environ.get("OPENROUTER_SITE_URL") or os.environ.get("OPENROUTER_HTTP_REFERER")
    title = os.environ.get("OPENROUTER_APP_NAME") or os.environ.get("OPENROUTER_TITLE")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-OpenRouter-Title"] = title
    return headers


def _selected_route_level(*, task_profile: Any | None = None, route_level: str | None = None) -> str | None:
    return resolve_routing_overrides(
        model_id="",
        task_profile=task_profile,
        route_level=route_level,
    ).get("route_level")


def _env_pick(*names: str, route_level: str | None = None) -> str | None:
    if route_level:
        suffix = route_client_env_suffix(route_level)
        for name in names:
            value = os.environ.get(f"{name}{suffix}")
            if value:
                return value
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _require_env(name: str, provider_name: str, *, route_level: str | None = None) -> str:
    value = _env_pick(name, route_level=route_level)
    if not value:
        raise ValueError(f"{provider_name} provider selected, but {name} is not set.")
    return value
