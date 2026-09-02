"""OpenAI GPT LLM provider."""

from __future__ import annotations

from typing import Any

from .base import LLMMessage, LLMProvider, LLMResponse

def _usage_dict(usage: Any) -> dict[str, int]:
    if usage is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_write_input_tokens": 0,
        }
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = 0
    if isinstance(prompt_details, dict):
        cached_tokens = int(prompt_details.get("cached_tokens", 0) or 0)
    elif prompt_details is not None:
        cached_tokens = int(getattr(prompt_details, "cached_tokens", 0) or 0)
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0) or input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cache_read_input_tokens": cached_tokens,
        "cache_write_input_tokens": 0,
    }

class OpenAIProvider(LLMProvider):
    """LLM provider using the OpenAI API (GPT / o-series models)."""

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        try:
            import openai
        except ImportError:
            raise ImportError("pip install openai  (or: pip install agent-evolve[openai])")

        self.model = model
        self.client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()

    def complete(
        self,
        messages: list[LLMMessage],
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **kwargs,
    ) -> LLMResponse:
        api_messages = [{"role": m.role, "content": m.content} for m in messages]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=api_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=choice.message.content or "",
            usage=_usage_dict(usage),
            raw=response,
        )

    def complete_with_tools(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]],
        max_tokens: int = 4096,
        **kwargs,
    ) -> LLMResponse:
        api_messages = [{"role": m.role, "content": m.content} for m in messages]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=api_messages,
            max_tokens=max_tokens,
            tools=tools,
        )
        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=choice.message.content or "",
            usage=_usage_dict(usage),
            raw=response,
        )
