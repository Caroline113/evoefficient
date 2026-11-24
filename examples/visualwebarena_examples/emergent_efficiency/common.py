"""Dependency-light helpers shared by the orchestrator and isolated VWA worker."""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


def append(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def usage_record(raw: dict | None, prices: dict | None, **meta: Any) -> dict:
    """OpenAI prompt_tokens include cache; reasoning is already in completion_tokens."""
    raw = raw or {}
    complete = all(
        isinstance(raw.get(k), int) and raw[k] >= 0 for k in ("prompt_tokens", "completion_tokens")
    )
    prompt = int(raw.get("prompt_tokens") or 0)
    output = int(raw.get("completion_tokens") or 0)
    cached = int(
        (raw.get("prompt_tokens_details") or {}).get("cached_tokens")
        or raw.get("prompt_cache_hit_tokens")
        or 0
    )
    if not 0 <= cached <= prompt:
        raise ValueError("cached_tokens is outside prompt_tokens")
    complete = complete and raw.get("total_tokens", prompt + output) == prompt + output
    record = dict(
        meta,
        input_tokens=prompt - cached,
        cache_read_input_tokens=cached,
        cache_write_input_tokens=0,
        output_tokens=output,
        total_tokens=prompt + output,
        reasoning_output_tokens=int(
            (raw.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        ),
        usage_record_quality="ok" if complete else "missing_or_inconsistent",
        usage_estimated=False,
        cost_units=None,
        cost_priced=False,
    )
    if prices is not None and complete:
        record["cost_units"] = (
            (prompt - cached) * prices["input_per_1m"]
            + cached * prices["cache_read_per_1m"]
            + output * prices["output_per_1m"]
        ) / 1e6
        record["cost_priced"] = True
        record["cost_currency"] = prices["currency"]
        record["cost_price_source"] = "explicit_experiment_price_file"
    return record


def load_prices(path: str | None) -> dict | None:
    if not path:
        return None
    obj = json.loads(Path(path).read_text())
    for phase in ("solver", "judge", "evolver"):
        if phase not in obj:
            continue
        entry = obj[phase]
        if entry.get("currency") != "USD":
            raise ValueError("Only explicit USD price profiles are supported")
        for key in ("input_per_1m", "output_per_1m", "cache_read_per_1m"):
            v = entry.get(key)
            if not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0:
                raise ValueError(f"Invalid {phase}.{key}")
    return obj


def aggregate(ledger: list[dict]) -> dict:
    records = [r for r in ledger if r.get("event") == "usage"]
    known = bool(records) and all(
        r.get("cost_priced") and not r.get("request_failed") for r in records
    )
    return {
        "requests_or_aggregates": len(records),
        "usage_complete": bool(records)
        and all(r.get("usage_record_quality") == "ok" for r in records),
        **{
            k: sum(r.get(k, 0) for r in records)
            for k in (
                "input_tokens",
                "cache_read_input_tokens",
                "output_tokens",
                "total_tokens",
                "reasoning_output_tokens",
            )
        },
        "cost_units": sum(r["cost_units"] for r in records) if known else None,
        "priced_partial_cost": sum(r.get("cost_units") or 0 for r in records),
        "cost_priced": known,
    }


class ChatClient:
    """OpenAI-compatible Chat Completions, with durable per-request usage logging."""

    def __init__(
        self,
        ledger: Path,
        *,
        base: str,
        key: str,
        model: str,
        phase: str,
        prices: dict | None,
        timeout: float = 120,
    ):
        self.ledger, self.base, self.key, self.model = ledger, base.rstrip("/"), key, model
        self.phase, self.prices, self.timeout = phase, prices, timeout

    def complete(
        self, messages: list, *, max_tokens: int = 2048, temperature: float = 0, top_p: float = 1
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        started = time.monotonic()
        request = urllib.request.Request(
            self.base + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.key},
        )
        meta = {"event": "usage", "phase": self.phase, "model_id": self.model}
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.load(response)
            rec = usage_record(body.get("usage"), self.prices, **meta)
            rec["elapsed"] = time.monotonic() - started
            rec["provider_model"] = body.get("model")
            append(self.ledger, rec)
            content = body["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("Chat endpoint must return string message.content")
            return content
        except Exception as exc:  # noqa: BLE001 -- persist usage at the transport boundary
            # Do not persist API keys, request images, or server error bodies.
            append(
                self.ledger,
                dict(
                    meta,
                    request_failed=True,
                    cost_priced=False,
                    usage_record_quality="request_failed",
                    error_type=type(exc).__name__,
                    elapsed=time.monotonic() - started,
                ),
            )
            raise RuntimeError(
                f"{self.phase} request failed ({type(exc).__name__}); see usage ledger"
            ) from None


def run_logged(
    command: list[str], *, cwd: Path, env: dict, timeout: int, stdout: Path, stderr: Path
) -> tuple[int, bool, float]:
    started = time.monotonic()
    stdout.parent.mkdir(parents=True, exist_ok=True)
    with stdout.open("w") as out, stderr.open("w") as err:
        proc = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=out, stderr=err, start_new_session=True
        )
        try:
            code = proc.wait(timeout=timeout)
            return code, False, time.monotonic() - started
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            return proc.returncode, True, time.monotonic() - started
