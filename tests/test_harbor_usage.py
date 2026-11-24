import json

from agent_evolve.agents.harbor_usage import extract_harbor_usage


def test_extract_harbor_usage_from_agent_result_fields() -> None:
    usage = extract_harbor_usage(
        {
            "agent_result": {
                "n_input_tokens": 100,
                "n_cache_tokens": 40,
                "n_output_tokens": 25,
                "cost_usd": 0.12,
            }
        },
        model_id="deepseek-flash",
        usage_source="osworld2_harbor",
    )

    assert usage["input_tokens"] == 100
    assert usage["cache_read_input_tokens"] == 40
    assert usage["output_tokens"] == 25
    assert usage["total_tokens"] == 125
    assert usage["cost_units"] == 0.12
    assert usage["model_id"] == "deepseek-flash"
    assert usage["usage_record_quality"] == "ok"


def test_extract_harbor_usage_from_codex_session_artifact(tmp_path) -> None:
    session_dir = tmp_path / "agent" / "sessions" / "2026" / "10" / "25"
    session_dir.mkdir(parents=True)
    session_path = session_dir / "rollout.jsonl"
    session_path.write_text(
        json.dumps(
            {
                "type": "token_usage_record",
                "payload": {
                    "thread_token_usage": {
                        "input_tokens": 415848,
                        "cached_input_tokens": 402688,
                        "cache_write_input_tokens": 0,
                        "output_tokens": 21060,
                        "reasoning_output_tokens": 8839,
                        "total_tokens": 436908,
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    usage = extract_harbor_usage(
        {},
        trial_dir=tmp_path,
        model_id="deepseek-flash",
        usage_source="terminal_bench_harbor",
        include_codex_artifacts=True,
    )

    assert usage["input_tokens"] == 13160
    assert usage["cache_read_input_tokens"] == 402688
    assert usage["output_tokens"] == 21060
    assert usage["total_tokens"] == 436908
    assert usage["reasoning_output_tokens"] == 8839
    assert usage["model_id"] == "deepseek-flash"
    assert usage["usage_source"] == "harbor_codex_artifact:session_jsonl"
