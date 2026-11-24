import json
import sys
import types
from pathlib import Path

import common
import pytest
import run_emergent_efficiency_vwa as runner
import vwa_worker


def test_usage_does_not_double_count_cache_or_reasoning():
    prices = {"input_per_1m": 1, "cache_read_per_1m": 0.1, "output_per_1m": 2, "currency": "USD"}
    row = common.usage_record(
        {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 60},
            "completion_tokens_details": {"reasoning_tokens": 5},
        },
        prices,
    )
    assert row["input_tokens"] == 40
    assert row["total_tokens"] == 120
    assert row["reasoning_output_tokens"] == 5
    assert row["cost_units"] == pytest.approx(0.000086)


@pytest.mark.parametrize(
    "raw,prices",
    [
        (None, None),
        ({"prompt_tokens": 10}, None),
        ({"prompt_tokens": 10, "completion_tokens": 2}, None),
    ],
)
def test_unknown_cost_never_becomes_zero(raw, prices):
    assert common.usage_record(raw, prices)["cost_units"] is None


def test_reject_inconsistent_cache():
    with pytest.raises(ValueError):
        common.usage_record(
            {
                "prompt_tokens": 3,
                "completion_tokens": 1,
                "prompt_tokens_details": {"cached_tokens": 4},
            },
            None,
        )


def test_partial_ledger_not_claimed_complete():
    good = common.usage_record(
        {"prompt_tokens": 10, "completion_tokens": 2},
        {"input_per_1m": 1, "output_per_1m": 1, "cache_read_per_1m": 0, "currency": "USD"},
        event="usage",
    )
    summary = common.aggregate(
        [good, {"event": "usage", "request_failed": True, "cost_priced": False}]
    )
    assert summary["cost_units"] is None
    assert summary["priced_partial_cost"] > 0
    assert not summary["usage_complete"]


def test_example_prices_require_user_values():
    with pytest.raises(ValueError):
        common.load_prices(str(runner.HERE / "prices.example.json"))


def test_task_identity_includes_site_and_selection_order(tmp_path):
    for site in ("classifieds", "shopping"):
        p = tmp_path / site / "0.json"
        p.parent.mkdir()
        common.dump(
            p, {"task_id": 0, "intent": "goal", "sites": [site], "eval": {"eval_types": []}}
        )
    args = runner.parse_args(
        ["--dry-run", "--tasks-dir", str(tmp_path), "--task-ids", "shopping/0,classifieds/0"]
    )
    tasks = runner.load_tasks(args)
    assert [t["id"] for t in tasks] == ["shopping/0", "classifieds/0"]
    assert all(t["checksum"] for t in tasks)
    args.task_ids = "shopping/0,shopping/0"
    with pytest.raises(ValueError, match="Duplicate"):
        runner.load_tasks(args)


def test_empty_task_directory_fails(tmp_path):
    with pytest.raises(ValueError, match="No generated"):
        runner.load_tasks(runner.parse_args(["--dry-run", "--tasks-dir", str(tmp_path)]))


def test_context_has_complete_skill_body_and_fail_closed_budget(tmp_path):
    common.dump(tmp_path / "memory/memories.jsonl", {})
    p = tmp_path / "skills/new/SKILL.md"
    p.parent.mkdir(parents=True)
    p.write_text("unique full technique")
    text, files = runner.workspace_context(tmp_path, 1000)
    assert "unique full technique" in text
    assert "skills/new/SKILL.md" in files
    with pytest.raises(ValueError, match="budget"):
        runner.workspace_context(tmp_path, 5)


def test_context_rejects_escape_symlink(tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("external")
    workspace = tmp_path / "ws"
    (workspace / "prompts").mkdir(parents=True)
    (workspace / "prompts/system.md").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        runner.workspace_context(workspace, 1000)


def test_skill_cap_preserves_existing_after_both_append_paths(tmp_path):
    ws = tmp_path / "workspace"
    for name in ("old", "new1", "new2"):
        p = ws / "skills" / name / "SKILL.md"
        p.parent.mkdir(parents=True)
        p.write_text(name)
    runner.enforce_skill_budget(ws, ["skills/old/SKILL.md"], 2, tmp_path, 1)
    assert (ws / "skills/old/SKILL.md").exists()
    assert len(list((ws / "skills").rglob("SKILL.md"))) == 2
    assert (tmp_path / "emergence/skill_budget.jsonl").exists()


@pytest.mark.parametrize(
    "score,status", [(1.0, "passed"), (0.0, "failed"), (None, "evaluation_missing")]
)
def test_worker_preserves_official_zero_and_missing_evaluation(
    tmp_path, monkeypatch, score, status
):
    agent_module = types.ModuleType("agent.agent")
    tokenizer_module = types.ModuleType("llms.tokenizers")
    helpers = types.ModuleType("evaluation_harness.helper_functions")

    class Tokenizer:
        def __init__(self, *a):
            pass

    tokenizer_module.Tokenizer = Tokenizer
    for name, module in [
        ("agent.agent", agent_module),
        ("llms.tokenizers", tokenizer_module),
        ("evaluation_harness.helper_functions", helpers),
    ]:
        monkeypatch.setitem(sys.modules, name, module)

    class Agent:
        def next_action(self, *a, **kw):
            assert self.multimodal_inputs is True
            return {"raw_prediction": "stop [fixture]", "action_type": "STOP"}

    native = types.SimpleNamespace()
    native.config = lambda: types.SimpleNamespace()
    native.construct_agent = lambda *a, **kw: Agent()
    native.evaluator_router = lambda *a, **kw: lambda *a, **kw: score

    def test(args, configs):
        agent = native.construct_agent(args)
        agent.next_action(
            [{"observation": {"text": "fixture observation"}, "info": {}}], "goal", meta_data={}
        )
        if score is not None:
            native.evaluator_router("config")(trajectory=[], config_file="config", page=None)

    native.test = test
    monkeypatch.setattr(vwa_worker, "load_native", lambda repo: native)
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    monkeypatch.setenv("VWA_API_KEY", "offline-fixture")
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent))
    job = {
        "result_dir": str(tmp_path / "result"),
        "vwa_repo": str(tmp_path),
        "model": "vision-fixture",
        "instruction_path": str(tmp_path / "instruction.json"),
        "config_file": str(tmp_path / "config.json"),
        "observation_type": "image_som",
        "action_set_tag": "som",
        "max_steps": 3,
        "max_tokens": 100,
        "temperature": 0,
        "max_obs_length": 100,
        "eval_captioning_model_device": "cpu",
        "sleep_after_execution": 0,
        "viewport_width": 100,
        "viewport_height": 100,
        "request_timeout": 5,
        "trajectory_text_chars": 100,
    }
    result = vwa_worker.execute(job)
    assert result["score"] == score
    assert result["status"] == status
    assert result["success"] == (score == 1.0)
    assert result["num_tool_calls"] == 1
    assert result["usage"]["cost_units"] is None


def test_ordinary_evolution_uses_shared_engine_and_keeps_usage(tmp_path, monkeypatch):
    from agent_evolve.algorithms.adaptive_skill import tools
    from agent_evolve.llm.base import LLMResponse

    class Provider:
        def converse_loop(self, **kw):
            assert '"success": true' in kw["user_message"]
            (ws / "skills/learned").mkdir(parents=True)
            (ws / "skills/learned/SKILL.md").write_text(
                "---\nname: learned\ndescription: browser technique\n---\nUse current page IDs."
            )
            return LLMResponse(content="done", usage={"input_tokens": 100, "output_tokens": 10})

    monkeypatch.setattr(tools, "create_default_llm", lambda config: Provider())
    import shutil

    ws = tmp_path / "workspace"
    shutil.copytree(runner.HERE / "seed_workspace", ws)
    args = runner.parse_args(["--dry-run", "--include-feedback"])
    rows = [
        {
            "instance_id": "fixture",
            "task_input": "goal",
            "success": True,
            "score": 1,
            "status": "passed",
            "steps": [{"action": "click [1]", "observation": "changed page"}],
        }
    ]
    runner.evolve(ws, rows, args, tmp_path, 1)
    assert (ws / "skills/learned/SKILL.md").exists()
    ledger = common.read_jsonl(tmp_path / "usage.jsonl")
    assert ledger[0]["phase"] == "normal_evolution"
    assert ledger[0]["raw_provider_usage"]["input_tokens"] == 100
    assert ledger[0]["cost_units"] is None


def test_efficiency_proposer_rejects_tools_and_applies_browser_skill(tmp_path, monkeypatch):
    from agent_evolve.algorithms.adaptive_skill import tools
    from agent_evolve.llm.base import LLMResponse

    class Provider:
        def complete(self, *a, **kw):
            return LLMResponse(
                content=json.dumps(
                    {
                        "proposals": [
                            {"name": "unregistered", "kind": "tool", "content": "def x(): pass"},
                            {
                                "name": "fresh_ids",
                                "kind": "skill",
                                "content": "Re-read IDs after page changes.",
                                "description": "After navigation",
                                "confidence": "MEDIUM",
                            },
                        ]
                    }
                ),
                usage={"input_tokens": 20, "output_tokens": 10},
            )

    monkeypatch.setattr(tools, "create_default_llm", lambda config: Provider())
    import shutil

    ws = tmp_path / "workspace"
    shutil.copytree(runner.HERE / "seed_workspace", ws)
    args = runner.parse_args(["--dry-run"])
    runner.propose(ws, [], args, tmp_path, 1)
    assert len(list((ws / "skills").glob("emergent*/SKILL.md"))) == 1
    rejected = json.loads((tmp_path / "emergence/batch_01_rejected.json").read_text())
    assert rejected[0]["status"] == "rejected_unsupported_kind"
    assert not list((ws / "tools").glob("*.py"))


def test_summary_does_not_count_synthetic_as_success():
    rows = [
        {
            "synthetic": True,
            "score": None,
            "success": False,
            "total_tokens": 0,
            "cost_priced": False,
            "num_tool_calls": 0,
            "elapsed": 0,
            "cost_units": None,
        }
    ]
    summary = runner.summarize(rows)
    assert summary["success_rate"] is None
    assert summary["total_cost"] is None
    assert summary["evaluated_tasks"] == 0


def test_live_worker_error_does_not_disappear_from_denominator():
    rows = [
        {
            "synthetic": False,
            "score": None,
            "success": False,
            "total_tokens": 0,
            "cost_priced": False,
            "num_tool_calls": 0,
            "elapsed": 1,
            "cost_units": None,
        }
    ]
    summary = runner.summarize(rows)
    assert summary["tasks"] == 1
    assert summary["success_rate"] == 0
    assert summary["total_cost"] is None


def test_chat_transport_keeps_images_and_accounts_server_usage(tmp_path, monkeypatch):
    import io

    seen = {}

    def urlopen(request, timeout):
        seen["body"] = json.loads(request.data)
        assert request.get_header("Authorization") == "Bearer offline-secret"
        return io.BytesIO(
            json.dumps(
                {
                    "model": "vision-fixture",
                    "choices": [{"message": {"content": "stop [done]"}}],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                        "total_tokens": 120,
                        "prompt_tokens_details": {"cached_tokens": 60},
                    },
                }
            ).encode()
        )

    monkeypatch.setattr(common.urllib.request, "urlopen", urlopen)
    ledger = tmp_path / "usage.jsonl"
    client = common.ChatClient(
        ledger,
        base="http://offline.invalid/v1",
        key="offline-secret",
        model="vision-fixture",
        phase="solver",
        prices=None,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "image goal"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,fixture"}},
            ],
        }
    ]
    assert client.complete(messages) == "stop [done]"
    assert seen["body"]["messages"] == messages
    assert seen["body"]["model"] == "vision-fixture"
    row = common.read_jsonl(ledger)[0]
    assert row["input_tokens"] == 40
    assert row["cache_read_input_tokens"] == 60
    assert "offline-secret" not in ledger.read_text()
    assert "data:image/png" not in ledger.read_text()


def test_transport_failure_is_unknown_not_free(tmp_path, monkeypatch):
    def urlopen(*a, **kw):
        raise OSError("a server error body that should not be persisted")

    monkeypatch.setattr(common.urllib.request, "urlopen", urlopen)
    ledger = tmp_path / "usage.jsonl"
    client = common.ChatClient(
        ledger,
        base="http://offline.invalid/v1",
        key="fixture",
        model="fixture",
        phase="solver",
        prices=None,
    )
    with pytest.raises(RuntimeError, match="request failed"):
        client.complete([])
    row = common.read_jsonl(ledger)[0]
    assert row["request_failed"]
    assert "server error body" not in ledger.read_text()
    assert common.aggregate([row])["cost_units"] is None


def test_worker_timeout_terminates_process_group(tmp_path):
    import os

    code, timed_out, elapsed = common.run_logged(
        [sys.executable, "-c", "import time;time.sleep(30)"],
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout=1,
        stdout=tmp_path / "stdout.log",
        stderr=tmp_path / "stderr.log",
    )
    assert timed_out
    assert code != 0
    assert elapsed < 10
