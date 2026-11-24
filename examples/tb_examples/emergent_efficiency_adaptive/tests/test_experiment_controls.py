from examples.tb_examples.emergent_efficiency_adaptive.trace_stats import consume, new_state

"""Offline regressions: selection fairness, failure feedback and live-guard boundaries."""

import argparse
import json
from types import SimpleNamespace

import pytest

from examples.tb_examples.emergent_efficiency_adaptive import experiment_controls as controls
from examples.tb_examples.emergent_efficiency_adaptive import (
    run_emergent_efficiency_tb_adaptive_harbor as runner,
)


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def selection_args(tmp_path, **kwargs):
    return argparse.Namespace(
        limit=2, history_root=[str(tmp_path)], allow_rerun=False, selection_file=None, **kwargs
    )


def fake_filter(tasks, args):
    return tasks if args.limit is None else tasks[: args.limit]


def test_exclude_before_limit_and_ignore_dry_runs(tmp_path):
    save(tmp_path / "old/results.json", [{"instance_id": "old", "success": False}])
    save(tmp_path / "smoke/experiment_manifest.json", {"dry_run_harbor": True})
    save(tmp_path / "smoke/results.json", [{"instance_id": "new1", "success": True}])
    tasks = [SimpleNamespace(name=n) for n in ["old", "new1", "new2", "new3"]]
    chosen, audit = controls.select_tasks(tasks, selection_args(tmp_path), fake_filter)
    assert [t.name for t in chosen] == ["new1", "new2"]
    assert audit["excluded_previously_run"] == ["old"]


def test_frozen_matrix_reuses_tasks_after_first_mode_finishes(tmp_path):
    args = selection_args(tmp_path)
    args.selection_file = str(tmp_path / "selection.json")
    tasks = [SimpleNamespace(name=n) for n in ["a", "b", "c"]]
    first, _ = controls.select_tasks(tasks, args, fake_filter)
    save(tmp_path / "first/results.json", [{"instance_id": t.name} for t in first])
    second, audit = controls.select_tasks(tasks, args, fake_filter)
    assert [t.name for t in first] == [t.name for t in second] == ["a", "b"]
    assert audit["selection_reused"]


def test_allow_rerun_is_explicit(tmp_path):
    save(tmp_path / "old/results.json", [{"instance_id": "a", "success": False}])
    args = selection_args(tmp_path)
    args.allow_rerun = True
    chosen, _ = controls.select_tasks([SimpleNamespace(name="a")], args, fake_filter)
    assert len(chosen) == 1


def test_bad_history_fails_instead_of_silently_rerunning(tmp_path):
    p = tmp_path / "old/results.json"
    p.parent.mkdir()
    p.write_text("{partial")
    with pytest.raises(ValueError):
        controls.select_tasks([SimpleNamespace(name="a")], selection_args(tmp_path), fake_filter)


def token_event(total=120):
    return {
        "type": "token_usage_record",
        "payload": {
            "thread_token_usage": {
                "input_tokens": total - 20,
                "cached_input_tokens": 60,
                "output_tokens": 20,
                "total_tokens": total,
            }
        },
    }


def failed_call(state, cid, code=1, cmd="make test"):
    consume(
        state,
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "call_id": cid,
                "name": "exec_command",
                "arguments": json.dumps({"cmd": cmd}),
            },
        },
    )
    consume(
        state,
        {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": cid,
                "output": f"Process exited with code {code}\nfixture",
            },
        },
    )


def test_trace_cache_counted_once_and_duplicate_usage_ignored():
    state = new_state()
    consume(state, token_event())
    consume(state, token_event())
    assert state["usage"]["total_tokens"] == 120
    assert state["usage"]["input_tokens"] == 40
    assert state["usage"]["cache_read_input_tokens"] == 60
    assert len(state["timeline"]) == 1


def test_success_or_new_attempt_resets_repeated_failure():
    state = new_state()
    for n in range(4):
        failed_call(state, str(n))
    assert state["repeated_failures"] == 4
    failed_call(state, "new", cmd="inspect new cause")
    assert state["repeated_failures"] == 1
    failed_call(state, "fixed", code=0)
    assert state["repeated_failures"] == 0


def test_patch_invalidates_stagnation():
    state = new_state()
    failed_call(state, "a")
    failed_call(state, "b")
    consume(
        state,
        {
            "type": "response_item",
            "payload": {
                "type": "custom_tool_call",
                "call_id": "edit",
                "name": "apply_patch",
                "input": "*** patch",
            },
        },
    )
    assert state["repeated_failures"] == 0


def test_unknown_exit_is_not_stagnation():
    state = new_state()
    for i in range(5):
        consume(
            state,
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "call_id": str(i),
                    "name": "exec_command",
                    "arguments": '{"cmd":"make"}',
                },
            },
        )
        consume(
            state,
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": str(i),
                    "output": "still running",
                },
            },
        )
    assert state["repeated_failures"] == 0


def test_failure_distribution_keeps_cache_output_and_no_fake_quartiles():
    rows = [
        {
            "instance_id": "fail",
            "success": False,
            "total_tokens": 100,
            "cumulative_input_tokens": 10,
            "cumulative_cache_read_input_tokens": 70,
            "cumulative_output_tokens": 20,
        },
        {"instance_id": "pass", "success": True, "total_tokens": 50},
    ]
    report = controls.failure_feedback(rows)
    assert report["groups"]["failed"]["token_shares"]["cached_input"] == 0.7
    assert report["failed_tasks"][0]["token_quartiles_by_model_updates"] is None


def test_failure_context_restored_on_proposer_error():
    fake = SimpleNamespace(
        _build_cost_feedback=lambda *a, **kw: "original",
        EMERGENCE_SYSTEM_PROMPT="normal",
        PREDEFINED_COST_AWARE_SYSTEM_PROMPT="predefined",
        CATEGORY_SEEDED_SYSTEM_PROMPT="category",
    )
    with (
        pytest.raises(RuntimeError),
        controls.failure_aware_proposer(fake, {"failed_tasks": ["fixture"]}),
    ):
        assert "failure_token_distribution" in fake._build_cost_feedback()
        assert "ONE OPTIONAL efficiency strategy" in fake.EMERGENCE_SYSTEM_PROMPT
        raise RuntimeError("fixture")
    assert fake._build_cost_feedback() == "original"
    assert fake.EMERGENCE_SYSTEM_PROMPT == "normal"


def test_static_main_never_runs_evolver_or_proposer(tmp_path, monkeypatch):
    from agent_evolve.agents.terminal import agent as agent_module
    from agent_evolve.agents.terminal import dataset
    from agent_evolve.engine import observer as observer_module

    args = runner.parse_args(
        ["--mode", "static-control", "--allow-rerun", "--limit", "2", "--batch-size", "1"]
    )
    monkeypatch.setattr(runner, "parse_args", lambda: args)
    tasks = [SimpleNamespace(name=n) for n in ["a", "b"]]
    monkeypatch.setattr(dataset, "load_all_tasks", lambda path: tasks)

    class Agent:
        def __init__(self, **kw):
            pass

        def reload_from_fs(self):
            pass

    monkeypatch.setattr(agent_module, "TerminalAgent", Agent)
    monkeypatch.setattr(observer_module, "Observer", lambda *a, **kw: object())

    def forbidden(*a, **kw):
        raise AssertionError("Static baseline called evolution")

    monkeypatch.setattr(runner, "_run_adaptive_evolution", forbidden)

    def seed(src, dst, **kw):
        dst.mkdir(parents=True)

    def manifest(out, args, tasks):
        save(out / "experiment_manifest.json", {"tasks": [t.name for t in tasks]})

    shared = SimpleNamespace(
        _strategy_leakage_scan=lambda p: {"passed": True},
        _behavior_hit_stats=lambda r: {},
        _record_workspace_snapshot=lambda *a: None,
        propose_emergent_efficiency=forbidden,
    )
    base = SimpleNamespace(
        _resolve_output_dir=lambda a: tmp_path / "run",
        _copy_seed_workspace=seed,
        _swe=shared,
        _filter_tasks=fake_filter,
        _task_to_core_task=lambda t: SimpleNamespace(id=t.name),
        _write_experiment_manifest=manifest,
        _build_system_prompt_for_terminal=lambda *a: "seed",
        _results_to_observations=lambda *a: [],
        _write_workspace_evolution_records=lambda **kw: None,
        _write_run_outputs=lambda out, rows, **kw: save(out / "results.json", rows),
    )

    def batch(**kw):
        return [
            {
                "instance_id": t.name,
                "success": True,
                "score": 1.0,
                "total_tokens": 1,
                "cost_units": 0.1,
                "steps": [],
            }
            for t in kw["batch_tasks"]
        ]

    harbor = SimpleNamespace(_solve_terminal_batch_harbor=batch)
    monkeypatch.setattr(runner, "_load_shared_runners", lambda: (base, harbor))
    assert runner.main() == 0
    manifest = controls.read_json(tmp_path / "run/experiment_manifest.json")
    assert manifest["normal_evolution_engine"] is None and not manifest["evolution_enabled"]
    assert len(controls.read_json(tmp_path / "run/results.json")) == 2
    assert not (tmp_path / "run/emergence/adaptive_evolution.jsonl").exists()


def test_failure_token_quartiles_are_deltas_not_repeated_cumulative_totals():
    result = controls.failure_feedback(
        [
            {
                "instance_id": "failure",
                "success": False,
                "total_tokens": 600,
                "token_distribution": {
                    "cumulative_samples": [{"total_tokens": x} for x in (120, 200, 500, 600)]
                },
            }
        ]
    )
    quartiles = result["failed_tasks"][0]["token_quartiles_by_model_updates"]
    assert quartiles == [120, 80, 300, 100]
    assert sum(quartiles) == 600


def test_failure_analysis_adds_data_without_changing_original_observer_input(tmp_path):
    import copy

    session = tmp_path / "session.jsonl"
    session.write_text(json.dumps(token_event()) + "\n")
    raw = tmp_path / "raw.json"
    save(raw, {"usage": {"usage_artifact_path": str(session)}})
    original = {
        "raw_harbor_result_path": str(raw),
        "steps": [{"tool": "original"}],
        "conversation": [{"role": "assistant", "content": "original trajectory"}],
        "num_tool_calls": 9,
        "num_turns": 8,
        "total_tokens": 120,
        "success": False,
        "score": 0,
        "behavior_stats": {"old": 1},
    }
    preserved = copy.deepcopy(original)
    result = controls.analyze_result(original)
    assert original == preserved
    for key, value in original.items():
        assert result[key] == value
    assert result["token_distribution"]["cumulative_samples"][0]["total_tokens"] == 120


def test_original_efficiency_instructions_and_feedback_remain_as_prefixes():
    originals = {
        "EMERGENCE_SYSTEM_PROMPT": "ORIGINAL all strategies, tool/skill/memory proposals",
        "PREDEFINED_COST_AWARE_SYSTEM_PROMPT": "ORIGINAL predefined methods",
        "CATEGORY_SEEDED_SYSTEM_PROMPT": "ORIGINAL categories",
    }
    fake = SimpleNamespace(
        **originals, _build_cost_feedback=lambda *a, **kw: "ORIGINAL complete feedback"
    )
    with controls.failure_aware_proposer(fake, {"failed_tasks": []}):
        assert fake._build_cost_feedback().startswith("ORIGINAL complete feedback")
        for name, prefix in originals.items():
            assert getattr(fake, name).startswith(prefix)
            assert "ONE OPTIONAL efficiency strategy" in getattr(fake, name)
            assert "retain all existing efficiency objectives" in getattr(fake, name)
    assert fake.EMERGENCE_SYSTEM_PROMPT == originals["EMERGENCE_SYSTEM_PROMPT"]


def test_normal_evolution_configuration_unchanged_by_cost_aware_additions():
    from dataclasses import asdict

    normal = runner._build_adaptive_config(runner.parse_args(["--mode", "normal-evolve"]))
    cost = runner._build_adaptive_config(runner.parse_args(["--mode", "cost-aware"]))
    assert asdict(normal) == asdict(cost)
