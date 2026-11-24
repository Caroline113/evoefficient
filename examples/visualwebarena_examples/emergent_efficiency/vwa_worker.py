#!/usr/bin/env python3
"""Run ONE task in the official VWA environment; import no A-Evolve dependencies."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

from common import ChatClient, aggregate, dump, read_jsonl


def load_native(repo: Path):
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    os.environ["DATASET"] = "visualwebarena"
    spec = importlib.util.spec_from_file_location("_aevolve_native_vwa", repo / "run.py")
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load VisualWebArena run.py")
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    return native


def execute(job: dict) -> dict:
    out = Path(job["result_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "traces").mkdir(exist_ok=True)
    steps, evaluated = [], []
    started = time.monotonic()
    ledger = out / "usage.jsonl"
    result = {"status": "evaluation_missing", "score": None, "success": False, "synthetic": False}
    try:
        repo = Path(job["vwa_repo"])
        # Official auto_login spawns `python`; make it use the same VWA environment.
        os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
        native = load_native(repo)
        sys.argv = [
            str(repo / "run.py"),
            "--provider",
            "openai",
            "--mode",
            "chat",
            "--model",
            job["model"],
            "--instruction_path",
            job["instruction_path"],
            "--result_dir",
            str(out),
            "--observation_type",
            job["observation_type"],
            "--action_set_tag",
            job["action_set_tag"],
            "--max_steps",
            str(job["max_steps"]),
            "--max_tokens",
            str(job["max_tokens"]),
            "--temperature",
            str(job["temperature"]),
            "--max_obs_length",
            str(job["max_obs_length"]),
            "--max_retry",
            "1",
            "--eval_captioning_model_device",
            job["eval_captioning_model_device"],
        ]
        args = native.config()
        args.render, args.render_screenshot = False, True
        args.save_trace_enabled, args.current_viewport_only = True, True
        args.sleep_after_execution = job["sleep_after_execution"]
        args.viewport_width, args.viewport_height = job["viewport_width"], job["viewport_height"]
        solver = ChatClient(
            ledger,
            base=os.environ.get("VWA_BASE_URL", "https://api.openai.com/v1"),
            key=os.environ["VWA_API_KEY"],
            model=job["model"],
            phase="solver",
            prices=(job.get("prices") or {}).get("solver"),
            timeout=job["request_timeout"],
        )
        agent_module = importlib.import_module("agent.agent")

        def call_solver(config, prompt):
            if not isinstance(prompt, list):
                raise TypeError("VisualWebArena bridge requires OpenAI chat messages")
            return solver.complete(
                prompt,
                max_tokens=config.gen_config["max_tokens"],
                temperature=config.gen_config["temperature"],
                top_p=config.gen_config.get("top_p", 1),
            )

        agent_module.call_llm = call_solver
        # Let modern model aliases use the native prompt constructor without its
        # historical model-name whitelist dropping screenshot inputs.
        tokenizers = importlib.import_module("llms.tokenizers")
        original_tokenizer = tokenizers.Tokenizer

        class CompatibleTokenizer(original_tokenizer):
            def __init__(self, provider, model_name):
                try:
                    super().__init__(provider, model_name)
                except KeyError:
                    if provider != "openai":
                        raise
                    import tiktoken

                    self.tokenizer = tiktoken.get_encoding("cl100k_base")

        agent_module.Tokenizer = CompatibleTokenizer
        construct = native.construct_agent

        def construct_instrumented(*a, **kw):
            agent = construct(*a, **kw)
            agent.multimodal_inputs = job["observation_type"] == "image_som"
            original_next = agent.next_action

            def next_action(trajectory, intent, meta_data, images=None, **extra):
                state = trajectory[-1]
                observation = state.get("observation", {})
                text = str(observation.get("text", ""))
                screen_name = f"screenshot_{len(steps):04d}.png"
                if observation.get("image") is not None:
                    from PIL import Image

                    Image.fromarray(observation["image"]).save(out / screen_name)
                action = original_next(
                    trajectory, intent, meta_data=meta_data, images=images, **extra
                )
                steps.append(
                    {
                        "action": str(action.get("raw_prediction", "")),
                        "action_type": str(action.get("action_type", "")),
                        "observation": text[: job["trajectory_text_chars"]],
                        "observation_chars": len(text),
                        "screenshot": screen_name if (out / screen_name).exists() else None,
                        "url": str(getattr(state.get("info", {}).get("page"), "url", "")),
                    }
                )
                dump(out / "steps.json", steps)
                return action

            agent.next_action = next_action
            return agent

        native.construct_agent = construct_instrumented
        # Preserve the official judge model and grading prompts; account for its
        # requests separately. Never pass evaluation references to the solver.
        helpers = importlib.import_module("evaluation_harness.helper_functions")

        def call_judge(messages, model, temperature=0, max_tokens=768, top_p=1, **kw):
            judge = ChatClient(
                ledger,
                base=os.environ.get(
                    "VWA_JUDGE_BASE_URL",
                    os.environ.get("VWA_BASE_URL", "https://api.openai.com/v1"),
                ),
                key=os.environ.get("VWA_JUDGE_API_KEY", os.environ["VWA_API_KEY"]),
                model=model,
                phase="judge",
                prices=(job.get("prices") or {}).get("judge"),
                timeout=job["request_timeout"],
            )
            return judge.complete(
                messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p
            )

        helpers.generate_from_openai_chat_completion = call_judge
        router = native.evaluator_router

        def capture_router(*a, **kw):
            official = router(*a, **kw)

            def grade(*grade_a, **grade_kw):
                score = float(official(*grade_a, **grade_kw))
                if not 0 <= score <= 1:
                    raise ValueError("Official evaluator returned invalid score")
                evaluated.append(score)
                return score

            return grade

        native.evaluator_router = capture_router
        native.test(args, [job["config_file"]])
        if len(evaluated) == 1:
            result.update(
                status="passed" if evaluated[0] == 1 else "failed",
                score=evaluated[0],
                success=evaluated[0] == 1,
            )
    except Exception as exc:  # noqa: BLE001 -- persist a failed isolated-worker result
        result["error_type"] = type(exc).__name__
        # Traceback stays in stderr; no API responses or credentials in result JSON.
        import traceback

        traceback.print_exc()
    finally:
        result.update(
            elapsed=time.monotonic() - started,
            steps=steps,
            num_tool_calls=len(steps),
            num_turns=len(steps),
            usage=aggregate([x for x in read_jsonl(ledger) if x.get("phase") == "solver"]),
            judge_usage=aggregate([x for x in read_jsonl(ledger) if x.get("phase") == "judge"]),
            usage_ledger=str(ledger),
        )
        dump(out / "worker_result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    result = execute(json.loads(args.job.read_text()))
    return 0 if result["score"] is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
