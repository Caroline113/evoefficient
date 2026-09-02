"""Strategy configuration helpers for EvoEfficient SWE experiments."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml


STRATEGY_ORDER = [
    # "simple_memory",
    "observation_masking",
    "verbatim_compaction",
    # "cognitive_router",
    "plan_template_cache",
    "adaptive_tool_trimming",
    "difficulty_model_routing",
    "stagnation_rescue_stop",
    # "context_reduction",
    # "reasoning_budget",
    # "plan_template",
    # "tool_lazy_loading",
    # "observation_filter",
    # "model_routing_prompt",
    "skill_compilation",
]


DEFAULT_STRATEGIES: dict[str, dict[str, Any]] = {
    # "simple_memory": {
    #     "module": "M1_context",
    #     "description": "Compress working memory into compact task facts, hypotheses, changes, and checks.",
    #     "max_memory_items": 3,
    # },
    "observation_masking": {
        "module": "M1_context",
        "description": "Replace stale large tool observations with stable placeholders after several tool calls.",
        "stale_after_calls": 5,
        "large_output_chars": 4000,
        "placeholder_chars": 360,
    },
    "verbatim_compaction": {
        "module": "M1_context",
        "description": "Delete low-value lines from long outputs while preserving every retained line verbatim.",
        "mode": "objective",
        "max_chars": 6000,
        "max_lines": 160,
    },
    # "cognitive_router": {
    #     "module": "M2_reasoning",
    #     "description": "Use ACT-R style L0-L3 cognitive-depth routing for action, diagnosis, and reflection.",
    # },
    "plan_template_cache": {
        "module": "M4_plan",
        "description": "Reuse compact task templates by intent, caching how to solve rather than answers.",
        "template_dir": "memory/plan_templates",
    },
    "adaptive_tool_trimming": {
        "module": "M5_tool",
        "description": "Apply task-aware output trimming for shell, editor, and Python tools.",
        "max_chars": 8000,
        "test_max_lines": 160,
    },
    "difficulty_model_routing": {
        "module": "M7_route",
        "description": "Run the main agent on the hard model and let it explicitly delegate only bounded simple substeps to the easy model.",
        "levels": ["easy", "hard"],
    },
    "stagnation_rescue_stop": {
        "module": "M8_stop",
        "description": "Near the step limit, interrupt only clear stagnant exploration: first force a rescue checkpoint, then hard-stop non-submit tools if stagnation continues.",
        "rescue_at_step_ratio": 0.90,
        "hard_stop_at_step_ratio": 0.98,
        "min_steps_before_rescue": 45,
        "min_steps_before_hard_stop": 55,
        "stagnant_window": 12,
        "min_signals": 3,
        "exploration_ratio": 0.70,
        "max_unique_recent_inputs": 5,
        "max_unique_recent_files": 2,
        "no_edit_after_steps": 45,
        "unvalidated_edit_steps": 10,
        "rescue_grace_steps": 4,
        "max_rescue_messages": 1,
    },
    # "context_reduction": {
    #     "module": "legacy_context",
    #     "description": "Use a smaller conversation window and avoid rereading unchanged context.",
    #     "window_size": 20,
    # },
    # "reasoning_budget": {
    #     "module": "legacy_reasoning",
    #     "description": "Use hypothesis-first, low-token reasoning with escalation only on uncertainty.",
    # },
    # "plan_template": {
    #     "module": "legacy_planning",
    #     "description": "Inject a compact locate-hypothesize-change-verify workflow template.",
    # },
    # "tool_lazy_loading": {
    #     "module": "legacy_tools",
    #     "description": "Expose only the most useful tools for the current task to reduce schema overhead.",
    #     "allowed_tools": ["bash", "text_editor", "submit"],
    # },
    # "observation_filter": {
    #     "module": "legacy_observation",
    #     "description": "Filter long tool output before it enters the model context.",
    #     "max_chars": 6000,
    #     "pytest_max_lines": 140,
    # },
    # "model_routing_prompt": {
    #     "module": "legacy_routing",
    #     "description": "Prompt the solver to reserve expensive reasoning for high-risk steps.",
    # },
    "skill_compilation": {
        "module": "legacy_compilation",
        "description": "Add a compact reusable task skill that compiles common workflow steps.",
    },
}


def efficiency_dir(workspace_dir: str | Path) -> Path:
    return Path(workspace_dir) / "efficiency"


def active_config_path(workspace_dir: str | Path) -> Path:
    return efficiency_dir(workspace_dir) / "active_strategies.yaml"
    # return efficiency_dir(workspace_dir) / "strategy_library.yaml"


def load_active_config(workspace_dir: str | Path) -> dict[str, Any]:
    """Load the active strategy config for a workspace."""
    path = active_config_path(workspace_dir)
    if not path.exists():
        return {"strategies": [], "parameters": {}, "metadata": {}}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    strategies = data.get("strategies", [])
    if isinstance(strategies, str):
        strategies = [strategies]
    data["strategies"] = [s for s in strategies if s in DEFAULT_STRATEGIES]
    data.setdefault("parameters", {})
    data.setdefault("metadata", {})
    return data


def write_active_config(
    workspace_dir: str | Path,
    strategies: list[str],
    *,
    metadata: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
) -> Path:
    """Write active strategies into a workspace."""
    clean = [s for s in strategies if s in DEFAULT_STRATEGIES]
    params: dict[str, Any] = {}
    for name in clean:
        defaults = {
            k: v
            for k, v in DEFAULT_STRATEGIES[name].items()
            if k not in {"module", "description"}
        }
        if parameters and isinstance(parameters.get(name), dict):
            defaults.update(parameters[name])
        params[name] = defaults

    path = active_config_path(workspace_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategies": clean,
        "parameters": params,
        "metadata": metadata or {},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def apply_strategy_set_to_workspace(
    seed_workspace: str | Path,
    target_workspace: str | Path,
    strategies: list[str],
    *,
    metadata: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
    overwrite: bool = True,
) -> Path:
    """Copy a seed workspace and activate a strategy set in the copy."""
    seed = Path(seed_workspace)
    target = Path(target_workspace)
    if overwrite and target.exists():
        shutil.rmtree(target)
    shutil.copytree(seed, target)
    write_active_config(target, strategies, metadata=metadata, parameters=parameters)
    materialize_strategy_artifacts(target, strategies)
    return target


def materialize_strategy_artifacts(workspace_dir: str | Path, strategies: list[str]) -> None:
    """Write runtime artifacts needed by active strategy ids."""
    workspace = Path(workspace_dir)
    _materialize_compiled_skill(workspace, strategies)
    _materialize_efficiency_docs(workspace, strategies)


def strategy_prompt_block(
    config: dict[str, Any],
    task_profile: dict[str, Any] | None = None,
) -> str:
    """Return a system-prompt block for active prompt-level strategies."""
    strategies = set(config.get("strategies", []))
    if not strategies:
        return ""

    sections: list[str] = ["\n\n## EvoEfficient Active Strategy Config"]
    sections.append("Use the active efficiency strategy only when it preserves correctness.")
    if task_profile:
        estimated_tokens = task_profile.get("estimated_tokens", 0)
        profile_notes = task_profile.get("notes", {}) if isinstance(task_profile.get("notes", {}), dict) else {}
        sections.append(
            "\n### Task Profile\n"
            f"- task type: {task_profile.get('task_type', 'general')}\n"
            f"- domain: {task_profile.get('domain', 'generic')}\n"
            f"- difficulty: {task_profile.get('difficulty', 'hard')}\n"
            f"- context pressure: {task_profile.get('context_pressure', 'medium')}\n"
            f"- tool pressure: {task_profile.get('tool_pressure', 'medium')}\n"
            f"- verification pressure: {task_profile.get('verification_pressure', 'medium')}\n"
            f"- output pressure: {task_profile.get('output_pressure', 'medium')}\n"
            f"- route level: {task_profile.get('route_level', 'hard')}\n"
            f"- estimated tokens: {estimated_tokens}\n"
            f"- task signal: {profile_notes.get('task_reason', {}).get('task_type', 'unknown')}"
        )
        sections.append(
            "\n### Profiling Budget\n"
            "- Treat the profile as a cheap hint, not a heavy analysis step.\n"
            "- If the task looks easy, keep the profile lightweight and avoid extra profiling passes.\n"
            "- If the profile says the task is hard, spend the saved budget on verification and repair, not on repeating the profile.\n"
            "- Compare any upfront profiling or routing overhead with the expected later savings before running extra work."
        )

    sections.append(
        "\n### Universal Task Playbook\n"
        "- Keep a small, explicit state: task goal, current hypothesis, changed artifact, evidence, and next check.\n"
        "- Prefer the smallest useful observation, the smallest useful change, and the smallest useful verification.\n"
        "- If a step is purely exploratory, do not let it consume the budget reserved for verification.\n"
        "- When a strategy saves tokens, preserve correctness first and spend the savings on evidence that actually changes the decision."
    )

    if task_profile:
        task_type = str(task_profile.get("task_type", "general")).lower()
        if task_type == "debug":
            sections.append(
                "\n### Debug Playbook\n"
                "- Reproduce the failure first, then isolate the smallest failing path.\n"
                "- Keep the traceback, assertion, or failing case in view.\n"
                "- Patch one cause at a time and verify the exact fix path.\n"
                "- Do not widen the search unless the evidence changes."
            )
        elif task_type == "summarization":
            sections.append(
                "\n### Summarization Playbook\n"
                "- Preserve facts, numbers, names, and causal links.\n"
                "- Drop repetition, greetings, and filler first.\n"
                "- Keep the output structure stable and short.\n"
                "- Prefer faithful compression over rewritten interpretation."
            )
        elif task_type == "translation":
            sections.append(
                "\n### Translation Playbook\n"
                "- Keep meaning, tone, and constraints aligned with the source.\n"
                "- Do not invent missing facts.\n"
                "- Preserve identifiers, code, symbols, and names exactly when needed.\n"
                "- If ambiguity matters, flag it instead of silently guessing."
            )
        elif task_type == "classification":
            sections.append(
                "\n### Classification Playbook\n"
                "- Decide the label from the evidence, then stop.\n"
                "- Keep label definitions and boundary cases explicit.\n"
                "- Prefer stable categories over ad hoc labels.\n"
                "- Return confidence only when the task asks for it."
            )
        elif task_type == "routing":
            sections.append(
                "\n### Routing Playbook\n"
                "- Route by task pressure, not by habit.\n"
                "- Use the cheapest model or path that still keeps correctness intact.\n"
                "- Escalate only when context, tool use, or verification demand it.\n"
                "- Record the reason for the route so it can be reused later."
            )
        elif task_type == "transformation":
            sections.append(
                "\n### Transformation Playbook\n"
                "- Make input shape and output shape explicit.\n"
                "- Preserve invariants while changing representation.\n"
                "- Verify one normal case and one edge case.\n"
                "- If a transform can be streamed or localized, do that instead of rewriting the whole artifact."
            )
        elif task_type == "analysis":
            sections.append(
                "\n### Analysis Playbook\n"
                "- Compare only the dimensions that matter.\n"
                "- Keep baseline, change, and effect separate.\n"
                "- State what changed, why it matters, and what remains uncertain.\n"
                "- Prefer evidence tables or short bullets over broad narrative when possible."
            )
        elif task_type == "generation":
            sections.append(
                "\n### Generation Playbook\n"
                "- Respect the output contract first.\n"
                "- Generate only the minimum content that satisfies the request.\n"
                "- Reuse any provided structure, examples, or style hints.\n"
                "- If the request is underspecified, ask for the missing constraint or make the smallest safe assumption."
            )
        elif task_type == "planning":
            sections.append(
                "\n### Planning Playbook\n"
                "- Turn the goal into a short ordered list of actions.\n"
                "- Keep dependencies and decision points visible.\n"
                "- Prefer one plan that can be executed directly over several abstract options.\n"
                "- Revisit the plan only when the evidence forces a change."
            )
        elif task_type == "qa":
            sections.append(
                "\n### QA Playbook\n"
                "- Answer the question directly first.\n"
                "- Use supporting evidence only as much as needed.\n"
                "- Avoid overexplaining when the task needs a short factual response.\n"
                "- If the answer depends on a hidden assumption, state it clearly."
            )

    if "simple_memory" in strategies:
        sections.append(
            "\n### Simple Memory\n"
            "- Keep compact working memory with only task facts, current hypothesis, changed artifacts, evidence, and final verification.\n"
            "- Replace old details mentally with a short state record instead of replaying the full trace.\n"
            "- Do not copy long prior observations into reasoning unless they contain a failure signal, a key evidence span, or a changed region.\n"
            "- When the task already has a prior history, summarize it as what was tried, what failed, and what remains unresolved.\n"
            "- If the history is noisy, keep only the latest useful slice."
        )

    if "observation_masking" in strategies:
        sections.append(
            "\n### Observation Masking\n"
            "- Older large tool outputs may be replaced by placeholders such as `[old output masked]`.\n"
            "- If you need masked content again, rerun a targeted command for exact lines, traces, or slices.\n"
            "- Prefer naturally small commands: targeted search, narrow reads, focused validation, and artifact diffs.\n"
            "- If a masked result hides the part you need, ask a narrower question instead of requesting the full output again.\n"
            "- Use masking only when the output is stale or unlikely to change the next decision."
        )

    if "verbatim_compaction" in strategies:
        sections.append(
            "\n### Verbatim Compaction\n"
            "- Long outputs are compacted by deleting low-value lines; retained lines are original text, not summaries.\n"
            "- Treat compacted output as reliable but incomplete, and ask for a narrower command when context is missing.\n"
            "- For tests, focus on traceback, assertion, failing test name, and short summary.\n"
            "- For text or logs, keep exact lines that mention inputs, outputs, errors, thresholds, decisions, or boundaries.\n"
            "- Keep the output verbatim whenever exact wording matters."
        )

    if "context_reduction" in strategies:
        sections.append(
            "\n### Context Reduction\n"
            "- Prefer targeted search and narrow reads over reading whole files.\n"
            "- Do not reread unchanged outputs unless a new failure gives a reason.\n"
            "- Keep only task facts, edited artifacts, failing checks, and current hypothesis in mind.\n"
            "- If the task has many artifacts, hold a compact index of what each artifact proved.\n"
            "- Treat unchanged context as a cost, not a free default."
        )

    if "cognitive_router" in strategies:
        sections.append(
            "\n### Cognitive Router\n"
            "- Level 0 instinct: execute obvious known actions directly.\n"
            "- Level 1 recall: briefly match known patterns before choosing a file, document, test, or artifact.\n"
            "- Level 2 deliberation: spend careful reasoning only when diagnosing root cause or designing a change.\n"
            "- Level 3 reflection: use only after repeated failure or contradictory evidence, then revise the hypothesis.\n"
            "- Spend more tokens only when the new evidence truly changes the decision space.\n"
            "- If the task is already solved by a known pattern, do not upgrade the reasoning level."
        )

    if "reasoning_budget" in strategies:
        sections.append(
            "\n### Reasoning Budget\n"
            "- Use concise hypotheses for low-risk navigation and file-reading steps.\n"
            "- Escalate reasoning only for patch design, ambiguous failures, and final verification.\n"
            "- Avoid long exploratory monologues before the first concrete code location.\n"
            "- Spend the saved budget on the step that actually changes the outcome."
        )

    if "plan_template_cache" in strategies:
        sections.append(
            "\n### Plan Template Cache\n"
            "- First classify this task intent as one of: bug, transformation, extraction, summarization, classification, comparison, generation, routing, planning, or QA.\n"
            "- Reuse the matching plan shape: inspect evidence, form root cause or target result, make the narrowest change, verify with a focused check.\n"
            "- Cache only the plan pattern, never the final answer or task-specific artifact.\n"
            "- Template reuse should preserve the order of actions, not the domain-specific content.\n"
            "- The template should help the next task choose a route, not copy the previous answer."
        )

    if "plan_template" in strategies:
        sections.append(
            "\n### SWE Plan Template\n"
            "1. Locate: identify the core evidence and at most three likely artifacts.\n"
            "2. Hypothesize: state the smallest likely root cause or best next action before editing.\n"
            "3. Patch: make the narrowest change that addresses the issue.\n"
            "4. Verify: run a focused check, then submit or continue.\n"
            "5. Record: keep the exact evidence that proved the change worked."
        )

    if "tool_lazy_loading" in strategies:
        allowed = config.get("parameters", {}).get("tool_lazy_loading", {}).get("allowed_tools", [])
        sections.append(
            "\n### Tool Lazy Loading\n"
            f"- Active tool surface is intentionally small: {', '.join(allowed) or 'default compact tools'}.\n"
            "- Load or use only the tools needed for the current step.\n"
            "- Prefer search and targeted operations before broad inspection or bulk mutation.\n"
            "- If the current tool is not enough, switch to a more specific one instead of dumping more output."
        )

    if "adaptive_tool_trimming" in strategies:
        sections.append(
            "\n### Adaptive Tool Trimming\n"
            "- Tool outputs may be trimmed with rules that keep tracebacks, assertions, diffs, matched lines, and summaries.\n"
            "- Avoid commands that dump whole directories or complete files; ask for the narrow region you need.\n"
            "- When a trimmed output is insufficient, issue a smaller command instead of repeating the broad one.\n"
            "- Trimming should be configured per tool class when possible.\n"
            "- Prefer exact evidence over large previews."
        )

    if "difficulty_model_routing" in strategies:
        sections.append(
            "\n### Difficulty Model Routing\n"
            "- You are the main agent and you run on the hard model by default for the whole task.\n"
            "- You decide when a bounded substep is safe to delegate with the `route_model` tool.\n"
            "- Use `route_model(route_level=\"easy\", ...)` only for simple substeps over already collected context: extracting names or numbers, compressing a short observation, formatting a small answer, or checking a mechanical condition.\n"
            "- Keep root-cause diagnosis, patch design, code edits, failed-test recovery, security-sensitive choices, and final verification in your own hard-model reasoning loop.\n"
            "- Do not route an entire task to easy just because the profile looks easy; easy is a reversible step-level delegation.\n"
            "- Give `route_model` a short task, a small context, an expected output shape, and a reason why the step is simple.\n"
            "- If the easy result is uncertain, says `ESCALATE_TO_HARD`, contradicts evidence, or fails, continue with the hard model and do not repeat the easy call.\n"
            "- The route table is controlled by environment variables, so model choices can change without editing code."
        )

    if "stagnation_rescue_stop" in strategies:
        params = config.get("parameters", {}).get("stagnation_rescue_stop", {})
        try:
            rescue_ratio = float(params.get("rescue_at_step_ratio", 0.90))
        except (TypeError, ValueError):
            rescue_ratio = 0.90
        try:
            hard_ratio = float(params.get("hard_stop_at_step_ratio", 0.98))
        except (TypeError, ValueError):
            hard_ratio = 0.98
        sections.append(
            "\n### Stagnation Rescue Stop\n"
            f"- This run has a high-threshold stagnation guard near the tool-call limit: rescue at about {rescue_ratio:.0%}, hard stop at about {hard_ratio:.0%}.\n"
            "- It should not affect normal early solving. It only fires when several signals show repeated exploration without useful progress.\n"
            "- When rescue is triggered, stop broad exploration and choose one of: make the smallest plausible patch, run the most focused verification, or submit the current best patch.\n"
            "- After rescue, if the guard blocks another non-submit tool, submit the best available patch or an empty patch only if no safe change exists.\n"
            "- Treat the rescue message as a budget alarm, not as proof the task is solved."
        )

    if "model_routing_prompt" in strategies:
        sections.append(
            "\n### Difficulty-Aware Routing\n"
            "- Treat navigation, listing, and simple reads as low-risk steps.\n"
            "- Keep the hard model as the default route for the task and use easy only when a simple step is explicitly marked.\n"
            "- Spend hard-model reasoning when choosing between competing fixes or diagnosing failures.\n"
            "- If a cheap path fails twice, switch to careful analysis instead of repeating it.\n"
            "- Prefer easy only when correctness does not depend on new diagnosis, code edits, or verification judgment.\n"
            "- Make the route decision reversible through environment variables."
        )

    if "skill_compilation" in strategies:
        sections.append(
            "\n### Compiled Workflow\n"
            "- If a matching efficiency skill is available, read it once and follow its compact workflow.\n"
            "- Do not expand the workflow into broad exploration.\n"
            "- The goal is to reuse a proven procedure, not to memorize one solved case.\n"
            "- Prefer a generic skill that can work across task types."
        )

    return "\n".join(sections)


def _materialize_compiled_skill(workspace_dir: Path, strategies: list[str]) -> None:
    if "skill_compilation" not in strategies:
        return
    skill_dir = workspace_dir / "skills" / "evoefficient-task-playbook"
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    if skill_file.exists():
        return
    skill_file.write_text(
        """---
name: evoefficient-task-playbook
description: TRIGGER when a task needs a compact reusable workflow; DO NOT TRIGGER when no procedure reuse is needed.
---

# EvoEfficient Task Playbook

1. Extract the concrete goal, constraint, error, or output shape from the task.
2. Classify the task quickly: debug, transform, summarize, compare, classify, route, generate, plan, or QA.
3. Locate the smallest relevant evidence or artifact.
4. Write a one-sentence hypothesis or plan before changing anything.
5. Make one narrow change.
6. Verify with the smallest useful check.
7. Record what changed, what proved it, and what should happen next.
""",
        encoding="utf-8",
    )
    alias_dir = workspace_dir / "skills" / "evoefficient-swe-debug-loop"
    alias_dir.mkdir(parents=True, exist_ok=True)
    alias_file = alias_dir / "SKILL.md"
    if not alias_file.exists():
        alias_file.write_text(
            """---
name: evoefficient-swe-debug-loop
description: TRIGGER when fixing SWE issues with failing tests; DO NOT TRIGGER for non-code tasks.
---

# EvoEfficient SWE Debug Loop

Use the generic task playbook and keep the change focused on the smallest failing region.
""",
            encoding="utf-8",
        )


def _materialize_efficiency_docs(workspace_dir: Path, strategies: list[str]) -> None:
    if not strategies:
        return

    prompt_dir = workspace_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "efficiency.md").write_text(
        strategy_prompt_block({"strategies": strategies, "parameters": {}}).strip() + "\n",
        encoding="utf-8",
    )

    if "plan_template_cache" in strategies:
        template_dir = workspace_dir / "memory" / "plan_templates"
        template_dir.mkdir(parents=True, exist_ok=True)
        templates = {
            "investigate.md": """# Investigation Plan

1. Classify the task intent.
2. Locate the smallest relevant evidence.
3. State the minimal hypothesis.
4. Check one focused counterexample or validation step.
5. Record what changed and what remains uncertain.
""",
            "transform.md": """# Transformation Plan

1. Identify input shape and output shape.
2. Find the smallest place where the transform should happen.
3. Make the narrowest change that preserves correctness.
4. Verify on one representative case and one edge case.
""",
            "compare.md": """# Comparison Plan

1. Define what should be compared.
2. Extract comparable evidence from each side.
3. Normalize the representation if needed.
4. Compare only the signal that matters for the task.
""",
            "verify.md": """# Verification Plan

1. Identify the claim to verify.
2. Run the smallest useful check first.
3. Expand only if the first check is inconclusive.
4. Keep exact failure evidence and stop when the claim is settled.
""",
            "debug.md": """# Debug Plan

1. Classify the task intent.
2. Locate the smallest source region with targeted search.
3. State the minimal root-cause hypothesis.
4. Make one narrow patch.
5. Run a focused reproduction or validation step.
6. Record the outcome and submit when appropriate.
""",
            "summarize.md": """# Summarization Plan

1. Identify the required output length and audience.
2. Keep facts, numbers, and decisions.
3. Drop repetition and filler first.
4. Preserve the original meaning.
5. Check that the shortened output still answers the task.
""",
            "classify.md": """# Classification Plan

1. List the labels or classes.
2. Extract the evidence that distinguishes them.
3. Decide from the strongest signal first.
4. Check the edge cases.
5. Return the label and confidence if needed.
""",
            "route.md": """# Routing Plan

1. Estimate the task pressure.
2. Keep the task on hard unless this is an explicitly simple step.
3. Keep the route decision reversible.
4. Record why the route was chosen.
5. Revisit only when the evidence changes.
""",
            "generate.md": """# Generation Plan

1. Identify the output contract.
2. Reuse any provided structure or style.
3. Generate only the minimum content that satisfies the request.
4. Check the result against the requested constraints.
5. Keep the output format stable.
""",
            "qa.md": """# QA Plan

1. Answer the question directly.
2. Use the smallest supporting evidence needed.
3. State assumptions if the answer depends on hidden context.
4. Avoid unnecessary expansion.
5. Keep the answer readable and short.
""",
        }
        for name, content in templates.items():
            (template_dir / name).write_text(content, encoding="utf-8")

    if "difficulty_model_routing" in strategies:
        cost_profile = workspace_dir / "efficiency" / "cost_profile.yaml"
        cost_profile.parent.mkdir(parents=True, exist_ok=True)
        if not cost_profile.exists() or not cost_profile.read_text(encoding="utf-8").strip():
            cost_profile.write_text(
                """models:
  easy:
    role: delegated_simple_step
    relative_cost: 1.0
  hard:
    role: main_agent_default_reasoning
    relative_cost: 4.0
route_env:
  easy:
    model_env: AEVOLVE_MODEL_EASY
    provider_env: AEVOLVE_MODEL_PROVIDER_EASY
  hard:
    model_env: AEVOLVE_MODEL_HARD
    provider_env: AEVOLVE_MODEL_PROVIDER_HARD
""",
                encoding="utf-8",
            )
        route_doc = workspace_dir / "efficiency" / "route_env.md"
        route_doc.write_text(
            """# Route Environment Variables

Use environment variables to change the model for each route level without editing code.
Task execution defaults to the hard route. When `difficulty_model_routing` is active,
the main agent receives a `route_model` tool and may call the easy route only for
bounded simple substeps over already collected context.

Common level selector:

- `AEVOLVE_ROUTE_FORCE_LEVEL=easy|hard`

Model selection:

- `AEVOLVE_MODEL_EASY`, `AEVOLVE_MODEL_HARD`
- `AEVOLVE_MODEL_ID_EASY`, `AEVOLVE_MODEL_ID_HARD`
- `AEVOLVE_MODEL_PROVIDER_EASY`, `AEVOLVE_MODEL_PROVIDER_HARD`

OpenAI-compatible settings:

- `AEVOLVE_OPENAI_API_KEY_EASY`, `AEVOLVE_OPENAI_API_KEY_HARD`
- `AEVOLVE_OPENAI_BASE_URL_EASY`, `AEVOLVE_OPENAI_BASE_URL_HARD`

OpenRouter settings:

- `AEVOLVE_OPENROUTER_API_KEY_EASY`, `AEVOLVE_OPENROUTER_API_KEY_HARD`
- `AEVOLVE_OPENROUTER_BASE_URL_EASY`, `AEVOLVE_OPENROUTER_BASE_URL_HARD`

Route selection uses an explicit step route or forced route first; otherwise it falls back to hard.
For experiments, prefer leaving `AEVOLVE_ROUTE_FORCE_LEVEL` unset so the main agent
stays hard and only explicit `route_model(route_level="easy", ...)` calls use the
small model.
""",
            encoding="utf-8",
        )
