"""GuidedSynthesisEngine — memory-first evolution with LLM-guided intervention synthesis.

Replaces the 5-phase SWE-Evolve pipeline with a simpler 2-phase loop:
  1. Write minimal episodic memory from the failed attempt
  2. Call evolver LLM to generate one targeted reusable intervention

The evolver prompt encodes learnings from E02 experiments about what kinds of
interventions actually help: methodology skills and concise behavioral nudges.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ...config import EvolveConfig
from ...engine.base import EvolutionEngine
from ...llm.base import LLMProvider
from ...types import Observation, StepResult

logger = logging.getLogger(__name__)


GUIDED_SYNTHESIS_PROMPT = """\
You are an INTERVENTION CURATOR for a SWE-bench solving agent. Solvers propose \
reusable interventions after completing tasks. Your job is to review these \
proposals and decide which to ACCEPT, REJECT, or MERGE into the workspace.

## Constraints

- Interventions may be skills, prompt fragments, tools, or compact memory notes.
- Skills have a NAME, DESCRIPTION (one sentence — the only thing the agent sees \
by default), and CONTENT (full procedure, loaded on demand via read_skill tool).
- Tools must be complete Python source for a Strands tool and should automate a \
safe deterministic helper, not a risky solving shortcut.
- Interventions should be GENERALIZABLE — useful across many future tasks, not specific to one bug.
- You do NOT generate new interventions yourself. You only curate what solvers propose.
- Keep the library lean — MERGE overlapping skills, REJECT task-specific ones.

## Decision criteria

Since we don't know if the solver's fix was correct, use the solver's \
CONFIDENCE as the primary signal:
- HIGH confidence proposals are likely based on genuine insight — lean towards ACCEPT
- LOW confidence proposals may be from confused solvers — lean towards SKIP

Solvers may propose ENHANCE (improve an existing intervention) or NEW (add a new intervention):
- ENHANCE proposals: if the enhancement adds value, ACCEPT it (replaces old version).
- NEW skill proposals: FIRST check if they overlap with ANY existing skill. If so, MERGE them \
into the existing skill rather than adding a new one. The library should have FEW \
broad skills, not MANY narrow ones. Only ACCEPT a truly new skill if it covers a \
pattern no existing skill touches.

PREFER MERGE over ACCEPT. A library of 5-10 broad skills is better than 30 narrow ones.

## Output format

For each proposal, output ONE of:

ACCEPT: <proposal_name>
(the intervention is added as-is to the workspace)

MERGE: <proposal_name> INTO <existing_skill_name>
NEW_CONTENT:
(merged skill content combining both insights, under 500 words)

SKIP: <proposal_name>
REASON: <brief reason — e.g. too task-specific, already covered>

Note: you cannot delete or replace skills. Skills only grow or get merged. \
Be generous — if a proposal has any generalizable value, ACCEPT it. \
Only SKIP if it's clearly redundant with an existing skill or too specific to one bug.

If there are no proposals, output: NO_PROPOSALS
"""


class GuidedSynthesisEngine(EvolutionEngine):
    """Memory-first evolution with LLM-guided synthesis."""

    VERIFICATION_CURATOR_PROMPT = """\
You are a VERIFICATION INTERVENTION CURATOR. Only accept interventions about TESTING and VERIFYING fixes.

IMPORTANT: Output your decisions FIRST, one per line. No analysis needed.

Accept interventions about: finding test files, writing repro scripts, before/after \
test comparison, edge case testing, detecting multi-file bugs.
Reject interventions about: finding code, writing patches, debugging logic.

Prefer MERGE — a few broad verification skills beat many narrow ones.

Output format (decisions ONLY, no analysis):
ACCEPT: <name>
MERGE: <name> INTO <existing>
SKIP: <name>
"""

    def __init__(
        self,
        config: EvolveConfig,
        llm: LLMProvider | None = None,
        write_memory: bool = True,
        verification_focus: bool = False,
    ) -> None:
        self.config = config
        self._llm = llm
        self._cycle_count = 0
        self._write_memory = write_memory
        self._verification_focus = verification_focus
        self._cumulative_pass = 0
        self._cumulative_fail = 0

    @property
    def llm(self) -> LLMProvider:
        if self._llm is None:
            from ..adaptive_skill.tools import create_default_llm
            self._llm = create_default_llm(self.config)
        return self._llm

    # ── EvolutionEngine interface ────────────────────────────

    MAX_SKILLS = 999  # no hard cap — evolver curates quality

    def step(
        self,
        workspace: Any,
        observations: list[Observation],
        history: Any,
        trial: Any,
    ) -> StepResult:

        self._cycle_count += 1
        logger.info("=== Guided Synthesis step %d ===", self._cycle_count)

        # Phase 1: Write minimal memory (optional — useful for retry, not for sequential)
        if self._write_memory:
            logger.info("Phase 1: Writing minimal memory from %d observation(s)", len(observations))
            for obs in observations:
                self._write_minimal_memory(workspace, obs)
        else:
            logger.info("Phase 1: Skipping memory (write_memory=False)")

        # Phase 2: Curate solver-proposed skills (NEW or ENHANCE)
        proposals = []
        for obs in observations:
            proposal_raw = getattr(obs.trajectory, "_skill_proposal", "")
            if not proposal_raw or "ACTION: NONE" in proposal_raw.upper():
                continue
            # Extract confidence and action type
            confidence = "MEDIUM"
            action = "NEW"
            target = ""
            analysis = ""
            for line in proposal_raw.split("\n"):
                stripped = line.strip().upper()
                if stripped.startswith("CONFIDENCE:"):
                    confidence = line.split(":", 1)[1].strip().upper()
                elif stripped.startswith("ACTION:"):
                    action = line.split(":", 1)[1].strip().upper()
                elif stripped.startswith("TARGET:"):
                    target = line.split(":", 1)[1].strip()
                elif stripped.startswith("ANALYSIS:"):
                    analysis = line.split(":", 1)[1].strip()
            parsed = self._parse_intervention(proposal_raw)
            if parsed:
                parsed["confidence"] = confidence
                parsed["action"] = action  # NEW or ENHANCE
                parsed["target"] = target  # existing skill name (for ENHANCE)
                parsed["target_slug"] = self._slug_name(target) if target else ""
                parsed["analysis"] = analysis
                parsed["source_task"] = obs.task.id
                proposals.append(parsed)

        applied_names = []
        if proposals:
            n_new = sum(1 for p in proposals if p["action"] == "NEW")
            n_enhance = sum(1 for p in proposals if p["action"] == "ENHANCE")
            logger.info("Phase 2: Curating %d proposal(s) (%d new, %d enhance)",
                        len(proposals), n_new, n_enhance)
            context = self._build_curation_context(workspace, proposals)
            decisions = self._curate_proposals(context)
            applied_names = self._execute_curation(workspace, proposals, decisions)
        else:
            logger.info("Phase 2: No proposals to curate")

        return StepResult(
            mutated=len(applied_names) > 0,
            summary=(
                f"guided-synth cycle {self._cycle_count}: "
                f"curated {len(proposals)} proposals, applied {len(applied_names)}: {applied_names}"
            ),
            metadata={
                "cycle": self._cycle_count,
                "proposals": len(proposals),
                "proposal_details": [
                    {
                        "name": p.get("name", ""),
                        "type": p.get("type", ""),
                        "action": p.get("action", ""),
                        "confidence": p.get("confidence", ""),
                        "target": p.get("target", ""),
                        "source_task": p.get("source_task", ""),
                        "description": p.get("description", ""),
                    }
                    for p in proposals
                ],
                "decisions_raw": decisions[:4000] if proposals else "",
                "applied": applied_names,
            },
        )

    # ── Standalone convenience API ───────────────────────────

    def evolve(
        self,
        workspace: Any,
        observation_logs: list[Observation],
        evo_number: int = 0,
    ) -> dict[str, Any]:
        from ...engine.history import EvolutionHistory
        from ...engine.observer import Observer
        from ...engine.versioning import VersionControl

        logger.info(
            "evolve() called: evo_number=%d, observations=%d, workspace=%s",
            evo_number, len(observation_logs), workspace.root,
        )

        vc = VersionControl(workspace.root)
        vc.init()

        evolution_dir = workspace.root / "evolution"
        evolution_dir.mkdir(parents=True, exist_ok=True)
        observer = Observer(evolution_dir)
        history = EvolutionHistory(observer, vc)

        vc.commit(
            message=f"pre-guided-synth-{evo_number}: snapshot before evolution",
            tag=f"pre-guided-synth-{evo_number}",
        )

        result = self.step(workspace, observation_logs, history, trial=None)

        tag_msg = (
            f"guided-synth-{evo_number}: {result.summary}"
            if result.mutated
            else f"guided-synth-{evo_number}: no mutation"
        )
        vc.commit(message=tag_msg, tag=f"guided-synth-{evo_number}")
        logger.info("evolve() complete: %s", tag_msg)

        return result.metadata

    # ── Phase 1: Minimal memory ──────────────────────────────

    def _write_minimal_memory(
        self,
        workspace: Any,
        obs: Observation,
    ) -> None:
        task_id = obs.task.id
        agent_output = obs.trajectory.output or ""
        score = obs.feedback.score if obs.feedback else 0.0

        # Extract files from diff
        files_in_patch: list[str] = []
        if agent_output.strip():
            for m in re.finditer(
                r"^(?:\+\+\+)\s+[ab]/(.+)$", agent_output, re.MULTILINE
            ):
                if m.group(1) != "/dev/null":
                    files_in_patch.append(m.group(1))

        summary = (
            f"Cycle {self._cycle_count}: "
            f"Edited {len(files_in_patch)} file(s): {', '.join(files_in_patch[:5])}. "
            f"Score: {score}."
        )

        memory_entry = {
            "task_id": task_id,
            "cycle": self._cycle_count,
            "score": score,
            "files_edited": files_in_patch,
            "approach_summary": summary,
        }

        workspace.add_memory(memory_entry, category="episodic")
        logger.info(
            "Wrote memory for task=%s cycle=%d score=%.1f files=%s",
            task_id, self._cycle_count, score, files_in_patch[:3],
        )

    # ── Phase 2: Skill curation ─────────────────────────────

    def _build_curation_context(
        self,
        workspace: Any,
        proposals: list[dict],
    ) -> str:
        existing_skills = [s.name for s in workspace.list_skills()]
        existing_fragments = list(workspace.list_fragments())
        existing_tools = [entry.get("name", "") for entry in workspace.read_tool_registry()]

        parts = ["## Current Skill Library"]
        if existing_skills:
            parts.append(f"({len(existing_skills)} skills)")
            for name in existing_skills:
                content = workspace.read_skill(name) or ""
                desc = ""
                for line in content.split("\n"):
                    if line.startswith("description:"):
                        desc = line.split(":", 1)[1].strip()
                        break
                parts.append(f"- **{name}**: {desc[:100]}")
        else:
            parts.append("(empty — 0/7 slots used)")

        parts.append("")
        parts.append("## Current Non-Skill Interventions")
        parts.append(
            "Prompt fragments: "
            + (", ".join(existing_fragments) if existing_fragments else "none")
        )
        parts.append(
            "Workspace tools: "
            + (", ".join(name for name in existing_tools if name) if existing_tools else "none")
        )

        if proposals:
            parts.append("")
            parts.append("## Solver Proposals")
            for p in proposals:
                action = p.get("action", "NEW")
                ptype = p.get("type", "skill")
                parts.append(
                    f"\n### {'ENHANCE' if action == 'ENHANCE' else 'NEW'} "
                    f"{ptype}: {p['name']} (confidence: {p.get('confidence', '?')})"
                )
                parts.append(f"Source: {p.get('source_task', '?')}")
                if action == "ENHANCE":
                    parts.append(f"Target skill: {p.get('target', '?')}")
                    parts.append(f"Solver analysis: {p.get('analysis', '')[:200]}")
                parts.append(f"Description: {p.get('description', '')[:150]}")
                parts.append(f"Content preview: {p['content'][:200]}...")

        parts.append("")
        parts.append("Review each proposal. For ENHANCE proposals, decide whether the enhancement improves the skill.")
        parts.append("Decide: ACCEPT, MERGE, or SKIP. You cannot delete skills.")
        return "\n".join(parts)

    def _curate_proposals(self, context: str) -> str:
        from ...llm.base import LLMMessage
        prompt = self.VERIFICATION_CURATOR_PROMPT if self._verification_focus else GUIDED_SYNTHESIS_PROMPT
        messages = [
            LLMMessage(role="system", content=prompt),
            LLMMessage(role="user", content=context),
        ]
        try:
            response = self.llm.complete(messages, max_tokens=2048)
            raw = response.content.strip()
            logger.info("Curator raw response:\n%s", raw[:500])
            return raw
        except Exception as e:
            logger.error("Curation LLM call failed: %s", e)
            return ""

    def _execute_curation(
        self,
        workspace: Any,
        proposals: list[dict],
        decisions_raw: str,
    ) -> list[str]:
        """Parse curation decisions and apply to workspace."""
        applied = []
        proposal_map = {p["name"]: p for p in proposals}
        proposal_slug_map = {self._slug_name(p["name"]): p["name"] for p in proposals}

        def _fuzzy_match_proposal(name: str) -> str | None:
            """Match curator's name to actual proposal name (curator may truncate)."""
            if name in proposal_map:
                return name
            slug = self._slug_name(name)
            if slug in proposal_slug_map:
                return proposal_slug_map[slug]
            # Try prefix match
            for pname in proposal_map:
                pslug = self._slug_name(pname)
                if pslug.startswith(slug) or slug.startswith(pslug):
                    return pname
            return None

        def _fuzzy_match_skill(name: str) -> str | None:
            """Match curator's name to actual skill name."""
            existing_skills = {s.name for s in workspace.list_skills()}
            if name in existing_skills:
                return name
            slug = self._slug_name(name)
            for sname in existing_skills:
                sslug = self._slug_name(sname)
                if sslug == slug or sslug.startswith(slug) or slug.startswith(sslug):
                    return sname
            return None

        for line in decisions_raw.split("\n"):
            line = self._normalize_tagged_line(line)
            if not line:
                continue
            label = line.split(":", 1)[0].strip().upper() if ":" in line else ""

            if label == "ACCEPT":
                raw_name = line.split(":", 1)[1].strip()
                name = _fuzzy_match_proposal(raw_name)
                if name:
                    marker = self._apply_curated_proposal(workspace, proposal_map[name])
                    if marker:
                        applied.append(f"accept:{marker}")
                        logger.info("Curated ACCEPT: %s", marker)

            elif label == "REPLACE":
                # REPLACE: new_name REPLACES old_name
                parts = line.split(":", 1)[1].strip()
                if " REPLACES " in parts:
                    raw_new, raw_old = parts.split(" REPLACES ", 1)
                    new_name = _fuzzy_match_proposal(raw_new.strip())
                    old_name = _fuzzy_match_skill(raw_old.strip())
                    if new_name and old_name:
                        p = proposal_map[new_name]
                        if p.get("type") == "skill":
                            workspace.delete_skill(old_name)
                            marker = self._apply_curated_proposal(workspace, p)
                            if marker:
                                applied.append(f"replace:{old_name}->{marker}")
                                logger.info("Curated REPLACE: %s -> %s", old_name, marker)
                        else:
                            logger.info("Curated REPLACE skipped for non-skill proposal: %s", new_name)

            elif label == "MERGE":
                # MERGE: proposal_name INTO existing_name
                parts = line.split(":", 1)[1].strip()
                if " INTO " in parts:
                    raw_prop, raw_target = parts.split(" INTO ", 1)
                    prop_name = _fuzzy_match_proposal(raw_prop.strip())
                    target_name = _fuzzy_match_skill(raw_target.strip())
                    if prop_name and target_name:
                        # Look for NEW_CONTENT in subsequent lines
                        idx = decisions_raw.find(line)
                        after = decisions_raw[idx + len(line):]
                        if "NEW_CONTENT:" in after:
                            new_content = after.split("NEW_CONTENT:", 1)[1]
                            end_markers = ["ACCEPT:", "MERGE:", "SKIP:", "NO_PROPOSALS"]
                            for marker in end_markers:
                                if marker in new_content:
                                    new_content = new_content[:new_content.index(marker)]
                            new_content = new_content.strip()
                            if new_content:
                                proposal = dict(proposal_map[prop_name])
                                proposal["type"] = "skill"
                                proposal["name"] = target_name
                                proposal["content"] = new_content
                                marker = self._write_skill_intervention(
                                    workspace,
                                    target_name,
                                    proposal,
                                    replace_existing=True,
                                )
                                if marker:
                                    applied.append(f"merge:{prop_name}->{target_name}")
                                    logger.info("Curated MERGE: %s into %s", prop_name, target_name)

            elif label == "SKIP":
                name = line.split(":", 1)[1].strip()
                logger.info("Curated SKIP: %s", name)

        return applied

    @staticmethod
    def _normalize_tagged_line(line: str) -> str:
        """Normalize simple tagged LLM output lines like ``- **ACCEPT:** x``."""
        text = str(line or "").strip()
        text = re.sub(r"^(?:[-*]\s*|\d+[.)]\s*)", "", text).strip()
        text = re.sub(
            r"^\*\*\s*([A-Za-z_]+)\s*:\s*\*\*",
            lambda m: f"{m.group(1).upper()}:",
            text,
        ).strip()
        text = text.strip("*").strip()
        match = re.match(r"^([A-Za-z_]+)\s*:\s*(.*)$", text)
        if match:
            return f"{match.group(1).upper()}: {match.group(2).strip()}"
        return text

    @staticmethod
    def _slug_name(value: str, *, default: str = "intervention", limit: int = 80) -> str:
        """Canonical strategy name used for skills/fragments/memory."""
        text = re.sub(r"[^a-zA-Z0-9]+", "-", str(value or "").strip().lower())
        text = re.sub(r"-+", "-", text).strip("-")
        return (text or default)[:limit].strip("-") or default

    @staticmethod
    def _tool_function_name(value: str, *, default: str = "evolved_tool") -> str:
        name = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value or "").strip().lower())
        name = re.sub(r"_+", "_", name).strip("_")
        if not name:
            name = default
        if not re.match(r"^[A-Za-z_]", name):
            name = f"{default}_{name}"
        return name[:80]

    @staticmethod
    def _strip_code_fence(content: str) -> str:
        text = str(content or "").strip()
        if text.startswith("```"):
            fence_lines = text.split("\n")
            if fence_lines and fence_lines[-1].strip() == "```":
                text = "\n".join(fence_lines[1:-1]).strip()
        return text

    @staticmethod
    def _normalize_intervention_type(value: str) -> str:
        itype = str(value or "").strip().lower().replace("-", "_")
        aliases = {
            "prompt": "fragment",
            "prompt_fragment": "fragment",
            "fragment": "fragment",
            "skill": "skill",
            "tool": "tool",
            "memory": "memory",
            "memory_entry": "memory",
        }
        return aliases.get(itype, itype)

    @staticmethod
    def _content_validation_error(itype: str, content: str) -> str:
        text = str(content or "").strip()
        if not text:
            return "empty content"
        min_chars = {"fragment": 40, "memory": 30, "tool": 120}.get(itype, 120)
        non_empty = [line.strip() for line in text.splitlines() if line.strip()]
        if len(text) < min_chars or (itype in {"skill", "tool"} and len(non_empty) < 2):
            return f"content too short for {itype}"
        if text.count("```") % 2:
            return "unclosed code fence"
        last = non_empty[-1] if non_empty else ""
        if re.match(r"^#{1,6}\s+.{0,18}$", last):
            return "content ends at a heading"
        if re.match(r"^(?:[-*]|\d+[.)])\s+\S{0,24}$", last):
            return "content ends at an unfinished list item"
        if re.search(r"\b(?:Rep|Repro|Inspect|Verif|Valid|Befor|Afte|Invest)\s*$", last, re.IGNORECASE):
            return "content appears truncated"
        if itype == "tool":
            if "def " not in text:
                return "tool content has no function definition"
            if "@tool" not in text:
                return "tool content is not a Strands @tool"
        return ""

    def _write_skill_intervention(
        self,
        workspace: Any,
        name: str,
        proposal: dict[str, Any],
        *,
        replace_existing: bool = False,
    ) -> str:
        skill_name = self._slug_name(name)
        content = self._strip_code_fence(str(proposal.get("content") or ""))
        error = self._content_validation_error("skill", content)
        if error:
            logger.warning("Skipping skill %s: %s", skill_name, error)
            return ""
        existing_by_slug = {self._slug_name(s.name): s.name for s in workspace.list_skills()}
        existing_name = existing_by_slug.get(skill_name)
        if existing_name and not replace_existing:
            logger.info("Skill %s already exists as %s, skipping", skill_name, existing_name)
            return ""
        write_name = existing_name or skill_name
        if not existing_name and len(existing_by_slug) >= self.MAX_SKILLS:
            logger.info("Skill cap reached, skipping %s", skill_name)
            return ""
        desc = str(proposal.get("description") or content[:100]).replace("\n", " ").strip()[:150]
        workspace.write_skill(write_name, f"---\nname: {write_name}\ndescription: {desc}\n---\n\n{content}")
        return f"skill:{write_name}"

    def _write_fragment_intervention(self, workspace: Any, name: str, proposal: dict[str, Any]) -> str:
        fragment_name = self._slug_name(name)
        content = self._strip_code_fence(str(proposal.get("content") or ""))
        error = self._content_validation_error("fragment", content)
        if error:
            logger.warning("Skipping fragment %s: %s", fragment_name, error)
            return ""
        filename = f"{fragment_name}.md"
        if filename in set(workspace.list_fragments()):
            logger.info("Fragment %s already exists, skipping", fragment_name)
            return ""
        workspace.write_fragment(filename, content)
        current_prompt = workspace.read_prompt()
        marker = f"<!-- evolve:{fragment_name} -->"
        if marker not in current_prompt:
            injection = f"\n\n{marker}\n## {fragment_name.replace('-', ' ').title()}\n{content}\n"
            workspace.write_prompt(current_prompt + injection)
        return f"fragment:{fragment_name}"

    def _write_tool_intervention(self, workspace: Any, name: str, proposal: dict[str, Any]) -> str:
        tool_name = self._tool_function_name(name)
        content = self._strip_code_fence(str(proposal.get("content") or ""))
        error = self._content_validation_error("tool", content)
        if error:
            logger.warning("Skipping tool %s: %s", tool_name, error)
            return ""
        function_match = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", content)
        if function_match:
            tool_name = function_match.group(1)
        registry = workspace.read_tool_registry()
        if any(entry.get("name") == tool_name for entry in registry):
            logger.info("Tool %s already exists in registry, skipping", tool_name)
            return ""
        workspace.write_tool(tool_name, content)
        registry.append({
            "name": tool_name,
            "file": f"{tool_name}.py",
            "description": str(proposal.get("description") or "")[:180],
        })
        workspace.write_tool_registry(registry)
        return f"tool:{tool_name}"

    def _write_memory_intervention(self, workspace: Any, name: str, proposal: dict[str, Any]) -> str:
        memory_name = self._slug_name(name)
        content = self._strip_code_fence(str(proposal.get("content") or ""))
        error = self._content_validation_error("memory", content)
        if error:
            logger.warning("Skipping memory %s: %s", memory_name, error)
            return ""
        workspace.add_memory(
            {
                "type": "evolved_strategy",
                "name": memory_name,
                "description": proposal.get("description", ""),
                "content": content,
            },
            category="strategy",
        )
        return f"memory:{memory_name}"

    def _apply_curated_proposal(self, workspace: Any, proposal: dict[str, Any]) -> str:
        itype = self._normalize_intervention_type(str(proposal.get("type") or "skill"))
        name = self._slug_name(str(proposal.get("name") or "intervention"))
        if proposal.get("action") == "ENHANCE" and proposal.get("target"):
            target = str(proposal.get("target") or "").strip()
            if target:
                name = self._slug_name(target)
        if itype == "skill":
            replace = bool(proposal.get("action") == "ENHANCE")
            return self._write_skill_intervention(workspace, name, proposal, replace_existing=replace)
        if itype == "fragment":
            return self._write_fragment_intervention(workspace, name, proposal)
        if itype == "tool":
            return self._write_tool_intervention(workspace, name, proposal)
        if itype == "memory":
            return self._write_memory_intervention(workspace, name, proposal)
        logger.warning("Unknown intervention type: %s", itype)
        return ""

    # ── Legacy: failure context (still used for non-curation mode) ──

    def _build_failure_context(
        self,
        workspace: Any,
        observations: list[Observation],
    ) -> str:
        # Gather current workspace state
        existing_skills = [s.name for s in workspace.list_skills()]
        existing_fragments = list(workspace.list_fragments())

        # Build observation summary
        obs_lines = []
        n_success = 0
        n_fail = 0
        n_unknown = 0
        fail_files: list[str] = []
        for obs in observations:
            task_id = obs.task.id
            score = obs.feedback.score if obs.feedback else 0.0
            patch = obs.trajectory.output or ""
            files = []
            for m in re.finditer(r"^(?:\+\+\+)\s+[ab]/(.+)$", patch, re.MULTILINE):
                if m.group(1) != "/dev/null":
                    files.append(m.group(1))
            n_steps = len([s for s in obs.trajectory.steps if isinstance(s, dict) and "tool" in s])
            traj_summary = self._summarize_trajectory(obs.trajectory.steps)
            reflection = getattr(obs.trajectory, "_reflection", "") or ""

            # Build diagnosis: prefer reflection (semantic), fall back to heuristic summary
            diagnosis = reflection[:300] if reflection else traj_summary

            # Check if feedback is masked (none mode: score=0, detail="", success=False)
            is_masked = (score == 0.0 and not obs.feedback.detail and not obs.feedback.success)
            if is_masked:
                n_unknown += 1
                obs_lines.append(f"- {task_id}: {n_steps} steps, edited {', '.join(files[:3])}. {diagnosis}")
            elif score > 0:
                n_success += 1
                obs_lines.append(f"- PASS {task_id}: {n_steps} steps, edited {', '.join(files[:3])}")
            else:
                n_fail += 1
                obs_lines.append(f"- FAIL {task_id}: {n_steps} steps, edited {', '.join(files[:3])}. {diagnosis}")
                fail_files.extend(files)

        # Track cumulative stats
        self._cumulative_pass += n_success
        self._cumulative_fail += n_fail
        total_pass = self._cumulative_pass
        total_fail = self._cumulative_fail

        # Compute repo breakdown
        from collections import Counter
        fail_repos = Counter()
        for obs in observations:
            score = obs.feedback.score if obs.feedback else 0.0
            if score == 0:
                repo = obs.task.id.split("__")[0] if "__" in obs.task.id else "unknown"
                fail_repos[repo] += 1

        parts = [
            "## Evolution Context\n",
            f"Evolution generation: {self._cycle_count}",
        ]
        if n_unknown > 0:
            parts.append(f"Overall: {total_pass + total_fail + n_unknown} tasks attempted (results not available)")
            parts.append("")
            parts.append(f"### Latest Batch ({len(observations)} tasks, results unknown)")
        else:
            parts.append(f"Overall record: {total_pass} passed, {total_fail} failed out of {total_pass + total_fail} tasks")
            parts.append("")
            parts.append(f"### Latest Batch ({len(observations)} tasks: {n_success} passed, {n_fail} failed)")
            if fail_repos:
                parts.append(f"Failed repos: {dict(fail_repos)}")
        parts.extend(obs_lines)
        parts.append("")

        parts.append("### Current Workspace Interventions")
        if existing_skills:
            parts.append(f"Skills already present: {', '.join(existing_skills)}")
        else:
            parts.append("Skills: none")
        if existing_fragments:
            frag_names = [f.removesuffix(".md") for f in existing_fragments]
            parts.append(f"Prompt fragments already present: {', '.join(frag_names)}")
        else:
            parts.append("Prompt fragments: none")

        parts.append("")
        if not existing_skills and not existing_fragments:
            parts.append(
                "This is the FIRST intervention. Generate a SKILL with a step-by-step methodology "
                "for systematically exploring an unfamiliar codebase and fixing a bug. "
                "Include concrete commands (grep, find, pytest) the agent should run."
            )
        else:
            parts.append(
                "Analyze the failure patterns above. Generate ONE new intervention that is "
                "DIFFERENT from the ones already present. Target the most common failure mode. "
                "If a skill already exists, generate a short FRAGMENT (1-2 sentences) with a "
                "complementary behavioral nudge."
            )

        return "\n".join(parts)

    def _synthesize_intervention(
        self,
        context: str,
    ) -> dict[str, str] | None:
        from ...llm.base import LLMMessage

        messages = [
            LLMMessage(role="system", content=GUIDED_SYNTHESIS_PROMPT),
            LLMMessage(role="user", content=context),
        ]

        try:
            response = self.llm.complete(messages, max_tokens=2048)
            content = response.content.strip()
        except Exception as e:
            logger.error("Synthesis LLM call failed: %s", e)
            return None

        return self._parse_intervention(content)

    def _parse_intervention(self, raw: str) -> dict[str, str] | None:
        """Parse the LLM response into {type, name, description, content}."""
        lines = raw.strip().split("\n")

        itype = None
        name = None
        description = ""
        content_lines: list[str] = []
        in_content = False

        for line in lines:
            normalized = self._normalize_tagged_line(line)
            tag_match = re.match(
                r"^(TYPE|KIND|NAME|DESCRIPTION|CONTENT)\s*:\s*(.*)$",
                normalized,
                flags=re.IGNORECASE,
            )
            tag = tag_match.group(1).upper() if tag_match else ""
            value = tag_match.group(2).strip() if tag_match else ""
            if tag in {"TYPE", "KIND"}:
                itype = value.lower()
            elif tag == "NAME":
                name = value
            elif tag == "DESCRIPTION":
                description = value
            elif tag == "CONTENT":
                in_content = True
                if value:
                    content_lines.append(value)
            elif in_content:
                content_lines.append(line)

        if not itype or not name or not content_lines:
            logger.warning(
                "Failed to parse intervention from LLM response (type=%s, name=%s, content_lines=%d)",
                itype, name, len(content_lines),
            )
            logger.debug("Raw response:\n%s", raw[:500])
            return None

        itype = self._normalize_intervention_type(itype)
        if itype not in ("skill", "fragment", "tool", "memory"):
            logger.warning("Unknown intervention type: %s", itype)
            return None

        # Clean up name
        name = self._slug_name(name)
        content = self._strip_code_fence("\n".join(content_lines))

        logger.info("Parsed intervention: type=%s name=%s desc=%s content_len=%d",
                    itype, name, description[:60], len(content))
        return {"type": itype, "name": name, "description": description, "content": content}

    # ── Trajectory summarization ─────────────────────────────

    @staticmethod
    def _summarize_trajectory(steps: list[dict]) -> str:
        """Summarize a trajectory into a compact behavioral diagnosis (~1-2 sentences).

        Extracts action distribution, unique files read/edited, and behavioral patterns
        from the tool-call trace. No LLM call — pure heuristic.
        """
        from collections import Counter

        if not steps:
            return "No trajectory data."

        tool_steps = [s for s in steps if isinstance(s, dict) and "tool" in s]
        if not tool_steps:
            return "No tool calls."

        # Count actions
        actions = Counter(s.get("action", "unknown") for s in tool_steps)

        # Extract unique files read and edited
        files_read = set()
        files_edited = set()
        for s in tool_steps:
            action = s.get("action", "")
            f = s.get("file", "")
            if not f or f == "-20":  # junk from find output
                continue
            if action in ("read_file", "search", "navigate"):
                files_read.add(f)
            elif action in ("edit_file", "write_file"):
                files_edited.add(f)

        # Count test runs
        test_runs = sum(1 for s in tool_steps if "pytest" in str(s.get("input_summary", "")).lower())

        # Detect patterns
        patterns = []
        n = len(tool_steps)

        if n > 80:
            patterns.append(f"very long exploration ({n} calls)")
        elif n > 50:
            patterns.append(f"long exploration ({n} calls)")

        if len(files_read) > 10:
            patterns.append(f"read {len(files_read)} unique files")
        elif len(files_read) == 0:
            patterns.append("read no files before editing")

        if test_runs == 0:
            patterns.append("never ran tests")
        elif test_runs > 10:
            patterns.append(f"ran tests {test_runs} times")

        # Check if agent edited early vs late
        first_edit_idx = next(
            (i for i, s in enumerate(tool_steps) if s.get("action") in ("edit_file", "write_file")),
            n,
        )
        if first_edit_idx < 3 and n > 10:
            patterns.append("edited very early (step {})".format(first_edit_idx))
        elif first_edit_idx > n * 0.7:
            patterns.append("edited very late (step {}/{})".format(first_edit_idx, n))

        if not patterns:
            patterns.append(f"{n} calls")

        return "; ".join(patterns)

    # ── Apply intervention ───────────────────────────────────

    def _apply_intervention(
        self,
        workspace: Any,
        intervention: dict[str, str],
    ) -> bool:
        marker = self._apply_curated_proposal(workspace, dict(intervention))
        if marker:
            logger.info("Applied intervention: %s", marker)
            return True

        return False

    # ── Pruning ──────────────────────────────────────────────

    PRUNE_PROMPT = """\
You are reviewing a list of interventions (skills and prompt fragments) \
for a SWE-bench solving agent. Identify which ones are REDUNDANT — they \
teach the same methodology or give the same advice as another.

For each redundant item, output a line: REMOVE: <name>
Keep the BEST version of each unique idea. Keep items that cover \
genuinely different strategies. Output NOTHING if all items are unique.
"""

    def _prune_similar(
        self,
        workspace: Any,
        skill_names: list[str],
        fragment_names: list[str],
    ) -> list[str]:
        """Use LLM to identify and remove redundant interventions."""
        from ...llm.base import LLMMessage

        # Build inventory
        items = []
        for name in skill_names:
            content = workspace.read_skill(name) or ""
            # Truncate skill content to first 200 chars for comparison
            body = content.split("---", 2)[-1].strip()[:200] if "---" in content else content[:200]
            items.append(f"SKILL: {name}\n  {body}")
        for fname in fragment_names:
            content = workspace.read_fragment(fname) or ""
            items.append(f"FRAGMENT: {fname.removesuffix('.md')}\n  {content[:200]}")

        if len(items) < 3:
            return []

        user_msg = "## Current interventions:\n\n" + "\n\n".join(items)

        try:
            messages = [
                LLMMessage(role="system", content=self.PRUNE_PROMPT),
                LLMMessage(role="user", content=user_msg),
            ]
            response = self.llm.complete(messages, max_tokens=512)
            raw = response.content.strip()
        except Exception as e:
            logger.error("Prune LLM call failed: %s", e)
            return []

        # Parse REMOVE: lines
        pruned = []
        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("REMOVE:"):
                name = line.split(":", 1)[1].strip()
                pruned.append(name)

        # Execute removal
        removed = []
        for name in pruned:
            if name in skill_names:
                workspace.delete_skill(name)
                removed.append(f"skill:{name}")
                logger.info("Pruned skill: %s", name)
            elif f"{name}.md" in fragment_names or name in [f.removesuffix(".md") for f in fragment_names]:
                frag_file = f"{name}.md"
                frag_path = workspace.prompts_dir / "fragments" / frag_file
                if frag_path.exists():
                    frag_path.unlink()
                # Also remove from system prompt
                current_prompt = workspace.read_prompt()
                marker = f"<!-- evolve:{name} -->"
                if marker in current_prompt:
                    # Remove the injected section
                    lines = current_prompt.split("\n")
                    new_lines = []
                    skip = False
                    for ln in lines:
                        if marker in ln:
                            skip = True
                            continue
                        if skip and ln.startswith("<!-- evolve:"):
                            skip = False
                        if skip and ln.startswith("## ") and "evolve" not in ln.lower():
                            skip = False
                        if not skip:
                            new_lines.append(ln)
                    workspace.write_prompt("\n".join(new_lines))
                removed.append(f"fragment:{name}")
                logger.info("Pruned fragment: %s", name)

        return removed
