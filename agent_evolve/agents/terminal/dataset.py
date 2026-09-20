"""Dataset loader for Terminal-Bench 2.0 and 2.1 tasks.

Reads eval.yaml + compose.yaml from the challenges directory to build
a list of tasks with their Docker images, prompts, test files, and metadata.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# Terminal-Bench 2.1 is the default.  TB2_CHALLENGES_DIR is retained for
# backwards compatibility with existing launch scripts.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CHALLENGES_DIR = os.environ.get(
    "TB2_CHALLENGES_DIR",
    str(_PROJECT_ROOT / "agent_evolve" / "benchmarks" / "tb21" / "tasks"),
)

# Pinned source for auto-download
TB21_REPO = "https://github.com/harbor-framework/terminal-bench-2-1.git"
TB21_SUBPATH = "tasks"

DEFAULT_TAG = "20251031"


def ensure_challenges(challenges_dir: str | Path) -> Path:
    """Ensure the challenges directory exists and is populated.

    If the directory is empty or missing, downloads challenges from the
    pinned GitHub commit using git sparse checkout.
    """
    challenges_dir = Path(challenges_dir)
    if challenges_dir.exists() and any(challenges_dir.iterdir()):
        return challenges_dir

    logger.info("Challenges directory empty or missing: %s", challenges_dir)
    logger.info("Downloading Terminal-Bench 2.1 from %s ...", TB21_REPO)

    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_dir = Path(tmp_dir) / "repo"
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             TB21_REPO, str(repo_dir)],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["git", "sparse-checkout", "set", TB21_SUBPATH],
            cwd=repo_dir, check=True, capture_output=True, text=True,
        )

        src = repo_dir / TB21_SUBPATH
        challenges_dir.mkdir(parents=True, exist_ok=True)

        import shutil
        for item in src.iterdir():
            dest = challenges_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

    count = sum(1 for d in challenges_dir.iterdir() if d.is_dir())
    logger.info("Downloaded %d challenges to %s", count, challenges_dir)
    return challenges_dir


@dataclass
class TB2Task:
    """A Terminal-Bench 2.x task."""

    name: str
    prompt: str
    docker_image: str
    test_sh_path: str
    test_py_path: Optional[str] = None
    files: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    timeout: int = 900
    tests_dir: Optional[str] = None


def load_task(challenge_dir: str | Path) -> TB2Task:
    """Load a single challenge from its directory."""
    challenge_dir = Path(challenge_dir)
    name = challenge_dir.name

    eval_path = challenge_dir / "eval.yaml"
    if not eval_path.exists():
        raise FileNotFoundError(f"No eval.yaml in {challenge_dir}")
    with open(eval_path) as f:
        eval_cfg = yaml.safe_load(f)

    compose_path = challenge_dir / "compose.yaml"
    if not compose_path.exists():
        raise FileNotFoundError(f"No compose.yaml in {challenge_dir}")
    with open(compose_path) as f:
        compose_cfg = yaml.safe_load(f)

    # Extract docker image from compose
    services = compose_cfg.get("services", {})
    default_svc = services.get("default", {})
    docker_image = default_svc.get("image", f"alexgshaw/{name}:{DEFAULT_TAG}")

    # Extract prompt from default variant
    variants = eval_cfg.get("variants", {})
    default_variant = variants.get("default", {})
    prompt = default_variant.get("prompt", "")

    # File mappings — resolve to absolute paths
    raw_files = eval_cfg.get("files", {})
    files = {}
    for container_path, local_rel in raw_files.items():
        local_abs = challenge_dir / local_rel
        files[container_path] = str(local_abs)

    # Test files
    test_sh = challenge_dir / "tests" / "test.sh"
    test_py = challenge_dir / "tests" / "test_outputs.py"

    metadata = eval_cfg.get("metadata", {})
    metadata["category"] = metadata.get("category", "unknown")
    metadata["difficulty"] = metadata.get("difficulty", "unknown")
    timeout = metadata.get("agent_timeout_sec", 900)

    return TB2Task(
        name=name,
        prompt=prompt.strip(),
        docker_image=docker_image,
        test_sh_path=str(test_sh) if test_sh.exists() else "",
        test_py_path=str(test_py) if test_py.exists() else None,
        files=files,
        metadata=metadata,
        timeout=timeout,
        tests_dir=str(challenge_dir / "tests"),
    )


def load_tb21_task(task_dir: str | Path) -> TB2Task:
    """Load a Terminal-Bench 2.1 Harbor-format task."""
    task_dir = Path(task_dir)
    with (task_dir / "task.toml").open("rb") as f:
        cfg = tomllib.load(f)

    task_cfg = cfg.get("task", {})
    metadata = dict(cfg.get("metadata", {}))
    agent_cfg = cfg.get("agent", {})
    verifier_cfg = cfg.get("verifier", {})
    environment_cfg = cfg.get("environment", {})
    agent_timeout = int(agent_cfg.get("timeout_sec", 900))
    verifier_timeout = int(verifier_cfg.get("timeout_sec", 900))
    metadata.update({
        "benchmark": "terminal_bench_2_1",
        "task_name": task_cfg.get("name", task_dir.name),
        "agent_timeout_sec": agent_timeout,
        "verifier_timeout_sec": verifier_timeout,
        "environment": environment_cfg,
        "verifier_env": verifier_cfg.get("env", {}),
    })

    instruction = task_dir / "instruction.md"
    tests_dir = task_dir / "tests"
    test_sh = tests_dir / "test.sh"
    test_py = tests_dir / "test_outputs.py"
    docker_image = environment_cfg.get("docker_image")
    if not docker_image:
        raise ValueError(f"No environment.docker_image in {task_dir / 'task.toml'}")
    if not instruction.exists():
        raise FileNotFoundError(f"No instruction.md in {task_dir}")
    if not test_sh.exists():
        raise FileNotFoundError(f"No tests/test.sh in {task_dir}")

    return TB2Task(
        name=task_dir.name,
        prompt=instruction.read_text(encoding="utf-8").strip(),
        docker_image=str(docker_image),
        test_sh_path=str(test_sh),
        test_py_path=str(test_py) if test_py.exists() else None,
        metadata=metadata,
        timeout=agent_timeout,
        tests_dir=str(tests_dir),
    )


def load_all_tasks(challenges_dir: str | None = None) -> list[TB2Task]:
    """Load Terminal-Bench tasks, defaulting to the official 2.1 repository.

    Auto-downloads challenges from GitHub if the directory is empty.
    """
    challenges_dir_path = ensure_challenges(Path(challenges_dir or CHALLENGES_DIR))
    tasks = []
    for d in sorted(challenges_dir_path.iterdir()):
        if not d.is_dir():
            continue
        try:
            if (d / "task.toml").exists():
                tasks.append(load_tb21_task(d))
            elif (d / "eval.yaml").exists():
                tasks.append(load_task(d))
        except Exception as e:
            print(f"WARNING: Failed to load {d.name}: {e}")
    return tasks


def get_task(name: str, challenges_dir: str | None = None) -> TB2Task:
    """Load a single challenge by name."""
    challenges_dir_path = Path(challenges_dir or CHALLENGES_DIR)
    challenge_dir = challenges_dir_path / name
    if not challenge_dir.exists():
        raise ValueError(f"Challenge '{name}' not found in {challenges_dir_path}")
    if (challenge_dir / "task.toml").exists():
        return load_tb21_task(challenge_dir)
    return load_task(challenge_dir)
