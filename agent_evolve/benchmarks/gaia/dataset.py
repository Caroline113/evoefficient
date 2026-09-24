"""Loader for the gated GAIA 2023 validation Level 3 split."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


EXPECTED_LEVEL3_VALIDATION_TASKS = 26
DATASET_ID = "gaia-benchmark/GAIA"


@dataclass
class GaiaTask:
    task_id: str
    question: str
    final_answer: str
    level: int = 3
    file_name: str = ""
    file_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _level(value: Any) -> int:
    text = str(value or "").strip().lower().replace("level", "").strip()
    return int(float(text))


def load_level3_validation(
    dataset_dir: str | Path | None = None,
    *,
    token: str | None = None,
    require_complete: bool = True,
) -> list[GaiaTask]:
    """Load GAIA 2023 Level 3 validation, downloading the gated snapshot if needed."""
    try:
        from datasets import load_dataset
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("GAIA requires `datasets` and `huggingface_hub` (install the mcp/all extra).") from exc

    auth_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if dataset_dir:
        root = Path(dataset_dir).expanduser().resolve()
    else:
        try:
            root = Path(snapshot_download(repo_id=DATASET_ID, repo_type="dataset", token=auth_token))
        except Exception as exc:
            raise RuntimeError(
                "Could not download gated GAIA. Accept the dataset terms at "
                "https://huggingface.co/datasets/gaia-benchmark/GAIA and run `hf auth login` "
                "or export HF_TOKEN."
            ) from exc

    dataset = load_dataset(str(root), "2023_level3", split="validation")
    tasks: list[GaiaTask] = []
    for row in dataset:
        level = _level(row.get("Level", 3))
        if level != 3:
            continue
        relative_attachment = str(row.get("file_path") or "")
        attachment = str((root / relative_attachment).resolve()) if relative_attachment else ""
        annotator = row.get("Annotator Metadata") or {}
        tasks.append(
            GaiaTask(
                task_id=str(row.get("task_id") or ""),
                question=str(row.get("Question") or "").strip(),
                final_answer=str(row.get("Final answer") or "").strip(),
                level=level,
                file_name=str(row.get("file_name") or ""),
                file_path=attachment,
                metadata={"annotator_metadata": annotator, "dataset_root": str(root)},
            )
        )
    if require_complete and len(tasks) != EXPECTED_LEVEL3_VALIDATION_TASKS:
        raise RuntimeError(
            f"Expected {EXPECTED_LEVEL3_VALIDATION_TASKS} GAIA validation Level 3 tasks, got {len(tasks)}."
        )
    return tasks
