"""Official-style deterministic GAIA answer normalization and scoring."""

from __future__ import annotations

import re
import string
import warnings


def normalize_number_str(value: str) -> float:
    text = str(value)
    for char in ("$", "%", ","):
        text = text.replace(char, "")
    try:
        return float(text)
    except ValueError:
        return float("inf")


def normalize_str(value: str, remove_punct: bool = True) -> str:
    text = re.sub(r"\s", "", str(value)).lower()
    return text.translate(str.maketrans("", "", string.punctuation)) if remove_punct else text


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def question_scorer(model_answer: str | None, ground_truth: str) -> bool:
    """Match the public GAIA leaderboard scorer for numbers, lists and strings."""
    answer = "None" if model_answer is None else str(model_answer)
    truth = str(ground_truth)
    if _is_float(truth):
        return normalize_number_str(answer) == float(truth)
    if any(char in truth for char in (",", ";")):
        gt_parts = re.split(r"[,;]", truth)
        answer_parts = re.split(r"[,;]", answer)
        if len(gt_parts) != len(answer_parts):
            warnings.warn("Answer lists have different lengths.", UserWarning)
            return False
        return all(
            normalize_number_str(a) == float(g) if _is_float(g)
            else normalize_str(a, remove_punct=False) == normalize_str(g, remove_punct=False)
            for a, g in zip(answer_parts, gt_parts)
        )
    return normalize_str(answer) == normalize_str(truth)
