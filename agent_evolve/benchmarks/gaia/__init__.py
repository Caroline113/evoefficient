"""GAIA benchmark adapter."""

from .dataset import GaiaTask, load_level3_validation
from .evaluator import question_scorer

__all__ = ["GaiaTask", "load_level3_validation", "question_scorer"]
