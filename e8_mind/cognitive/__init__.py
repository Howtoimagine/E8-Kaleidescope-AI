"""Cognitive subsystem: insight, dreaming, affect, scheduling, and agent society.

This package extracts cognitive classes from the legacy monolith files into modular
components. Prefer importing from e8_mind.cognitive.*.
"""

from .insight import InsightAgent, NoveltyScorer, HypothesisValidator  # noqa: F401
from .dream import DreamEngine, DreamReplayService  # noqa: F401
from .affect import MoodEngine, SubconsciousLayer, GoalField, DriveSystem  # noqa: F401
from .scheduler import CognitiveScheduler  # noqa: F401
from .agents import SACMPOAgent, SocietyOfMind, BaseAgentAdapter  # noqa: F401

__all__ = [
    'InsightAgent', 'NoveltyScorer', 'HypothesisValidator',
    'DreamEngine', 'DreamReplayService',
    'MoodEngine', 'SubconsciousLayer', 'GoalField', 'DriveSystem',
    'CognitiveScheduler',
    'SACMPOAgent', 'SocietyOfMind', 'BaseAgentAdapter',
]
