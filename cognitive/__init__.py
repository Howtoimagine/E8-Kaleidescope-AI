"""
Cognitive Systems Module

This module contains the high-level cognitive architectures and systems that
drive the E8Mind's reasoning, mood, goals, and consciousness-like behaviors.
"""

from .mood_engine import MoodEngine
from .subconscious import SubconsciousLayer  
from .goal_field import GoalField
from .drive_system import DriveSystem
from .emergence import detect_emergence_minima
from .reflection import ReflectionEngine

__all__ = [
    'MoodEngine',
    'SubconsciousLayer',
    'GoalField',
    'DriveSystem',
    'ReflectionEngine',
    'detect_emergence_minima'
]
