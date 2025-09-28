"""Monolith forwarder for modular physics (refactor complete for physics).

This module now serves as a stable import surface that re-exports the
modular physics implementations under ``e8_mind.physics``. Downstream code
can safely import from ``e8_mind.monolith`` while the actual source of truth
remains the modular packages.

Deprecation note:
- Legacy implementations in ``e8_mind_server_M24.1.py`` / ``M24.2.py`` are
    preserved only for historical context. At import time, those files rebind
    symbols to the modular implementations to avoid drift.

Preferred imports (examples):
        from e8_mind.physics.e8 import E8Physics
        from e8_mind.physics.mantle import HyperdimensionalFieldMantle
        from e8_mind.physics.horizon import HorizonManager, build_cross_horizon_kernel
    from e8_mind.physics.quantum import QuantumConfig, QuantumEngine
    from e8_mind.memory import MemoryManager, DimensionalShell, ProximityEngine, PathAsset, VariationalAutoencoder
"""

from __future__ import annotations

import os, sys, asyncio, math, json, time, logging, random, re

from .core.config import *  # noqa: F401,F403 - expose config constants temporarily
from .core.utils import *   # noqa: F401,F403 - expose utility helpers temporarily

# Physics subsystem imports (extracted and stabilized)
from .physics.e8 import E8Physics
from .physics.rotor import CliffordRotorGenerator
from .physics.mantle import HyperdimensionalFieldMantle
from .physics.horizon import (
    HorizonLayer,
    HorizonManager,
    build_e8_horizon,
    build_cross_horizon_kernel,
)
from .physics.quantum import (QuantumConfig, QuantumEngine)

# Memory subsystem (in-progress modularization)
from .memory import (
    MemoryManager,
    DimensionalShell,
    ProximityEngine,
    PathAsset,
    VariationalAutoencoder,
)

# Cognitive subsystem (new modularization)
from .cognitive import (
    InsightAgent,
    NoveltyScorer,
    HypothesisValidator,
    DreamEngine,
    DreamReplayService,
    MoodEngine,
    SubconsciousLayer,
    GoalField,
    DriveSystem,
    CognitiveScheduler,
    SACMPOAgent,
    SocietyOfMind,
    BaseAgentAdapter,
)

# Orchestrator (facade to legacy implementation for now)
try:
    from .core.mind import E8Mind, new_default_mind  # noqa: F401
except Exception:
    E8Mind = None  # type: ignore
    new_default_mind = None  # type: ignore

# Console initialization (wrapped)
try:
    from rich.console import Console
    _base_console = Console(record=True, force_terminal=True, color_system="auto")
except Exception:  # pragma: no cover
    try:
        from core.utils import Console as _FallbackConsole  # type: ignore
        _base_console = _FallbackConsole()
    except Exception:
        class _Dummy:
            def print(self, *a, **k): print(*a)
            def log(self, *a, **k): print(*a)
        _base_console = _Dummy()

console = DimmedLoggerConsole(SafeConsole(_base_console))

# Placeholders for future extracted classes/logic from legacy file.
def placeholder_start():
    console.log("[e8_mind] monolith placeholder start invoked.")

# Re-export key physics for compatibility when imported via monolith
__all__ = [
    'E8Physics',
    'CliffordRotorGenerator',
    'HyperdimensionalFieldMantle',
    'HorizonLayer', 'HorizonManager', 'build_e8_horizon', 'build_cross_horizon_kernel',
    'QuantumConfig', 'QuantumEngine',
    # memory
    'MemoryManager', 'DimensionalShell', 'ProximityEngine', 'PathAsset', 'VariationalAutoencoder',
    # cognitive
    'InsightAgent', 'NoveltyScorer', 'HypothesisValidator',
    'DreamEngine', 'DreamReplayService',
    'MoodEngine', 'SubconsciousLayer', 'GoalField', 'DriveSystem',
    'CognitiveScheduler',
    'SACMPOAgent', 'SocietyOfMind', 'BaseAgentAdapter',
    # orchestrator
    'E8Mind', 'new_default_mind',
]

if __name__ == "__main__":
    placeholder_start()
