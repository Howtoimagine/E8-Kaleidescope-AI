"""Memory subsystem abstractions (manager, shells, proximity, VAE).

This package modularizes memory components originally embedded in legacy server files.
It provides a stable import surface for:
- MemoryManager
- DimensionalShell
- ProximityEngine
- PathAsset
- VariationalAutoencoder (optional)

Downstream code should prefer importing from e8_mind.memory.* over legacy modules.
"""
from .manager import MemoryManager, GraphDB, GeometryHygieneMixin
from .shell import DimensionalShell
from .proximity import ProximityEngine, PathAsset
try:
    from .vae import VariationalAutoencoder
except Exception:  # Optional dependency (torch)
    VariationalAutoencoder = None  # type: ignore

__all__ = [
    "MemoryManager",
    "GraphDB",
    "GeometryHygieneMixin",
    "DimensionalShell",
    "ProximityEngine",
    "PathAsset",
    "VariationalAutoencoder",
]
