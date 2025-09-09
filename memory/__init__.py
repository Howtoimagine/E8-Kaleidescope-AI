"""
Memory Management System for E8Mind

This module provides the core memory management infrastructure including:
- Graph database management
- Vector storage and retrieval
- Novelty scoring and coherence evaluation
- Advanced memory consolidation systems
"""

from .manager import MemoryManager
from .graph import GraphDB
from .structures import (
    NoveltyScorer, HopfieldModern, KanervaSDM, 
    VSA, MicroReranker, EmergenceSeed
)

__all__ = [
    'MemoryManager',
    'GraphDB', 
    'NoveltyScorer',
    'HopfieldModern',
    'KanervaSDM',
    'VSA', 
    'MicroReranker',
    'EmergenceSeed'
]
