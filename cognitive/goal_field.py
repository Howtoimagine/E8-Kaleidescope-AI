"""
Goal Field - Dynamic goal activation and management system.

The GoalField manages multiple abstract goals that compete for attention
and influence the mind's behavior through activation levels.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from utils.helpers import sanitize_line


class GoalField:
    """
    Manages and tracks the activation levels of abstract goals.
    Goals compete for attention and influence the mind's cognitive processes.
    """
    
    def __init__(self, embedding_fn: Callable, console: Any):
        self.console = console
        self.embedding_fn = embedding_fn
        self.goals: Dict[str, Dict[str, Any]] = {}
        self.is_initialized = False
        self.activation_decay = 0.98

    async def initialize_goals(self):
        """Initialize the core set of goals with their embeddings."""
        if self.is_initialized:
            return
            
        goal_definitions = {
            "synthesis": "Achieve synthesis and coherence; find the unifying pattern.",
            "novelty": "Look at novelty and the unknown; break existing patterns.",
            "stability": "Reinforce core identity and create a stable self-model.",
            "curiosity": "Understand the 'why'; ask questions and follow causal chains."
        }
        
        for name, desc in goal_definitions.items():
            vec = await self.embedding_fn(desc)
            self.goals[name] = {
                "description": desc,
                "embedding": vec,
                "activation": 0.25
            }
            
        self.is_initialized = True
        self.console.log("🌻 Goal-Field Initialized with attractors.")

    def decay(self):
        """Apply temporal decay to all goal activations."""
        for name in self.goals:
            self.goals[name]["activation"] *= self.activation_decay

    def update_from_embedding(self, vector: np.ndarray, weight: float = 0.1):
        """Update goal activations based on similarity to an input vector."""
        if not self.is_initialized or vector is None:
            return
            
        total_similarity, sims = 0.0, {}
        
        def _cos(a, b):
            a = np.asarray(a, dtype=np.float32)
            b = np.asarray(b, dtype=np.float32)
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-9 or nb < 1e-9:
                return 0.0
            return float(np.dot(a, b) / (na * nb))
        
        for name, goal_data in self.goals.items():
            sim = _cos(vector, goal_data.get("embedding", np.zeros_like(vector)))
            sims[name], total_similarity = sim, total_similarity + sim
            
        if total_similarity > 1e-9:
            for name, sim in sims.items():
                self.goals[name]["activation"] += weight * (sim / total_similarity)
                
        self._normalize_activations()

    def update_from_mood(self, mood_vector: dict):
        """Update goal activations based on current mood state."""
        if not self.is_initialized:
            return
            
        def mood_get(mood_dict: dict, key: str, default: float = 0.5) -> float:
            return float(mood_dict.get(key, default))
            
        mood_updates = {
            "synthesis": mood_get(mood_vector, "coherence", 0.5) * 0.05,
            "novelty": mood_get(mood_vector, "entropy", 0.5) * 0.05,
            "stability": (1.0 - mood_get(mood_vector, "intensity", 0.5)) * 0.03,
            "curiosity": mood_get(mood_vector, "intelligibility", 0.5) * 0.04
        }
        
        for name, update in mood_updates.items():
            if name in self.goals:
                self.goals[name]["activation"] += update
                
        self._normalize_activations()

    def _normalize_activations(self):
        """Ensure all goal activations sum to 1.0."""
        total_activation = sum(g["activation"] for g in self.goals.values())
        if total_activation > 1e-9:
            for name in self.goals:
                self.goals[name]["activation"] /= total_activation

    def get_top_goals(self, k: int = 2) -> List[Tuple[str, str]]:
        """Get the k most active goals with their descriptions."""
        if not self.is_initialized:
            return [("nascent", "The mind is still forming its goals.")]
            
        if not self.goals:
            return [("empty", "No goals defined.")]
            
        sorted_goals = sorted(
            self.goals.items(), 
            key=lambda item: -item[1].get("activation", 0.0)
        )
        
        return [
            (name, data.get("description", "No description")) 
            for name, data in sorted_goals[:k]
        ]
