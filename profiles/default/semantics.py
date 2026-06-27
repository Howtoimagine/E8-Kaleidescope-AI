# new semantics.py

from __future__ import annotations
from typing import Mapping, List, Tuple, Any, Dict, Callable
import os
import re
import unicodedata
import numpy as np
import yaml  # Added dependency: PyYAML

# A helper function for cosine similarity, used in reranking.
def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculates cosine similarity between two vectors."""
    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-9 or norm2 < 1e-9:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

class Prompts:
    """
    A simple class to load and render prompt templates from a dictionary.
    This handles the 'prompts' section of the YAML profile.
    """
    def __init__(self, prompt_data: Dict[str, str]):
        self._templates = prompt_data

    def render(self, key: str, **variables: Any) -> str:
        """Renders a prompt template with the given variables."""
        template = self._templates.get(key)
        if not template:
            # Fallback for keys not in the profile, like 'ask'
            if key == 'ask' and 'question' in variables:
                return variables['question']
            return f"Prompt key '{key}' not found in profile."
        
        # Replace placeholders like {variable} with their values
        return template.format(**variables)

class DynamicScienceSemantics:
    """
    A dynamic semantics plugin for the E8 Mind.

    This version loads its base configuration from a YAML file and connects
    to the live mind's state to provide context-aware persona, embedding hints,
    and result reranking.
    """
    def __init__(self, semantics_data: Dict[str, Any]):
        self.mind = None  # Attached later by the E8Mind instance
        
        # Load static configuration from the YAML data
        self.name: str = semantics_data.get("name", "unnamed_profile")
        self.base_domain: str = semantics_data.get("base_domain", "")
        
        # The persona_prefix is a lambda function stored as a string in YAML.
        # We use eval() to compile it into a callable Python function.
        # This is a powerful technique for dynamic configuration.
        persona_lambda_str = semantics_data.get("persona_prefix", "lambda mood: 'Default persona.'")
        try:
            self._persona_fn: Callable[[Dict], str] = eval(persona_lambda_str)
        except Exception as e:
            print(f"[Semantics] Failed to evaluate persona_prefix lambda: {e}")
            self._persona_fn = lambda mood: "Default persona (eval failed)."

    def attach_mind(self, mind_instance):
        """Attaches the main E8Mind instance to this semantics object."""
        self.mind = mind_instance
        print(f"[Semantics] '{self.name}' semantics are now attached to the E8Mind instance.")

    def persona_prefix(self, mood_vector: Mapping[str, float]) -> str:
        """🧠 Generates a dynamic persona by executing the loaded lambda."""
        if not mood_vector:
            mood_vector = {}
        return self._persona_fn(mood_vector)

    def pre_embed(self, text: str) -> str:
        """🎯 Adds a goal-aware hint and normalizes text before embedding."""
        hint = ""
        if self.mind and hasattr(self.mind, 'goal_field') and self.mind.goal_field.is_initialized:
            try:
                top_goal_name, _ = self.mind.goal_field.get_top_goals(k=1)[0]
                goal_hints = {
                    "synthesis": "Focus on unification, coherence, and underlying patterns.",
                    "novelty": "Focus on the unknown, anomalies, and breaking patterns.",
                    "stability": "Focus on core identity, reinforcement, and self-models.",
                    "curiosity": "Focus on causality, first principles, and asking 'why'."
                }
                hint = f" | Goal hint: {goal_hints.get(top_goal_name, '')}"
            except (IndexError, TypeError):
                pass
        
        # Normalize text
        t = (text or "").replace("\u00AD", "")
        t = unicodedata.normalize("NFKC", t)
        t = re.sub(r"\s+", " ", t).strip()
        return f"{t}{hint}"

    def post_embed(self, vec, host=None, dim=None) -> np.ndarray:
        """Normalizes and optionally snaps the vector to the E8 lattice."""
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(v)
        if norm > 1e-9:
            v = v / norm
        
        mind_host = host or self.mind
        if mind_host and hasattr(mind_host, "_snap_to_lattice"):
            try:
                v = mind_host._snap_to_lattice(v, dim or len(v))
            except Exception:
                pass
        return v

    def rerank(self, candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """✨ Reranks candidates based on the mind's current primary goal."""
        if not self.mind or not hasattr(self.mind, 'goal_field') or not self.mind.goal_field.is_initialized or not candidates:
            return candidates

        try:
            top_goal_name, _ = self.mind.goal_field.get_top_goals(k=1)[0]
        except (IndexError, TypeError):
            return candidates

        if top_goal_name == "novelty":
            return self._rerank_for_novelty(candidates)
        elif top_goal_name == "synthesis":
            return self._rerank_for_synthesis(candidates)
        else: # For stability, curiosity, or default
            return self._rerank_for_relevance(candidates)

    def _rerank_for_relevance(self, candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Default rerank: boost candidates similar to the current goal vector."""
        top_goal_name, _ = self.mind.goal_field.get_top_goals(k=1)[0]
        goal_vec = self.mind.goal_field.goals[top_goal_name]['embedding']
        
        scored = []
        for text, base_score in candidates:
            # Assumes text is the node label
            node_id = self.mind.memory.label_to_node_id.get(text)
            text_vec = self.mind.memory.main_vectors.get(node_id)
            
            if text_vec is None:
                scored.append((text, base_score))
                continue
                
            similarity = _cosine_similarity(text_vec, goal_vec)
            final_score = base_score + (similarity * 0.2) # Boost score based on relevance
            scored.append((text, final_score))
            
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def _rerank_for_novelty(self, candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Boosts candidates that are semantically different from existing memories."""
        scored = []
        for text, base_score in candidates:
            node_id = self.mind.memory.label_to_node_id.get(text)
            text_vec = self.mind.memory.main_vectors.get(node_id)
            if text_vec is None:
                scored.append((text, base_score))
                continue

            similar_nodes = self.mind.memory.find_similar_in_main_storage(text_vec, k=2) # k=2 to get nearest *other* node
            if len(similar_nodes) > 1:
                distance_to_nearest = similar_nodes[1][1]
                novelty_bonus = min(0.3, distance_to_nearest * 0.5)
                scored.append((text, base_score + novelty_bonus))
            else:
                scored.append((text, base_score + 0.3)) # Max bonus if no neighbors
        
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def _rerank_for_synthesis(self, candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Boosts candidates that bridge different clusters of concepts in memory."""
        # A full implementation requires community detection. Falling back to relevance.
        return self._rerank_for_relevance(candidates)

def load_profile(name: str) -> Tuple[DynamicScienceSemantics, Prompts]:
    """
    Loads a complete profile from a YAML file, returning separate
    semantics and prompts objects.
    """
    # This assumes the profile YAML files are in a 'profiles' subdirectory
    # relative to where this script is located.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, f"{name}.yml")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Profile '{name}' not found at {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        profile_data = yaml.safe_load(f)
        
    semantics_data = profile_data.get("semantics", {})
    prompts_data = profile_data.get("prompts", {})
    
    semantics_instance = DynamicScienceSemantics(semantics_data)
    prompts_instance = Prompts(prompts_data)
    
    return semantics_instance, prompts_instance