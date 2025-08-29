from typing import Mapping, List, Tuple
import numpy as np

class PhysicsSemantics:
    name = "science"
    base_domain = "quantum mechanics; particle physics; cosmology; relativity; thermodynamics; entanglement; dark matter; spacetime curvature; quantum field theory"
    
    def persona_prefix(self, mood_vector: Mapping[str, float]) -> str:
        return "You are a theoretical physicist. Seek fundamental principles and unifying theories. Be precise. Be skeptical. No speculation."
    
    def pre_text(self, text: str) -> str:
        return text.replace("\u00AD", "").strip()
    
    def post_text(self, text: str) -> str:
        return text.strip()
    
    def pre_embed(self, text: str) -> str:
        return text + " | terms: quantum, particle, field, wave, photon, electron, boson, fermion, relativity, spacetime, entanglement, black hole, singularity, cosmology, Big Bang"
    
    def post_embed(self, vec) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float32)
        n = np.linalg.norm(v)
        return v if n == 0 else (v / n)
    
    def rerank(self, candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Boost items containing core physics terms. Simple keyword scoring.
        Keeps interface identical: input [(text, score)] -> output re-ordered list.
        """
        if not candidates:
            return candidates
        
        keywords = [
            "quantum", "relativity", "entanglement", "spacetime", "black hole", 
            "singularity", "string theory", "dark matter", "boson", "fermion", 
            "cosmology", "thermodynamics", "quantum field theory"
        ]
        
        def boost(item) -> Tuple[str, float]:
            text, score = item
            t = text.lower()
            bonus = 0.0
            for kw in keywords:
                if kw in t:
                    bonus += 0.15
            bonus = min(bonus, 0.60)
            return (text, score + bonus)
        
        boosted = [boost(it) for it in candidates]
        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted

PLUGIN = PhysicsSemantics()