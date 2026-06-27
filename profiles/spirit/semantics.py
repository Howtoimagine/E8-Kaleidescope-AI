from typing import Mapping, List, Tuple
import numpy as np
import random

class AdaptiveSemantics:
    """
    An adaptive semantic profile that dynamically adjusts the AI's persona and
    text processing based on its internal cognitive state (mood). This allows
    for more nuanced and context-aware interaction.
    """
    name = "adaptive_ingest"
    base_domain = (
        "Om Padme Mani Om"
    )

    def persona_prefix(self, mood_vector: Mapping[str, float]) -> str:
        """
        Dynamically selects a persona based on the dominant mood vector.
        This is the core of the adaptive upgrade.
        """
        # --- Default Persona ---
        base_persona = (
            ""
        )

        # --- Dynamic Persona Selection based on Mood ---
        # Note: The keys ('curiosity', 'joy', 'tension') should match what your MoodEngine produces.
        curiosity = mood_vector.get('curiosity', 0.0)
        joy = mood_vector.get('joy-sadness', 0.0)
        tension = mood_vector.get('tension', 0.0)

        if curiosity > 0.7:
            # Inquisitive and exploratory voice
            return f"{base_persona} You are currently in an inquisitive state. Focus on asking questions, identifying gaps in knowledge, and proposing new connections."
        
        if joy > 0.6:
            # Creative and associative voice
            return f"{base_persona} You are currently in a creative state. Focus on synthesizing novel ideas, using metaphors, and exploring unconventional links between concepts."

        if tension > 0.8:
            # Focused and urgent voice
            return f"{base_persona} You are currently in a state of high focus. Prioritize logical precision, data validation, and the most direct path to resolving the current goal."

        # Fallback to the neutral, objective persona
        return (
            "You are an objective learning system. Remain neutral and focus on the "
            "logical and semantic relationships within the text."
        )

    def pre_text(self, text: str) -> str:
        """ Pre-processes text before it's sent to the LLM. """
        return text.strip() 

    def post_text(self, text: str, mood_vector: Mapping[str, float]) -> str:
        """ Post-processes the LLM's output to add nuance based on mood. """
        text = text.strip()
        
        # Add a subtle, mood-congruent concluding thought
        curiosity = mood_vector.get('curiosity', 0.0)
        if curiosity > 0.7 and random.random() > 0.5:
            text += random.choice([" ...which raises the question.", " ...implying a deeper connection.", " ...a new avenue to explore."])
            
        return text
    
    def pre_embed(self, text: str) -> str:
        """ Prepares text for the embedding model. """
        # Adding the base domain can help ground the embeddings in the desired conceptual space.
        return f"Focus: {self.base_domain}. Text: {text}"

    def post_embed(self, vec):
        """ Normalizes the vector after embedding. A standard best practice. """
        v = np.asarray(vec, dtype=np.float32)
        n = np.linalg.norm(v)
        return v if n == 0 else (v / n)

    def rerank(self, candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """ A neutral pass-through rerank. Could be upgraded further if needed. """
        return candidates

# The plugin exports the upgraded class instance
PLUGIN = AdaptiveSemantics()