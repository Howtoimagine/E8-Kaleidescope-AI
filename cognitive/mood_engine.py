"""
Mood Engine - Affective Computing System

The MoodEngine simulates emotional states and mood dynamics for the E8Mind,
processing events and maintaining a multi-dimensional mood vector that influences
all other cognitive processes.
"""

import math
import time
import json
import zlib
import numpy as np
from collections import deque
from typing import Any, Dict, Optional

def mood_get(mood_vector: dict, key: str, default: float = 0.5) -> float:
    """Safely retrieves a float value from the mood vector dictionary."""
    return float(mood_vector.get(key, default))

def normalize_vector(v):
    """Helper function to ensure vectors have unit length."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-9 else v

class MoodEngine:
    """
    Manages the emotional state and mood dynamics of the E8Mind through a
    multi-dimensional mood vector that evolves based on cognitive events.
    """
    
    def __init__(self, console: Any, baseline=0.5, decay_rate=0.995):
        self.console = console
        self.baseline = baseline
        self.decay_rate = decay_rate
        self.mood_vector = {
            "intensity": 0.5, 
            "entropy": 0.5, 
            "coherence": 0.5, 
            "positivity": 0.5, 
            "fluidity": 0.5, 
            "intelligibility": 0.5
        }
        self.event_queue = deque()
        self._wx_last_code = None
        self._wx_repeat = 0
        if self.console:
            self.console.log("🌦️  Affective WeatherEngine Initialized.")

    def _nudge(self, key: str, amount: float):
        """Safely adjusts a mood dimension by the given amount."""
        if key in self.mood_vector: 
            self.mood_vector[key] = np.clip(self.mood_vector[key] + amount, 0.0, 1.0)

    def process_event(self, event_type: str, **kwargs):
        """Queues an event for processing during the next update cycle."""
        self.event_queue.append((event_type, kwargs))

    def update(self):
        """Processes all queued events and applies mood decay."""
        while self.event_queue:
            event_type, kwargs = self.event_queue.popleft()
            
            if event_type == "movement":
                mag = kwargs.get("magnitude", 0.0)
                self._nudge("intensity", 0.05 * min(mag, 5.0))
                themes = kwargs.get("themes", [])
                if any(t in themes for t in ["disorder", "burst"]): 
                    self._nudge("entropy", 0.15)
                    self._nudge("coherence", -0.10)
                if any(t in themes for t in ["integration", "stasis"]): 
                    self._nudge("coherence", 0.10)
                    self._nudge("entropy", -0.05)
                if "growth" in themes: 
                    self._nudge("fluidity", 0.08)
                    
            elif event_type == "new_concept":
                rating = kwargs.get("rating", 0.5)
                if rating > 0.75: 
                    self._nudge("coherence", 0.05*rating)
                    self._nudge("positivity", 0.10*rating)
                    self._nudge("intelligibility", 0.06*rating)
                else: 
                    self._nudge("entropy", 0.05 * (1.0 - rating))
                    
            elif event_type == "dream":
                self._nudge("entropy", 0.30)
                self._nudge("fluidity", 0.25)
                self._nudge("coherence", -0.15)
                self._nudge("intensity", 0.10)
                
            elif event_type == "reflection":
                self._nudge("coherence", 0.20)
                self._nudge("entropy", -0.10)
                self._nudge("positivity", 0.05)
                self._nudge("intelligibility", 0.08)
                
            elif event_type == "weather_tick":
                step = kwargs.get("step", 0)
                bh = float(kwargs.get("bh", 0.0))
                osc = 0.03 * math.sin(2.0 * math.pi * ((step % 240) / 240.0))
                noise = float(np.random.normal(0.0, 0.01))
                self._nudge("entropy", osc + noise + 0.15 * bh)
                self._nudge("intensity", 0.02 + 0.10 * bh)
                self._nudge("coherence", -0.5 * (osc + 0.10 * bh))
                
            elif event_type == "blackhole":
                m = float(kwargs.get("magnitude", 0.0))
                self._nudge("entropy", 0.25 + 0.10 * min(m, 5.0))
                self._nudge("intensity", 0.20)
                self._nudge("coherence", -0.15)
                
            elif event_type == "insight":
                r = float(kwargs.get("reward", 0.0))
                self._nudge("coherence", 0.12 * r)
                self._nudge("positivity", 0.05 * r)
                self._nudge("entropy", -0.04 * r)

        # Apply decay toward baseline
        for k, v in self.mood_vector.items():
            self.mood_vector[k] = v * self.decay_rate + self.baseline * (1.0 - self.decay_rate)

    def describe(self) -> str:
        """Returns a natural language description of the current mood state."""
        high = sorted(self.mood_vector.items(), key=lambda x: -x[1])
        low = sorted(self.mood_vector.items(), key=lambda x: x[1])
        return f"The mind feels predominantly {high[0][0]}, with undertones of {high[1][0]}. The least active state is {low[0][0]}."

    def get_symbolic_weather(self) -> str:
        """Returns a poetic, symbolic description of the current mood as 'weather'."""
        e = mood_get(self.mood_vector, "entropy")
        i = mood_get(self.mood_vector, "intensity") 
        c = mood_get(self.mood_vector, "coherence")
        
        def bin_with_hysteresis(value, thresholds, last_bin):
            padding = 0.05
            current_bin = sum(value > t for t in thresholds)
            if last_bin is not None:
                if current_bin != last_bin:
                    if current_bin > last_bin:
                        if value < thresholds[last_bin] + padding: 
                            return last_bin
                    else:
                        if value > thresholds[current_bin] - padding: 
                            return last_bin
            return current_bin

        b_e = bin_with_hysteresis(e, (0.25, 0.5, 0.75), getattr(self, "_b_e", None))
        b_i = bin_with_hysteresis(i, (0.25, 0.5, 0.75), getattr(self, "_b_i", None))
        b_c = bin_with_hysteresis(c, (0.25, 0.5, 0.75), getattr(self, "_b_c", None))
        
        self._b_e, self._b_i, self._b_c = b_e, b_i, b_c
        code = (b_e << 4) | (b_i << 2) | b_c
        
        if code == self._wx_last_code: 
            self._wx_repeat += 1
        else: 
            self._wx_repeat, self._wx_last_code = 0, code

        variants = {
            "storm": ["Volatile, sharp swings.", "Choppy, energy spikes.", "Jittery air, quick flips."],
            "calm":  ["Calm, steady drift.", "Gentle, small ripples.", "Soft, even flow."],
            "flow":  ["In-flow, coherent.", "Rolling, smooth arcs.", "Aligned, easy motion."],
            "turbulent": ["Turbulent, scattered.", "Noisy, low signal.", "Foggy, fragmented."],
        }

        if b_i >= 2 and b_e >= 2 and b_c <= 1: 
            bucket = "storm"
        elif b_c >= 2 and b_e <= 1: 
            bucket = "flow"
        elif b_e <= 1 and b_i <= 1: 
            bucket = "calm"
        else: 
            bucket = "turbulent"

        idx = (self._wx_repeat // 8) % len(variants[bucket])
        return variants[bucket][idx]

    def get_entropy_level(self) -> float:
        """Returns the current entropy level for external systems."""
        return mood_get(self.mood_vector, "entropy")

    def get_llm_persona_prefix(self) -> str:
        """Generates a persona prefix for LLM interactions based on current mood."""
        i = mood_get(self.mood_vector, 'intensity', 0.5)
        e = mood_get(self.mood_vector, 'entropy', 0.5)
        c = mood_get(self.mood_vector, 'coherence', 0.5)
        
        if e > 0.7 and i > 0.6:
            return "You are feeling chaotic, fragmented, and electric. Your response should be surreal and full of unexpected connections."
        elif c > 0.75:
            return "You are feeling exceptionally clear, logical, and focused. Your response should be precise and structured."
        elif i < 0.3:
            return "You are feeling calm, quiet, and introspective. Your response should be gentle and thoughtful."
        else:
            return "You are in a balanced state of mind. Your response should be clear and considered."

    def get_mood_modulation_vector(self, dim: int) -> np.ndarray:
        """Generates a mood-influenced modulation vector for embedding adjustment."""
        seed = zlib.adler32(json.dumps(self.mood_vector, sort_keys=True).encode())
        rng = np.random.default_rng(seed)
        
        coherence = mood_get(self.mood_vector, 'coherence', 0.5)
        entropy = mood_get(self.mood_vector, 'entropy', 0.5)
        
        modulation = rng.standard_normal(dim).astype(np.float32)
        modulation *= (1.0 + 0.5 * (coherence - 0.5))
        modulation += rng.standard_normal(dim).astype(np.float32) * 0.2 * entropy
        
        return normalize_vector(modulation)
