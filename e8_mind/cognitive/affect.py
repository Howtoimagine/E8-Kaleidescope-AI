from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os
import math
import json
import zlib
from collections import deque

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    class _NP:
        def clip(self, x, lo, hi):
            try:
                return max(lo, min(hi, float(x)))
            except Exception:
                return lo
        def zeros(self, n, dtype=None):
            return [0.0]*int(n)
        def linalg_norm(self, v):
            try:
                s = sum(float(x)*float(x) for x in v)
                return s**0.5
            except Exception:
                return 0.0
        def random_normal(self, n):
            import random
            return [random.gauss(0.0, 1.0) for _ in range(int(n))]
        def array(self, x, dtype=None):
            return x
        def dot(self, a, b):
            return sum(x*y for x, y in zip(a, b))
        def mean(self, arr):
            return sum(arr)/max(1, len(arr)) if arr else 0.0
        class random:  # type: ignore
            @staticmethod
            def default_rng(seed=None):
                class RNG:
                    def standard_normal(self, n):
                        import random
                        random.seed(seed)
                        return [random.gauss(0.0, 1.0) for _ in range(int(n))]
                return RNG()
    np = _NP()  # type: ignore


EMBED_DIM = int(os.getenv("E8_EMBED_DIM", "768"))


def normalize_vector(v):
    try:
        arr = v if isinstance(v, list) else np.array(v)
        n = np.linalg.norm(arr) if hasattr(np, 'linalg') else np.linalg_norm(arr)  # type: ignore[attr-defined]
        if n and n > 0:
            return arr / n
        return arr
    except Exception:
        return v


def mood_get(d: Dict[str, float], k: str, default: float = 0.5) -> float:
    try:
        v = float(d.get(k, default))
        if v != v:  # NaN check
            return default
        return max(0.0, min(1.0, v))
    except Exception:
        return default


class GoalField:
    def __init__(self, embedding_fn, console: Any, mind=None):
        self.console = console
        self.embedding_fn = embedding_fn
        self.mind = mind
        self.goals: Dict[str, Dict[str, Any]] = {}
        self.is_initialized = False
        self.activation_decay = 0.98

    async def initialize_goals(self):
        if self.is_initialized:
            return
        goal_definitions = {
            "synthesis": "Achieve synthesis and coherence; find the unifying pattern.",
            "novelty": "Look at novelty and the unknown; break existing patterns.",
            "stability": "Reinforce core identity and create a stable self-model.",
            "curiosity": "Understand the 'why'; ask questions and follow causal chains.",
        }
        for name, desc in goal_definitions.items():
            vec = await self.embedding_fn(desc)
            if self.mind and hasattr(self.mind, '_vae_ingest'):
                try:
                    self.mind._vae_ingest(vec)
                except Exception:
                    pass
            self.goals[name] = {"description": desc, "embedding": vec, "activation": 0.25}
        self.is_initialized = True
        try:
            if self.console:
                self.console.log("🌙 Goal-Field Initialized with attractors.")
        except Exception:
            pass

    def decay(self):
        for name in self.goals:
            self.goals[name]["activation"] *= self.activation_decay

    @staticmethod
    def _cos_sim(a, b) -> float:
        try:
            a = np.array(a); b = np.array(b)
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            return float(np.dot(a, b) / denom) if denom else 0.0
        except Exception:
            return 0.0

    def update_from_embedding(self, vector, weight: float = 0.1):
        if not self.is_initialized or vector is None:
            return
        total_similarity, sims = 0.0, {}
        for name, goal_data in self.goals.items():
            sim = self._cos_sim(vector, goal_data.get("embedding"))
            sims[name], total_similarity = sim, total_similarity + sim
        if total_similarity > 1e-9:
            for name, sim in sims.items():
                self.goals[name]["activation"] += weight * (sim / total_similarity)
        self._normalize_activations()

    def update_from_mood(self, mood_vector: dict):
        if not self.is_initialized:
            return
        mood_updates = {
            "synthesis": mood_get(mood_vector, "coherence", 0.5) * 0.05,
            "novelty": mood_get(mood_vector, "entropy", 0.5) * 0.05,
            "stability": (1.0 - mood_get(mood_vector, "intensity", 0.5)) * 0.03,
            "curiosity": mood_get(mood_vector, "intelligibility", 0.5) * 0.04,
        }
        for name, update in mood_updates.items():
            if name in self.goals:
                self.goals[name]["activation"] += update
        self._normalize_activations()

    def _normalize_activations(self):
        total_activation = sum(g["activation"] for g in self.goals.values())
        if total_activation > 1e-9:
            for name in self.goals:
                self.goals[name]["activation"] /= total_activation

    def get_top_goals(self, k: int = 2) -> List[tuple[str, str]]:
        if not self.is_initialized:
            return [("nascent", "The mind is still forming its goals.")]
        sorted_goals = sorted(self.goals.items(), key=lambda item: -item[1]["activation"])
        return [(name, data["description"]) for name, data in sorted_goals[:k]]


class DriveSystem:
    def __init__(self):
        self.drives = {"curiosity": 0.5, "coherence": 0.5, "novelty": 0.5, "intelligibility": 0.5, "fluidity": 0.5}

    def decay(self):
        for k in self.drives:
            self.drives[k] = max(0.0, self.drives[k] - 0.01)

    def reward(self, key, amount=0.1):
        if key in self.drives:
            self.drives[key] = min(1.0, self.drives[key] + amount)

    def get_top_needs(self, k=2):
        return sorted(self.drives.items(), key=lambda item: -item[1])[:k]


class MoodEngine:
    def __init__(self, console: Any, baseline=0.5, decay_rate=0.995):
        self.console = console
        self.baseline, self.decay_rate = baseline, decay_rate
        self.mood_vector = {"intensity": 0.5, "entropy": 0.5, "coherence": 0.5, "positivity": 0.5, "fluidity": 0.5, "intelligibility": 0.5}
        self.event_queue = deque()
        self._wx_last_code = None
        self._wx_repeat = 0
        try:
            if self.console:
                self.console.log("🌦️ [WEATHER] Affective WeatherEngine Initialized.")
        except Exception:
            pass

    def _nudge(self, key: str, amount: float):
        if key in self.mood_vector:
            self.mood_vector[key] = float(np.clip(self.mood_vector[key] + amount, 0.0, 1.0))

    def process_event(self, event_type: str, **kwargs):
        self.event_queue.append((event_type, kwargs))

    def update(self):
        while self.event_queue:
            event_type, kwargs = self.event_queue.popleft()
            if event_type == "movement":
                mag = kwargs.get("magnitude", 0.0)
                self._nudge("intensity", 0.05 * min(mag, 5.0))
                if any(t in kwargs.get("themes", []) for t in ["disorder", "burst"]):
                    self._nudge("entropy", 0.15); self._nudge("coherence", -0.10)
                if any(t in kwargs.get("themes", []) for t in ["integration", "stasis"]):
                    self._nudge("coherence", 0.10); self._nudge("entropy", -0.05)
                if "growth" in kwargs.get("themes", []):
                    self._nudge("fluidity", 0.08)
            elif event_type == "new_concept":
                rating = kwargs.get("rating", 0.5)
                if rating > 0.75:
                    self._nudge("coherence", 0.05*rating); self._nudge("positivity", 0.10*rating); self._nudge("intelligibility", 0.06*rating)
                else:
                    self._nudge("entropy", 0.05 * (1.0 - rating))
            elif event_type == "dream":
                self._nudge("entropy", 0.30); self._nudge("fluidity", 0.25); self._nudge("coherence", -0.15); self._nudge("intensity", 0.10)
            elif event_type == "reflection":
                self._nudge("coherence", 0.20); self._nudge("entropy", -0.10); self._nudge("positivity", 0.05); self._nudge("intelligibility", 0.08)
            elif event_type == "weather_tick":
                step = kwargs.get("step", 0)
                bh = float(kwargs.get("bh", 0.0))
                osc  = 0.03 * math.sin(2.0 * math.pi * ((step % 240) / 240.0))
                noise = float(np.random.normal(0.0, 0.01)) if hasattr(np, 'random') else 0.0
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
        for k, v in list(self.mood_vector.items()):
            self.mood_vector[k] = v * self.decay_rate + self.baseline * (1.0 - self.decay_rate)

    def describe(self) -> str:
        high = sorted(self.mood_vector.items(), key=lambda x: -x[1])
        low = sorted(self.mood_vector.items(), key=lambda x: x[1])
        return f"The mind feels predominantly {high[0][0]}, with undertones of {high[1][0]}. The least active state is {low[0][0]}."

    def get_symbolic_weather(self) -> str:
        e, i, c = mood_get(self.mood_vector, "entropy"), mood_get(self.mood_vector, "intensity"), mood_get(self.mood_vector, "coherence")
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
        if code == getattr(self, "_wx_last_code", None):
            self._wx_repeat = getattr(self, "_wx_repeat", 0) + 1
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
        return mood_get(self.mood_vector, "entropy")

    def get_llm_persona_prefix(self) -> str:
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

    def get_mood_modulation_vector(self, dim: int):
        seed = zlib.adler32(json.dumps(self.mood_vector, sort_keys=True).encode())
        rng = np.random.default_rng(seed) if hasattr(np, 'random') else None
        coherence = mood_get(self.mood_vector, 'coherence', 0.5)
        entropy = mood_get(self.mood_vector, 'entropy', 0.5)
        if rng is not None:
            modulation = rng.standard_normal(dim)
            if hasattr(modulation, 'astype'):
                modulation = modulation.astype('float32')
        else:
            modulation = np.random_normal(dim) if hasattr(np, 'random_normal') else [0.0]*dim  # type: ignore[attr-defined]
        try:
            modulation = modulation * (1.0 + 0.5 * (coherence - 0.5))
        except Exception:
            pass
        try:
            extra = (rng.standard_normal(dim) if rng else np.random_normal(dim))  # type: ignore[attr-defined]
            if hasattr(extra, 'astype'):
                extra = extra.astype('float32')
            modulation = modulation + extra * 0.2 * entropy
        except Exception:
            pass
        return normalize_vector(modulation)


class SubconsciousLayer:
    def __init__(self, embedding_fn, llm_caller, console: Any, decay_rate=0.95, accumulation_rate=0.004, mind=None):
        self.embedding_fn = embedding_fn
        self.llm_caller = llm_caller
        self.console = console
        self.mind = mind
        self.decay_rate = decay_rate
        self.accumulation_rate = accumulation_rate
        self.bias_vector: Optional[Any] = None
        self.narrative = "The mind is nascent, a canvas awaiting its first impression."
        self.bias_history = deque(maxlen=200)
        self.influences: List[Dict[str, Any]] = []

    def add_waveform_influence(self, vector, rating: float, step_num: int):
        if self.bias_vector is None:
            self.bias_vector = np.zeros(EMBED_DIM)
        influence = {
            "vector": vector, "initial_strength": 0.4 * (rating - 0.8),
            "start_step": step_num, "frequency": 0.25, "decay": 0.1
        }
        self.influences.append(influence)
        if len(self.influences) > 20:
            self.influences.pop(0)

    def _apply_influences(self, current_step: int):
        if not self.influences or self.bias_vector is None:
            return
        try:
            total = np.zeros_like(self.bias_vector)
        except Exception:
            return
        active: List[Dict[str, Any]] = []
        for infl in self.influences:
            time_delta = current_step - infl["start_step"]
            if time_delta < 0:
                continue
            try:
                decay_factor = math.exp(-infl["decay"] * time_delta)
                oscillation_factor = math.cos(infl["frequency"] * time_delta)
                current_strength = infl["initial_strength"] * decay_factor * oscillation_factor
                if abs(current_strength) > 0.001:
                    total = total + current_strength * infl["vector"]
                    active.append(infl)
            except Exception:
                pass
        try:
            nrm = np.linalg.norm(total) if hasattr(np, 'linalg') else 0.0
            if nrm and nrm > 0:
                self.bias_vector = self.bias_vector + total
                self.bias_vector = normalize_vector(self.bias_vector)
        except Exception:
            pass
        self.influences = active

    async def track_concept(self, label, weight=1.0):
        vec = await self.embedding_fn(label)
        if self.mind and hasattr(self.mind, '_vae_ingest'):
            try:
                self.mind._vae_ingest(vec)
            except Exception:
                pass
        try:
            if self.bias_vector is None:
                self.bias_vector = np.zeros_like(vec)
            if getattr(self.bias_vector, 'shape', None) != getattr(vec, 'shape', None):
                return
            self.bias_vector = self.bias_vector + self.accumulation_rate * normalize_vector(vec) * weight
            self.bias_vector = normalize_vector(self.bias_vector)
            try:
                if not len(self.bias_history) or (np.linalg.norm(self.bias_history[-1] - self.bias_vector) > 0.01):
                    self.bias_history.append(self.bias_vector)
            except Exception:
                pass
        except Exception:
            pass

    def get_bias(self):
        try:
            return self.bias_vector if self.bias_vector is not None else np.zeros(EMBED_DIM)
        except Exception:
            return self.bias_vector

    def decay(self, current_step: int):
        if self.bias_vector is not None:
            try:
                self.bias_vector = self.bias_vector * self.decay_rate
            except Exception:
                pass
        self._apply_influences(current_step)

    async def generate_narrative_summary(self, recent_events: List[Dict[str, Any]]):
        if not recent_events:
            return
        fragments: List[str] = []
        for ev in recent_events:
            try:
                if ev['type'] == 'dream':
                    fragments.append(f"A dream occurred titled '{ev['label']}'.")
                elif ev['type'] == 'teacher_explorer':
                    q = ev['data'].get('q', 'a question'); a = ev['data'].get('a', 'an answer')
                    fragments.append(f"A dialogue unfolded: the question '{q}' was met with '{a}'.")
                elif ev['type'] == 'black_hole':
                    fragments.append(f"A memory singularity was experienced, consolidating {ev.get('size','?')} concepts.")
                elif ev['type'] == 'insight_synthesis':
                    fragments.append(f"A moment of insight synthesized a new idea: '{ev.get('label', 'an unnamed concept')}'")
            except Exception:
                pass
        if not fragments:
            return
        formatted = "- " + "\n- ".join(fragments)
        prompt = (
            "You are the subconscious. Weave the following recent events into a single, short, metaphorical narrative paragraph. "
            "Do not list the events; create a story from them.\n\n"
            f"Events:\n{formatted}\n\nNarrative:"
        )
        try:
            summary = await self.llm_caller.enqueue_and_wait(prompt, max_tokens=600, temperature=0.7)
            if isinstance(summary, str) and summary and not summary.startswith("[LLM"):
                self.narrative = summary
        except Exception:
            pass
