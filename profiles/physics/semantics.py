from __future__ import annotations
from typing import Mapping, List, Tuple
import os, re, unicodedata
import numpy as np

class PhysicsSemantics:
    """
    Semantics plugin for physics simulation, focusing on particles, forces, and units.
    - Reranks candidates to favor physically grounded concepts and correct unit usage.
    - Adapts persona to a computational physicist.
    - Normalizes vectors and text for consistency.
    """

    name = "physics_simulation"
    base_domain = (
        "Classical Mechanics; Newtonian Physics; Force; Mass; Acceleration; Velocity; Momentum; Energy; "
        "Work; Power; Conservation Laws; Electromagnetism; Electric Field; Magnetic Field; Maxwell's Equations; "
        "Quantum Mechanics; Wave-particle duality; Particle Physics; Standard Model; "
        "SI Units (meter, kilogram, second, Ampere, Kelvin, mole, candela); "
        "Numerical simulation; Differential equations; Vector calculus."
    )

    # ---- weights (env-tunable) ----
    KW_BONUS = float(os.getenv("PHYS_KW_BONUS", "0.15"))     # per-hit bonus for core physics keywords
    KW_MAX   = float(os.getenv("PHYS_KW_MAX", "0.60"))       # cap for keyword bonuses
    UNIT_BONUS = float(os.getenv("PHYS_UNIT_BONUS", "0.25")) # bonus for mentioning SI units
    CITE_BONUS = float(os.getenv("PHYS_CITE_BONUS", "0.20")) # arXiv/doi/url bonus
    SPEC_PEN   = float(os.getenv("PHYS_SPEC_PEN", "0.10"))   # speculation penalty per hit
    OFFTOP_PEN = float(os.getenv("PHYS_OFFTOP_PEN", "0.20")) # weak signal penalty
    EQUATION_BONUS = float(os.getenv("PHYS_EQ_BONUS", "0.15")) # bonus for equations/symbols

    # Vocabulary for physics simulation
    _KW = [
        "force", "mass", "acceleration", "velocity", "momentum", "energy", "work", "power",
        "newton", "joule", "watt", "pascal", "hertz", "coulomb", "volt", "ohm", "tesla", "weber",
        "farad", "henry", "meter", "kilogram", "second", "ampere", "kelvin", "mole", "candela",
        "vector", "scalar", "tensor", "field", "particle", "wave", "gravity", "electric", "magnetic",
        "strong force", "weak force", "maxwell's equations", "schrodinger", "heisenberg",
        "lagrangian", "hamiltonian", "conservation", "inertia", "friction", "torque", "angular momentum"
    ]
    _UNITS = [" m ", " kg ", " s ", " A ", " K ", " mol ", " cd ", " N ", " Pa ", " J ", " W ", " C ", " V ", " F ", " H ", " T ", " Wb ", " Hz "]
    _SPEC = ["maybe","might","could","probably","i think","i believe","seems","appears", "hypothetically"]

    def persona_prefix(self, mood_vector: Mapping[str, float]) -> str:
        # Adapt tone to mood if provided
        coherence = float(mood_vector.get("coherence", 0.5)) if mood_vector else 0.5
        precision = "Be meticulous about physical laws and mathematical consistency." if coherence > 0.6 else "Focus on first principles."
        return (
            "You are a computational physicist specializing in high-fidelity simulations of physical systems. "
            "Your expertise lies in modeling particles, forces, fields, and their interactions. "
            f"{precision} Your goal is to build and analyze simulations that accurately reflect physical reality."
        )

    # ---- text normalization ----
    def pre_text(self, text: str) -> str:
        if not text:
            return ""
        # Remove soft hyphens and normalize unicode; collapse whitespace
        t = text.replace("\u00AD", "")
        t = unicodedata.normalize("NFKC", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def post_text(self, text: str) -> str:
        return (text or "").strip()

    # ---- embedding hooks ----
    def pre_embed(self, text: str) -> str:
        # Append light vocabulary hints to stabilize embeddings
        vocab_hint = (
            " | concepts: force, mass, energy, field, momentum, SI units, vector, equation, "
            "conservation laws, electromagnetism, classical mechanics, quantum mechanics"
        )
        return f"{self.pre_text(text)}{vocab_hint}"

    def post_embed(self, vec, host=None, dim=None) -> np.ndarray:
        """
        Normalize the vector to unit length.
        """
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        if n > 0:
            v = v / n
        return v

    # ---- reranker ----
    def rerank(self, candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Deterministic reranker for physics simulation:
        - Boost relevant keywords and SI units (capped)
        - Boost citations (arXiv/DOI/URLs) and equations
        - Penalize speculation words
        - Penalize off-topic if signal is weak
        Input/Output shape preserved: List[(text, score)]
        """
        if not candidates:
            return candidates

        def score_one(item: Tuple[str, float]) -> Tuple[str, float]:
            text, base = item
            t = (text or "").lower()

            # Keyword bonus
            kw_hits = sum(1 for w in self._KW if w in t)
            kw_bonus = min(kw_hits * self.KW_BONUS, self.KW_MAX)

            # Unit bonus
            unit_hits = sum(1 for u in self._UNITS if u in t)
            unit_bonus = min(unit_hits * self.UNIT_BONUS, self.KW_MAX) # Capped by the same max

            # Citation bonus
            cite_bonus = self.CITE_BONUS if ("arxiv" in t or "doi:" in t or re.search(r"https?://", t)) else 0.0

            # Speculation penalty (scaled by occurrences)
            spec_hits = sum(t.count(w) for w in self._SPEC)
            spec_pen = min(spec_hits * self.SPEC_PEN, 0.5)

            # Off-topic penalty if few keywords
            physics_signal = kw_hits >= 2 or unit_hits >= 1 or any(k in t for k in ("lagrangian","hamiltonian","maxwell","newton"))
            offtop_pen = 0.0 if physics_signal else self.OFFTOP_PEN

            # Equations / symbols lightweight bonus
            eq_bonus = self.EQUATION_BONUS if re.search(r"[=∑∫∂λΩμνħ∇]|\b(F|E|B)\s*=\s*", t) else 0.0

            total_bonus = kw_bonus + unit_bonus + cite_bonus + eq_bonus
            total_penalty = spec_pen + offtop_pen
            
            return (text, float(base + total_bonus - total_penalty))

        scored = [score_one(it) for it in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

PLUGIN = PhysicsSemantics()