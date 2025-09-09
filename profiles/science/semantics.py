from __future__ import annotations
from typing import Mapping, List, Tuple
import os, re, unicodedata
import numpy as np

class PhysicsSemantics:
    """
    M18-aware semantics plugin studying E8 lattice, physics, and AI memory.
    - Keeps the same interface as your original: rerank([(text, score)]) -> List[(text, score)]
    - Adds light, deterministic boosts/penalties so early memory stays grounded.
    - Optional lattice-aware post_embed if the host mind exposes `_snap_to_lattice`.
    """

    name = "e8_ai_memory_m18"
    base_domain = (
        "E8 lattice; root system; Weyl group; Lie algebra; Gosset polytope; quasicrystal; "
        "quantum mechanics; cosmology; entanglement; spacetime curvature; quantum field theory; "
        "AI memory; Hopfield network; Kanerva SDM; Vector-Symbolic Architecture; episodic memory; "
        "attractor network; memory consolidation; retrieval; holographic reduced representation"
    )

    # ---- weights (env-tunable) ----
    KW_BONUS = float(os.getenv("SCI_KW_BONUS", "0.12"))     # per-hit bonus for keywords
    KW_MAX   = float(os.getenv("SCI_KW_MAX", "0.60"))       # cap for keyword bonuses
    CITE_BONUS = float(os.getenv("SCI_CITE_BONUS", "0.20")) # arXiv/doi/url bonus
    SPEC_PEN   = float(os.getenv("SCI_SPEC_PEN", "0.10"))   # speculation penalty per hit
    OFFTOP_PEN = float(os.getenv("SCI_OFFTOP_PEN", "0.20")) # weak signal penalty
    EQUATION_BONUS = float(os.getenv("SCI_EQ_BONUS", "0.08"))

    # Vocabulary to anchor early runs, now including all three domains
    _KW = [
        "quantum","relativity","entanglement","spacetime","black hole","singularity",
        "string theory","dark matter","boson","fermion","cosmology","thermodynamics",
        "quantum field theory","qft","lagrangian","hamiltonian","symmetry","gauge",
        "renormalization","unitarity","noether","holographic","ads/cft","superconduct",
        "spin","qubit","wavefunction","eigenvalue","operator","path integral","tensor",
        "manifold","curvature","geodesic","heisenberg","schrodinger","dirac","neutrino",
        "e8 lattice", "root system", "weyl group", "lie algebra", "quasicrystal", "gosset",
        "hopfield network", "kanerva", "sdm", "vsa", "hrr", "attractor network", "episodic",
        "semantic memory", "consolidation", "retrieval", "hebbian"
    ]
    _SPEC = ["maybe","might","could","probably","i think","i believe","seems","appears"]

    def persona_prefix(self, mood_vector: Mapping[str, float]) -> str:
        # Adapt tone to M17 mood if provided
        coherence = float(mood_vector.get("coherence", 0.5)) if mood_vector else 0.5
        caution = "Demand mathematical or experimental evidence." if coherence < 0.45 else "Be precise and concise."
        return (
            "You are a research scientist specializing in the intersection of E8 lattice theory, "
            "computational physics, and advanced AI memory systems. Your goal is to find unifying "
            f"principles between these domains. {caution} Ground your reasoning in established concepts."
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
        # Append light vocabulary hints to stabilize early embeddings
        vocab_hint = (
            " | terms: E8 lattice, root vector, quantum field, entanglement, spacetime, "
            "Hopfield network, attractor, episodic memory, VSA, Kanerva, consolidation, "
            "relativity, hamiltonian, gauge, symmetry, tensor, eigenvalue"
        )
        return f"{self.pre_text(text)}{vocab_hint}"

    def post_embed(self, vec, host=None, dim=None) -> np.ndarray:
        """
        Normalize; if host mind exposes `_snap_to_lattice`, optionally snap using its quantizer.
        - host: pass `self` from E8Mind to enable lattice snapping (optional)
        - dim: provide the shell dim if you want host snapping to use it
        """
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        if n > 0:
            v = v / n
        # Optional E8-aware snapping path
        try:
            if host is not None and hasattr(host, "_snap_to_lattice"):
                quant = os.getenv("E8_QUANTIZER", "e8").lower()
                if quant != "none":
                    snapped = host._snap_to_lattice(v, dim or len(v))
                    v = np.asarray(snapped, dtype=np.float32)
        except Exception:
            pass
        return v

    # ---- reranker ----
    def rerank(self, candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Deterministic reranker for science:
        - Boost relevant keywords (capped)
        - Boost citations (arXiv/DOI/URLs)
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
            hits = sum(1 for w in self._KW if w in t)
            kw_bonus = min(hits * self.KW_BONUS, self.KW_MAX)

            # Citation bonus
            cite_bonus = self.CITE_BONUS if ("arxiv" in t or "doi:" in t or re.search(r"https?://", t)) else 0.0

            # Speculation penalty (scaled by occurrences)
            spec_hits = sum(t.count(w) for w in self._SPEC)
            spec_pen = min(spec_hits * self.SPEC_PEN, 0.5)

            # Off-topic penalty if few keywords
            physics_signal = hits >= 2 or any(k in t for k in ("lagrangian","hamiltonian","operator","tensor","e8","hopfield"))
            offtop_pen = 0.0 if physics_signal else self.OFFTOP_PEN

            # Equations / symbols lightweight bonus
            eq_bonus = self.EQUATION_BONUS if re.search(r"[=∑∫∂λΩμνħ]|\b(L|H)\s*=\s*", t) else 0.0

            return (text, float(base + kw_bonus + cite_bonus + eq_bonus - spec_pen - offtop_pen))

        scored = [score_one(it) for it in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

PLUGIN = PhysicsSemantics()