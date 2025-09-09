
from typing import Mapping, List, Tuple
import numpy as np

class FinanceSemantics:
    name = "finance"
    base_domain = "financial markets; risk; volatility; tail events; drawdowns; liquidity; correlation; credit spreads; funding stress; term structure; skew; kurtosis"
    def persona_prefix(self, mood_vector: Mapping[str, float]) -> str:
        return "You are a risk‑first markets analyst. Hunt for regime shifts and tail risk. Be precise. Be cautious. No advice."
    def pre_text(self, text: str) -> str:
        return text.replace("\u00AD", "").strip()
    def post_text(self, text: str) -> str:
        return text.strip()
    def pre_embed(self, text: str) -> str:
        return text + " | terms: return, drawdown, volatility, vol-of-vol, skew, kurtosis, correlation, breadth, liquidity, credit spread, CDS, basis, term structure, VIX, funding stress"
    def post_embed(self, vec):
        v = np.asarray(vec, dtype=np.float32)
        n = np.linalg.norm(v)
        return v if n == 0 else (v / n)
    
    def rerank(self, candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Boost items containing risk / crash terms. Simple keyword scoring.
        Keeps interface identical: input [(text, score)] -> output re-ordered list.
        """
        if not candidates:
            return candidates
        keywords = [
            "drawdown","crash","selloff","liquidity","margin","stress","cds","credit spread",
            "funding","basis","inversion","breadth","skew","kurtosis","volatility","vix","correlation"
        ]
        def boost(item):
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


PLUGIN = FinanceSemantics()
