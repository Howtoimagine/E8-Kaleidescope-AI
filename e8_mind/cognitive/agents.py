from __future__ import annotations

from typing import Any, Dict
import math

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    class _NP:
        def zeros(self, n, dtype=None): return [0.0]*int(n)
        def argmin(self, arr):
            m=min((float(v),i) for i,v in enumerate(arr)); return m[1]
        def argmax(self, arr):
            m=max((float(v),i) for i,v in enumerate(arr)); return m[1]
        def asarray(self, x, dtype=None): return x
        def linalg_norm(self, v):
            try: return (sum(float(x)*float(x) for x in v))**0.5
            except Exception: return 0.0
        def exp(self, x):
            import math
            return [math.exp(float(v)) for v in x]
        def ones_like(self, x): return [1.0 for _ in x]
        def isfinite(self, x): return True
    np = _NP()  # type: ignore


class BaseAgentAdapter:
    def __init__(self, agent, name: str = "base"):
        self.agent = agent
        self.name = name

    async def select_action(self, state, mind):
        try:
            return self.agent.select_action(state)
        except Exception:
            ad = int(getattr(mind, 'action_dim', 0))
            return np.zeros(ad)


class SACMPOAgent:
    """Compatibility stub. Full torch-based agent isn't included in modular package.
    Exposes select_action/store/update signatures used by the mind.
    """
    def __init__(self, *a, **k):
        self.available = False
        self.action_dim = int(k.get('action_dim', 0) or (a[1] if len(a) > 1 else 0))

    def select_action(self, state, deterministic: bool = False):
        return np.zeros(self.action_dim)

    def store(self, *a, **k):
        return None

    def update(self):
        return None


class ActionCandidateSampler:
    def __init__(self, mind, K: int = 12, mag: float = 0.04):
        self.mind = mind
        self.K = int(K)
        self.mag = float(mag)

    def sample(self):
        ad = int(getattr(self.mind, 'action_dim', 0))
        c = [np.zeros(ad)]
        for i in range(min(self.K, ad)):
            v = np.zeros(ad); v[i] = self.mag
            # Attempt unary negation, fall back to elementwise for lists
            try:
                neg = -v
            except Exception:
                neg = [ -float(x) for x in v ]
            c.append(v); c.append(neg)
        return c[:max(1, self.K)]


class NoveltyAgent:
    def __init__(self, sampler):
        self.sampler = sampler
        self.name = "nov"

    async def select_action(self, state, mind):
        wm = getattr(mind, 'world_model', None)
        c = self.sampler.sample()
        best, score = None, -1e9
        for a in c:
            try:
                s = float(wm.score_transition(state, a)) if (wm and getattr(wm, 'ready', False)) else float(_norm(a))
            except Exception:
                s = float(_norm(a))
            if s > score:
                best, score = a, s
        return best if best is not None else c[0]


class StabilityAgent:
    def __init__(self, sampler):
        self.sampler = sampler
        self.name = "stab"

    async def select_action(self, state, mind):
        c = self.sampler.sample()
        try:
            idx = int(np.argmin([_norm(a) for a in c]))
        except Exception:
            idx = 0
        return c[idx]


class SynthesisAgent:
    def __init__(self, sampler):
        self.sampler = sampler
        self.name = "syn"

    async def select_action(self, state, mind):
        c = self.sampler.sample()
        anchors = getattr(mind, 'anchors', None)
        target = None
        try:
            if anchors and getattr(anchors, 'anchors', None):
                vecs = [np.asarray(v) for v, _ in anchors.anchors]
                if len(vecs) >= 2:
                    dmax, pair = -1.0, (vecs[0], vecs[0])
                    for i in range(len(vecs)):
                        for j in range(i + 1, len(vecs)):
                            try:
                                d = float(_norm(vecs[i] - vecs[j]))
                            except Exception:
                                d = float(_norm([vi - vj for vi, vj in zip(vecs[i], vecs[j])]))
                            if d > dmax:
                                dmax, pair = d, (vecs[i], vecs[j])
                    target = 0.5 * (pair[0] + pair[1])
        except Exception:
            target = None
        if target is None:
            return await NoveltyAgent(self.sampler).select_action(state, mind)
        wm = getattr(mind, 'world_model', None)

        def score(a):
            try:
                if wm and getattr(wm, 'ready', False):
                    return -abs(wm.score_transition(state, a))
            except Exception:
                pass
            return float(_norm(a))

        try:
            idx = int(np.argmin([score(a) for a in c]))
        except Exception:
            idx = 0
        return c[idx]


def _softmax_b(values, beta: float):
    vs = [float(v) for v in values]
    if not vs:
        return []
    m = max(vs)
    exps = [math.exp(beta * (v - m)) for v in vs]
    s = sum(exps)
    if s <= 0 or not math.isfinite(s):
        return [1.0/len(vs) for _ in vs]
    return [e/s for e in exps]


def _norm(a) -> float:
    try:
        # numpy array path
        return float((a if hasattr(a, '__array__') else np.asarray(a)).astype('float64'))  # type: ignore
    except Exception:
        try:
            # numpy with linalg
            import numpy as _n
            return float(_n.linalg.norm(a))
        except Exception:
            try:
                # python list/vector
                return (sum(float(x)*float(x) for x in a))**0.5
            except Exception:
                return 0.0


class MetaArbiter:
    def __init__(self, agents: dict, drive_system, beta: float = 3.0, console=None, metrics=None):
        self.agents = agents
        self.drive_system = drive_system
        self.beta = float(beta)
        self.console = console
        self.metrics = metrics

    def utilities(self, state, mind) -> Dict[str, float]:
        ds = self.drive_system

        def _safe(fn, fallback):
            try:
                return float(fn(state))
            except Exception:
                return float(fallback)

        try:
            sims = mind.memory.find_similar_in_main_storage(state, k=5)
            nn = sims[0][1] if sims else 0.0
        except Exception:
            nn = 0.25
        u_nov = _safe(getattr(ds, 'novelty_need', lambda s: nn), nn)
        u_syn = _safe(getattr(ds, 'synthesis_need', lambda s: nn), nn)
        u_stab = _safe(getattr(ds, 'stability_need', lambda s: 0.3), 0.3)
        u_base = _safe(getattr(ds, 'exploit_need', lambda s: 0.5 * (1.0 - u_nov)), 0.5 * (1.0 - u_nov))
        return {"nov": u_nov, "syn": u_syn, "stab": u_stab, "base": u_base}

    async def choose(self, state, mind):
        u = self.utilities(state, mind)
        names, vals = zip(*u.items())
        probs = _softmax_b(vals, self.beta)
        try:
            idx = max(range(len(probs)), key=lambda i: float(probs[i]))
        except Exception:
            idx = 0
        choice = names[idx]
        try:
            if self.metrics and hasattr(self.metrics, 'increment'):
                self.metrics.increment(f"society.usage.{choice}")
        except Exception:
            pass
        return self.agents[choice]

    async def step(self, state, mind):
        agent = await self.choose(state, mind)
        return await agent.select_action(state, mind)


class SocietyOfMind:
    def __init__(self, mind, beta: float = 3.0, K: int = 12):
        sampler = ActionCandidateSampler(mind, K=K)
        agents = {
            "base": BaseAgentAdapter(getattr(mind, 'agent', None), name="base"),
            "nov": NoveltyAgent(sampler),
            "syn": SynthesisAgent(sampler),
            "stab": StabilityAgent(sampler),
        }
        drives = getattr(mind, 'drives', None) or getattr(mind, 'drive_system', None)
        self.arbiter = MetaArbiter(agents, drives, beta=beta, console=getattr(mind, 'console', None), metrics=getattr(mind, 'metrics', None))

    async def step(self, state, mind):
        return await self.arbiter.step(state, mind)
