
def seed_all(seed: int):
    import random as _random
    import numpy as _np
    _random.seed(seed)
    try:
        _np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch as _torch
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

import traceback


def run_hypothesis_validation(insight: dict) -> dict:
    """
    Validates a generated insight and returns a dict with:
      - type: classification of the insight
      - test_plan: {method, confidence, next_step}
    """
    text = (insight.get("text") or insight.get("label") or str(insight))[:400]
    # naive classification
    if any(k in text.lower() for k in ["simulate", "model", "predict"]):
        method = "simulate"
        confidence = 0.65
        next_step = "Run internal simulation and compare deltas."
    elif any(k in text.lower() for k in ["pattern", "cluster", "trend"]):
        method = "observe pattern"
        confidence = 0.6
        next_step = "Observe graph neighborhoods and telemetry trends."
    elif any(k in text.lower() for k in ["compare", "versus", "vs "]):
        method = "compare"
        confidence = 0.55
        next_step = "Compare to nearest neighbors and past analogs."
    else:
        method = "interview"
        confidence = 0.5
        next_step = "Ask one clarifying question via Teacher."
    return {
        "type": "hypothesis",
        "test_plan": {"method": method, "confidence": confidence, "next_step": next_step}
    }


import os, sys, math, json, time, random, re, logging, tempfile, io, glob, hashlib, contextlib, traceback, threading, faulthandler, zlib, heapq
from typing import Any, Dict, List, Optional, Tuple, Union, Iterable, TYPE_CHECKING, cast
from collections import deque, defaultdict
import inspect
from dataclasses import dataclass

import numpy as np
try:
    import websockets  # type: ignore[import-not-found]
except Exception:
    websockets = None  # type: ignore

# Rich library imports with fallbacks
try:
    from rich.console import Console as RichConsole
    from rich.panel import Panel as RichPanel  
    from rich.progress import Progress as RichProgress, SpinnerColumn as RichSpinnerColumn, BarColumn as RichBarColumn, TextColumn as RichTextColumn, TimeElapsedColumn as RichTimeElapsedColumn
    from rich.markup import escape as _rich_escape
    RICH_AVAILABLE = True
    SpinnerColumn = RichSpinnerColumn
    BarColumn = RichBarColumn
    TextColumn = RichTextColumn
    TimeElapsedColumn = RichTimeElapsedColumn
except ImportError:
    RichConsole = None  # type: ignore[assignment]
    RichPanel = None   # type: ignore[assignment]
    RichProgress = None  # type: ignore[assignment]
    SpinnerColumn = None  # type: ignore[assignment]
    BarColumn = None  # type: ignore[assignment]
    TextColumn = None  # type: ignore[assignment]
    TimeElapsedColumn = None  # type: ignore[assignment]
    _rich_escape = None  # type: ignore[assignment]
    RICH_AVAILABLE = False

# Time utilities (safe imports; used across telemetry and run IDs)
try:
    from datetime import datetime, timezone
except Exception:
    datetime = None  # type: ignore
    timezone = None  # type: ignore

# Combinatorics helper for E8 root generation
try:
    from itertools import combinations
except Exception:
    combinations = None  # type: ignore

# Optional GA backend (Clifford)
try:
    import clifford  # type: ignore
    CLIFFORD_AVAILABLE = True
except Exception:
    clifford = None  # type: ignore
    CLIFFORD_AVAILABLE = False

# Optional LLM backends that may not be installed
try:
    import ollama  # type: ignore
except Exception:
    ollama = None  # type: ignore
try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None  # type: ignore

if TYPE_CHECKING:
    # Lightweight stubs to satisfy type checkers for forward refs
    class _E8Mind:  # pragma: no cover - stub only
        def __getattr__(self, name: str) -> Any: ...


def load_profile(name):
    # Ensure fallback classes exist, even if original fallback block wasn't imported yet
    try:
        _FallbackSem  # type: ignore[name-defined]
        _FallbackPrompts  # type: ignore[name-defined]
    except Exception:
        class _FallbackPrompts:  # minimal prompt pack
            def render(self, key, **vars):
                q = vars.get("question") or vars.get("topic") or vars.get("text") or ""
                persona = vars.get("persona", "")
                domain_hint = vars.get("domain_hint", "")
                return f"{persona}\n\n{domain_hint}\n\n{q}"
        class _FallbackSem:
            name = "default"
            base_domain = SEMANTIC_DOMAIN
            def persona_prefix(self, mood):
                intensity = (mood or {}).get('intensity', 0.5)
                entropy = (mood or {}).get('entropy', 0.5)
                coherence = (mood or {}).get('coherence', 0.5)
                if entropy > 0.7 and intensity > 0.6:
                    return "You are feeling chaotic, fragmented, and electric."
                elif coherence > 0.75:
                    return "You are feeling exceptionally clear, logical, and focused."
                elif intensity < 0.3:
                    return "You are feeling calm, quiet, and introspective."
                else:
                    return "You are in a balanced and considered state of mind."
            def pre_embed(self, t):  # light domain tag + identity
                base = getattr(self, "base_domain", None)
                if base and isinstance(t, str): return f"{base}: {t}"
                return t
            def post_embed(self, v):  # unit-norm
                try:
                    import numpy as _np
                    n = float(_np.linalg.norm(v))
                    return v / n if n > 1e-9 else v
                except Exception:
                    return v
            def rerank(self, c):  # stability-friendly no-op
                return c

    # Flag handling
    if (not E8_PROFILE_ENABLED) or (E8_PROFILE_MODE in ("none","off","disable","disabled")):
        try:
            console.log("[profiles] Profiles disabled by flags; using fallback semantics.")
        except Exception:
            pass
        return _FallbackSem(), _FallbackPrompts()

    if _real_load_profile is None:
        try:
            console.log("[profiles] profiles.loader not available; using fallback semantics.")
        except Exception:
            pass
        return _FallbackSem(), _FallbackPrompts()

    # Try real loader; fallback on any error
    try:
        return _real_load_profile(name)
    except Exception as _e:
        try:
            console.log(f"[profiles] load_profile failed ({type(_e).__name__}): {_e}; using fallback semantics.")
        except Exception:
            pass
        return _FallbackSem(), _FallbackPrompts()
# --- end wrapper ---



import sys as _sys, asyncio

# Base directory (used by other constants)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Runtime and tuning defaults
RUNTIME_DIR = os.path.join(BASE_DIR, "runtime")
POOL_WORKER_TIMEOUT = float(os.getenv("E8_POOL_WORKER_TIMEOUT", "30"))
POOL_RESULT_TIMEOUT = float(os.getenv("E8_POOL_RESULT_TIMEOUT", "60"))
TEMP_HALF_LIFE_VIVID = float(os.getenv("E8_TEMP_HALF_LIFE_VIVID", "1200"))
TEMP_HALF_LIFE_HOT = float(os.getenv("E8_TEMP_HALF_LIFE_HOT", "2400"))
TEMP_HALF_LIFE_WARM = float(os.getenv("E8_TEMP_HALF_LIFE_WARM", "3600"))
TEMP_HALF_LIFE_COLD = float(os.getenv("E8_TEMP_HALF_LIFE_COLD", "5400"))
BH_DIFFUSION_ETA = float(os.getenv("E8_BH_DIFFUSION_ETA", "0.1"))
BH_SPREAD_FRAC = float(os.getenv("E8_BH_SPREAD_FRAC", "0.05"))
DREAM_MODE_ENABLED = os.getenv("E8_DREAM_ENABLED", "1") == "1"
DREAM_MIN_INTERVAL_SEC = float(os.getenv("E8_DREAM_MIN_INTERVAL", "15"))
LOCAL_GEN_WORKERS = int(os.getenv("E8_LOCAL_GEN_WORKERS", "2"))
TEACHER_ASK_EVERY = int(os.getenv("E8_TEACHER_EVERY", "50"))
TEACHER_OFFSET = int(os.getenv("E8_TEACHER_OFFSET", "10"))
ACTION_SIZE_NO_LOCK = int(os.getenv("E8_ACTION_DIM", "6"))
CONSOLE_EXPORT_EVERY_STEPS = int(os.getenv("E8_CONSOLE_EXPORT_EVERY", "500"))
DIMENSIONAL_SHELL_SIZES = [4, 6, 8]
# Use env default directly here to avoid early reference before EMBED_DIM is defined later
AUTOENCODER_LAYER_SIZES = [int(os.getenv("E8_EMBED_DIM", "1536")), 512, 128, 32]

# Dialogue cadence and timeouts
EXPLORER_OFFSET = int(os.getenv("E8_EXPLORER_OFFSET", str(TEACHER_OFFSET + 5)))
TEACHER_STEP_TIMEOUT = float(os.getenv("E8_TEACHER_STEP_TIMEOUT", "25"))
EXPLORER_STEP_TIMEOUT = float(os.getenv("E8_EXPLORER_STEP_TIMEOUT", "25"))

# LLM timeouts
LLM_CALL_TIMEOUT_SEC = float(os.getenv("E8_LLM_CALL_TIMEOUT", "30"))
EMBEDDING_TIMEOUT_SEC = float(os.getenv("E8_EMBEDDING_TIMEOUT", "15"))

# Black hole/collapse parameters
BLACK_HOLE_COOLDOWN_STEPS = int(os.getenv("E8_BH_COOLDOWN_STEPS", "200"))
BH_PRESSURE_THRESHOLD = float(os.getenv("E8_BH_PRESSURE_THRESHOLD", "0.85"))
BLACK_HOLE_K = int(os.getenv("E8_BH_LINKS_K", "10"))
CONSOLIDATE_MIN = int(os.getenv("E8_BH_CONSOLIDATE_MIN", "3"))

# Console export format
CONSOLE_EXPORT_FORMAT = os.getenv("E8_CONSOLE_EXPORT_FORMAT", "both").lower()  # text|json|both

# External data sources configuration (name -> config)
DATA_SOURCES = {}

# Action layout helper and default
# Ensure commonly used names are exported (guard if Rich* names missing)
if not RICH_AVAILABLE:
    class _FallbackProgress:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
        def add_task(self, *args, **kwargs): return 0
        def update(self, *args, **kwargs): pass
    class _FallbackConsole:
        def __init__(self, record=False): pass
        def log(self, *a, **k): print(*a)
        def print(self, *a, **k): print(*a)
        def rule(self, *a, **k): print("-" * 20)
        def export_text(self): return ""
        def print_exception(self, *a, **k):
            import traceback as _tb
            _tb.print_exc()
    class _FallbackPanel(str):
        def __new__(cls, content, **kwargs): return str(content)
        @classmethod
        def fit(cls, content, **kwargs):
            return str(content)
    class _FallbackSpinnerColumn:
        def __init__(self, *args, **kwargs): pass
    class _FallbackBarColumn:
        def __init__(self, *args, **kwargs): pass
    class _FallbackTextColumn:
        def __init__(self, *args, **kwargs): pass
    class _FallbackTimeElapsedColumn:
        def __init__(self, *args, **kwargs): pass
    def _fallback_rich_escape(s): return s

    # Map fallbacks to primary names
    RichConsole = _FallbackConsole  # type: ignore[assignment]
    RichPanel = _FallbackPanel  # type: ignore[assignment]
    RichProgress = _FallbackProgress  # type: ignore[assignment]
    SpinnerColumn = _FallbackSpinnerColumn  # type: ignore[assignment]
    BarColumn = _FallbackBarColumn  # type: ignore[assignment]
    TextColumn = _FallbackTextColumn  # type: ignore[assignment]
    TimeElapsedColumn = _FallbackTimeElapsedColumn  # type: ignore[assignment]
    _rich_escape = _fallback_rich_escape  # type: ignore[assignment]

from typing import cast as _cast_any
# Universal shims so callers can always use Panel/Progress/Columns safely
_rich_escape_impl = _rich_escape

def rich_escape(s):
    try:
        if callable(_rich_escape_impl):
            return _rich_escape_impl(s)
    except Exception:
        pass
    return str(s)

class _PanelShim:
    def __call__(self, content, **kwargs):
        try:
            if RICH_AVAILABLE and RichPanel is not None:
                return RichPanel(content, **kwargs)  # type: ignore[misc]
        except Exception:
            pass
        return str(content)
    def fit(self, content, **kwargs):
        try:
            if RICH_AVAILABLE and RichPanel is not None and hasattr(RichPanel, 'fit'):
                return RichPanel.fit(content, **kwargs)  # type: ignore[attr-defined]
        except Exception:
            pass
        return str(content)

class _NoProgress:
    def __init__(self, *args, **kwargs):
        self._task_id = 0
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False
    def add_task(self, *args, **kwargs):
        return 0
    def update(self, *args, **kwargs):
        return None

def Progress(*args, **kwargs):
    try:
        if RICH_AVAILABLE and RichProgress is not None:
            return RichProgress(*args, **kwargs)  # type: ignore[misc]
    except Exception:
        pass
    return _NoProgress()

def SpinnerColumn(*args, **kwargs):
    try:
        if RICH_AVAILABLE and 'RichSpinnerColumn' in globals() and RichSpinnerColumn is not None:
            return RichSpinnerColumn(*args, **kwargs)  # type: ignore[misc]
    except Exception:
        pass
    return object()

def BarColumn(*args, **kwargs):
    try:
        if RICH_AVAILABLE and 'RichBarColumn' in globals() and RichBarColumn is not None:
            return RichBarColumn(*args, **kwargs)  # type: ignore[misc]
    except Exception:
        pass
    return object()

def TextColumn(*args, **kwargs):
    try:
        if RICH_AVAILABLE and 'RichTextColumn' in globals() and RichTextColumn is not None:
            return RichTextColumn(*args, **kwargs)  # type: ignore[misc]
    except Exception:
        pass
    return object()

def TimeElapsedColumn(*args, **kwargs):
    try:
        if RICH_AVAILABLE and 'RichTimeElapsedColumn' in globals() and RichTimeElapsedColumn is not None:
            return RichTimeElapsedColumn(*args, **kwargs)  # type: ignore[misc]
    except Exception:
        pass
    return object()

Panel = _PanelShim()

# Initialize console early (used throughout the codebase)
console = RichConsole(record=True)  # type: ignore[call-arg]

# --- Global defaults and constants (env-tunable) ---
GLOBAL_SEED = int(os.getenv("E8_SEED", "42"))
EMBED_DIM = int(os.getenv("E8_EMBED_DIM", "1536"))
SEMANTIC_DOMAIN = os.getenv("E8_SEMANTIC_DOMAIN", "general")
# Default OpenAI chat model (can be overridden via OPENAI_MODEL). Using GPT-5 mini (Preview) by default.
DEFAULT_OPENAI_CHAT_MODEL = os.getenv("OPENAI_MODEL", os.getenv("E8_OPENAI_MODEL", "gpt-5-mini-preview"))

# Profiles flags
E8_PROFILE_ENABLED = os.getenv("E8_PROFILE_ENABLED", "1") == "1"
E8_PROFILE_MODE = os.getenv("E8_PROFILE_MODE", "auto").lower()

# Seed domain bootstrapping flags
E8_SEED_DOMAIN = os.getenv("E8_SEED_DOMAIN", "1") == "1"
E8_SEED_LABEL = os.getenv("E8_SEED_LABEL", "")

# === inserted flags ===
E8_SELF_PROJECT = os.getenv("E8_SELF_PROJECT", "1") == "1"   # 1=enabled, 0=disabled
E8_INGEST = os.getenv("E8_INGEST", "1") == "1"               # 1=enabled, 0=disabled
# ======================

# Action system constants
ACTION_SIZE_NO_LOCK = int(os.getenv("E8_ACTION_DIM", "6"))

def _default_action_layout(total_dim: int):
    try:
        td = int(total_dim)
    except Exception:
        td = 6
    if td >= 6:
        # Two layers, non-overlapping slices + two angle indices
        return [
            {"dim": 4, "biv_start": 0, "biv_len": 2, "angle_idx": 4},
            {"dim": 6, "biv_start": 2, "biv_len": 2, "angle_idx": 5},
        ]
    elif td >= 3:
        # Single layer: use all but last for bivector, last as angle
        return [
            {"dim": 4, "biv_start": 0, "biv_len": max(td - 1, 1), "angle_idx": td - 1},
        ]
    else:
        # Minimal fallback
        return [
            {"dim": 4, "biv_start": 0, "biv_len": 1, "angle_idx": 0},
        ]

# Global ACTION_LAYOUT derived from configured action size
ACTION_LAYOUT = _default_action_layout(ACTION_SIZE_NO_LOCK)
try:
    import networkx as nx
    from networkx.algorithms import community as nx_comm
    from networkx.readwrite import json_graph
except Exception:
    nx = None; nx_comm = None
    class _JG:
        def node_link_data(self, g): return {"nodes": [], "links": []}
        def node_link_graph(self, d): return None
    json_graph = _JG()
try:
    from aiohttp import web as _web_mod
    import aiohttp_cors as _aiohttp_cors_mod
    import aiohttp as _aiohttp_mod
    import xml.etree.ElementTree as _ET_mod
    web = cast(Any, _web_mod)
    aiohttp_cors = cast(Any, _aiohttp_cors_mod)
    aiohttp = cast(Any, _aiohttp_mod)
    ET = cast(Any, _ET_mod)
except Exception:
    web = cast(Any, None); aiohttp_cors = cast(Any, None); aiohttp = cast(Any, None); ET = cast(Any, None)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    class _NN:
        Module = object
        ModuleList = list
        class Linear:
            def __init__(self, *args, **kwargs): pass
            def __call__(self, x): return x
            def to(self, device): return self
        class ReLU:
            def __call__(self, x): return x
        class Sequential:
            def __init__(self, *layers): self.layers = layers
            def __call__(self, x): return x
            def to(self, device): return self
        class GRU:
            def __init__(self, *args, **kwargs): pass
            def __call__(self, x, h=None): return x, h
            def to(self, device): return self
        class utils:
            @staticmethod
            def clip_grad_norm_(*args, **kwargs): return None
    class _Normal:
        def __init__(self, *args): pass
        def sample(self): return None
    class _F:
        @staticmethod
        def relu(x): return x
        @staticmethod
        def mse_loss(a, b):
            a_np, b_np = np.asarray(a), np.asarray(b)
            diff = a_np - b_np
            return float(np.mean(diff * diff))
    class _Torch:
        class _Cuda:
            @staticmethod
            def is_available(): return False
        cuda = _Cuda()
        class _Optim:
            class Adam:
                def __init__(self, *args, **kwargs): pass
                def zero_grad(self): pass
                def step(self): pass
        optim = _Optim()
        float32 = float
        @staticmethod
        def device(x): return "cpu"
        @staticmethod
        def exp(x): return x
        @staticmethod
        def randn_like(x): return 0
        @staticmethod
        def tensor(x, dtype=None): return np.asarray(x)
        @staticmethod
        def cat(tensors, dim=0): return np.concatenate([np.asarray(t) for t in tensors], axis=dim)
        @staticmethod
        def mean(x): return float(np.mean(np.asarray(x)))
        class _NoGrad:
            def __enter__(self): return None
            def __exit__(self, exc_type, exc, tb): return False
        @staticmethod
        def no_grad(): return _Torch._NoGrad()
        class nn:
            utils = _NN.utils
    # assign placeholders
    nn = _NN()
    F = _F()
    torch = _Torch()
    Normal = _Normal
    
# Relax typing for torch/nn/F so attribute access doesn't raise static errors when unavailable
from typing import cast
torch = cast(Any, torch)
nn = cast(Any, nn)
F = cast(Any, F)
nn = cast(Any, nn)
F = cast(Any, F)
try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN
except Exception:
    PCA, DBSCAN = None, None
try:
    import faiss  # optional ANN backend
    _FAISS = True
except Exception:
    _FAISS = False

try:
    from sklearn.neighbors import KDTree as _SKKDTree
except Exception:
    _SKKDTree = None
try:
    from sklearn.metrics.pairwise import cosine_distances as _sk_cosine_distances, cosine_similarity as _sk_cosine_similarity
except Exception:
    _sk_cosine_distances, _sk_cosine_similarity = None, None
try:
    from scipy.spatial import KDTree as _SPKDTree
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import eigsh, expm_multiply
    import scipy as sp
except Exception:
    _SPKDTree, csr_matrix, diags, eigsh, expm_multiply, sp = None, None, None, None, None, None

# Ensure VariationalAutoencoder exists even when torch is available
try:
    VariationalAutoencoder  # type: ignore[name-defined]
except Exception:
    class VariationalAutoencoder:
        """Lightweight fallback VAE with shape-only projections.

        Methods are no-ops suitable for environments without the full VAE impl.
        """
        def __init__(self, layer_sizes=None, console=None):
            self._trained = False
            self.layer_sizes = layer_sizes or []
            self.console = console

        @property
        def is_trained(self) -> bool:
            return bool(self._trained)

        def train_on_batch(self, x):
            # Pretend to train; return empty loss dict to keep callers happy
            return {"total_loss": 0.0, "recon_loss": 0.0, "kld_loss": 0.0}

        def _to_np(self, x):
            try:
                import numpy as _np
                if x is None:
                    return None
                if hasattr(x, "detach"):
                    x = x.detach().cpu().numpy()
                return _np.asarray(x, dtype=_np.float32)
            except Exception:
                return x

        def project_to_dim(self, x, target_dim: int):
            """Pad/truncate to target_dim; keeps interface stable.
            Accepts 1D or 2D arrays and returns matching batch if 2D.
            """
            x_np = self._to_np(x)
            import numpy as _np
            if x_np is None:
                return None
            x2 = _np.atleast_2d(x_np).astype(_np.float32)
            out = _np.zeros((x2.shape[0], int(target_dim)), dtype=_np.float32)
            n = min(x2.shape[1], int(target_dim))
            out[:, :n] = x2[:, :n]
            return out if x2.ndim == 2 else out[0]

        def project_between_dim(self, x, source_dim: int, target_dim: int):
            # Ignore source_dim; just reshape as needed
            return self.project_to_dim(x, target_dim)



# --- inserted: lightweight stubs for optional planners/managers (satisfy Pylance/runtime when missing) ---
try:
    LatentDiffusionProposer  # type: ignore[name-defined]
except NameError:
    class LatentDiffusionProposer:
        def __init__(self, action_dim:int, horizon:int=8, samples:int=16, **kwargs): pass
        def propose(self, *args, **kwargs): return None

try:
    _ = MacroManager  # type: ignore[name-defined]
except NameError:
    class MacroManager:
        def __init__(self, layout, action_dim:int, pick_every:int=20, **kwargs):
            self.layout = layout
            self.action_dim = int(action_dim)
            self.pick_every = int(pick_every)
            self._t = 0
        def on_action_executed(self, action_vec):
            # no-op hook in stub
            return None

        async def select_action(self, state, mind):
            """Greedy one-step lookahead using the world model, falling back to a small impulse set."""
            import numpy as _np
            K = min(8, self.action_dim)
            cands = [_np.zeros(self.action_dim, dtype=_np.float32)]
            for i in range(K):
                v = _np.zeros(self.action_dim, dtype=_np.float32); v[i] = 0.04
                cands.append(v); cands.append(-v)
            best, best_s = None, -1e9
            wm = getattr(mind, "world_model", None)
            for a in cands:
                s = 0.0
                try:
                    if wm and getattr(wm, "available", False) and getattr(wm, "ready", False):
                        s = float(wm.score_transition(state, a))
                except Exception:
                    s = 0.0
                if s > best_s:
                    best, best_s = a, s
            return best if best is not None else cands[0]

    try:
        LatentCEMPlanner  # type: ignore[name-defined]
    except NameError:
        class LatentCEMPlanner:
            def __init__(self, *args, **kwargs): pass
            def plan(self, *args, **kwargs): return None
# --- end inserted ---


# --- Helper Functions and Classes ---

def get_run_id() -> str:
    """Generates a unique run ID based on the current timestamp."""
    try:
        if datetime is not None and timezone is not None:  # type: ignore[truthy-bool]
            return datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    except Exception:
        pass
    from datetime import datetime as _dt
    return _dt.utcnow().strftime("run_%Y%m%d_%H%M%S")

def get_path(rel: str, run_id: str) -> str:
    """Constructs an absolute path within the current run's directory."""
    base = os.path.join(RUNTIME_DIR, str(run_id)) if run_id else RUNTIME_DIR
    path = os.path.join(base, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

def mood_get(mood_vector: dict, key: str, default: float = 0.5) -> float:
    """Safely retrieves a float value from the mood vector dictionary."""
    return float(mood_vector.get(key, default))

def sanitize_line(text: str, max_chars: int = 80) -> str:
    """Cleans a string to be a single, sanitized line."""
    if not isinstance(text, str): return ""
    text = text.replace('\n', ' ').replace('\r', '').strip()
    return text[:max_chars]

def sanitize_block(text: str, max_sentences: int = 5, max_chars: int = 500) -> str:
    """Cleans and truncates a block of text."""
    if not isinstance(text, str): return ""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    truncated_text = " ".join(sentences[:max_sentences])
    return truncated_text[:max_chars]

def safe_json_write(filepath: str, data: Any):
    """Safely writes data to a JSON file, creating directories if needed."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with tempfile.NamedTemporaryFile('w', delete=False, dir=os.path.dirname(filepath), encoding='utf-8') as tf:
            json.dump(data, tf, ensure_ascii=False, indent=2, cls=NumpyEncoder)
            tempname = tf.name
        os.replace(tempname, filepath)
    except Exception as e:
        logging.warning(f"Failed to write JSON to {filepath}: {e}")

def safe_json_read(filepath: str, default: Any = None) -> Any:
    """Safely reads data from a JSON file."""
    if not os.path.exists(filepath):
        return default
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logging.warning(f"Failed to read JSON from {filepath}: {e}")
        return default

def _parse_json_object(text: str) -> Dict:
    """Robustly finds and parses a JSON object from a string."""
    if not text: return {}
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}

def export_graph(graph: Any) -> Dict[str, Any]:
    """Exports a NetworkX graph to a serializable dictionary."""
    if nx is None: return {"nodes": [], "links": []}
    data = json_graph.node_link_data(graph)
    return dict(data)

def classify_geometry_theme(delta_vector: np.ndarray) -> list[str]:
    """
    Classifies an 8D movement vector into descriptive themes based on its
    geometric and statistical properties.
    """
    themes = set()
    magnitude = np.linalg.norm(delta_vector)

    # Rule 1: Stasis (very little movement)
    if magnitude < 0.05:
        themes.add("stasis")
        return list(themes)

    # Rule 2: Magnitude-based themes
    if magnitude > 1.5:
        themes.add("burst")
    elif magnitude > 0.8:
        themes.add("growth")
    else:
        themes.add("drift")

    # Rule 3: Sparsity (how many dimensions are involved)
    non_zero_elements = np.count_nonzero(np.abs(delta_vector) > 1e-4)
    if non_zero_elements <= 2:
        themes.add("focus")
    elif non_zero_elements >= 7:
        themes.add("integration")
    else:
        themes.add("shift")

    # Rule 4: Coherence vs Disorder (variance of components)
    component_variance = np.var(delta_vector)
    if component_variance > 0.2:
        themes.add("disorder")
    else:
        themes.add("coherence")

    return list(themes)

def teacher_prompt_ok(prompt: str, graph_terms: set[str]) -> bool:
    tokens = {t.lower() for t in re.findall(r"[A-Za-z0-9]+", prompt or "")}
    return len(tokens & graph_terms) >= 2

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        return super(NumpyEncoder, self).default(obj)

@dataclass
class EmergenceSeed:
    remnant_id: str
    embedding_vector: np.ndarray
    projected_vector: np.ndarray
    mass: float
    absorbed_ids: List[str]
    step_created: int

class UniversalEmbeddingAdapter:
    def __init__(self, in_dim, out_dim):
        self.in_dim, self.out_dim = in_dim, out_dim
        if in_dim == out_dim:
            self.W = np.eye(in_dim, dtype=np.float32)
        else:
            rng = np.random.default_rng(GLOBAL_SEED)
            self.W = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
            self.W /= np.linalg.norm(self.W, axis=0, keepdims=True)

    def __call__(self, vector: np.ndarray) -> np.ndarray:
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)
        if vector.shape[0] != self.in_dim:
            # Pad or truncate if there's a mismatch
            padded_vec = np.zeros(self.in_dim, dtype=np.float32)
            size_to_copy = min(vector.shape[0], self.in_dim)
            padded_vec[:size_to_copy] = vector[:size_to_copy]
            vector = padded_vec
        return vector @ self.W
# --- Main Code ---

LAST_INTRINSIC = {}

import sys as _sys, asyncio as _asyncio
if _sys.platform.startswith("win"):
    try:
        if not isinstance(_asyncio.get_event_loop_policy(), _asyncio.WindowsSelectorEventLoopPolicy):
             _asyncio.set_event_loop_policy(_asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

try:
    from profiles.loader import load_profile as _real_load_profile
except ImportError:
    _real_load_profile = None
    class _FallbackPrompts:
        def render(self, key, **vars):
            q = vars.get("question") or vars.get("topic") or vars.get("text") or ""
            persona, domain_hint = vars.get("persona", ""), vars.get("domain_hint", "")
            return f"{persona}\n\n{domain_hint}\n\n{q}"

    class _FallbackSem:
        name = "default"
        base_domain = SEMANTIC_DOMAIN

        def persona_prefix(self, mood):
            intensity = (mood or {}).get('intensity', 0.5)
            entropy = (mood or {}).get('entropy', 0.5)
            coherence = (mood or {}).get('coherence', 0.5)
            if entropy > 0.7 and intensity > 0.6:
                return "You are feeling chaotic, fragmented, and electric."
            elif coherence > 0.75:
                return "You are feeling exceptionally clear, logical, and focused."
            elif intensity < 0.3:
                return "You are feeling calm, quiet, and introspective."
            else:
                return "You are in a balanced and considered state of mind."

        def pre_embed(self, t):
            if self.base_domain and isinstance(t, str):
                return f"{self.base_domain}: {t}"
            return t

        def post_embed(self, v):
            try:
                n = float(np.linalg.norm(v))
                return v / n if n > 1e-9 else v
            except Exception:
                return v

        def rerank(self, c):
            return c

class KDTree:
    """A wrapper for scikit-learn/scipy KDTree with optional FAISS and a NumPy fallback."""
    def __init__(self, data):
        X = np.asarray(data, dtype=np.float32)
        if '_FAISS' in globals() and _FAISS and X.ndim == 2 and X.size:
            self._is_faiss = True
            self._dim = X.shape[1]
            self._faiss_index = faiss.IndexFlatIP(self._dim)
            # normalize for cosine similarity via dot
            norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
            Xn = X / norms
            self._faiss_index.add(Xn)  # type: ignore[attr-defined]
            self.n = X.shape[0]
            self._is_fallback = False
            self._impl = None
        elif _SKKDTree is not None:
            self._impl = _SKKDTree(X)
            self.n = self._impl.data.shape[0]
            self._is_fallback = False
            self._is_faiss = False
        elif _SPKDTree is not None:
            self._impl = _SPKDTree(X)
            self.n = self._impl.n
            self._is_fallback = False
            self._is_faiss = False
        else:
            self._impl = X
            self.n = self._impl.shape[0]
            self._is_fallback = True
            self._is_faiss = False
    # End of backend selection

    def query(self, q, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
        t0 = time.perf_counter()
        q_arr = np.asarray(q, dtype=np.float32)
        is_single_query = q_arr.ndim == 1
        q_2d = np.atleast_2d(q_arr)

        if getattr(self, '_is_faiss', False):
            q2 = q_2d
            norms = np.linalg.norm(q2, axis=1, keepdims=True) + 1e-12
            qn = q2 / norms
            try:
                qfaiss = qn.astype(np.float32)
                if qfaiss.ndim == 1:
                    qfaiss = qfaiss.reshape(1, -1)
                D, I = self._faiss_index.search(qfaiss, int(k))  # type: ignore[attr-defined]
            except Exception:
                # Fallback: no results
                D = np.ones((q_2d.shape[0], int(k)), dtype=np.float32)
                I = -np.ones((q_2d.shape[0], int(k)), dtype=np.int64)
            # Convert cosine sim to distance
            d = 1.0 - D
            i = I
        elif not self._is_fallback and hasattr(self._impl, 'query'):
            d, i = self._impl.query(q_2d, k=k)  # type: ignore[union-attr, attr-defined]
            d = np.asarray(d, dtype=np.float32)
            i = np.asarray(i, dtype=np.int64)
        # In KDTree.query
        else:
            # --- CORRECTED LOGIC ---
            # Corrected NumPy fallback for both single and batch queries
            all_dists = []
            all_indices = []
            data_points = self._impl
            
            for query_vector in q_2d:
                # Calculate Euclidean distances from the current query vector to all data points
                distances = np.sqrt(np.sum((data_points - query_vector)**2, axis=1))
                
                # Get the indices of the k smallest distances
                # Use argpartition for efficiency, as it's faster than a full sort.
                if k < self.n:
                    # Find the k nearest indices (unsorted)
                    nearest_idx = np.argpartition(distances, k-1)[:k]
                    # Now, sort only that small partition by distance to get the correct order.
                    sorted_partition_indices = np.argsort(distances[nearest_idx])
                    idx = nearest_idx[sorted_partition_indices]
                else: # If k is as large as the dataset, just sort everything
                    idx = np.argsort(distances)[:k]

                all_indices.append(idx)
                all_dists.append(distances[idx])
            
            d = np.array(all_dists, dtype=np.float32)
            i = np.array(all_indices, dtype=np.int64)

        # Return results in the expected shape
        d_out = d.ravel() if is_single_query else d
        i_out = i.ravel() if is_single_query else i
        # Track latency rolling average
        try:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if not hasattr(self, '_latency_ms'): self._latency_ms = []
            self._latency_ms.append(dt_ms)
            if len(self._latency_ms) > 128: self._latency_ms.pop(0)
        except Exception:
            pass
        return d_out, i_out

def cosine_distances(A, B):
    if _sk_cosine_distances is not None:
        return _sk_cosine_distances(A, B)
    A = np.asarray(A, dtype=np.float32); B = np.asarray(B, dtype=np.float32)
    def _norm(x): return np.sqrt((x*x).sum(axis=1, keepdims=True)) + 1e-12
    A2, B2 = A / _norm(A), B / _norm(B)
    return 1.0 - (A2 @ B2.T)

def cosine_similarity(A, B):
    if _sk_cosine_similarity is not None:
        return _sk_cosine_similarity(A, B)
    A = np.asarray(A, dtype=np.float32); B = np.asarray(B, dtype=np.float32)
    def _norm(x): return np.sqrt((x*x).sum(axis=1, keepdims=True)) + 1e-12
    A2, B2 = A / _norm(A), B / _norm(B)
    return A2 @ B2.T

# Prefer modular KDTree/cosine if available
try:
    from memory.index import KDTree as _ModKDTree, cosine_distances as _mod_cosine_distances, cosine_similarity as _mod_cosine_similarity
    KDTree = _ModKDTree
    Connects to a real-time financial data websocket (Finnhub) to stream
    tick data into the mind, triggering cognitive events on market activity.
    """
    def __init__(self, symbols: List[str], api_key: str, on_tick, on_bar):
        # If websockets is unavailable, disable feed gracefully
        if getattr(websockets, 'connect', None) is None:
            raise ImportError("The 'websockets' library is required for MarketFeed.")
        self.ws_url = f"wss://ws.finnhub.io?token={api_key}"
        self.symbols = symbols
        self.on_tick = on_tick  # Callback for individual trades
        self.on_bar = on_bar    # Callback for aggregated 1-minute bars
        self.running = False
        self._task = None
        self._bar_aggregator = {} # {symbol: [trades]}

    async def start(self):
        """Starts the websocket connection and data processing task."""
        if self.running: return
        console.log(f"📈 [MarketFeed] Starting connection for symbols: {self.symbols}")
        self.running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        """Stops the websocket connection."""
        self.running = False
        if self._task:
            self._task.cancel()
        console.log("📉 [MarketFeed] Connection stopped.")

    async def _run(self):
        """The main loop for connecting, subscribing, and processing messages."""
        while self.running:
            try:
                ws_connect = getattr(websockets, 'connect', None)
                if ws_connect is None:
                    await asyncio.sleep(5)
                    continue
                async with ws_connect(self.ws_url) as websocket:
                    for symbol in self.symbols:
                        await websocket.send(json.dumps({"type": "subscribe", "symbol": symbol}))
                        console.log(f"[MarketFeed] Subscribed to {symbol}")

                    async for message in websocket:
                        data = json.loads(message)
                        if data.get("type") == "trade":
                            for trade in data.get("data", []):
                                self.on_tick(trade["s"], trade)
                                self._aggregate_bar(trade)
                        elif data.get("type") == "ping":
                            await websocket.send(json.dumps({"type": "pong"}))
            except Exception as e:
                # If websockets is available and exception is a connection close, note and loop
                try:
                    if websockets is not None and hasattr(websockets, 'exceptions') and isinstance(e, getattr(websockets.exceptions, 'ConnectionClosed', ())):
                        console.log("[MarketFeed] Connection closed.")
                    elif isinstance(e, asyncio.CancelledError):
                        console.log("[MarketFeed] Connection cancelled.")
                        break
                    else:
                        console.log(f"[bold red][MarketFeed] Error: {e}. Reconnecting in 15 seconds.[/bold red]")
                        await asyncio.sleep(15)
                except Exception:
                    console.log(f"[bold red][MarketFeed] Error: {e}. Reconnecting in 15 seconds.[/bold red]")
                    await asyncio.sleep(15)

    def _aggregate_bar(self, trade: Dict):
        """Aggregates trades into 1-minute bars (OHLC)."""
        symbol = trade["s"]
        price = trade["p"]
        volume = trade["v"]
        timestamp_ms = trade["t"]
        current_minute = int(timestamp_ms / 60000)

        if symbol not in self._bar_aggregator:
            self._bar_aggregator[symbol] = {"minute": current_minute, "o": price, "h": price, "l": price, "c": price, "v": 0}

        bar = self._bar_aggregator[symbol]
        if current_minute > bar["minute"]:
            # Finalize and send the previous bar
            final_bar = Bar(o=bar["o"], h=bar["h"], l=bar["l"], c=bar["c"], v=bar["v"], ts=bar["minute"]*60)
            self.on_bar(symbol, "1m", final_bar)
            # Start a new bar
            bar.update({"minute": current_minute, "o": price, "h": price, "l": price, "c": price, "v": volume})
        else:
            # Update the current bar
            bar["h"] = max(bar["h"], price)
            bar["l"] = min(bar["l"], price)
            bar["c"] = price
            bar["v"] += volume
class Bar:
    def __init__(self, **kwargs):
        for k,v in kwargs.items(): setattr(self, k, v)

class OUNoise:
    def __init__(self, size, theta=0.05, sigma=0.06):
        import numpy as np
        self.size = size
        self.theta = theta
        self._sigma0 = sigma
        self.sigma = sigma
        self.state = np.zeros(self.size, dtype=np.float32)
    def reset(self):
        import numpy as np
        self.state = np.zeros(self.size, dtype=np.float32)
    def sample(self):
        import numpy as np
        dx = self.theta * (-self.state) + self.sigma * np.random.randn(self.size).astype(np.float32)
        self.state = self.state + dx
        return self.state

def clamp_action(vec, max_norm=0.04):
    import numpy as np
    n = float(np.linalg.norm(vec))
    if n == 0.0 or n <= max_norm:
        return vec
    return (vec * (max_norm / n)).astype(np.float32)

def shaped_reward_components(bh, bh_ma50, action, prev_action, extras):
    """
    Return a dict of reward components to be summed by the caller.
    extras may include: goal_resonance, avg_tension, valence, surprise,
    and optionally intrinsic signals like free_energy, epistemic, topo.
    """
    import numpy as np
    w_grad, w20, w40, w60, w_act, w_smooth = 0.8, 0.02, 0.05, 0.10, 0.5, 0.25
    grad_term = w_grad * max(0.0, bh - (bh_ma50 or 0.0))
    dwell = (w20 if bh > 0.20 else 0.0) + (w40 if bh > 0.40 else 0.0) + (w60 if bh > 0.60 else 0.0)
    force_pen = w_act * (float(np.linalg.norm(action)) ** 2)
    smooth_pen = w_smooth * float(np.sum((action - prev_action) ** 2))
    goal_term = 0.4 * float(extras.get('goal_resonance', 0.0))
    tension_term = 0.1 * float(extras.get('avg_tension', 0.0))
    valence_term = 0.1 * float(extras.get('valence', 0.0))
    surprise_term = 0.4 * float(extras.get('surprise', 0.0))

    try:
        fe = float(extras.get('free_energy', LAST_INTRINSIC.get('free_energy', 0.0)))
        epi = float(extras.get('epistemic', LAST_INTRINSIC.get('epistemic', 0.0)))
        topo = float(extras.get('topo', LAST_INTRINSIC.get('topo', 0.0)))
    except Exception:
        fe = epi = topo = 0.0
    fe_term = 0.2 * fe
    epi_term = 0.3 * epi
    topo_term = 0.3 * topo
    return {
        'grad': grad_term, 'dwell': dwell, 'force_pen': -force_pen, 'smooth_pen': -smooth_pen,
        'goal': goal_term, 'tension': tension_term, 'valence': valence_term, 'surprise': surprise_term,
        'free_energy': fe_term, 'epistemic': epi_term, 'topo': topo_term
    }

def normalize_vector(v):
    """Helper function to ensure vectors have unit length."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-9 else v

class CliffordRotorGenerator:
    """
    Generates a mathematically precise Geometric Algebra rotor using the Clifford library.
    This version includes safety checks for collinear or zero-magnitude vectors.
    """
    def __init__(self, mind_instance: 'E8Mind', layout, blades):
        self.mind = mind_instance
        self.layout = layout
        self.blades = blades
        self.basis_vectors = [self.blades[f'e{i+1}'] for i in range(layout.dims)]

    def _random_unit_bivector(self):
        """Returns a simple, random unit bivector (e.g., e1^e2)."""
        n = len(self.basis_vectors)
        i, j = np.random.choice(np.arange(n), size=2, replace=False)
        B = self.basis_vectors[i] ^ self.basis_vectors[j]
        return B.normal()

    def _select_dynamic_pair(self, shell: 'DimensionalShell') -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Selects a pair of vectors from the shell to define the plane of rotation."""
        nodes = list(shell.vectors.keys())
        if len(nodes) < 2:
            return None

        candidates = []
        for nid in nodes:
            node_data = self.mind.memory.graph_db.get_node(nid)
            if node_data:
                vec_np = shell.get_vector(nid)
                if vec_np is not None and np.linalg.norm(vec_np) > 1e-9:
                    candidates.append({'id': nid, 'temp': node_data.get('temperature', 0.1), 'vec': vec_np})

        if len(candidates) < 2:
            return None

        candidates.sort(key=lambda x: x['temp'], reverse=True)
        anchor_a = candidates[0]

        best_partner = None
        max_dist = -1.0
        for partner_candidate in candidates[1:min(len(candidates), 15)]:
            dist = 1.0 - abs(np.dot(normalize_vector(anchor_a['vec']), normalize_vector(partner_candidate['vec'])))
            if dist > max_dist:
                max_dist = dist
                best_partner = partner_candidate

        if best_partner is None:
            return None

        return anchor_a['vec'], best_partner['vec']

    def generate_rotor(self, shell: 'DimensionalShell', angle: float) -> Any:
        """Generates a rotor that rotates by 'angle' in the plane defined by two vectors."""
        pair = self._select_dynamic_pair(shell)

        if pair is None:
            random_bivector = self._random_unit_bivector()
            return (-(random_bivector) * (angle / 2.0)).exp()

        a_vec, b_vec = pair
        a = sum(val * bv for val, bv in zip(a_vec, self.basis_vectors))
        b = sum(val * bv for val, bv in zip(b_vec, self.basis_vectors))
        a = a.normal() if hasattr(a, 'normal') else a  # type: ignore[attr-defined]
        b = b.normal() if hasattr(b, 'normal') else b  # type: ignore[attr-defined]

        B = (a ^ b)

        try:
            magB = float(abs(B))
        except Exception:
            magB = 0.0
        if magB < 1e-9:
            try:
                return (-(self._random_unit_bivector()) * (angle / 2.0)).exp()
            except Exception:
                return self.layout.scalar if hasattr(self.layout, 'scalar') else 1.0

        B_normalized = B.normal() if hasattr(B, 'normal') else B  # type: ignore[attr-defined]

        try:
            rotor = (-B_normalized * angle / 2.0).exp()  # type: ignore[attr-defined]
        except Exception:
            rotor = self.layout.scalar if hasattr(self.layout, 'scalar') else 1.0
        return rotor

class DimensionalShell:
    """
    Represents a dimensional space where concepts exist as Geometric Algebra multivectors.
    """
    def __init__(self, dim: int, mind_instance: 'E8Mind'):
        if not CLIFFORD_AVAILABLE:
            raise ImportError("The 'clifford' library is required for DimensionalShell.")
        self.dim = dim
        self.mind = mind_instance
        self.layout, self.blades = clifford.Cl(dim)  # type: ignore[union-attr]
        self.vectors = {}
        self.basis_vectors = [self.blades[f'e{i+1}'] for i in range(dim)]
        self.rotor_generator = CliffordRotorGenerator(mind_instance, self.layout, self.blades)
        self.orientation = self.layout.scalar if hasattr(self.layout, 'scalar') else 1.0

        try:
            self._build_bivector_basis()
        except Exception:
            # If GA basis build fails, keep an empty list so other methods can guard on it
            self.bivector_basis = []
    def add_vector(self, node_id: str, vector: np.ndarray):
        """Converts a numpy vector to a multivector and adds it to the shell."""
        if vector.shape[0] != self.dim:
            padded_vector = np.zeros(self.dim)
            size_to_copy = min(vector.shape[0], self.dim)
            padded_vector[:size_to_copy] = vector[:size_to_copy]
            vector = padded_vector

        snapped_vector = self.mind._snap_to_lattice(vector, self.dim)
        mv = sum(val * bv for val, bv in zip(snapped_vector, self.basis_vectors))
        self.vectors[node_id] = mv

    def get_vector(self, node_id: str) -> Optional[np.ndarray]:
        """Retrieves a vector as a numpy array for external compatibility."""
        multivector = self.vectors.get(node_id)
        if multivector is None:
            return None

        return np.array([float(multivector[bv]) for bv in self.basis_vectors], dtype=np.float32)

    def spin(self, action_angle: float = 0.1):
        """Applies a concept-driven rotation to the entire shell."""
        if len(self.vectors) < 2:
            return

        incremental_rotor = self.rotor_generator.generate_rotor(self, action_angle)
        new_orientation = (incremental_rotor * self.orientation)
        self.orientation = new_orientation.normal() if hasattr(new_orientation, 'normal') else new_orientation

        orientation_reverse = ~self.orientation

        for node_id, mv in self.vectors.items():
            rotated_mv = self.orientation * mv * orientation_reverse
            rotated_vec_np = np.array([float(rotated_mv[bv]) for bv in self.basis_vectors])
            snapped_vec_np = self.mind._snap_to_lattice(rotated_vec_np, self.dim)
            self.vectors[node_id] = sum(val * bv for val, bv in zip(snapped_vec_np, self.basis_vectors))

    def get_all_vectors_as_matrix(self) -> tuple[Optional[np.ndarray], Optional[list[str]]]:
        """Returns all vectors as a single NumPy matrix."""
        if not self.vectors:
            return None, None

        node_ids = list(self.vectors.keys())
        matrix_list = []
        for mv in self.vectors.values():
            matrix_list.append([float(mv[bv]) for bv in self.basis_vectors])

        return np.array(matrix_list, dtype=np.float32), node_ids

    def _build_bivector_basis(self):
        try:
            self.bivector_basis = []
            for i in range(self.dim):
                for j in range(i+1, self.dim):
                    self.bivector_basis.append(self.basis_vectors[i] ^ self.basis_vectors[j])
        except Exception:
            self.bivector_basis = []

    def spin_with_bivector(self, bivector_coeffs, angle):
        try:
            if not hasattr(self, "bivector_basis") or not self.bivector_basis:
                self._build_bivector_basis()
            if len(self.vectors) < 1:
                return
            B = 0
            k = min(len(self.bivector_basis), len(bivector_coeffs))
            for idx in range(k):
                try:
                    B = B + float(bivector_coeffs[idx]) * self.bivector_basis[idx]
                except Exception:
                    pass
            try:
                Bn = B.normal() if hasattr(B, 'normal') else B  # type: ignore[attr-defined]
            except Exception:
                Bn = None
            if Bn is None:
                try:
                    self.spin(float(angle))
                except Exception:
                    pass
                return
            try:
                R = (-Bn * (float(angle) / 2.0)).exp()  # type: ignore[attr-defined]
                self.orientation = (R * self.orientation).normal() if hasattr(self.orientation, 'normal') else (R * self.orientation)
                Rrev = ~self.orientation
                new_vecs = {}
                for node_id, mv in self.vectors.items():
                    mv2 = self.orientation * mv * Rrev
                    vec_np = []
                    for bv in self.basis_vectors:
                        try:
                            vec_np.append(float(mv2[bv]))
                        except Exception:
                            vec_np.append(0.0)
                    snapped = self.mind._snap_to_lattice(np.array(vec_np), self.dim) if hasattr(self.mind, "_snap_to_lattice") else np.array(vec_np)
                    try:
                        new_vecs[node_id] = sum(val * bv for val, bv in zip(snapped, self.basis_vectors))
                    except Exception:
                        new_vecs[node_id] = mv
                self.vectors = new_vecs
            except Exception:
                try:
                    self.spin(float(angle))
                except Exception:
                    pass
        except Exception:
            pass

class ProximityEngine:
    """
    An engine for finding approximate nearest neighbors within and across dimensional shells.
    """
    def __init__(self, shell_dims: List[int], mind_instance: 'E8Mind', console: Any):
        self.console = console
        self.shell_dims = shell_dims
        self.mind = mind_instance
        self.indices: Dict[int, Optional[KDTree]] = {dim: None for dim in shell_dims}
        self.id_maps: Dict[int, List[str]] = {dim: [] for dim in shell_dims}

    def update_shell_index(self, dim: int, shell: DimensionalShell):
        """Rebuilds the KDTree index for a specific dimensional shell."""
        if dim not in self.indices: return

        matrix, node_ids = shell.get_all_vectors_as_matrix()
        if matrix is not None and node_ids and matrix.shape[0] > 0:
            try:
                self.indices[dim] = KDTree(matrix)
                self.id_maps[dim] = node_ids
            except Exception as e:
                self.console.log(f"[ProximityEngine] Failed to build KDTree for dim {dim}: {e}")
                self.indices[dim] = None
                self.id_maps[dim] = []
        else:
            self.indices[dim] = None
            self.id_maps[dim] = []

    def find_similar_in_shell(self, query_vector: np.ndarray, dim: int, k: int = 5) -> List[tuple[str, float]]:
        """Finds k-nearest neighbors for a query vector within its own shell."""
        kdtree = self.indices.get(dim)
        id_map = self.id_maps.get(dim)
        if kdtree is None or not id_map: return []

        num_points = kdtree.n
        if k > num_points: k = num_points
        if k == 0: return []

        distances, indices = kdtree.query(query_vector, k=k)

        if k == 1 and isinstance(indices, (int, np.integer)):
            return [(id_map[int(indices)], float(distances))]
        return [(id_map[int(i)], float(d)) for d, i in zip(distances, indices)]

    def cross_dimensional_query(self, query_vector: np.ndarray, source_dim: int, target_dim: int, k: int = 1) -> List[tuple[str, float]]:
        """Finds nearest neighbors for a vector from a source shell in a target shell."""
        if not TORCH_AVAILABLE or self.mind.autoencoder is None or not self.mind.autoencoder.is_trained: return []
        if source_dim == target_dim:
            return self.find_similar_in_shell(query_vector, target_dim, k)

        # Local import to satisfy type-checkers only when torch is available
        import torch as _torch
        with _torch.no_grad():
            source_tensor = _torch.from_numpy(query_vector).float().unsqueeze(0)

            projected_tensor = self.mind.autoencoder.project_between_dim(source_tensor, source_dim=source_dim, target_dim=target_dim)

            if projected_tensor is None:
                return []
            if hasattr(projected_tensor, 'cpu'):
                projected_vector = projected_tensor.squeeze(0).cpu().numpy()  # type: ignore[attr-defined]
            else:
                projected_vector = np.asarray(projected_tensor).squeeze(0)

        return self.find_similar_in_shell(projected_vector, target_dim, k)

    def hybrid_rerank_query(self, query_vector: np.ndarray, rerank_shell_dim: int, initial_k: int = 20, final_k: int = 5) -> List[tuple[str, float]]:
        """
        Performs a hybrid search:
        1. Fast ANN search in high-dimensional main memory to get initial candidates.
        2. Projects candidates and query into a lower-dimensional shell.
        3. Re-ranks candidates based on distance in the abstract shell space.
        """
        if not TORCH_AVAILABLE or self.mind.autoencoder is None or not self.mind.autoencoder.is_trained:
            return self.mind.memory.find_similar_in_main_storage(query_vector, k=final_k)

        initial_candidates = self.mind.memory.find_similar_in_main_storage(query_vector, k=initial_k)
        if not initial_candidates:
            return []

        candidate_ids = [nid for nid, _ in initial_candidates]

        rerank_shell = self.mind.dimensional_shells[rerank_shell_dim]
        candidate_vectors_low_dim = []
        valid_candidate_ids = []
        for nid in candidate_ids:
            vec = rerank_shell.get_vector(nid)
            if vec is not None:
                candidate_vectors_low_dim.append(vec)
                valid_candidate_ids.append(nid)

        if not valid_candidate_ids:
            return initial_candidates[:final_k]

        import torch as _torch
        with _torch.no_grad():
            query_tensor_high_dim = _torch.from_numpy(query_vector).float().unsqueeze(0)
            query_tensor_low_dim = self.mind.autoencoder.project_to_dim(query_tensor_high_dim, rerank_shell_dim)

        if query_tensor_low_dim is None:
            return initial_candidates[:final_k]

        if hasattr(query_tensor_low_dim, 'cpu'):
            query_vector_low_dim = query_tensor_low_dim.squeeze(0).cpu().numpy()  # type: ignore[attr-defined]
        else:
            query_vector_low_dim = np.asarray(query_tensor_low_dim).squeeze(0)

        candidate_matrix = np.array(candidate_vectors_low_dim)
        distances = cosine_distances(query_vector_low_dim.reshape(1, -1), candidate_matrix).flatten()

        reranked_results = sorted(zip(valid_candidate_ids, distances.tolist()), key=lambda item: item[1])

        return reranked_results[:final_k]

class ShellAttention:
    def __init__(self, out_dim: int = 32, keep_k: int = 3):
        self.out_dim = int(out_dim); self.keep_k = int(max(1, keep_k))

    @staticmethod
    def _ten(vec: np.ndarray) -> np.ndarray:
        if vec is None or vec.size == 0: return np.zeros(10, dtype=np.float32)
        if vec.size >= 10: return vec[:10].astype(np.float32)
        out = np.zeros(10, dtype=np.float32); out[:vec.size] = vec.astype(np.float32); return out

    def _weights(self, tensions: dict, mood: "MoodEngine") -> dict:
        eps = 1e-6
        coh = float(mood.mood_vector.get("coherence", 0.5)) if hasattr(mood, "mood_vector") else 0.5
        raw = {d: coh / (eps + float(t)) for d,t in tensions.items()}
        if not raw: return {}
        xs = np.array(list(raw.values()), dtype=np.float32); xs = np.exp(xs - xs.max()); xs /= (xs.sum() + 1e-12)
        return {d: float(w) for d,w in zip(raw.keys(), xs.tolist())}

    def build(self, mind: "E8Mind", out_dim: int = 32, keep_k: int = 3) -> np.ndarray:
        out_dim = int(out_dim or self.out_dim); keep_k = int(keep_k or self.keep_k)
        tensions = {}
        for dim, shell in mind.dimensional_shells.items():
            try:
                M,_ = shell.get_all_vectors_as_matrix()
                if M is not None and M.shape[0] > 1: tensions[dim] = float(np.linalg.norm(M - M.mean(0), axis=1).mean())
                else: tensions[dim] = 0.0
            except Exception: tensions[dim] = 0.0
        ws = self._weights(tensions, mind.mood)
        top = sorted(ws.items(), key=lambda kv: -kv[1])[:keep_k]
        parts = []
        for dim, w in top:
            try:
                M,_ = mind.dimensional_shells[dim].get_all_vectors_as_matrix()
                v = M.mean(0).astype(np.float32) if (M is not None and M.size>0) else np.zeros(dim, dtype=np.float32)
            except Exception:
                v = np.zeros(max(1,int(dim)), dtype=np.float32)
            parts.append(self._ten(v) * float(w))
        head = np.concatenate(parts, axis=0) if parts else np.zeros(0, dtype=np.float32)
        need = 10*keep_k
        if head.size < need: head = np.pad(head, (0, need-head.size))
        elif head.size > need: head = head[:need]
        gten = float(sum(tensions.values())/len(tensions)) if tensions else 0.0
        coh = float(mind.mood.mood_vector.get("coherence", 0.5))
        out = np.concatenate([head.astype(np.float32), np.array([gten, coh], dtype=np.float32)])
        if out.size < out_dim: out = np.pad(out, (0, out_dim - out.size))
        elif out.size > out_dim: out = out[:out_dim]
        return out.astype(np.float32)

class ArbiterGate:
    def __init__(self):
        self._last_tv = 0.0
    def decide(self, telemetry: dict, mood_vec: dict) -> float:
        tv = float((telemetry or {}).get("tv", 0.0))
        energy = float((telemetry or {}).get("energy", 0.0))
        norm = float((telemetry or {}).get("norm", 1.0))
        coh = float((mood_vec or {}).get("coherence", 0.5))
        ent = float((mood_vec or {}).get("entropy", 0.5))
        d_tv = tv - self._last_tv; self._last_tv = tv
        g = 0.5 + 0.35 * (0.3*coh - 0.7*max(0.0, d_tv)) - 0.15*abs(1.0 - norm) + 0.05*(0.5 - ent)
        return float(np.clip(g, 0.0, 1.0))

@dataclass
class AutoTask:
    id: str; label: str; reason: str; novelty: float; coherence: float; status: str = "pending"; created_step: int = 0

class AutoTaskManager:
    def __init__(self, console: Any):
        self.console = console; self.queue: list[AutoTask] = []
    def maybe_spawn(self, step: int, novelty: float, coherence: float, top_labels: list[str]):
        if novelty >= 1.10 and coherence <= 0.50:
            lid = f"task-{step}-{len(self.queue)+1}"
            label = (top_labels[0] if top_labels else "Consolidate new pattern")
            reason = f"Novelty {novelty:.2f} high, coherence {coherence:.2f} low. Add grounding task."
            t = AutoTask(id=lid, label=label, reason=reason, novelty=float(novelty), coherence=float(coherence), created_step=int(step))
            self.queue.append(t)
            try: self.console.log(f"[Curriculum] Spawned: {t.label} · {reason}")
            except Exception: pass
            return t
        return None
    def complete_if_related(self, node_label: str) -> float:
        for t in self.queue:
            if t.status == "pending" and node_label and (node_label.lower() in t.label.lower() or t.label.lower() in node_label.lower()):
                t.status = "done"
                return float(np.clip(0.15*(t.novelty - 0.8) + 0.15*(0.6 - t.coherence), 0.0, 0.5))
        return 0.0

class NoveltyScorer:
    """
    Calculates novelty and coherence scores for new concepts.
    This version queries the correct high-dimensional memory space and uses
    adaptive normalization to evaluate novelty relative to the memory's current density.
    """
    def __init__(self, memory_manager: 'MemoryManager', llm_pool: 'AsyncLLMPool', console: Any):
        self.console = console
        self.memory_manager = memory_manager
        self.llm_pool = llm_pool

    def calculate_novelty(self, new_vector: np.ndarray) -> float:
        """
        Calculates novelty based on the normalized distance to the nearest neighbor
        in the full, high-dimensional memory space.
        """
        similar_nodes = self.memory_manager.find_similar_in_main_storage(new_vector, k=1)
        if not similar_nodes:
            return 2.0

        distance_to_nearest = similar_nodes[0][1]
        avg_distance = self.memory_manager.get_average_nearest_neighbor_distance()
        if avg_distance < 1e-6:
            return 2.0

        novelty_score = distance_to_nearest / avg_distance
        return np.clip(novelty_score, 0.0, 2.0)

    async def calculate_coherence(self, new_concept_text: str) -> float:
        """Uses an LLM to rate the coherence (usefulness/well-formedness) of the new concept."""
        if not new_concept_text: return 0.0
        prompt = (
            f'On a scale from 0.0 to 1.0, how coherent and meaningful is the following idea? '
            f'A coherent idea is well-formed, logical, and potentially useful. '
            f'Respond with ONLY the numeric score.\n\n'
            f'Idea: "{new_concept_text}"\n\n'
            f'Coherence Score:'
        )
        try:
            response = await self.llm_pool.enqueue_and_wait(prompt, max_tokens=10, temperature=0.1)
            match = re.search(r"[-+]?\d*\.\d+|\d+", response)
            if match:
                return np.clip(float(match.group()), 0.0, 1.0)
            return 0.5
        except Exception as e:
            self.console.log(f"[NoveltyScorer] Coherence check failed: {e}")
            return 0.5

# [UPGRADE 5] Neural Episodic Memory (DND) for prioritized replay
class EpisodicMemory:
    """Stores high-reward episodes in a max-heap for prioritized replay."""
    def __init__(self, max_size=500):
        self.max_size = max_size
        # Max-heap is simulated with a min-heap of (-priority, data)
        self.heap = []

    def add_episode(self, episode_data, reward):
        """Adds an episode to the memory, prioritized by reward."""
        priority = -reward  # Use negative reward for max-heap behavior
        heapq.heappush(self.heap, (priority, episode_data))
        if len(self.heap) > self.max_size:
            # Remove the episode with the lowest reward (highest priority value)
            heapq.heappop(self.heap)

    def get_top_episodes(self, k=1):
        """Retrieves the top k episodes without removing them."""
        if not self.heap:
            return []
        top_k = heapq.nsmallest(k, self.heap)
        # Return just the episode data, not the priority
        return [data for priority, data in top_k]


    def sample_prioritized(self, k=32, mind=None, alpha=1.0, beta=1.0, gamma=1.0, eta=0.6):
        """Return up to k episodes sampled with priority p_i^eta, where
        p_i = alpha*reward + beta*surprise + gamma*novelty.
        If fields are missing, compute simple heuristics from memory.
        """
        if not self.heap:
            return []
        # Gather
        episodes = [ep for (_prio, ep) in self.heap]
        # Compute components
        rewards, surprises, novelties = [], [], []
        for ep in episodes:
            r = float(ep.get('reward', ep.get('rating', 0.0)))
            rewards.append(r)
            # Surprise: distance between child embedding and mean(parent embeddings)
            try:
                if mind is not None and 'node_id' in ep and 'parent_ids' in ep:
                    child_vec = mind.memory.main_vectors.get(ep['node_id']) or ep.get('embedding')
                    parents = [mind.memory.main_vectors.get(pid) for pid in ep['parent_ids'] if pid in mind.memory.main_vectors]
                    if child_vec is not None and parents:
                        import numpy as _np
                        pmean = _np.mean(_np.stack(parents), axis=0)
                        surprises.append(float(_np.linalg.norm(_np.array(child_vec) - pmean)))
                    else:
                        surprises.append(0.0)
                else:
                    surprises.append(float(ep.get('surprise', 0.0)))
            except Exception:
                surprises.append(0.0)
            # Novelty: prefer logged novelty else compute NN distance proxy
            try:
                nov = float(ep.get('novelty', 0.0))
                if nov == 0.0 and mind is not None and 'node_id' in ep:
                    # fallback: distance to nearest neighbor in memory index if available
                    try:
                        sim = mind.memory.find_similar_in_main_storage(mind.memory.main_vectors.get(ep['node_id']), k=2)
                        d = 1.0 - (sim[0][1] if sim else 0.0)
                        nov = float(d)
                    except Exception:
                        pass
                novelties.append(nov)
            except Exception:
                novelties.append(0.0)
        rewards = np.asarray(rewards, dtype=np.float64)
        surprises = np.asarray(surprises, dtype=np.float64)
        novelties = np.asarray(novelties, dtype=np.float64)
        pri = alpha*rewards + beta*surprises + gamma*novelties
        pri = np.maximum(pri, 1e-8)
        w = np.power(pri, eta)
        w = w / (w.sum() + 1e-12)
        # Sample without replacement
        idxs = np.random.choice(np.arange(len(episodes)), size=min(k, len(episodes)), replace=False, p=w)
        return [episodes[i] for i in idxs]
class InsightAgent:
    """An agent that generates new concepts and learns from an insight-driven reward signal."""
    def __init__(self, llm_pool: 'AsyncLLMPool', novelty_scorer: NoveltyScorer, console: Any):
        self.console = console
        self.llm_pool = llm_pool
        self.novelty_scorer = novelty_scorer
        self.reward_history = deque(maxlen=100)
        # [UPGRADE 5] Instantiate EpisodicMemory
        self.episodic_memory = EpisodicMemory()

    # In the InsightAgent class

    async def create_hybrid_concept(self, concept_a: Dict, concept_b: Dict) -> str:
        """Uses an LLM to synthesize a new, hybrid concept from two source concepts."""
        
        # --- START: New logic to get mind context ---
        mind_instance = self.novelty_scorer.memory_manager.mind
        try:
            # Get the mind's current top-level goal for direction.
            _, top_goal_desc = mind_instance.goal_field.get_top_goals(k=1)[0]
        except (IndexError, TypeError):
            top_goal_desc = "achieve greater understanding"

        # Get the subconscious narrative for thematic context.
        subconscious_narrative = mind_instance.subconscious.narrative
        # --- END: New logic ---

        # --- The new, more powerful prompt ---
        prompt = (
            f"You are a creative synthesizer of ideas. Your current high-level objective is to '{top_goal_desc}'.\n"
            f"The mind's current internal narrative is: \"{subconscious_narrative}\"\n\n"
            f"Synthesize the following two concepts into a single, novel idea that ADVANCES THE OBJECTIVE. "
            f"Describe the new idea in one or two clear sentences.\n\n"
            f"Concept A: '{concept_a.get('metaphor', concept_a.get('label', ''))}'\n"
            f"Concept B: '{concept_b.get('metaphor', concept_b.get('label', ''))}'\n\n"
            f"Hybrid Concept:"
        )
        new_concept_text = await self.llm_pool.enqueue_and_wait(prompt, max_tokens=100, temperature=0.85)
        return new_concept_text.strip()

    def learn_from_reward(self, reward: float, episode_data: Optional[Dict] = None):
        """
        Stores the reward and logs the full episode to the EpisodicMemory.
        """
        self.reward_history.append(reward)
        if len(self.reward_history) > 10:
            avg_reward = np.mean(self.reward_history)
            self.console.log(f"[InsightAgent] Average Insight Reward: {avg_reward:.3f}")

        # [UPGRADE 5] Log the complete episode if data is provided
        if episode_data:
            self.episodic_memory.add_episode(episode_data, reward)

class GraphDB:
    """A graph database wrapper around NetworkX for managing conceptual relationships."""
    def __init__(self):
        if nx is None: raise ImportError("networkx library is required for GraphDB.")
        self.graph = nx.Graph()
    def add_node(self, node_id: str, **attrs):
        """Adds a node to the graph with the given attributes."""
        self.graph.add_node(node_id, **attrs)
    def add_edge(self, source_id: str, target_id: str, **attrs):
        """Adds an edge between two nodes with the given attributes."""
        self.graph.add_edge(source_id, target_id, **attrs)
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a node's data."""
        if self.graph.has_node(node_id):
            return self.graph.nodes[node_id]
        return None
    def get_neighbors(self, node_id: str) -> List[str]:
        """Gets the neighbors of a node."""
        if self.graph.has_node(node_id):
            return list(self.graph.neighbors(node_id))
        return []

    # [UPGRADE 4] GraphRAG: Community Detection
    def compute_and_store_communities(self, partition_key: str = "community_id"):
        """Computes Louvain communities and stores the partition ID on each node."""
        if nx_comm is None or self.graph.number_of_nodes() < 10:
            return
        try:
            communities_iter = nx_comm.louvain_communities(self.graph, seed=GLOBAL_SEED)
            communities = list(cast(Iterable, communities_iter))  # type: ignore[arg-type]
            for i, community_nodes in enumerate(communities):
                for node_id in community_nodes:
                    if self.graph.has_node(node_id):
                        self.graph.nodes[node_id][partition_key] = i
            console.log(f"[GraphDB] Computed {len(communities)} communities.")
        except Exception as e:
            console.log(f"[GraphDB] Community detection failed: {e}")

    def increment_edge_weight(self, u, v, delta=0.1, min_w=0.0, max_w=10.0, **attrs):
        """Create edge if absent; add delta to 'weight' clamped to [min_w, max_w]."""
        try:
            if not self.graph.has_edge(u, v):
                self.graph.add_edge(u, v, weight=max(min_w, delta), **attrs)
            else:
                w = float(self.graph.get_edge_data(u, v, default={'weight': 0.0}).get('weight', 0.0)) + float(delta)
                w = min(max_w, max(min_w, w))
                self.graph[u][v]['weight'] = w
                for k,val in attrs.items():
                    self.graph[u][v][k] = val
        except Exception as e:
            try: console.log(f"[GraphDB] increment_edge_weight failed: {e}")
            except Exception: pass
    

if TORCH_AVAILABLE:
    import torch as _torch
    import torch.nn as _nn

    class GaussianActor(_nn.Module):
        def __init__(self, state_dim, action_dim, max_action):
            super().__init__()
            self.l1 = _nn.Linear(state_dim, 256)
            self.l2 = _nn.Linear(256, 256)
            self.mu = _nn.Linear(256, action_dim)
            self.log_std = _nn.Linear(256, action_dim)
            self.max_action = float(max_action)
        def forward(self, state):
            h = _nn.functional.relu(self.l1(state))
            h = _nn.functional.relu(self.l2(h))
            mu = self.mu(h)
            log_std = _torch.clamp(self.log_std(h), -5.0, 2.0)
            return mu, log_std
        def sample(self, state):
            mu, log_std = self.forward(state)
            std = _torch.exp(log_std)
            normal = _torch.distributions.Normal(mu, std)
            x_t = normal.rsample()
            y_t = _torch.tanh(x_t)
            action = y_t * self.max_action
            log_prob = normal.log_prob(x_t) - _torch.log(1 - y_t.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=1, keepdim=True)
            mu_action = _torch.tanh(mu) * self.max_action
            return action, log_prob, mu_action

    class QCritic(_nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.l1 = _nn.Linear(state_dim + action_dim, 256)
            self.l2 = _nn.Linear(256, 256)
            self.l3 = _nn.Linear(256, 1)
        def forward(self, state, action):
            x = _torch.cat([state, action], dim=-1)
            x = _nn.functional.relu(self.l1(x))
            x = _nn.functional.relu(self.l2(x))
            return self.l3(x)

else:
    # Minimal placeholders when torch is missing. Avoid redefining class names to prevent obscuring.
    GaussianActor = object  # type: ignore[assignment]
    QCritic = object  # type: ignore[assignment]

class ReplayBuffer:
    """Simple FIFO replay buffer."""
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.done = np.zeros((self.max_size, 1), dtype=np.float32)

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            np.asarray(self.state[ind]),
            np.asarray(self.action[ind]),
            np.asarray(self.next_state[ind]),
            np.asarray(self.reward[ind]),
            np.asarray(self.done[ind]),
        )

class PrioritizedReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e5), alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_frames = int(beta_frames)
        self.frame = 1
        self.eps = 1e-6
        self.state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.done = np.zeros((self.max_size, 1), dtype=np.float32)
        self.priorities = np.zeros((self.max_size,), dtype=np.float32)

    def add(self, state, action, next_state, reward, done):
        max_prio = self.priorities.max() if self.size > 0 else 1.0
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = float(done)
        self.priorities[self.ptr] = max_prio
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        if self.size == self.max_size:
            prios = self.priorities
        else:
            prios = self.priorities[:self.size]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(self.size, batch_size, p=probs)
        beta = self.beta_start + (1.0 - self.beta_start) * min(1.0, self.frame / self.beta_frames)
        self.frame += 1
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        return (
            self.state[indices],
            self.action[indices],
            self.next_state[indices],
            self.reward[indices],
            self.done[indices],
            weights.reshape(-1,1).astype(np.float32),
            indices
        )

    def update_priorities(self, indices, td_errors):
        prios = np.abs(td_errors) + self.eps
        self.priorities[indices] = prios

class SACMPOAgent:
        def __init__(self, state_dim, action_dim, max_action, console=None, tau=0.005, use_per=True, device=None):
            self.state_dim = int(state_dim)
            self.action_dim = int(action_dim)
            self.max_action = float(max_action)
            self.console = console
            self.tau = float(tau)
            if TORCH_AVAILABLE:
                self.device = device or (_torch.device("cuda") if _torch.cuda.is_available() else _torch.device("cpu"))
            else:
                raise ImportError("Torch is required for SACMPOAgent but is not available.")
            self.actor = GaussianActor(state_dim, action_dim, max_action).to(self.device)  # type: ignore[attr-defined]
            self.actor_old = GaussianActor(state_dim, action_dim, max_action).to(self.device)  # type: ignore[attr-defined]
            self.actor_old.load_state_dict(self.actor.state_dict())
            self.critics = _nn.ModuleList([QCritic(state_dim, action_dim).to(self.device) for _ in range(4)])  # type: ignore[attr-defined]
            self.critics_target = _nn.ModuleList([QCritic(state_dim, action_dim).to(self.device) for _ in range(4)])  # type: ignore[attr-defined]
            for i in range(4):
                self.critics_target[i].load_state_dict(self.critics[i].state_dict())
            self.active_critics = 2
            self.actor_opt = _torch.optim.Adam(self.actor.parameters(), lr=3e-4)
            self.critic_opts = [_torch.optim.Adam(self.critics[i].parameters(), lr=3e-4) for i in range(4)]
            self.log_alpha = _nn.Parameter(_torch.tensor(0.0, device=self.device))
            self.alpha_opt = _torch.optim.Adam([self.log_alpha], lr=3e-4)
            self.alpha_min, self.alpha_max = 1e-4, 1.0
            self.replay = PrioritizedReplayBuffer(state_dim, action_dim, max_size=int(2e5)) if use_per else ReplayBuffer(state_dim, action_dim, max_size=int(2e5))
            self._train_steps = 0
            self.batch_size = 256
            self.gamma = 0.99
            self.bh_pressure = 0.0
            self.kl_beta = 0.01

        def set_active_critics(self, n:int):
            self.active_critics = int(max(1, min(4, n)))

        @property
        def alpha(self):
            a = float(self.log_alpha.exp().item())
            return float(max(self.alpha_min, min(self.alpha_max, a)))

        def _target_entropy(self):
            bh = float(max(0.0, min(1.5, self.bh_pressure)))
            base = -float(self.action_dim) * 0.60
            scale = 0.60 + 0.25 * bh
            return float(base * scale)

        def select_action(self, state, deterministic=False):
            s = _torch.tensor(state, dtype=_torch.float32, device=self.device).unsqueeze(0)
            with _torch.no_grad():
                if deterministic:
                    mu, _ = self.actor.forward(s)
                    a = _torch.tanh(mu) * self.max_action
                else:
                    a, _, _ = self.actor.sample(s)
            return a.squeeze(0).cpu().numpy().astype("float32")

        def store(self, state, action, next_state, reward, done):
            self.replay.add(state, action, next_state, reward, done)

        def epistemic_std(self, state, action):
            try:
                s = _torch.tensor(state, dtype=_torch.float32, device=self.device).unsqueeze(0)
                a = _torch.tensor(action, dtype=_torch.float32, device=self.device).unsqueeze(0)
                qs = []
                with _torch.no_grad():
                    for i in range(self.active_critics):
                        qs.append(self.critics[i](s, a).cpu().item())
                if len(qs) <= 1:
                    return 0.0
                import numpy as _np
                return float(_np.std(_np.array(qs)))
            except Exception:
                return 0.0

        def _soft_update(self, net, target):
            for p, tp in zip(net.parameters(), target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        def update(self):
            if self.replay.size < max(1024, self.batch_size): return
            
            use_per = isinstance(self.replay, PrioritizedReplayBuffer)
            batch = self.replay.sample(self.batch_size)
            
            if use_per:
                state_np, action_np, next_state_np, reward_np, done_np, weights_np, indices = batch  # type: ignore[misc]
                weights = _torch.tensor(weights_np, dtype=_torch.float32, device=self.device)
            else:
                state_np, action_np, next_state_np, reward_np, done_np = batch  # type: ignore[misc]
                weights, indices = None, None

            state = _torch.tensor(state_np, dtype=_torch.float32, device=self.device)
            action = _torch.tensor(action_np, dtype=_torch.float32, device=self.device)
            next_state = _torch.tensor(next_state_np, dtype=_torch.float32, device=self.device)
            reward = _torch.tensor(reward_np, dtype=_torch.float32, device=self.device)
            done = _torch.tensor(done_np, dtype=_torch.float32, device=self.device)

            with _torch.no_grad():
                next_a, next_logp, _ = self.actor.sample(next_state)
                q_next = []
                for i in range(self.active_critics):
                    q_next.append(self.critics_target[i](next_state, next_a))
                q_next = _torch.min(_torch.stack(q_next, dim=0), dim=0).values
                target_v = q_next - self.log_alpha.exp() * next_logp
                target_q = reward + (1.0 - done) * self.gamma * target_v

            td_errors_for_buffer = []
            for i in range(self.active_critics):
                qi = self.critics[i](state, action)
                if i == 0 and use_per: # Calculate TD errors once for buffer update
                    td_errors = _torch.abs(qi - target_q).detach().cpu().numpy().flatten()
                    td_errors_for_buffer = td_errors
                
            if use_per and weights is not None:
                li = (_nn.functional.mse_loss(qi, target_q, reduction='none') * weights).mean()
            else:
                li = _nn.functional.mse_loss(qi, target_q)
            
            self.critic_opts[i].zero_grad()
            li.backward()
            _nn.utils.clip_grad_norm_(self.critics[i].parameters(), 1.0)
            self.critic_opts[i].step()

            if use_per and len(td_errors_for_buffer) > 0 and hasattr(self.replay, 'update_priorities'):
                self.replay.update_priorities(indices, td_errors_for_buffer)  # type: ignore[attr-defined]
            a, logp, _ = self.actor.sample(state)
            q_pi = []
            for i in range(self.active_critics):
                q_pi.append(self.critics[i](state, a))
            q_pi = _torch.min(_torch.stack(q_pi, dim=0), dim=0).values
            with _torch.no_grad():
                mu_old, logstd_old = self.actor_old.forward(state)
            mu_new, logstd_new = self.actor.forward(state)
            kl = 0.5 * (
                (logstd_old.exp().pow(2) + (mu_old - mu_new).pow(2)) / (logstd_new.exp().pow(2) + 1e-8)
                + 2*(logstd_new - logstd_old) - 1.0
            ).sum(dim=1, keepdim=True).mean()
            actor_loss = (self.log_alpha.exp() * logp - q_pi).mean() + self.kl_beta * kl
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            target_ent = self._target_entropy()
            alpha_loss = -(self.log_alpha * (logp.detach() + target_ent)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            for i in range(self.active_critics):
                self._soft_update(self.critics[i], self.critics_target[i])
            self._soft_update(self.actor, self.actor_old)

class BaseAgentAdapter:
    """Wrap baseline agent to the society interface."""
    def __init__(self, agent, name="base"):
        self.agent = agent; self.name = name
    async def select_action(self, state, mind):
        try: return self.agent.select_action(state)
        except Exception: 
            import numpy as _np; return _np.zeros(mind.action_dim, dtype=_np.float32)

class ActionCandidateSampler:
    def __init__(self, mind, K=12, mag=0.04):
        self.mind = mind; self.K=int(K); self.mag=float(mag)
    def sample(self):
        import numpy as _np
        ad = int(self.mind.action_dim); c=[_np.zeros(ad, dtype=_np.float32)]
        for i in range(min(self.K, ad)):
            v=_np.zeros(ad, dtype=_np.float32); v[i]=self.mag; c.append(v); c.append(-v)
        return c[:max(1,self.K)]

class NoveltyAgent:
    def __init__(self, sampler): self.sampler=sampler; self.name="nov"
    async def select_action(self, state, mind):
        import numpy as _np
        wm = getattr(mind, 'world_model', None); c = self.sampler.sample()
        best,score=None,-1e9
        for a in c:
            s = float(wm.score_transition(state,a)) if (wm and getattr(wm,'available',False) and getattr(wm,'ready',False)) else float(_np.linalg.norm(a))
            if s>score: best,score=a,s
        return best or c[0]

class StabilityAgent:
    def __init__(self, sampler): self.sampler=sampler; self.name="stab"
    async def select_action(self, state, mind):
        import numpy as _np
        c=self.sampler.sample()
        idx=int(_np.argmin([_np.linalg.norm(a) for a in c]))
        return c[idx]

class SynthesisAgent:
    def __init__(self, sampler): self.sampler=sampler; self.name="syn"
    async def select_action(self, state, mind):
        import numpy as _np
        c=self.sampler.sample(); anchors=getattr(mind,'anchors',None)
        target=None
        try:
            if anchors and getattr(anchors,'anchors',None):
                vecs=[_np.asarray(v,dtype=_np.float32) for v,_ in anchors.anchors]
                if len(vecs)>=2:
                    dmax,pair=-1,(vecs[0],vecs[0])
                    for i in range(len(vecs)):
                        for j in range(i+1,len(vecs)):
                            d=float(_np.linalg.norm(vecs[i]-vecs[j])); 
                            if d>dmax: dmax,pair=d,(vecs[i],vecs[j])
                    target=0.5*(pair[0]+pair[1])
        except Exception: target=None
        if target is None:
            return await NoveltyAgent(self.sampler).select_action(state, mind)
        wm = getattr(mind, 'world_model', None)
        def score(a):
            try:
                if wm and getattr(wm,'available',False) and getattr(wm,'ready',False):
                    return -abs(wm.score_transition(state,a))
            except Exception: pass
            return float(_np.linalg.norm(a))
        idx=int(_np.argmin([score(a) for a in c]))
        return c[idx]

def _softmax_b(values, beta: float):
    import numpy as _np
    x=_np.asarray(values, dtype=_np.float64); x=x-x.max(); ex=_np.exp(beta*x); s=ex.sum()
    return ex/s if s>0 and _np.isfinite(s) else _np.ones_like(ex)/len(ex)

class MetaArbiter:
    def __init__(self, agents: dict, drive_system, beta: float = 3.0, console=None, metrics=None):
        self.agents=agents; self.drive_system=drive_system; self.beta=float(beta); self.console=console; self.metrics=metrics
    def utilities(self, state, mind):
        ds=self.drive_system
        def _safe(fn, fallback):
            try: return float(fn(state))
            except Exception: return float(fallback)
        # Heuristics if drives are missing
        try:
            sims = mind.memory.find_similar_in_main_storage(state, k=5)
            nn = sims[0][1] if sims else 0.0
        except Exception: nn=0.25
        u_nov = _safe(getattr(ds,'novelty_need', lambda s: nn), nn)
        u_syn = _safe(getattr(ds,'synthesis_need', lambda s: nn), nn)
        u_stab= _safe(getattr(ds,'stability_need', lambda s: 0.3), 0.3)
        u_base= _safe(getattr(ds,'exploit_need', lambda s: 0.5*(1.0-u_nov)), 0.5*(1.0-u_nov))
        return {"nov":u_nov,"syn":u_syn,"stab":u_stab,"base":u_base}
    # [note] Context management is handled by InstrumentedLock; MetaArbiter isn't a lock.

class AsyncOpenAIClient:
    def __init__(self, api_key: str, console: Any):
        from openai import AsyncOpenAI, BadRequestError
        self.client = AsyncOpenAI(api_key=api_key)
        self.BadRequestError = BadRequestError
        self.console = console

    async def chat(self, messages, model=None, max_tokens=None, temperature=None):
        try:
            _model = model or DEFAULT_OPENAI_CHAT_MODEL
            cc = await self.client.chat.completions.create(
                model=_model, messages=messages, max_tokens=max_tokens, temperature=temperature)
            if cc.choices:
                return (cc.choices[0].message.content or "").strip()
            return "[LLM ERROR] No choices returned from API."
        except self.BadRequestError as e:
            # Try a safe fallback model if preview is not available
            try:
                fallback_model = "gpt-4o-mini"
                if (model or DEFAULT_OPENAI_CHAT_MODEL) != fallback_model:
                    cc = await self.client.chat.completions.create(
                        model=fallback_model, messages=messages, max_tokens=max_tokens, temperature=temperature)
                    if cc.choices:
                        self.console.log(f"[yellow]Fell back to {fallback_model} after BadRequestError for model '{model or DEFAULT_OPENAI_CHAT_MODEL}'.[/yellow]")
                        return (cc.choices[0].message.content or "").strip()
            except Exception:
                pass
            self.console.log(f"[bold red]OpenAI API Error: {e}[/bold red]")
            return f"[LLM ERROR] {e}"

    async def get_logprobs_and_tokens(self, messages, **kwargs):
        return -99.0, []

    async def embedding(self, text, model=None, dimensions=None):
        try:
            _model = model or "text-embedding-3-small"
            if dimensions is None:
                res = await self.client.embeddings.create(input=[text], model=_model)
            else:
                res = await self.client.embeddings.create(input=[text], model=_model, dimensions=int(dimensions))
            return res.data[0].embedding
        except Exception as e:
            self.console.log(f"[bold red]OpenAI Embedding Error: {e}[/bold red]")
            return np.zeros(EMBED_DIM).tolist()

    async def batch_embedding(self, texts, model=None, dimensions=None):
        try:
            _model = model or "text-embedding-3-small"
            if dimensions is None:
                res = await self.client.embeddings.create(input=texts, model=_model)
            else:
                res = await self.client.embeddings.create(input=texts, model=_model, dimensions=int(dimensions))
            return [d.embedding for d in res.data]
        except Exception as e:
            self.console.log(f"[bold red]OpenAI Batch Embedding Error: {e}[/bold red]")
            return [np.zeros(EMBED_DIM).tolist() for _ in texts]

class OllamaClient:
    def __init__(self, ollama_model: str, console: Any):
        if ollama is None:
            raise RuntimeError("Python package 'ollama' not installed. Please `pip install ollama`.")
        self.client = ollama.AsyncClient()
        self.model = ollama_model
        self.console = console

    async def chat(self, messages, **kwargs):
        try:
            res = await self.client.chat(model=self.model, messages=messages)
            return res["message"]["content"].strip()
        except Exception as e:
            self.console.log(f"[bold red]Ollama Chat Error: {e}[/bold red]")
            return f"[LLM ERROR] Could not connect to Ollama or model '{self.model}' not found."

    async def get_logprobs_and_tokens(self, messages, **kwargs):
        return -99.0, []

    async def embedding(self, text, model=None, dimensions=None):
        try:
            res = await self.client.embeddings(model=model or self.model, prompt=text)
            emb = res["embedding"]
            if dimensions:
                if len(emb) > dimensions:
                    emb = emb[:dimensions]
                elif len(emb) < dimensions:
                    emb = emb + [0.0] * (dimensions - len(emb))
            return emb
        except Exception as e:
            self.console.log(f"[bold red]Ollama Embedding Error: {e}[/bold red]")
            v = np.random.standard_normal(EMBED_DIM).astype(np.float32)
            v /= (np.linalg.norm(v) + 1e-12)
            return v.tolist()

    async def batch_embedding(self, texts, model=None, dimensions=None):
        try:
            tasks = [self.embedding(t, model, dimensions) for t in texts]
            return await asyncio.gather(*tasks)
        except Exception as e:
            self.console.log(f"[bold red]Ollama Batch Embedding Error: {e}[/bold red]")
            out = []
            for _ in texts:
                v = np.random.standard_normal(EMBED_DIM).astype(np.float32)
                v /= (np.linalg.norm(v) + 1e-12)
                out.append(v.tolist())
            return out

class GeminiClient:
    def __init__(self, api_key: str, model_name: str, console: Any):
        if genai is None:
            raise RuntimeError("google-generativeai is not installed. Please `pip install google-generativeai`. ")
        if not api_key:
            raise ValueError("Gemini API key is required.")
        # Configure using documented API. Fallback if attributes differ.
        try:
            if hasattr(genai, "configure"):
                genai.configure(api_key=api_key)  # type: ignore[call-arg]
        except Exception:
            pass
        try:
            self.model = genai.GenerativeModel(model_name)  # type: ignore[attr-defined]
        except Exception:
            # Fallback: some versions use genai.GenerativeModel with model= kw
            try:
                self.model = genai.GenerativeModel(model=model_name)  # type: ignore[call-arg]
            except Exception:
                self.model = None
        self.console = console

    async def chat(self, messages, max_tokens=None, temperature=None, **kwargs):
        try:
            gemini_messages = []
            for msg in messages:
                role = "model" if msg["role"] == "assistant" else "user"
                gemini_messages.append({"role": role, "parts": [msg["content"]]})
            if len(gemini_messages) > 1:
                deduped = [gemini_messages[0]]
                for i in range(1, len(gemini_messages)):
                    if gemini_messages[i]['role'] != deduped[-1]['role']:
                        deduped.append(gemini_messages[i])
                    else:
                        deduped[-1] = gemini_messages[i]
                gemini_messages = deduped
            # Build generation config defensively
            config = None
            try:
                types_mod = getattr(genai, "types", None)
                if types_mod is not None:
                    config = types_mod.GenerationConfig(max_output_tokens=max_tokens, temperature=temperature)
            except Exception:
                config = None
            # Some SDKs are sync only; call in a thread if needed
            if hasattr(self.model, "generate_content_async"):
                response = await self.model.generate_content_async(gemini_messages, generation_config=config)  # type: ignore[call-arg]
            else:
                import asyncio as _asyncio
                response = await _asyncio.to_thread(self.model.generate_content, gemini_messages, generation_config=config)  # type: ignore[call-arg]

            text_out = ""
            try:
                candidates = getattr(response, "candidates", []) or []
                chosen = None
                for c in candidates:
                    content = getattr(c, "content", None)
                    parts = getattr(content, "parts", None) if content is not None else None
                    if parts:
                        chosen = c
                        break
                if chosen is not None:
                    parts = chosen.content.parts
                    chunk_list = []
                    for p in parts:
                        t = getattr(p, "text", None)
                        if isinstance(t, str):
                            chunk_list.append(t)
                    text_out = "".join(chunk_list).strip()
                if not text_out:
                    try:
                        text_out = (response.text or "").strip()
                    except Exception:
                        text_out = ""
                if not text_out:
                    try:
                        fr = None
                        if candidates:
                            fr = getattr(candidates[0], "finish_reason", None)
                        self.console.log(f"[bold red]Gemini returned no text. finish_reason={fr}[/bold red]")
                    except Exception:
                        pass
                    return ""
            except Exception as e:
                self.console.log(f"[bold red]Gemini Parse Error: {e}[/bold red]")
                return ""
            return text_out
        except Exception as e:
            self.console.log(f"[bold red]Gemini Chat Error: {e}[/bold red]")
            return ""

    async def get_logprobs_and_tokens(self, messages, **kwargs):
        return -99.0, []

    async def embedding(self, text, model="models/embedding-001", **kwargs):
        try:
            # Use to_thread to call sync embed_content on versions without async
            _embed_async = getattr(genai, "embed_content_async", None)
            _embed_sync = getattr(genai, "embed_content", None)
            if _embed_async is not None:
                result = await _embed_async(model=model, content=text, task_type="retrieval_document")  # type: ignore[call-arg]
            elif _embed_sync is not None:
                import asyncio as _asyncio
                result = await _asyncio.to_thread(_embed_sync, model=model, content=text, task_type="retrieval_document")  # type: ignore[call-arg]
            else:
                return np.zeros(EMBED_DIM).tolist()
            # result may be dict-like or object; handle generically
            emb = getattr(result, "embedding", None)
            if emb is None and isinstance(result, dict):
                emb = result.get("embedding")
            return emb if emb is not None else np.zeros(EMBED_DIM).tolist()
        except Exception as e:
            self.console.log(f"[bold red]Gemini Embedding Error: {e}[/bold red]")
            return np.zeros(EMBED_DIM).tolist()

    async def batch_embedding(self, texts, model="models/embedding-001", **kwargs):
        try:
            _embed_async = getattr(genai, "embed_content_async", None)
            _embed_sync = getattr(genai, "embed_content", None)
            if _embed_async is not None:
                result = await _embed_async(model=model, content=texts, task_type="retrieval_document")  # type: ignore[call-arg]
            elif _embed_sync is not None:
                import asyncio as _asyncio
                result = await _asyncio.to_thread(_embed_sync, model=model, content=texts, task_type="retrieval_document")  # type: ignore[call-arg]
            else:
                return [np.zeros(EMBED_DIM).tolist() for _ in texts]
            emb = getattr(result, "embedding", None)
            if emb is None and isinstance(result, dict):
                emb = result.get("embedding")
            if isinstance(emb, list):
                return emb
            return [np.zeros(EMBED_DIM).tolist() for _ in texts]
        except Exception as e:
            self.console.log(f"[bold red]Gemini Batch Embedding Error: {e}[/bold red]")
            return [np.zeros(EMBED_DIM).tolist() for _ in texts]

class AsyncLLMPool:
    def __init__(self, mind_instance, worker_count):
        self.mind = mind_instance
        self.queue = asyncio.Queue(maxsize=worker_count * 4)
        self.workers = []
        self.worker_count = worker_count
        self.lock = asyncio.Lock()
        self._results: Dict[int, Any] = {}
        self._next_id = 0
        self.running = True
        self._sem = asyncio.Semaphore(int(os.getenv('E8_MAX_INFLIGHT', '32')))
        try:
            self._timeout = float(os.getenv('E8_LLM_TIMEOUT', '45'))
        except Exception:
            self._timeout = 45.0

    async def _worker(self):
        while self.running:
            try:
                prompt_id, prompt, args = await self.queue.get()
                if prompt_id is None:
                    self.queue.task_done()
                    break

                result = "[LLM UNKNOWN ERROR]"
                try:
                    self.mind.console.log(f"[LLM POOL] Worker starting task id={prompt_id} key={args.get('_prompt_key','ask')} model={self.mind.client_model}")
                    result = await asyncio.wait_for(
                        self.mind._async_call_llm_internal(prompt, **(args or {})),
                        timeout=POOL_WORKER_TIMEOUT
                    )
                    self.mind.console.log(f"[LLM POOL] Worker finished task id={prompt_id}")
                except asyncio.TimeoutError:
                    result = f"[LLM TIMEOUT] Task {prompt_id} exceeded {POOL_WORKER_TIMEOUT}s."
                except asyncio.CancelledError:
                    result = "[LLM CANCELLED]"
                    break
                except Exception as e:
                    result = f"[LLM ERROR] {e}"
                    self.mind.console.log(f"[LLM POOL] Worker error on task id={prompt_id}: {e}")
                finally:
                    async with self.lock:
                        self._results[prompt_id] = result or ""
                    self.queue.task_done()
            except asyncio.CancelledError:
                break

    async def start(self):
        if self.workers and any(not w.done() for w in self.workers): return
        self.running = True
        self.workers = [w for w in self.workers if not w.done()]
        for _ in range(self.worker_count - len(self.workers)):
            self.workers.append(asyncio.create_task(self._worker()))
        self.mind.console.log(f"[LLM POOL] Started {len(self.workers)} workers.")

    async def stop(self):
        self.running = False
        for _ in range(len(self.workers)):
            try:
                await self.queue.put((None, None, None))
            except asyncio.CancelledError:
                pass
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()

    async def submit(self, prompt, **kwargs) -> int:
        async with self.lock:
            prompt_id = self._next_id; self._next_id += 1
            self._results[prompt_id] = None
        await self.queue.put((prompt_id, prompt, kwargs))
        return prompt_id

    async def get_result(self, prompt_id, timeout=None):
        start = time.time()
        if timeout is None:
            timeout = POOL_RESULT_TIMEOUT
        while True:
            async with self.lock:
                result = self._results.get(prompt_id)
            if result is not None:
                async with self.lock:
                    if prompt_id in self._results:
                        del self._results[prompt_id]
                if isinstance(result, str) and result.startswith('[LLM'):
                    raise Exception(result)
                return result
            if time.time() - start > timeout:
                raise asyncio.TimeoutError(f"Pool timeout for prompt_id {prompt_id}")

            await asyncio.sleep(0.01)

    async def enqueue_and_wait(self, prompt, **kwargs):
        async def _inner():
            pid = await self.submit(prompt, **kwargs)
            return await self.get_result(pid, timeout=self._timeout)
        async def _wrapped():
            async with self._sem:
                try:
                    # Py3.10 compatibility: use wait_for instead of asyncio.timeout CM
                    return await asyncio.wait_for(_inner(), timeout=self._timeout)
                except Exception:
                    return "[LLM:DEGRADED]"
        return await _wrapped()

# --- Minimal instrumentation and optional-module stubs ---
class InstrumentedLock:
    """An asyncio.Lock wrapper that logs acquire/release events via Probe."""
    def __init__(self, name: str, probe: Optional["Probe"] = None):
        import asyncio as _asyncio
        self._lock = _asyncio.Lock()
        self.name = str(name)
        self.probe = probe
        self._t_acq = None

    async def __aenter__(self):
        import time as _time
        t0 = _time.time()
        await self._lock.acquire()
        self._t_acq = _time.time()
        if self.probe is not None:
            try:
                await self.probe.log(ev="lock_acquire", lock=self.name, wait_ms=round((_time.time()-t0)*1000.0,2))
            except Exception:
                self.probe.log_sync(ev="lock_acquire", lock=self.name)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        import time as _time
        if self.probe is not None:
            try:
                await self.probe.log(ev="lock_release", lock=self.name, held_ms=round((_time.time()-float(self._t_acq or _time.time()))*1000.0,2))
            except Exception:
                self.probe.log_sync(ev="lock_release", lock=self.name)
        self._lock.release()

    # Provide explicit acquire/release for direct use
    async def acquire(self):
        return await self._lock.acquire()
    def release(self):
        return self._lock.release()

class Probe:
    """Lightweight async probe that writes newline-delimited JSON events."""
    def __init__(self, run_id: str):
        self.run_id = str(run_id)
        self.path = os.path.join(RUNTIME_DIR, self.run_id, "debug", "probe.ndjson")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._lock = asyncio.Lock()

    def _write(self, record: dict):
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")
        except Exception:
            pass

    async def log(self, **fields):
        rec = {"ts": (datetime.now(timezone.utc).isoformat() if (datetime is not None and timezone is not None) else __import__('datetime').datetime.utcnow().isoformat())}
        rec.update(fields)
        async with self._lock:
            await asyncio.to_thread(self._write, rec)

    def log_sync(self, **fields):
        rec = {"ts": (datetime.now(timezone.utc).isoformat() if (datetime is not None and timezone is not None) else __import__('datetime').datetime.utcnow().isoformat())}
        rec.update(fields)
        self._write(rec)

def set_asyncio_exception_logger(probe: Probe):
    """Install an asyncio exception handler that forwards to the probe."""
    try:
        loop = asyncio.get_event_loop()
        def _handler(loop, context):
            msg = context.get("message") or str(context.get("exception"))
            try:
                loop.create_task(probe.log(ev="asyncio_exception", message=str(msg)))
            except Exception:
                probe.log_sync(ev="asyncio_exception", message=str(msg))
        loop.set_exception_handler(_handler)
    except Exception:
        pass

class SocietyOfMind:
    """Minimal stub: selects a neutral action when advanced society is unavailable."""
    def __init__(self, mind, beta: float = 3.0, K: int = 12):
        self.mind = mind
        self.beta = float(beta)
        self.K = int(K)
    async def step(self, state: np.ndarray, mind):
        return np.zeros(getattr(mind, 'action_dim', 6), dtype=np.float32)

# Wavey integration (guarded import)
try:
    from wavey import WaveyE8Bridge, integrate_one_cycle, PotentialFunction  # type: ignore
except Exception:
    class WaveyE8Bridge:  # type: ignore
        def __init__(self, embed_dim: int = EMBED_DIM, seed: int = 0, topk: int = 8):
            self.embed_dim = int(embed_dim)
            self.seed = int(seed)
            self.topk = int(topk)
    def integrate_one_cycle(mind, bridge):  # type: ignore
        return {
            "hamiltonian_bias": np.zeros(240, dtype=np.float32),
            "attention_weights": np.zeros((0,), dtype=np.float32),
            "potentials": [],
            "events": [],
            "seed_used": False,
            "node_potentials": np.zeros(240, dtype=np.float32),
        }
def neuro_to_engine(DA: float, NE: float, ACh: float, S5: float):
    DA, NE, ACh, S5 = np.clip([DA, NE, ACh, S5], 0.0, 1.0)
    sigma = float(np.clip(1.25 * (1.0 + 0.6*NE - 0.3*ACh + 0.3*S5), 0.8, 2.2))
    alpha_cur = float(np.clip(0.12 * (1.0 + 0.8*NE - 0.3*S5 + 0.4*DA), 0.02, 0.35))
    zeta = max(0.0, 0.03 * (1.0 + 0.7*S5 - 0.7*NE))
    sensory_gain = 1.0 + 0.8*ACh + 0.3*DA
    prior_gain = 1.0 + 0.6*S5 - 0.5*ACh
    phi0 = float(np.clip(0.10 * (1.0 + 0.6*DA + 0.3*NE - 0.2*S5), 0.02, 0.25))
    J = float(np.clip(0.08 * (1.0 + 0.5*NE + 0.2*ACh - 0.2*S5), 0.0, 0.2))
    return dict(sigma=sigma, alpha_cur=alpha_cur, zeta=zeta, sensory_gain=sensory_gain, prior_gain=prior_gain, phi0=phi0, J=J)

def theta_phase(step_idx: int, theta_len: int = 8):
    ph = step_idx % theta_len
    return ph, (ph < 5), (ph == 5), (ph == 6), (ph == 7)

class MPS:
    def __init__(self, M: int, d: int, chi: int = 8):
        self.M = int(M); self.d = int(d); self.chi = int(chi)
        self.A = []
        for k in range(M):
            chiL = 1 if k == 0 else chi
            chiR = 1 if k == M-1 else chi
            T = np.zeros((chiL, d, chiR), dtype=np.complex64)
            for i in range(min(chiL, chiR)):
                T[i, 0, i] = 1.0 + 0j
            self.A.append(T)

    def state_vector(self):
        v = self.A[0].reshape(-1, self.d * self.A[0].shape[-1])
        for k in range(1, self.M):
            T = self.A[k]
            v = v @ T.reshape(T.shape[0], -1)
            v = v.reshape(-1, T.shape[-1]*self.d)
        return v.reshape(-1)

def generate_e8_roots():
    roots = set()
    if combinations is None: return np.array([])
    for i, j in combinations(range(8), 2):
        for s1, s2 in [(1,1), (1,-1), (-1,1), (-1,-1)]:
            vec = [0]*8; vec[i], vec[j] = s1, s2
            roots.add(tuple(vec))
    for signs in range(2**8):
        vec, neg_count = [], 0
        for i in range(8):
            if (signs >> i) & 1: vec.append(-0.5); neg_count += 1
            else: vec.append(0.5)
        if neg_count % 2 == 0: roots.add(tuple(vec))
    return np.array(list(roots))

def build_weighted_adjacency(roots, atol=1e-6):
    R = roots.astype(np.float32); N = R.shape[0]
    mask = np.isclose(np.abs(R @ R.T), 1.0, atol=atol)
    np.fill_diagonal(mask, False)
    W = np.zeros((N, N), dtype=np.float32); W[mask] = 1.0
    int_roots = {tuple((2*r).astype(np.int8)) for r in R}
    for i in range(N):
        for j in np.where(mask[i])[0]:
            ri2, rj2 = (2*R[i]).astype(np.int8), (2*R[j]).astype(np.int8)
            s, d = tuple((ri2 + rj2).tolist()), tuple((ri2 - rj2).tolist())
            W[i, j] += 0.15 * (s in int_roots) + 0.10 * (d in int_roots)
    return W

def build_diff_adjacency(roots):
    R = roots.astype(np.float32)
    N = R.shape[0]
    int_roots = set(tuple((2*r).astype(np.int8)) for r in R)
    Wd = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        ri2 = tuple((2*R[i]).astype(np.int8))
        for j in range(N):
            if i == j: continue
            rj2 = tuple((2*R[j]).astype(np.int8))
            d = tuple(a - b for a, b in zip(ri2, rj2))
            if d in int_roots: Wd[i, j] = 1.0
    Wd = 0.5 * (Wd + Wd.T)
    np.fill_diagonal(Wd, 0.0)
    return Wd

def all_pairs_hops(A_bool):
    N = A_bool.shape[0]
    nbrs = [np.where(A_bool[i] > 0)[0] for i in range(N)]
    dist = np.full((N, N), np.inf, dtype=np.float32)
    for s in range(N):
        dist[s, s] = 0.0; q = deque([s])
        while q:
            u = q.popleft()
            for v in nbrs[u]:
                if dist[s, v] == np.inf:
                    dist[s, v] = dist[s, u] + 1.0
                    q.append(v)
    return dist

def weyl_average_potential(physics, anchors, draws=3, seed=None):
    rng = np.random.default_rng(seed)
    V_acc = np.zeros(physics.weights.shape[0], dtype=np.float32)
    def rand_sign_perm(rng):
        P = np.eye(8, dtype=np.float32); rng.shuffle(P)
        signs = rng.choice([-1.0, 1.0], size=(8,), replace=True).astype(np.float32)
        if (signs < 0).sum() % 2 == 1: signs[0] *= -1.0
        return (P.T * signs).T
    for _ in range(draws):
        A = rand_sign_perm(rng)
        transformed = []
        for (s, lam) in anchors.anchors:
            sA = (A @ s).astype(np.float32)
            sA /= np.linalg.norm(sA) + 1e-12
            transformed.append((sA, lam))
        tmp = MultiAnchorField(physics, kernel=anchors.kernel, rbf_sigma=anchors.rbf_sigma)
        tmp.set(transformed)
        V_acc += tmp.potential()
    return (V_acc / float(draws)).astype(np.float32)

def add_curiosity_penalty(V, visits, alpha=0.12):
    try:
        cur = -alpha * np.log1p(visits.astype(np.float32))
        return (V + cur).astype(np.float32)
    except Exception:
        return V

class E8Physics:
    def __init__(self, console):
        self.console = console
        self.roots = generate_e8_roots()
        self.roots_unit = self.roots / (np.linalg.norm(self.roots, axis=1, keepdims=True) + 1e-12)
        self.roots_kdtree = KDTree(self.roots)
        self.weights = build_weighted_adjacency(self.roots)
        self.adj_bool = (self.weights > 0).astype(np.int8)
        self.hops = all_pairs_hops(self.adj_bool)
        self.L_norm = self._build_normalized_laplacian()
        self._mask_cache = {}
        self.projection_matrix = None
        self.console.log(f"[INIT] E8Physics: roots={len(self.roots)}, edges={(self.adj_bool.sum())//2}")

    def find_nearest_root_index(self, vector_8d: np.ndarray) -> Optional[int]:
        if vector_8d is None or vector_8d.shape[0] != 8:
            return None
        try:
            _, index = self.roots_kdtree.query(vector_8d.reshape(1, -1), k=1)
            result_index = index[0] if isinstance(index, np.ndarray) else index
            return int(result_index)
        except Exception as e:
            self.console.log(f"[E8Physics] Error finding nearest root: {e}")
            return None

    def generate_quasicrystal_blueprint(self, seed: int = GLOBAL_SEED):
        P, pts = None, None
        uniqueness_threshold = 230
        max_tries = 32

        for i in range(max_tries):
            current_seed = seed + i
            rng = np.random.default_rng(current_seed)
            M = rng.normal(size=(8, 3)).astype(np.float32)
            Q, _ = np.linalg.qr(M)
            P_candidate = Q[:, :3]

            pts_candidate = self.roots @ P_candidate
            unique_pts = np.unique(np.round(pts_candidate, 3), axis=0)

            if len(unique_pts) >= uniqueness_threshold:
                P = P_candidate
                pts = pts_candidate
                self.console.log(f"[INIT] Quasicrystal projection found after {i+1} tries. Uniqueness: {len(unique_pts)}/240.")
                break

        if P is None:
            self.console.log(f"[bold yellow][WARN] Quasicrystal projection failed to meet uniqueness threshold after {max_tries} tries. Using last attempt.[/bold yellow]")
            rng = np.random.default_rng(seed + max_tries - 1)
            M = rng.normal(size=(8, 3)).astype(np.float32)
            Q, _ = np.linalg.qr(M)
            P = Q[:, :3]
            pts = self.roots @ P

        # At this point, pts should not be None; assert for type-checkers and safety
        assert pts is not None, "Projection points (pts) must be computed before normalization."
        pts -= pts.mean(axis=0, keepdims=True)
        pts /= (np.abs(pts).max() + 1e-6)
        self.projection_matrix = P

        blueprint_coords = []
        rounded_coords = np.round(pts, 4)
        coord_groups = defaultdict(list)
        for i, coord in enumerate(rounded_coords):
            coord_groups[tuple(coord)].append(i)

        for i in range(pts.shape[0]):
            base_x, base_y, base_z = float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])
            group = coord_groups[tuple(rounded_coords[i])]
            render_x, render_y = base_x, base_y

            if len(group) > 1:
                k = group.index(i) + 1
                epsilon = 0.005
                radius = epsilon * math.sqrt(k)
                theta = k * math.pi * (3 - math.sqrt(5))
                render_x += radius * math.cos(theta)
                render_y += radius * math.sin(theta)

            blueprint_coords.append({
                "id": i, "x": base_x, "y": base_y, "z": base_z,
                "render_x": render_x, "render_y": render_y, "render_z": base_z
            })

        try:
            kdtree = KDTree(pts)
            distances, _ = kdtree.query(pts, k=2)
            min_dist = np.min(distances[:, 1])
            self.console.log(f"[INIT] Min nearest-neighbor distance in blueprint: {min_dist:.4f}")
        except Exception as e:
            self.console.log(f"[INIT] Could not calculate min distance: {e}")

        return blueprint_coords

    def _build_normalized_laplacian(self):
        if csr_matrix is None or diags is None:
            # Pure numpy fallback for normalized Laplacian
            W = np.asarray(self.weights, dtype=np.float32)
            deg = np.sum(W, axis=1)
            d_is = 1.0 / np.sqrt(np.maximum(deg, 1e-9))
            D_inv_sqrt = np.diag(d_is)
            return np.eye(W.shape[0], dtype=np.float32) - D_inv_sqrt @ W @ D_inv_sqrt
        W = csr_matrix(self.weights, dtype=np.float32)
        deg = np.asarray(W.sum(axis=1)).ravel()
        D_inv_sqrt = diags(1.0 / np.sqrt(np.maximum(deg, 1e-9)))
        n = int(np.shape(self.weights)[0])
        return diags(np.ones(n, dtype=np.float32)) - D_inv_sqrt @ W @ D_inv_sqrt

    def heat_mask_cached(self, center_idx, sigma=1.25):
        key = (int(center_idx), round(float(sigma), 2))
        m = self._mask_cache.get(key)
        if m is None:
            d = self.hops[center_idx]
            m = np.exp(- (d * d) / (2.0 * sigma * sigma)).astype(np.float32)
            self._mask_cache[key] = m
        return m

class ClassicalConfig:
    def __init__(self, seed=None):
        self.seed = seed

class QuantumConfig:
    def __init__(self, gamma: float = 0.03, dt: float = 0.25, batch: int = 9,
                 dephase: float = 0.0, locality_sigma: float = 1.5,
                 seed=None, topk_amp: int = 5, non_linearity_strength: float = 2.5):
        self.gamma = gamma
        # Graph-aware extensions
        self.mode = os.getenv('E8_QMODE', 'lattice')  # 'lattice' or 'graph'
        self.lambda_V = float(os.getenv('E8_Q_LAMBDA', '0.2'))
        self.decay_tau = float(os.getenv('E8_Q_DECAY_TAU', '600'))
        self.rebuild_every = int(os.getenv('E8_Q_REBUILD_EVERY', '10'))
        self.alpha_edge = float(os.getenv('E8_Q_ALPHA_EDGE', '1.0'))
        self.reward_gain = float(os.getenv('E8_Q_REWARD_GAIN', '0.5'))

        self.dt = dt
        self.batch = batch
        self.dephase = dephase
        self.locality_sigma = locality_sigma
        self.seed = seed
        self.topk_amp = topk_amp
        self.non_linearity_strength = non_linearity_strength

# === Graph-aware Hamiltonian helpers (CTQW on memory graph projected to E8) ===
def _safe_node_to_root_idx(nid, memory, mind):
    try:
        node = memory.graph_db.get_node(nid)
        loc = node.get('blueprint_location_id') if node else None
        if isinstance(loc, (int, np.integer)) and 0 <= int(loc) < 240:
            return int(loc)
        vec = getattr(memory, 'main_vectors', {}).get(nid) if hasattr(memory, 'main_vectors') else None
        if vec is not None and TORCH_AVAILABLE and getattr(mind, 'autoencoder', None) and getattr(mind.autoencoder, 'is_trained', False):
            import torch as _torch
            with _torch.no_grad():
                z8 = mind.autoencoder.project_to_dim(_torch.from_numpy(np.asarray(vec, dtype=np.float32)).float().unsqueeze(0), 8)
                if z8 is not None:
                    try:
                        z8_np = z8.squeeze(0).cpu().numpy()
                    except Exception:
                        z8_np = np.asarray(z8).squeeze()
                    idx = mind.physics.find_nearest_root_index(z8_np)
                    return int(idx) if idx is not None else None
    except Exception:
        return None
    return None

def build_adjacency_240_from_memory(memory, mind, alpha=1.0, decay_tau=600.0, reward_gain=0.5):
    rows, cols, data = [], [], []
    try:
        G = memory.graph_db.graph
        now = getattr(mind, 'step_num', 0)
        for u, v, attr in G.edges(data=True):
            iu = _safe_node_to_root_idx(u, memory, mind)
            iv = _safe_node_to_root_idx(v, memory, mind)
            if iu is None or iv is None or iu == iv:
                continue
            w = float(attr.get('weight', 1.0))
            ts = float(attr.get('ts', 0.0))
            lu = float((G.nodes[u] or {}).get('last_step', 0)) if u in G.nodes else 0.0
            lv = float((G.nodes[v] or {}).get('last_step', 0)) if v in G.nodes else 0.0
            last_seen = max(ts, lu, lv)
            rec = float(np.exp(-(max(0.0, now - last_seen))/max(1e-6, decay_tau)))
            ru = float((G.nodes[u] or {}).get('insight_reward', 0.0)) if u in G.nodes else 0.0
            rv = float((G.nodes[v] or {}).get('insight_reward', 0.0)) if v in G.nodes else 0.0
            rewd = float(attr.get('reward', 0.0))
            rbar = (ru + rv + rewd) / 3.0
            wij = alpha * w * rec * (1.0 + reward_gain * rbar)
            if wij <= 0: 
                continue
            rows.append(iu); cols.append(iv); data.append(wij)
            rows.append(iv); cols.append(iu); data.append(wij)
        if csr_matrix is None:
            # Dense fallback
            A = np.zeros((240,240), dtype=np.float32)
            if data:
                r = np.asarray(rows, dtype=np.int32)
                c = np.asarray(cols, dtype=np.int32)
                d = np.asarray(data, dtype=np.float32)
                A[r, c] = d
            return A
        else:
            if data:
                A = csr_matrix((np.asarray(data, dtype=np.float32), (np.asarray(rows), np.asarray(cols))), shape=(240,240))
            else:
                A = csr_matrix((240,240), dtype=np.float32)
            return A
    except Exception as e:
        try:
            console.log(f"[Quantum] build_adjacency_240_from_memory failed: {e}")
        except Exception:
            pass
        if csr_matrix is None:
            return np.zeros((240,240), dtype=np.float32)
        return csr_matrix((240,240), dtype=np.float32)
# === End Graph-aware helpers ===
class QuantumEngine:
    def attach_mind(self, mind_instance):
        self.mind = mind_instance
        return self
    def __init__(self, physics, config, console: Any):
        self.mind = None
        self._last_graph_build_step = -1

        self.console = console
        self.physics, self.config = physics, config
        self.psi = np.ones((config.batch, 240), dtype=np.complex64) / np.sqrt(240)
        self.rng = np.random.default_rng(config.seed)
        self.H: Any = None
        self._last_H: Any = None
        self._last_potential: Optional[np.ndarray] = None
        self._last_norm = np.nan
        self._last_energy = np.nan
        self.build_hamiltonian()
        self.console.log("[INIT] Quantum Engine online (Non-Linear Edition).")

    def build_hamiltonian(self, V: Optional[np.ndarray] = None):
        if diags is None or csr_matrix is None:
            # Fallback: store dense H using numpy
            if V is None:
                V = np.zeros(240, dtype=np.float32)
            H = (self.config.gamma * np.asarray(self.physics.L_norm, dtype=np.complex64)) + np.diag(V)
            self.H = H
            self._last_H = H
            self._last_potential = np.asarray(V).copy()
            return
        if V is None:
            V = np.zeros(240, dtype=np.float32)
        # Safer guard for optional mind/memory
        _mind = getattr(self, 'mind', None)
        _mem = getattr(_mind, 'memory', None) if _mind is not None else None
        use_graph = (getattr(self.config, 'mode', 'lattice') == 'graph' and _mind is not None and _mem is not None)
        if use_graph and (self._last_graph_build_step < 0 or (getattr(_mind, 'step_num', 0) - self._last_graph_build_step) >= getattr(self.config,'rebuild_every',10)):
            A = build_adjacency_240_from_memory(_mem, _mind,
                                               alpha=getattr(self.config,'alpha_edge',1.0),
                                               decay_tau=getattr(self.config,'decay_tau',600.0),
                                               reward_gain=getattr(self.config,'reward_gain',0.5))
            if csr_matrix is None or diags is None:
                H = (-self.config.gamma * np.asarray(A, dtype=np.complex64)) + getattr(self.config,'lambda_V',0.2) * np.diag(V)
                self.H = H
            else:
                H = (-self.config.gamma * A.astype(np.complex64)) + getattr(self.config,'lambda_V',0.2) * diags(V)
                self.H = csr_matrix(H)
            self._last_graph_build_step = getattr(_mind, 'step_num', 0)
        else:
            if diags is None or csr_matrix is None:
                H = (self.config.gamma * np.asarray(self.physics.L_norm, dtype=np.complex64)) + np.diag(V)
                self.H = H
            else:
                H = (self.config.gamma * self.physics.L_norm.astype(np.complex64)) + diags(V)
                self.H = csr_matrix(H)
        self._last_H = self.H
        self._last_potential = np.asarray(V).copy()


    def _probs(self):
        p = np.abs(self.psi)**2
        return p / (np.sum(p, axis=1, keepdims=True) + 1e-12)

    def step_adaptive(self, tv_target=0.07, dt_min=0.02, dt_max=1.2):
        if self.H is None:
            return 0.0

        p0 = self._probs()
        H_eff = self.H.copy()
        if self.config.non_linearity_strength != 0:
            feedback = self.config.non_linearity_strength * p0[0]
            if diags is not None and hasattr(H_eff, 'shape') and csr_matrix is not None:
                try:
                    H_eff = H_eff + diags(feedback.astype(np.float32), 0)
                except Exception:
                    pass
            else:
                # dense numpy path
                H_eff = np.asarray(H_eff) + np.diag(feedback.astype(np.float32))

        if expm_multiply is not None and hasattr(H_eff, 'dot'):
            psi_new = expm_multiply(-1j * H_eff * self.config.dt, self.psi.T).T
        else:
            # crude Euler step fallback for environments without scipy
            H_dense = np.asarray(H_eff)
            psi_new = (self.psi + (-1j * self.config.dt) * (self.psi @ H_dense.T)).astype(np.complex64)
        nrm = np.linalg.norm(psi_new, axis=1, keepdims=True)
        self.psi = psi_new / np.maximum(nrm, 1e-12)
        p1 = self._probs()

        tv = 0.5 * float(np.abs(p0 - p1).sum(axis=1).mean())
        if tv < 0.5*tv_target: self.config.dt = min(dt_max, self.config.dt*1.25)
        elif tv > 1.5*tv_target: self.config.dt = max(dt_min, self.config.dt*0.66)

        if self.config.dephase > 0:
            mag = np.abs(self.psi)
            self.psi = (1.0 - self.config.dephase) * self.psi + self.config.dephase * mag
            nrm = np.linalg.norm(self.psi, axis=1, keepdims=True)
            self.psi /= np.maximum(nrm, 1e-12)

        try:
            self._last_norm = float(np.mean(np.sum(np.abs(self.psi)**2, axis=1)))
            Href = self._last_H
            if Href is not None and getattr(Href, 'ndim', 0) == 2:
                Energies = []
                for b in range(self.psi.shape[0]):
                    v = self.psi[b].reshape(-1,1)
                    E = (np.conjugate(v).T @ (Href @ v)).ravel()[0]
                    Energies.append(np.real(E))
                self._last_energy = float(np.mean(Energies))
        except Exception:
            self._last_norm = np.nan
            self._last_energy = np.nan
        return tv

    def measure_local(self, prev_idx, sigma=None):
        sigma = sigma or self.config.locality_sigma
        P = self._probs()
        masks = np.stack([self.physics.heat_mask_cached(i, sigma) for i in prev_idx]) if isinstance(prev_idx, (list, np.ndarray)) else np.tile(self.physics.heat_mask_cached(int(prev_idx), sigma), (self.config.batch, 1))
        P *= masks
        P /= np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
        return np.array([self.rng.choice(P.shape[1], p=p) for p in P], dtype=np.int32)

    def measure_hybrid(self, prev_idx=None, sigma=None, topk=None):
        """Hybrid measurement: combine engine amplitudes with a soft projection mask
        derived from the last potential (attractive wells), then apply local heat-mask
        around the previous index. Falls back to measure_local if data is missing.
        Returns a list of chosen indices (len=batch).
        """
        if prev_idx is None:
            prev_idx = 0

        if not hasattr(self, "psi"):
            return self.measure_local([prev_idx] * self.config.batch, sigma)
        B, N = self.psi.shape

        P = np.abs(self.psi)**2
        P = P / (np.sum(P, axis=1, keepdims=True) + 1e-12)

        Vlast = self._last_potential
        if Vlast is not None and np.size(Vlast) == N:

            soft = np.maximum(0.0, -np.real(np.asarray(Vlast).reshape(1, -1)))
            if topk is None:
                topk = int(getattr(self.config, "topk_amp", 5) or 5)

            idx = np.argpartition(soft[0], -topk)[-topk:]
            mask = np.zeros_like(P)
            mask[:, idx] = 1.0

            Amp = np.sqrt(P) * np.sqrt(soft + 1e-12)
            P = (Amp**2) * mask
            P = P / (np.sum(P, axis=1, keepdims=True) + 1e-12)

        if sigma is None:
            sigma = float(getattr(self.config, "locality_sigma", 1.5) or 1.5)

        hops = self.physics.hops
        w = np.exp(-(hops[prev_idx]**2) / (2.0 * sigma * sigma))

        P = P * w.reshape(1, -1)
        P = P / (np.sum(P, axis=1, keepdims=True) + 1e-12)

        choices = []
        for b in range(B):
            choices.append(int(self.rng.choice(N, p=P[b])))
        return choices

    def telemetry_state(self):
        """Return latest quantum telemetry values."""
        return {
            "dt": float(getattr(self.config, "dt", 0.01)),
            "gamma": float(getattr(self.config, "gamma", 1.0)),
            "dephase": float(getattr(self.config, "dephase", 0.0)),
            "norm": float(self._last_norm),
            "energy": float(self._last_energy),
            "topk_amp": int(getattr(self.config, "topk_amp", 5)),
            "locality_sigma": float(getattr(self.config, "locality_sigma", 1.5)),
        }

    def measure_ablation(self, prev_idx:int, sigma: Optional[float]=None, window:int=5, trials:int=512):
        """Compare local vs hybrid measurement near prev_idx.
        Returns dict with hit counts and rates inside ±window."""
        if sigma is None:
            sigma = float(getattr(self.config, "locality_sigma", 1.5) or 1.5)
        B, N = getattr(self, "psi", np.zeros((1,1))).shape
        total = trials * B
        if total == 0:
            return {}

        local_choices = []
        hybrid_choices = []
        for _ in range(trials):
            local_choices.extend(self.measure_local([prev_idx] * B, sigma=sigma))
            hybrid_choices.extend(self.measure_hybrid(prev_idx=prev_idx, sigma=sigma))
        local_counts = np.bincount(np.asarray(local_choices), minlength=N)
        hybrid_counts = np.bincount(np.asarray(hybrid_choices), minlength=N)

        lo = max(0, prev_idx-window); hi = min(N-1, prev_idx+window)
        local_win = int(local_counts[lo:hi+1].sum())
        hybrid_win = int(hybrid_counts[lo:hi+1].sum())
        return {
            "prev_idx": int(prev_idx),
            "window": int(window),
            "sigma": float(sigma),
            "trials": int(trials),
            "batch": int(B),
            "N": int(N),
            "local_win": local_win,
            "hybrid_win": hybrid_win,
            "local_rate": float(local_win/total),
            "hybrid_rate": float(hybrid_win/total),
        }

class ClassicalEngine:
    def __init__(self, physics, config, console: Any):
        self.console = console
        self.physics, self.config = physics, config
        self.rng = np.random.default_rng(config.seed)
        self.console.log("[INIT] Classical Engine online.")

    def next_index(self, prev_idx, sensor8):
        nbrs = np.where(self.physics.weights[prev_idx] > 0)[0]
        if nbrs.size > 0:
            if np.linalg.norm(sensor8) > 0:
                scores = self.physics.roots[nbrs] @ sensor8
                p = np.exp(2.5 * scores); p /= np.sum(p)
                return self.rng.choice(nbrs, p=p)
            return self.rng.choice(nbrs)
        return self.rng.integers(0, 240)

class E8BoundaryFabric:
    def __init__(self, physics: "E8Physics", seed: int = 1337):
        self.physics = physics
        self.N = physics.roots.shape[0]
        self.A = (physics.weights > 0).astype(np.float32)
        self.pos2d: Optional[np.ndarray] = None
        self.z1d: Optional[np.ndarray] = None
        self.rng = np.random.default_rng(seed)

    def layout_2d(self):
        W = self.A; deg = W.sum(axis=1)
        Dm12 = 1.0 / np.sqrt(np.maximum(deg, 1e-9))
        L = np.eye(self.N, dtype=np.float32) - (Dm12[:,None] * W * Dm12[None,:])
        try:
            if csr_matrix is None or eigsh is None: raise RuntimeError("scipy is required for layout_2d")
            _, vecs = eigsh(csr_matrix(L), k=4, which='SM')
            P = vecs[:, 1:4]
        except Exception:
            _, vecs = np.linalg.eigh(L)
            P = vecs[:, 1:4]
        P = (P - P.mean(axis=0)) / (P.std(axis=0) + 1e-6)
        self.pos2d = P[:, :2].astype(np.float32)
        self.z1d = P[:, 2].astype(np.float32)

    def neighbors(self, i: int) -> np.ndarray:
        return np.where(self.A[i] > 0)[0].astype(np.int32)

    def to_json(self):
        if self.pos2d is None: self.layout_2d()
        edges = np.column_stack(np.where(np.triu(self.A, 1) > 0)).tolist()
        if self.pos2d is None or self.z1d is None:
            return {"nodes": [], "edges": []}
        return {
            "nodes": [{"id": int(i), "x": float(self.pos2d[i,0]), "y": float(self.pos2d[i,1]), "z": float(self.z1d[i])} for i in range(self.N)],
            "edges": [{"s": int(i), "t": int(j)} for i, j in edges]
        }

class SliceStack:
    def __init__(self, n_slices: int = 24, zmin: float = -1.5, zmax: float = 1.5):
        self.n, self.zmin, self.zmax = n_slices, zmin, zmax
        self.bin = np.linspace(self.zmin, self.zmax, self.n + 1)

    def index(self, z: float) -> int:
        return int(np.clip(np.searchsorted(self.bin, z, side="right") - 1, 0, self.n - 1))

class HoloEncoder:
    def __init__(self, fabric: E8BoundaryFabric, feat_dim: int = 8, shadow_k: int = 12, seed: int = 1337):
        self.fabric, self.feat_dim, self.shadow_k = fabric, feat_dim, shadow_k
        self.rng = np.random.default_rng(seed)
        self._U_cache: Dict[Tuple, np.ndarray] = {}
        self.store: Dict[Tuple[int, int], float] = {}

    def shadow_set(self, bulk_idx: int, pos_hint_xy: Optional[np.ndarray] = None) -> np.ndarray:
        if pos_hint_xy is not None and self.fabric.pos2d is not None:
            d = np.sum((self.fabric.pos2d - pos_hint_xy[None,:])**2, axis=1)
            return np.argsort(d)[:self.shadow_k].astype(np.int32)
        nb = self.fabric.neighbors(int(bulk_idx))
        if nb.size >= self.shadow_k: return nb[:self.shadow_k]
        pool = np.setdiff1d(np.arange(self.fabric.N), np.append(nb, bulk_idx))
        if not pool.size > 0: return nb
        extra_count = self.shadow_k - nb.size
        extra = self.rng.choice(pool, size=min(extra_count, pool.size), replace=False)
        return np.concatenate([nb, extra]).astype(np.int32)

    def _U(self, shadow_ids: np.ndarray):
        key = tuple(sorted(shadow_ids.tolist()))
        if key not in self._U_cache:
            K, D = len(shadow_ids), self.feat_dim
            if K < D: self._U_cache[key] = np.zeros((K,D), dtype=np.float32)
            else:
                R = self.rng.standard_normal((K, D)).astype(np.float32)
                Q, _ = np.linalg.qr(R, mode='reduced')
                self._U_cache[key] = Q[:, :D]
        return self._U_cache[key]

    def encode_bulk(self, feat: np.ndarray, shadow_ids: np.ndarray, slice_id: int):
        U = self._U(shadow_ids)
        y = U @ feat
        payload = {"f": y.astype(np.float32).tolist()}
        for nid, val in zip(shadow_ids, payload["f"]):
            self.store[(int(nid), int(slice_id))] = float(val)
        return payload

    def decode_boundary(self, shadow_ids: np.ndarray, slice_id: int, payload: dict) -> np.ndarray:
        U = self._U(shadow_ids)
        y = np.array(payload.get("f", []), dtype=np.float32)
        if y.size == 0:
            return np.zeros(self.feat_dim, dtype=np.float32)
        y = y[:U.shape[0]]
        return (U.T @ y).astype(np.float32)

    def encode(self, embedding: np.ndarray) -> np.ndarray:
        """Compress a high-D embedding to feat_dim (default 8). Safe fallbacks if compressor isn't ready."""
        try:
            v = np.asarray(embedding, dtype=np.float32).reshape(-1)
            comp = getattr(self, "_compressor", None)
            if comp is None:
                in_dim = int(v.size)
                try:
                    comp = TinyCompressor(in_dim=in_dim, code_dim=int(self.feat_dim))
                except Exception:
                    comp = None
                setattr(self, "_compressor", comp)
            if comp is not None:
                try:
                    z = comp.encode(v)
                except Exception:
                    z = None
            else:
                z = None
            if z is None:
                z = v[: self.feat_dim]
            z = np.asarray(z, dtype=np.float32).reshape(-1)
            if z.size < self.feat_dim:
                z = np.pad(z, (0, self.feat_dim - z.size))
            elif z.size > self.feat_dim:
                z = z[: self.feat_dim]
            return z.astype(np.float32)
        except Exception:
            v = np.asarray(embedding, dtype=np.float32).reshape(-1)
            if v.size < self.feat_dim:
                v = np.pad(v, (0, self.feat_dim - v.size))
            return v[: self.feat_dim].astype(np.float32)
class EntropyMap:
    def __init__(self, fabric: "E8BoundaryFabric", k_bits_per_edge: float = 4.0):
        self.fabric, self.k = fabric, float(k_bits_per_edge)
        self.A = (fabric.A > 0).astype(np.float32)
        self.N = int(self.A.shape[0])

    def perimeter(self, region_nodes: np.ndarray) -> float:
        mask = np.zeros(self.N, dtype=np.float32)
        mask[region_nodes] = 1.0
        cut = np.sum(self.A[region_nodes], axis=0) * (1.0 - mask)
        return float(cut.sum())

    def budget_bits(self, region_nodes: np.ndarray) -> float:
        return self.k * self.perimeter(region_nodes)

    def usage_bits(self, store: dict, region_nodes: np.ndarray, slice_id: Optional[int] = None) -> float:
        rset = set(int(i) for i in region_nodes.tolist())
        bits = 0.0
        for (nid, sid), val in store.items():
            if nid in rset and (slice_id is None or sid == int(slice_id)):
                bits += 32.0
        return float(bits)

    def deficit_ratio(self, store: dict, region_nodes: np.ndarray, slice_id: Optional[int] = None) -> float:
        B = self.budget_bits(region_nodes) + 1e-6
        U = self.usage_bits(store, region_nodes, slice_id)
        return float((U - B) / B)

class SensorProjector:
    def __init__(self, in_dim, out_dim=8, seed=None):
        self.in_dim, self.out_dim = in_dim, out_dim
        self.rng = np.random.default_rng(seed)
        self.W = self.rng.standard_normal((in_dim, out_dim)).astype(np.float32) * 0.1
        self.mu = np.zeros(in_dim, dtype=np.float32)

    def pca_bootstrap(self, embeddings: np.ndarray, top_k=240):
        if embeddings.shape[0] < self.out_dim or PCA is None: return
        try:
            pca = PCA(n_components=self.out_dim)
            pca.fit(embeddings[:top_k])
            self.W, self.mu = pca.components_.T, pca.mean_
            console.log(f"[PROJ] Bootstrapped with PCA on {top_k} embeddings.")
        except Exception as e:
            console.log(f"[PROJ] PCA bootstrap failed: {e}. Falling back to random init.")

    def project(self, embedding):
        if embedding.shape[0] != self.in_dim:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.in_dim}, got {embedding.shape[0]}.")
        return normalize_vector((embedding - self.mu) @ self.W)

    def train(self, embeddings, labels, roots_unit, epochs=3, lr=5e-3, batch_size=64, **kwargs):
        if embeddings.shape[0] < batch_size: return
        console.log(f"[PROJ] Starting training burst on {embeddings.shape[0]} samples.")
        for _ in range(epochs):
            indices = self.rng.integers(0, embeddings.shape[0], size=batch_size)
            for i in indices:
                e, y = embeddings[i], labels[i]
                s = normalize_vector((e - self.mu) @ self.W)
                delta_W = lr * np.outer(e - self.mu, roots_unit[y] - s)
                self.W += delta_W

class TinyCompressor:
    def __init__(self, in_dim=1536, code_dim=8):
        self.in_dim, self.code_dim = in_dim, code_dim
        self.ready, self._pca = False, None
        self._use_torch = TORCH_AVAILABLE
        # If torch is available, build a tiny linear autoencoder
        if self._use_torch and TORCH_AVAILABLE:
            import torch as _torch  # local alias
            import torch.nn as _nn
            class AE(_nn.Module):
                def __init__(self, D: int, C: int):
                    super().__init__()
                    self.enc = _nn.Linear(D, C, bias=False)
                    self.dec = _nn.Linear(C, D, bias=False)
                def forward(self, x):
                    z = self.enc(x)
                    xh = self.dec(z)
                    return z, xh
            self.net = AE(self.in_dim, self.code_dim)
            for p in self.net.parameters():
                try:
                    _nn.init.xavier_uniform_(p.data)
                except Exception:
                    pass
            self.opt = _torch.optim.Adam(self.net.parameters(), lr=3e-3)
            self.ready = True

    def fit(self, X: np.ndarray, epochs=5, bs=64):
        if X.shape[0] < max(bs, self.code_dim + 1): return
        if self._use_torch and TORCH_AVAILABLE:
            import torch as _torch
            import torch.nn as _nn
            self.net.train()
            loss_fn = _nn.MSELoss()
            # Shuffle each epoch
            for _ in range(epochs):
                idx = np.random.permutation(X.shape[0])
                for i in range(0, X.shape[0], bs):
                    b = _torch.from_numpy(X[idx[i:i+bs]])
                    self.opt.zero_grad()
                    _, xh = self.net(b)
                    loss = loss_fn(xh, b)
                    loss.backward()
                    self.opt.step()
            self.ready = True
        elif PCA is not None:
            # Fallback to PCA when torch is not available
            self._pca = PCA(n_components=self.code_dim).fit(X)
            self.ready = True

    def encode(self, x: np.ndarray) -> np.ndarray:
        x = x.reshape(1, -1)
        if self._use_torch and self.ready and TORCH_AVAILABLE:
            import torch as _torch
            self.net.eval()
            with _torch.no_grad(): z, _ = self.net(_torch.from_numpy(x))
            return z.cpu().numpy().ravel()
        if self._pca and self.ready: return self._pca.transform(x).ravel()
        return x.ravel()[:self.code_dim]

class MultiAnchorField:
    def __init__(self, physics, kernel='cosine', rbf_sigma=0.8):
        self.physics, self.kernel, self.rbf_sigma = physics, kernel, rbf_sigma
        self.anchors: List[Tuple[np.ndarray, float]] = []

    def set(self, anchor_list: List[Tuple[np.ndarray, float]]):
        self.anchors = []
        if not anchor_list: return
        total_weight = sum(w for _, w in anchor_list)
        if total_weight > 1e-9:
            self.anchors = [(vec, w / total_weight) for vec, w in anchor_list]

    def potential(self):
        V = np.zeros(240, dtype=np.float32)
        if not self.anchors: return V
        for vec, weight in self.anchors:
            if self.kernel == 'cosine':
                scores = self.physics.roots_unit @ vec
            else:
                dists = np.linalg.norm(self.physics.roots - vec, axis=1)
                scores = np.exp(-dists**2 / (2 * self.rbf_sigma**2))
            V -= weight * scores
        return V

class GoalField:
    def __init__(self, embedding_fn, console: Any):
        self.console = console
        self.embedding_fn = embedding_fn
        self.goals: Dict[str, Dict[str, Any]] = {}
        self.is_initialized = False
        self.activation_decay = 0.98

    async def initialize_goals(self):
        if self.is_initialized: return
        goal_definitions = {
            "synthesis": "Achieve synthesis and coherence; find the unifying pattern.",
            "novelty": "Look at novelty and the unknown; break existing patterns.",
            "stability": "Reinforce core identity and create a stable self-model.",
            "curiosity": "Understand the 'why'; ask questions and follow causal chains."
        }
        for name, desc in goal_definitions.items():
            vec = await self.embedding_fn(desc)
            self.goals[name] = {
                "description": desc, "embedding": vec, "activation": 0.25
            }
        self.is_initialized = True
        self.console.log("🌻 Goal-Field Initialized with attractors.")

    def decay(self):
        for name in self.goals:
            self.goals[name]["activation"] *= self.activation_decay

    def update_from_embedding(self, vector: np.ndarray, weight: float = 0.1):
        if not self.is_initialized or vector is None: return
        total_similarity, sims = 0.0, {}
        def _cos(a, b):
            a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-9 or nb < 1e-9: return 0.0
            return float(np.dot(a, b) / (na * nb))
        for name, goal_data in self.goals.items():
            sim = _cos(vector, goal_data.get("embedding", np.zeros_like(vector)))
            sims[name], total_similarity = sim, total_similarity + sim
        if total_similarity > 1e-9:
            for name, sim in sims.items():
                self.goals[name]["activation"] += weight * (sim / total_similarity)
        self._normalize_activations()

    def update_from_mood(self, mood_vector: dict):
        if not self.is_initialized: return
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
        total_activation = sum(g["activation"] for g in self.goals.values())
        if total_activation > 1e-9:
            for name in self.goals:
                self.goals[name]["activation"] /= total_activation

    def get_top_goals(self, k: int = 2) -> List[tuple[str, str]]:
        if not self.is_initialized: return [("nascent", "The mind is still forming its goals.")]
        if not self.goals: return [("empty", "No goals defined.")]
        sorted_goals = sorted(self.goals.items(), key=lambda item: -item[1].get("activation", 0.0))
        return [(name, data.get("description", "No description")) for name, data in sorted_goals[:k]]

class StatePotentialEvaluator:
    def __init__(self, dimensional_shells: Dict[int, 'DimensionalShell'], goal_field: 'GoalField'):
        self.dimensional_shells = dimensional_shells
        self.goal_field = goal_field
        self.last_potential = 0.0

    def _calculate_goal_resonance(self) -> float:
        if not self.goal_field.is_initialized:
            return 0.0
        total_resonance = 0.0
        goal_vec = np.zeros(EMBED_DIM, dtype=np.float32)
        for name, data in self.goal_field.goals.items():
            goal_vec += data.get("activation", 0.0) * data.get("embedding", np.zeros(EMBED_DIM, dtype=np.float32))
        if np.linalg.norm(goal_vec) == 0:
            return 0.0

        resonance_count = 0
        for dim, shell in self.dimensional_shells.items():
            matrix, _ = shell.get_all_vectors_as_matrix()
            if matrix is None:
                continue

            projected_goal_vec = np.zeros(dim)
            size_to_copy = min(EMBED_DIM, dim)
            projected_goal_vec[:size_to_copy] = goal_vec[:size_to_copy]

            similarities = cosine_similarity(matrix, projected_goal_vec.reshape(1, -1))
            shell_resonance = np.mean(similarities)
            total_resonance += shell_resonance
            resonance_count += 1
        return float(total_resonance / resonance_count) if resonance_count > 0 else 0.0

    def calculate_potential_and_get_reward(self) -> float:
        goal_resonance_potential = self._calculate_goal_resonance()
        current_potential = goal_resonance_potential
        reward = current_potential - self.last_potential
        self.last_potential = current_potential
        return reward

class DriveSystem:
    def __init__(self):
        self.drives = {"curiosity": 0.5, "coherence": 0.5, "novelty": 0.5, "intelligibility": 0.5, "fluidity": 0.5}

    def decay(self):
        for k in self.drives: self.drives[k] = max(0.0, self.drives[k] - 0.01)

    def reward(self, key, amount=0.1):
        if key in self.drives: self.drives[key] = min(1.0, self.drives[key] + amount)

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
        if self.console:
            self.console.log("🌦️  Affective WeatherEngine Initialized.")

    def _nudge(self, key: str, amount: float):
        if key in self.mood_vector: self.mood_vector[key] = np.clip(self.mood_vector[key] + amount, 0.0, 1.0)

    def process_event(self, event_type: str, **kwargs):
        self.event_queue.append((event_type, kwargs))

    def update(self):
        while self.event_queue:
            event_type, kwargs = self.event_queue.popleft()
            if event_type == "movement":
                mag = kwargs.get("magnitude", 0.0)
                self._nudge("intensity", 0.05 * min(mag, 5.0))
                if any(t in kwargs.get("themes", []) for t in ["disorder", "burst"]): self._nudge("entropy", 0.15); self._nudge("coherence", -0.10)
                if any(t in kwargs.get("themes", []) for t in ["integration", "stasis"]): self._nudge("coherence", 0.10); self._nudge("entropy", -0.05)
                if "growth" in kwargs.get("themes", []): self._nudge("fluidity", 0.08)
            elif event_type == "new_concept":
                rating = kwargs.get("rating", 0.5)
                if rating > 0.75: self._nudge("coherence", 0.05*rating); self._nudge("positivity", 0.10*rating); self._nudge("intelligibility", 0.06*rating)
                else: self._nudge("entropy", 0.05 * (1.0 - rating))
            elif event_type == "dream":
                self._nudge("entropy", 0.30); self._nudge("fluidity", 0.25); self._nudge("coherence", -0.15); self._nudge("intensity", 0.10)
            elif event_type == "reflection":
                self._nudge("coherence", 0.20); self._nudge("entropy", -0.10); self._nudge("positivity", 0.05); self._nudge("intelligibility", 0.08)
            elif event_type == "weather_tick":
                step = kwargs.get("step", 0)
                bh = float(kwargs.get("bh", 0.0))
                osc  = 0.03 * math.sin(2.0 * math.pi * ((step % 240) / 240.0))
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
        for k, v in self.mood_vector.items():
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
                        if value < thresholds[last_bin] + padding: return last_bin
                    else:
                        if value > thresholds[current_bin] - padding: return last_bin
            return current_bin
        b_e, b_i, b_c = bin_with_hysteresis(e, (0.25, 0.5, 0.75), getattr(self, "_b_e", None)), bin_with_hysteresis(i, (0.25, 0.5, 0.75), getattr(self, "_b_i", None)), bin_with_hysteresis(c, (0.25, 0.5, 0.75), getattr(self, "_b_c", None))
        self._b_e, self._b_i, self._b_c = b_e, b_i, b_c
        code = (b_e << 4) | (b_i << 2) | b_c
        if code == self._wx_last_code: self._wx_repeat += 1
        else: self._wx_repeat, self._wx_last_code = 0, code
        variants = {
            "storm": ["Volatile, sharp swings.", "Choppy, energy spikes.", "Jittery air, quick flips."],
            "calm":  ["Calm, steady drift.", "Gentle, small ripples.", "Soft, even flow."],
            "flow":  ["In-flow, coherent.", "Rolling, smooth arcs.", "Aligned, easy motion."],
            "turbulent": ["Turbulent, scattered.", "Noisy, low signal.", "Foggy, fragmented."],
        }
        if b_i >= 2 and b_e >= 2 and b_c <= 1: bucket = "storm"
        elif b_c >= 2 and b_e <= 1: bucket = "flow"
        elif b_e <= 1 and b_i <= 1: bucket = "calm"
        else: bucket = "turbulent"
        idx = (self._wx_repeat // 8) % len(variants[bucket])
        return variants[bucket][idx]

    def get_entropy_level(self) -> float:
        return mood_get(self.mood_vector, "entropy")

    def get_llm_persona_prefix(self) -> str:
        i, e, c = mood_get(self.mood_vector, 'intensity', 0.5), mood_get(self.mood_vector, 'entropy', 0.5), mood_get(self.mood_vector, 'coherence', 0.5)
        if e > 0.7 and i > 0.6: return "You are feeling chaotic, fragmented, and electric. Your response should be surreal and full of unexpected connections."
        elif c > 0.75: return "You are feeling exceptionally clear, logical, and focused. Your response should be precise and structured."
        elif i < 0.3: return "You are feeling calm, quiet, and introspective. Your response should be gentle and thoughtful."
        else: return "You are in a balanced state of mind. Your response should be clear and considered."

    def get_mood_modulation_vector(self, dim: int) -> np.ndarray:
        seed = zlib.adler32(json.dumps(self.mood_vector, sort_keys=True).encode())
        rng = np.random.default_rng(seed)
        coherence, entropy = mood_get(self.mood_vector, 'coherence', 0.5), mood_get(self.mood_vector, 'entropy', 0.5)
        modulation = rng.standard_normal(dim).astype(np.float32)
        modulation *= (1.0 + 0.5 * (coherence - 0.5))
        modulation += rng.standard_normal(dim).astype(np.float32) * 0.2 * entropy
        return normalize_vector(modulation)

class SubconsciousLayer:
    def __init__(self, embedding_fn, llm_caller, console: Any, decay_rate=0.95, accumulation_rate=0.004):
        self.embedding_fn = embedding_fn
        self.llm_caller = llm_caller
        self.console = console
        self.decay_rate = decay_rate
        self.accumulation_rate = accumulation_rate
        self.bias_vector: Optional[np.ndarray] = None
        self.narrative = "The mind is nascent, a canvas awaiting its first impression."
        self.bias_history = deque(maxlen=200)
        self.influences: List[Dict[str, Any]] = []

    def add_waveform_influence(self, vector: np.ndarray, rating: float, step_num: int):
        if self.bias_vector is None: self.bias_vector = np.zeros_like(vector)
        influence = {
            "vector": vector, "initial_strength": 0.4 * (rating - 0.8),
            "start_step": step_num, "frequency": 0.25, "decay": 0.1
        }
        self.influences.append(influence)
        if len(self.influences) > 20: self.influences.pop(0)

    def _apply_influences(self, current_step: int):
        if not self.influences or self.bias_vector is None: return
        total_influence_vec = np.zeros_like(self.bias_vector, dtype=np.float32)
        active_influences = []
        for influence in self.influences:
            time_delta = current_step - influence.get("start_step", current_step)
            if time_delta < 0: continue
            decay_factor = math.exp(-influence.get("decay", 0.1) * time_delta)
            oscillation_factor = math.cos(influence.get("frequency", 0.25) * time_delta)
            current_strength = influence.get("initial_strength", 0.0) * decay_factor * oscillation_factor
            if abs(current_strength) > 0.001:
                total_influence_vec += current_strength * influence.get("vector", np.zeros_like(self.bias_vector))
                active_influences.append(influence)
        if np.linalg.norm(total_influence_vec) > 0:
             self.bias_vector += total_influence_vec
             self.bias_vector = normalize_vector(self.bias_vector)
        self.influences = active_influences

    async def track_concept(self, label, weight=1.0):
        vec = await self.embedding_fn(label)
        if np.linalg.norm(vec) > 0:
            # Ensure we have a bias vector to work with and shapes are compatible
            if self.bias_vector is None:
                self.bias_vector = np.zeros_like(vec)
            bv = self.bias_vector
            if getattr(bv, 'shape', None) != vec.shape:
                return
            bv = bv + self.accumulation_rate * normalize_vector(vec) * weight
            self.bias_vector = normalize_vector(bv)
            if len(self.bias_history) == 0 or np.linalg.norm(self.bias_history[-1] - self.bias_vector) > 0.01:
                try:
                    self.bias_history.append(np.array(self.bias_vector, copy=True))
                except Exception:
                    self.bias_history.append(self.bias_vector)

    def get_bias(self):
        return self.bias_vector if self.bias_vector is not None else np.zeros(EMBED_DIM)

    def decay(self, current_step: int):
        if self.bias_vector is not None:
            self.bias_vector *= self.decay_rate
        self._apply_influences(current_step)

    async def generate_narrative_summary(self, recent_events: List[Dict[str, Any]]):
        if not recent_events: return
        event_fragments = []
        for event in recent_events:
            if event['type'] == 'dream': event_fragments.append(f"A dream occurred titled '{event['label']}'.")
            elif event['type'] == 'teacher_explorer':
                q, a = event['data'].get('q', 'a question'), event['data'].get('a', 'an answer')
                event_fragments.append(f"A dialogue unfolded: the question '{q}' was met with '{a}'.")
            elif event['type'] == 'black_hole': event_fragments.append(f"A memory singularity was experienced, consolidating {event['size']} concepts.")
            elif event['type'] == 'insight_synthesis': event_fragments.append(f"A moment of insight synthesized a new idea: '{event.get('label', 'an unnamed concept')}'")
        if not event_fragments: return
        formatted_events = "- " + "\n- ".join(event_fragments)
        prompt = (
            "You are the subconscious. Weave the following recent events into a single, short, metaphorical narrative paragraph. "
            "Do not list the events; create a story from them.\n\n"
            f"Events:\n{formatted_events}\n\nNarrative:"
        )
        try:
            summary = await self.llm_caller.enqueue_and_wait(prompt, max_tokens=150, temperature=0.7)
            if summary and not summary.startswith("[LLM"):
                self.narrative = sanitize_block(summary, max_sentences=3, max_chars=300)
                self.console.print(Panel(self.narrative, title="[bold #5B4F97]Subconscious Narrative[/]", border_style="#5B4F97"))
        except Exception as e:
            self.console.log(f"[Subconscious] Narrative generation failed: {e}")

# --- NEW HELPER CLASSES FOR UPGRADES ---

# [UPGRADE 1] Modern Hopfield Nets for attractor clean-up
class HopfieldModern:
    """Applies an energy minimization step to clean up synthesized vectors."""
    def __init__(self, memory_manager, top_k=50, tau=0.1):
        self.memory = memory_manager
        self.top_k = top_k
        self.tau = tau
        self.prototypes = np.array([], dtype=np.float32)

    def update_prototypes(self):
        """Selects the top-K highest-rated nodes as stable memory patterns."""
        nodes = self.memory.graph_db.graph.nodes(data=True)
        high_reward_nodes = sorted(
            [(nid, data) for nid, data in nodes if 'rating' in data and nid in self.memory.main_vectors],
            key=lambda item: item[1]['rating'],
            reverse=True
        )[:self.top_k]

        if high_reward_nodes:
            vecs = [self.memory.main_vectors[nid] for nid, _ in high_reward_nodes]
            # Prototypes P should be shape [D, K] for P @ v
            self.prototypes = np.array(vecs, dtype=np.float32).T
            console.log(f"[Hopfield] Updated prototypes with {self.prototypes.shape[1]} vectors.")

    def clean_up(self, vector, steps=3):
        """Pulls a vector towards the nearest energy minimum (prototype)."""
        if self.prototypes.shape[1] == 0:
            return vector
        v = vector.copy()
        for _ in range(steps):
            # v_t+1 = softmax(P.T @ v_t / tau) @ P
            energy = self.prototypes.T @ v / self.tau
            exp_energy = np.exp(energy - np.max(energy)) # Softmax
            softmax_energy = exp_energy / np.sum(exp_energy)
            v_new = self.prototypes @ softmax_energy
            v_new /= (np.linalg.norm(v_new) + 1e-9) # Normalize
            if np.linalg.norm(v_new - v) < 1e-4:
                break
            v = v_new
        return v

# [UPGRADE 2] Kanerva Sparse Distributed Memory
class KanervaSDM:
    """A sparse, error-correcting secondary memory system."""
    def __init__(self, mind_instance, num_addresses=4096, dim=8, radius=0.85):
        self.mind = mind_instance
        self.num_addresses = num_addresses
        self.dim = dim
        self.radius = radius
        rng = np.random.default_rng(GLOBAL_SEED)
        self.addresses = rng.standard_normal((num_addresses, dim)).astype(np.float32)
        self.addresses /= np.linalg.norm(self.addresses, axis=1, keepdims=True)
        self.kdtree = KDTree(self.addresses)
        self.memory = np.zeros((num_addresses, dim), dtype=np.float32)
        self.hits = np.zeros(num_addresses, dtype=np.int32)

    def _get_vec8d(self, vec_embed):
        """Projects a high-dimensional vector to the 8D address space."""
        if (not TORCH_AVAILABLE) or getattr(self.mind, 'autoencoder', None) is None or (not getattr(self.mind.autoencoder, 'is_trained', False)):
            return None
        try:
            import torch as _torch
            with _torch.no_grad():
                source_tensor = _torch.from_numpy(np.asarray(vec_embed, dtype=np.float32)).float().unsqueeze(0)
                proj_tensor = self.mind.autoencoder.project_to_dim(source_tensor, self.dim)
                if proj_tensor is None: return None
                try:
                    return proj_tensor.squeeze(0).detach().cpu().numpy()
                except Exception:
                    return np.asarray(proj_tensor).squeeze()
        except Exception:
            return None

    def write(self, vec_embed):
        """Sparse multi-write: reinforces memory at multiple nearby addresses."""
        vec8d = self._get_vec8d(vec_embed)
        if vec8d is None: return
        impl = getattr(self.kdtree, '_impl', None)
        if impl is not None and hasattr(impl, 'query_radius'):
            indices = impl.query_radius(vec8d.reshape(1, -1), r=self.radius)[0]
            if len(indices) > 0:
                update_vec = np.sign(vec8d)
                for idx in indices:
                    self.memory[idx] = 0.95 * self.memory[idx] + 0.05 * update_vec
                    self.hits[idx] += 1

    def read_strength(self, vec_embed):
        """Radius-gated read: aggregates hits to determine memory strength."""
        vec8d = self._get_vec8d(vec_embed)
        if vec8d is None: return 0.5
        impl = getattr(self.kdtree, '_impl', None)
        if impl is not None and hasattr(impl, 'query_radius'):
            indices = impl.query_radius(vec8d.reshape(1, -1), r=self.radius)[0]
            if len(indices) == 0: return 0.0
            total_hits = np.sum(self.hits[indices])
            # Sigmoid scaling: strength approaches 1 as hits increase
            strength = 1 / (1 + np.exp(-(total_hits - 10) / 5.0))
            return strength
        return 0.5 # Fallback if radius query is unavailable

# [UPGRADE 3] Vector-Symbolic Architecture / Holographic Reduced Representations
class VSA:
    """Encodes and decodes compositional structure in vectors using FFT."""
    def __init__(self, dim, seed=GLOBAL_SEED):
        self.dim = dim
        rng = np.random.default_rng(seed)
        self.roles = {
            "PARENT_A": self._make_rand_vec(rng), "PARENT_B": self._make_rand_vec(rng),
            "CAUSE": self._make_rand_vec(rng), "EFFECT": self._make_rand_vec(rng),
        }
    def _make_rand_vec(self, rng):
        vec = rng.standard_normal(self.dim).astype(np.float32)
        return vec / np.linalg.norm(vec)
    def bind(self, role_key, filler_vec):
        """Binds a role to a filler vector using circular convolution."""
        role_vec = self.roles[role_key]
        return np.fft.ifft(np.fft.fft(role_vec) * np.fft.fft(filler_vec)).real.astype(np.float32)
    def unbind(self, role_key, bound_vec):
        """Approximately recovers the filler vector."""
        role_vec = self.roles[role_key]
        return np.fft.ifft(np.fft.fft(bound_vec) * np.conj(np.fft.fft(role_vec))).real.astype(np.float32)
    def encode_parentage(self, vec_a, vec_b):
        """Combines two parent vectors into a single structural representation."""
        bound_a = self.bind("PARENT_A", vec_a)
        bound_b = self.bind("PARENT_B", vec_b)
        combined = bound_a + bound_b
        return combined / (np.linalg.norm(combined) + 1e-9)

# [UPGRADE 7] Contrastive Micro-Reranker
class MicroReranker:
    """A small, local reranker to separate true insights from near-misses."""
    def __init__(self, memory_manager):
        self.memory = memory_manager
        # Simple linear model weights: positive for good features, negative for bad
        self.weights = np.array([
            0.4,  # coherence (good)
            0.3,  # novelty (good)
            -0.1, # ppl (lower is better, feature is placeholder)
            -0.2, # dup_rate (lower is better, feature is placeholder)
            0.15, # parent_coh_mean (good)
        ], dtype=np.float32)

    def _get_features(self, novelty, coherence, parent_ids):
        """Extracts features for a candidate concept."""
        ppl = 100.0 # Placeholder: perplexity would need a language model
        dup_rate = 0.1 # Placeholder: duplication rate would need text analysis
        parent_coh_mean = 0.5
        if parent_ids:
            ratings = [self.memory.graph_db.get_node(pid).get('rating', 0.5) for pid in parent_ids if self.memory.graph_db.get_node(pid)]
            if ratings: parent_coh_mean = np.mean(ratings) if ratings else 0.5
        return np.array([coherence, novelty, ppl, dup_rate, parent_coh_mean], dtype=np.float32)

    def score(self, features):
        return np.dot(self.weights, features)

    def validate(self, candidate_vec, parent_ids, novelty, coherence, margin=0.1):
        """Validates a candidate against known low-coherence "hard negatives"."""
        candidate_features = self._get_features(novelty, coherence, parent_ids)
        candidate_score = self.score(candidate_features)

        # Find nearby nodes with low coherence scores (ratings)
        similar_nodes = self.memory.find_similar_in_main_storage(candidate_vec, k=10)
        hard_negatives = [(nid, d) for nid, _ in similar_nodes if (d := self.memory.graph_db.get_node(nid)) and d.get('rating', 1.0) < 0.45][:3]

        if not hard_negatives: return True # No negatives to compare against

        for neg_id, neg_data in hard_negatives:
            # Create a proxy feature vector for the negative node
            neg_features = self._get_features(0.5, neg_data.get('rating', 0.0), [])
            neg_score = self.score(neg_features)
            if candidate_score <= neg_score + margin:
                self.memory.mind.console.log(f"[Reranker] Rejected. Too similar to low-coherence node '{neg_data.get('label', '')}' (Score {candidate_score:.2f} <= {neg_score:.2f} + {margin})")
                return False
        return True

# =======================================================================================
# ======================== FULLY CORRECTED MEMORY MANAGER CLASS =========================
# =======================================================================================
# =======================================================================================
# ======================== FULLY CORRECTED MEMORY MANAGER CLASS =========================
# =======================================================================================
class MemoryManager:
    """
    An advanced, high-performance memory management system for the E8Mind.
    This class orchestrates the mind's Long-Term Memory (LTM), managing a graph
    database for relationships (GraphDB) and a high-dimensional vector space for
    semantic content. It features batched index updates for performance, dynamic
    reranking for intelligent retrieval, and a suite of advanced cognitive modules
    for memory processing, including Hopfield networks, VSA, and a MicroReranker.
    """
    def __init__(self, mind_instance: 'E8Mind', **kwargs):
        # --- CORRECTED: Initialize all attributes FIRST ---
        self.mind = mind_instance
        self.embedding_fn = self.mind.get_embedding
        self.mood = self.mind.mood
        self.subconscious = self.mind.subconscious
        self.run_id = self.mind.run_id
        self.probe = self.mind.probe
        self.llm_caller = self.mind.llm_pool
        self.console = self.mind.console
        self.lock = InstrumentedLock("memory", probe=self.probe)

        # Initialize data structures and state
        self.graph_db = GraphDB()
        self.main_vectors: Dict[str, np.ndarray] = {}
        self.main_kdtree: Optional[KDTree] = None
        self._main_storage_ids: List[str] = []
        self._main_storage_matrix = np.empty((0, EMBED_DIM), dtype=np.float32)
        self.pending_additions: List[Tuple[str, np.ndarray]] = []
        self.pending_embeddings: List[np.ndarray] = []
        self.label_to_node_id: Dict[str, str] = {}
        self.consolidation_buffer: List[Dict] = []
        self.consolidation_task: Optional[asyncio.Task] = None
        self.field: Dict[str, float] = defaultdict(float)
        self.background_temp = 0.0
        self.active_locks: Dict[Tuple[str, str], int] = {}

        # Performance and configuration parameters
        self.KDTREE_REBUILD_THRESHOLD = int(os.getenv("E8_KDTREE_BATCH_SIZE", "32"))
        self.memory_consolidation_min = int(kwargs.get("memory_consolidation_min", 50))

        # --- Advanced Cognitive Modules ---
        self.hopfield = HopfieldModern(self)
        self.sdm = KanervaSDM(self.mind)
        self.vsa = VSA(EMBED_DIM)
        self.reranker = MicroReranker(self)

        self.console.log("🧠 [MemoryManager] New advanced instance initialized.")
        

        self._kdtree_latency_ms_window = []
        self._kdtree_latency_threshold_ms = float(os.getenv('E8_KDTREE_LATENCY_MS', '25'))
        self._novelty_stats_path = get_path('novelty_stats.json', self.run_id)
        self._avg_nn_ema = None
        try:
            if os.path.exists(self._novelty_stats_path):
                d = json.load(open(self._novelty_stats_path, 'r'))
                self._avg_nn_ema = d.get('avg_nn_ema')
        except Exception:
            self._avg_nn_ema = None
    def _rebuild_main_kdtree(self):
        matrix_list = [self.main_vectors[nid] for nid in self._main_storage_ids]
        if matrix_list:
            self._main_storage_matrix = np.array(matrix_list, dtype=np.float32)
            self.main_kdtree = KDTree(self._main_storage_matrix)
        else:
            self._main_storage_matrix = np.empty((0, EMBED_DIM), dtype=np.float32)
            self.main_kdtree = None
    @staticmethod
    def _cos_sim(v1, v2) -> float:
        """Calculates cosine similarity between two vectors."""
        v1 = np.asarray(v1, dtype=np.float32)
        v2 = np.asarray(v2, dtype=np.float32)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-9 or norm2 < 1e-9:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    async def add_entry(self, entry_data: dict, parent_ids: Optional[List[str]] = None, target_shells: Optional[List[int]] = None, precomputed_vec: Optional[np.ndarray] = None) -> str:
        """
        Adds a new concept to memory. This is the primary entry point for creating new memories.
        The process involves embedding, validation, and queuing for indexed retrieval.
        """
        # --- 1. Vector Computation & Normalization ---
        if precomputed_vec is not None:
            vec = precomputed_vec
        else:
            text_to_embed = f"{entry_data.get('label', '')}: {entry_data.get('metaphor', '')}".strip()
            if text_to_embed:
                raw_vec = await self.embedding_fn(text_to_embed)
                mood_mod = self.mood.get_mood_modulation_vector(raw_vec.shape[0])
                mood_blend = mood_get(self.mood.mood_vector, "intensity") * 0.15
                vec = normalize_vector(raw_vec + mood_blend * mood_mod)
            else:
                vec = np.zeros(EMBED_DIM, dtype=np.float32)

        if np.linalg.norm(vec) > 1e-9:
            vec = normalize_vector(vec)
            # Add to buffer for VAE training
            self.pending_embeddings.append(vec)

        entry_data["embedding"] = vec

        # --- 2. Autoencoder-based Blueprint Location ---
        if TORCH_AVAILABLE and getattr(self.mind, 'autoencoder', None) is not None and getattr(self.mind.autoencoder, 'is_trained', False):
            try:
                import torch as _torch
                with _torch.no_grad():
                    source_tensor = _torch.from_numpy(vec).float().unsqueeze(0)
                    z8_tensor = self.mind.autoencoder.project_to_dim(source_tensor, 8)
                    if z8_tensor is not None:
                        try:
                            z_np = np.asarray(z8_tensor).squeeze()
                        except Exception:
                            z_np = None
                        if z_np is not None and np.size(z_np) > 0:
                            location_id = self.mind.physics.find_nearest_root_index(z_np)
                            entry_data["blueprint_location_id"] = location_id
            except Exception:
                pass

        # --- 3. Add to Graph and Queue for KD-Tree ---
        async with self.lock:
            node_id = entry_data.get('idx')
            if not node_id:
                content_str = f"{entry_data.get('label', '')}{entry_data.get('metaphor', '')}{time.time()}"
                node_id = hashlib.sha1(content_str.encode()).hexdigest()[:16]
                entry_data['idx'] = node_id

            if self.graph_db.graph.has_node(node_id):
                self.spike_temperature(node_id, amount=0.5)
                try:
                    nd = self.graph_db.get_node(node_id)
                    if isinstance(nd, dict):
                        nd['last_step'] = self.mind.step_num
                except Exception:
                    pass
                return node_id
            
            entry_data.setdefault("temperature", 1.0)
            entry_data.setdefault("age", 0)
            entry_data["last_step"] = self.mind.step_num
            entry_data["mood_context"] = self.mood.mood_vector.copy()
            
            self.graph_db.add_node(node_id, **entry_data)
            self.main_vectors[node_id] = vec
            if entry_data.get('label'):
                self.label_to_node_id[entry_data['label']] = node_id

            shells_to_update = [self.mind.dimensional_shells[d] for d in target_shells] if target_shells else self.mind.dimensional_shells.values()
            for shell in shells_to_update:
                shell.add_vector(node_id, vec)
        # Auto-schedule validation of new insight/concept nodes
        try:
            hv = getattr(self.mind, 'hypothesis_validator', None)
            if os.getenv('E8_AUTO_VALIDATE_INSIGHTS', '1') == '1' and hv is not None and hasattr(hv, 'validate_insight'):
                asyncio.create_task(hv.validate_insight(node_id))
        except Exception:
            pass

        if parent_ids:
            for parent_id in parent_ids:
                if self.graph_db.get_node(parent_id):
                    self.graph_db.add_edge(node_id, parent_id, type="reflection_source", weight=0.9)
        
        self.pending_additions.append((node_id, vec))
        if len(self.pending_additions) >= self.KDTREE_REBUILD_THRESHOLD:
            self._commit_pending_additions_locked()

        # AFTER
        # --- 4. Post-processing and Cognitive Hooks ---
        self.sdm.write(vec)
        self.mood.process_event("new_concept", rating=entry_data.get("rating", 0.5))
        if entry_data.get("label"):
            await self.subconscious.track_concept(entry_data["label"], weight=entry_data.get("rating", 0.5))

        # Add this line to create connections to older, similar concepts
       
        asyncio.create_task(self.mind.perform_retro_relink(node_id, vec))
        # Add this at the end of the method, inside the lock or just after
        self.mind.new_node_id_queue.append(node_id)
        return node_id

    def _commit_pending_additions_locked(self):
        """
        Rebuilds the main KD-Tree index using batched additions.
        Must be called from within a locked context.
        """
        if not self.pending_additions:
            return

        self.console.log(f"🧠 [MemoryManager] Committing {len(self.pending_additions)} new vectors to KD-Tree index...")
        self._main_storage_ids = list(self.main_vectors.keys())
        matrix_list = [self.main_vectors[nid] for nid in self._main_storage_ids]
        
        if matrix_list:
            self._main_storage_matrix = np.array(matrix_list, dtype=np.float32)
            self.main_kdtree = KDTree(self._main_storage_matrix)
        else:
            self._main_storage_matrix = np.empty((0, EMBED_DIM), dtype=np.float32)
            self.main_kdtree = None
            
        self.pending_additions.clear()

    def find_similar_in_main_storage(self, query_vector: np.ndarray, k: int = 5) -> List[tuple[str, float]]:
        """
        Finds k-nearest neighbors using a dynamic reranking strategy that considers
        distance, recency, temperature, and semantic community.
        """
        if self.main_kdtree is None or not self._main_storage_ids:
            return []
        
        initial_k = min(k * 5, len(self._main_storage_ids))
        if initial_k == 0: return []
        
        t0=time.perf_counter(); distances, indices = self.main_kdtree.query(query_vector, k=initial_k)
        try:
            dt_ms=(time.perf_counter()-t0)*1000.0
            self._kdtree_latency_ms_window.append(dt_ms)
            try:
                self.mind.metrics.timing('kdtree.query_ms', dt_ms)
            except Exception:
                pass
            if len(self._kdtree_latency_ms_window)>64:
                self._kdtree_latency_ms_window.pop(0)
            if len(self._kdtree_latency_ms_window)>=16 and (sum(self._kdtree_latency_ms_window)/len(self._kdtree_latency_ms_window))>self._kdtree_latency_threshold_ms:
                self._rebuild_main_kdtree()
                self._kdtree_latency_ms_window.clear()
        except Exception:
            pass
        distances, indices = np.atleast_1d(distances), np.atleast_1d(indices)

        query_community = (self.graph_db.get_node(self._main_storage_ids[indices[0]]) or {}).get("community_id", -1)

        reranked_candidates = []
        for dist, idx in zip(distances, indices):
            nid = self._main_storage_ids[idx]
            node_data = self.graph_db.get_node(nid)
            if not node_data: continue

            score = float(dist)
            temp = node_data.get('temperature', 0.1)
            score *= (1.2 - (temp * 0.2))
            recency_penalty = max(0, self.mind.step_num - node_data.get('last_step', 0)) * 0.0001
            score += recency_penalty
            if query_community != -1 and node_data.get("community_id") == query_community:
                score *= 0.90

            reranked_candidates.append((nid, score))

        reranked_candidates.sort(key=lambda item: item[1])
        return reranked_candidates[:k]
    
    async def apply_decay(self):
        """Applies time-based decay to concept temperatures and runs maintenance."""
        async with self.lock:
            decay_vivid, decay_hot, decay_warm, decay_cold = 0.5**(1.0/TEMP_HALF_LIFE_VIVID), 0.5**(1.0/TEMP_HALF_LIFE_HOT), 0.5**(1.0/TEMP_HALF_LIFE_WARM), 0.5**(1.0/TEMP_HALF_LIFE_COLD)
            for node_id, data in self.graph_db.graph.nodes(data=True):
                temp = data.get('temperature', 1.0)
                if data.get("vivid_until_step", -1) > self.mind.step_num: temp *= decay_vivid
                elif temp > 1.5: temp *= decay_hot
                elif temp > 0.5: temp *= decay_warm
                else: temp *= decay_cold
                data['temperature'] = max(0.01, temp)
                data['age'] = data.get('age', 0) + 1
        
        if self.mind.step_num > 0 and self.mind.step_num % 200 == 0:
            asyncio.create_task(self._run_maintenance_cycle())

    async def _run_maintenance_cycle(self):
        """Performs periodic housekeeping tasks on the memory graph."""
        self.console.log("🧹 [MemoryManager] Running maintenance cycle...")
        async with self.lock:
            self._commit_pending_additions_locked()
            self.graph_db.compute_and_store_communities()
            self._evict_if_needed_locked()
            self.hopfield.update_prototypes()
        self.console.log("🧹 [MemoryManager] Maintenance cycle complete.")
    
    def _calculate_importance_score(self, node_id: str, data: Dict) -> float:
        """Calculates a holistic importance score for a node for eviction purposes."""
        rating = data.get('rating', 0.5)
        temp = data.get('temperature', 0.1)
        age = data.get('age', 1)
        degree = self._safe_degree(node_id)
        return (rating + temp * 0.5) / (np.log1p(age) * (degree + 1))

    def _evict_if_needed_locked(self, max_nodes: int = 50000, eviction_ratio: float = 0.05):
        """Prunes the least important nodes. Must be called from within a locked context."""
        num_nodes = self.graph_db.graph.number_of_nodes()
        if num_nodes <= max_nodes: return

        self.console.log(f"🧠 [MemoryManager] Memory limit ({max_nodes}) exceeded. Evicting nodes...")
        num_to_evict = int(num_nodes * eviction_ratio)
        candidates = []
        for nid, data in self.graph_db.graph.nodes(data=True):
            if data.get('type') in ['blackhole_remnant', 'self_code']: continue
            candidates.append((self._calculate_importance_score(nid, data), nid))
        
        candidates.sort()
        for _, node_id_to_evict in candidates[:num_to_evict]:
            try:
                self.graph_db.graph.remove_node(node_id_to_evict)
                self.main_vectors.pop(node_id_to_evict, None)
            except Exception: pass
        
        self._commit_pending_additions_locked()
        self.console.log(f"🧠 [MemoryManager] Evicted {num_to_evict} nodes.")

    def spike_temperature(self, node_id: str, amount: float = 1.0):
        """Safely increases a node's temperature."""
        node = self.graph_db.get_node(node_id)
        if node:
            node['temperature'] = node.get('temperature', 1.0) + amount

    def find_event_horizon(self, density_threshold=0.20, temp_threshold=1.05, age_threshold=1):
        """Identifies the most unstable point in memory, the epicenter for a black hole event."""
        graph, candidates = self.graph_db.graph, []
        for nid, d in graph.nodes(data=True):
            if d.get('age', 0) < age_threshold: continue
            temp = d.get('temperature', 0.0)
            density = self._local_density(nid)
            pressure = float(temp) * float(density)
            if (temp > temp_threshold) and (density >= density_threshold):
                candidates.append((pressure, nid))
        if not candidates: return None, None
        best_pressure, best_id = max(candidates)
        return best_id, best_pressure

    def _local_density(self, center_id: str, radius: int = 4) -> float:
        """Calculates the connection density of a node's local neighborhood."""
        try:
            if nx is None:
                return 0.0
            try:
                nodes_in_radius = set(nx.ego_graph(self.graph_db.graph, center_id, radius=radius).nodes())
            except Exception:
                return 0.0
            subgraph = self.graph_db.graph.subgraph(nodes_in_radius)
            num_nodes, num_edges = subgraph.number_of_nodes(), subgraph.number_of_edges()
            possible_edges = num_nodes * (num_nodes - 1) / 2
            return num_edges / possible_edges if possible_edges > 0 else 0.0
        except Exception:
            return 0.0

    def collect_cluster(self, center_id: str, radius: int = 4) -> List[str]:
        """Collects a cluster of nodes around a center using DBSCAN, with a robust fallback."""
        if DBSCAN is None:
            try:
                if nx is not None:
                    return list(nx.ego_graph(self.graph_db.graph, center_id, radius=2).nodes())
            except Exception:
                pass
            return [center_id]
        try:
            try:
                if nx is not None:
                    nodes_in_radius = list(nx.ego_graph(self.graph_db.graph, center_id, radius=radius).nodes())
                else:
                    return [center_id]
            except Exception:
                return [center_id]
            vectors = [self.main_vectors.get(nid) for nid in nodes_in_radius]
            valid_indices = [i for i, v in enumerate(vectors) if v is not None and np.any(v)]
            if len(valid_indices) < 3: return [nodes_in_radius[i] for i in valid_indices]
            
            matrix = np.array([vectors[i] for i in valid_indices])
            node_ids = [nodes_in_radius[i] for i in valid_indices]
            
            clustering = DBSCAN(eps=0.85, min_samples=2, metric='cosine').fit(matrix)
            center_idx = node_ids.index(center_id)
            center_label = clustering.labels_[center_idx]

            if center_label == -1 or np.sum(clustering.labels_ == center_label) < 3:
                try:
                    if nx is not None:
                        return list(nx.ego_graph(self.graph_db.graph, center_id, radius=2).nodes())
                except Exception:
                    pass
                return [center_id]
            
            return [node_ids[i] for i, label in enumerate(clustering.labels_) if label == center_label]
        except Exception:
            try:
                if nx is not None:
                    return list(nx.ego_graph(self.graph_db.graph, center_id, radius=2).nodes())
            except Exception:
                pass
            return [center_id]

    async def synthesize_remnant(self, cluster_nodes: List[str], label_hint: str, is_macro: bool = False) -> Tuple[Optional[Dict], Optional[np.ndarray], float]:
        """Synthesizes a new 'remnant' concept from a cluster of nodes during a collapse event."""
        if not cluster_nodes: return None, None, 0.0
        
        cluster_data = [d for nid in cluster_nodes if (d := self.graph_db.get_node(nid))]
        cluster_vectors = [v for nid in cluster_nodes if (v := self.main_vectors.get(nid)) is not None]
        
        if not cluster_data or not cluster_vectors: return None, None, 0.0

        weights = np.array([d.get('temperature', 1.0) for d in cluster_data])
        if weights.sum() < 1e-9: weights = np.ones(len(cluster_data))
        weights /= weights.sum()

        avg_rating = np.average([d.get('rating', 0.5) for d in cluster_data], weights=weights)
        mass = sum(d.get('temperature', 1.0) for d in cluster_data) * avg_rating
        
        remnant_vec = np.average(np.array(cluster_vectors), axis=0, weights=weights)
        remnant_vec = self.hopfield.clean_up(remnant_vec)

        fragments = [d.get('metaphor', d.get('label', '')) for d in cluster_data]
        prompt = (f"Synthesize the following fragmented ideas into a single, dense, core concept. "
                  f"Provide a short, evocative label and a one-sentence metaphor for the new idea.\n\n"
                  f"Ideas: {'; '.join(fragments[:10])}\n\nRespond in JSON format with keys 'label' and 'metaphor'.")
        
        try:
            response = await self.llm_caller.enqueue_and_wait(prompt, max_tokens=120)
            parsed = _parse_json_object(response)
            new_label, new_metaphor = parsed.get('label', label_hint), parsed.get('metaphor', 'A consolidated memory.')
        except Exception:
            new_label, new_metaphor = label_hint, "A synthesized concept."
            
        return {
            "type": "blackhole_remnant", "label": new_label, "metaphor": new_metaphor,
            "embedding": remnant_vec, "rating": avg_rating, "step": self.mind.step_num, "is_macro": is_macro
        }, remnant_vec, mass

    def fold_and_prune(self, cluster_nodes: List[str]):
        """Marks nodes as 'folded' after a collapse, effectively archiving them."""
        for node_id in cluster_nodes:
            node = self.graph_db.get_node(node_id)
            if node:
                node['folded'] = True
                node['temperature'] *= 0.1
                for shell in self.mind.dimensional_shells.values():
                    shell.vectors.pop(node_id, None)

    async def snapshot(self):
        """Saves a snapshot of the mind's current state to a file."""
        async with self.lock:
            self._commit_pending_additions_locked()
            filepath = get_path(f"snapshot_step_{self.mind.step_num:06d}.json", self.run_id)
            snapshot_data = {
                "graph": export_graph(self.graph_db.graph),
                "main_vectors": {nid: vec.tolist() for nid, vec in self.main_vectors.items()},
                "step": self.mind.step_num,
                "mood": self.mind.mood.mood_vector,
                "subconscious_narrative": self.mind.subconscious.narrative,
                "novelty_stats": {"avg_nn_ema": self._avg_nn_ema},
                "embedding_adapter": {
                    "in_dim": getattr(self.mind.embed_adapter, "in_dim", None),
                    "out_dim": getattr(self.mind.embed_adapter, "out_dim", None),
                    "W": (np.asarray(getattr(getattr(self.mind, 'embed_adapter', None), 'W', None)).tolist() if getattr(getattr(self.mind, 'embed_adapter', None), 'W', None) is not None else None)
                },
                "kd_index_meta": {
                    "backend": "faiss" if getattr(self.main_kdtree, "_is_faiss", False) else ("sklearn/scipy" if not getattr(self.main_kdtree, "_is_fallback", True) else "numpy"),
                    "n": int(getattr(self.main_kdtree, "n", 0)),
                    "dim": int(EMBED_DIM)
                },
                "metrics_counters": self.mind.metrics.snapshot_counters(),
            }
        safe_json_write(filepath, snapshot_data)
        try:
            sz = os.path.getsize(filepath)
            self.mind.metrics.gauge('snapshot.bytes', sz)
        except Exception:
            pass
        all_snapshots = sorted(glob.glob(get_path("snapshot_step_*.json", self.run_id)), key=os.path.getmtime)
        while len(all_snapshots) > 10:
            os.remove(all_snapshots.pop(0))

    def diffuse_field(self):
        """
        Simulates the diffusion of a generic activation field across the memory graph.
        This allows for spreading influence and temperature between connected concepts.
        """
        eta = BH_DIFFUSION_ETA
        next_field = defaultdict(float)
        nodes_with_field = list(self.field.keys())
        for node_id in nodes_with_field:
            current_val = self.field.get(node_id, 0.0)
            if current_val < 1e-4: continue
            retained_value = current_val * (1.0 - eta)
            next_field[node_id] += retained_value
            value_to_spread = current_val * eta
            neighbors = list(self.graph_db.get_neighbors(node_id))
            if neighbors:
                share_per_neighbor = value_to_spread / len(neighbors)
                for neighbor_id in neighbors:
                    next_field[neighbor_id] += share_per_neighbor
        self.field = next_field

    def decay_locks(self):
        """
        Decrements the timer on all active locks and removes any that have expired.
        This allows for temporary, decaying associations between concepts in memory.
        """
        expired_locks = [key for key, timer in self.active_locks.items() if timer <= 1]
        for key in expired_locks:
            del self.active_locks[key]
        for key in self.active_locks:
            self.active_locks[key] -= 1

    async def _cosmological_spread(self, remnant_vec: np.ndarray, mass: float):
        """Spreads influence from a black hole event to distant but similar concepts."""
        similar_nodes = self.find_similar_in_main_storage(remnant_vec, k=50)
        spread_factor = mass * BH_SPREAD_FRAC
        for node_id, dist in similar_nodes:
            # Spread is stronger for closer nodes
            energy = spread_factor * (1.0 - dist)
            if energy > 0:
                self.spike_temperature(node_id, amount=energy)
    
    def get_average_nearest_neighbor_distance(self) -> float:
        """Calculates the average distance between nearest neighbors in the main KD-tree."""
        if self.main_kdtree is None or self._main_storage_matrix.shape[0] < 2:
            return 1.0 # Default value if memory is sparse
        
        # Query for the 2 nearest neighbors: the point itself (dist=0) and its true nearest neighbor
        distances, _ = self.main_kdtree.query(self._main_storage_matrix, k=2)
        
        # The second column contains the distances to the actual nearest neighbors
        nearest_distances = distances[:, 1]
        avg=float(np.mean(nearest_distances))
        try:
            alpha=float(os.getenv('E8_NOVELTY_EMA_ALPHA','0.2'))
            if self._avg_nn_ema is None:
                self._avg_nn_ema=avg
            else:
                self._avg_nn_ema=alpha*avg+(1-alpha)*self._avg_nn_ema
            json.dump({'avg_nn_ema': self._avg_nn_ema}, open(self._novelty_stats_path,'w'))
        except Exception:
            pass
        return avg

    def _allow_edge(self, u: str, v: str) -> bool:
        """Checks if an edge can be safely added between two nodes."""
        g = self.graph_db.graph
        return g.has_node(u) and g.has_node(v) and not g.has_edge(u, v)

    def _safe_degree(self, node_id: str) -> int:
        """Returns node degree safely without relying on dict(DegreeView) conversions."""
        try:
            g = self.graph_db.graph
            d_attr = getattr(g, 'degree', None)
            if callable(d_attr):
                try:
                    val = d_attr(node_id)
                    if isinstance(val, (int, np.integer)):
                        return int(val)
                except Exception:
                    pass
            if d_attr is not None:
                try:
                    val2 = d_attr[node_id]  # type: ignore[index]
                    if isinstance(val2, (int, np.integer)):
                        return int(val2)
                except Exception:
                    pass
            try:
                return int(len(list(g.neighbors(node_id))))
            except Exception:
                return 0
        except Exception:
            return 0

    def _trim_degree(self, node_id: str, max_deg: int = 16):
        """If a node has too many connections, prune the weakest ones."""
        g = self.graph_db.graph
        try:
            deg = self._safe_degree(node_id)
            if (not g.has_node(node_id)) or deg <= max_deg:
                return
            edges = sorted(
                list(g.edges(node_id, data=True)),
                key=lambda e: e[2].get('weight', 0.0)  # Sort by weight, weakest first
            )
            num_to_prune = max(0, deg - max_deg)
            for u, v, _ in edges[:num_to_prune]:
                g.remove_edge(u, v)
        except Exception:
            return

class DreamEngine:
    """
    Generates synthetic memories by running thought experiments about future possibilities,
    allowing the AI to learn from events that haven't happened.
    """
    ALLOWED_TYPES = ("explorer_insight", "insight_synthesis", "meta_reflection", "phase_summary", "concept", "external_concept", "mutation", "synthetic_memory", "self_code", "self_code_section")

    def __init__(self, memory, mind_instance):
        self.memory = memory
        self.mind = mind_instance
        self.console = mind_instance.console

    def _eligible_concepts(self):
        G = self.memory.graph_db.graph
        out = []
        for nid, d in G.nodes(data=True):
            if d.get("folded"): continue
            if d.get("type") not in self.ALLOWED_TYPES: continue
            if self.memory.main_vectors.get(nid) is None: continue
            out.append((nid, d))
        return out

    def _pick_from_tension(self, elig, k=1):
        if not elig: return []
        tension_candidates = sorted(elig, key=lambda item: item[1].get('shell_tension', 0.0), reverse=True)
        high_tension_seeds = [item for item in tension_candidates if item[1].get('shell_tension', 0.0) > 0.1]
        if high_tension_seeds: return high_tension_seeds[:k]
        else: return self._pick_neutral(elig, k)


    def _pick_neutral(self, elig, k=1):
        if not elig:
            return []

        # Sort by temperature to prioritize "hot" concepts
        elig.sort(key=lambda item: (item[1].get("temperature", 0.0), item[1].get("step", 0)), reverse=True)
        
        # Create a small pool of top candidates and choose randomly from it.
        pool_size = min(len(elig), 5) # Take the top 5 or fewer if not enough concepts exist.
        if pool_size == 0:
            return []

        top_candidates = elig[:pool_size]
        
        # Randomly select k concepts from the top pool without replacement.
        num_to_sample = min(k, len(top_candidates))
        return random.sample(top_candidates, num_to_sample)

    async def run_dream_sequence(self, depth=1):
        # --- HOTFIX INTEGRATED: Thought Experiment Gating ---
        min_for_thought_exp = int(os.getenv("E8_MIN_FOR_THOUGHT_EXP", "40"))
        current_nodes = self.memory.graph_db.graph.number_of_nodes()
        if current_nodes < min_for_thought_exp:
            return # Silently skip if not enough concepts

        if not DREAM_MODE_ENABLED: return
        now = time.monotonic()
        if self.mind._dream_lock.locked() or (now - self.mind._last_dream_at < DREAM_MIN_INTERVAL_SEC):
            return

        async with self.mind._dream_lock:
            if time.monotonic() - self.mind._last_dream_at < DREAM_MIN_INTERVAL_SEC: return

            self.mind._last_dream_at = time.monotonic()

            elig = self._eligible_concepts()
            if not elig:
                self.console.log("[Thought Experiment] No suitable concepts found.")
                return

            seed = self._pick_neutral(elig, k=1)
            if not seed:
                self.console.log("[Thought Experiment] Seed picking failed.")
                return

            seed_node_id, seed_node_data = seed[0]

            try:
                _, top_goal_desc = self.mind.goal_field.get_top_goals(k=1)[0]
            except (IndexError, TypeError):
                top_goal_desc = "achieve a greater understanding"

            try:
                experiment_prompt = self.mind.prompts.render(
                    "thought_experiment",
                    concept=seed_node_data.get('label', 'a concept'),
                    details=seed_node_data.get('metaphor', ''),
                    goal=top_goal_desc
                )

                narrative = await asyncio.wait_for(self.mind.llm_pool.enqueue_and_wait(
                    experiment_prompt, max_tokens=300, temperature=0.85
                ), timeout=30)

                if narrative and not narrative.startswith("[LLM"):
                    # --- HOTFIX INTEGRATED: Vocabulary Constraint Check ---
                    try:
                        seed_neighbors = self.memory.graph_db.get_neighbors(seed_node_id)
                        neighbor_data = [self.memory.graph_db.get_node(n) for n in seed_neighbors if self.memory.graph_db.get_node(n)]
                        neighbor_labels = [d.get('label', '') for d in neighbor_data]
                        local_terms = set(re.findall(r"[A-Za-z0-9]+", " ".join(neighbor_labels).lower()))

                        if local_terms and not any(term in narrative.lower() for term in local_terms):
                            narrative = f"(loose) {narrative}"
                    except Exception as e:
                        self.console.log(f"[Thought Experiment] Vocab check failed: {e}")
                    # --- END HOTFIX ---

                    new_node_id = await self.mind.memory.add_entry({
                        "type": "synthetic_memory",
                        "label": f"Experiment: {seed_node_data.get('label')}",
                        "metaphor": narrative,
                        "rating": 0.75,
                        "is_synthetic": True,
                        "step": self.mind.step_num
                    }, parent_ids=[seed_node_id])

                    self.console.print(Panel(f"[bold]Seed Concept:[/] {seed_node_data.get('label')}\n[bold]Hypothetical Narrative:[/] {narrative}",
                        title="[bold blue]THOUGHT EXPERIMENT[/]", border_style="blue"))

                    self.mind.subconscious_event_log.append({
                        'type': 'thought_experiment',
                        'label': f"Experiment on {seed_node_data.get('label')}",
                        'step': self.mind.step_num,
                        'data': {'summary': narrative}
                    })

            except Exception as e:
                self.console.log(f"[Thought Experiment] Failed to run experiment: {e}")

class DreamReplayService:
    """Offline consolidation via prioritized replay (PER) into world model and graph memory."""
    def __init__(self, mind, batch=32, steps=50):
        self.mind = mind
        self.batch = int(batch)
        self.steps = int(steps)
        self.alpha = float(os.getenv("E8_REPLAY_ALPHA","1.0"))
        self.beta = float(os.getenv("E8_REPLAY_BETA","0.8"))
        self.gamma = float(os.getenv("E8_REPLAY_GAMMA","0.6"))
        self.eta = float(os.getenv("E8_REPLAY_PER_ETA","0.6"))
        self.edge_eta = float(os.getenv("E8_REINFORCE_ETA","0.08"))
        self.edge_decay = float(os.getenv("E8_REINFORCE_DECAY","0.01"))

    def _episodes(self):
        ia = getattr(self.mind, "insight_agent", None)
        if ia is None or not hasattr(ia, "episodic_memory"):
            return []
        return ia.episodic_memory.sample_prioritized(self.batch, mind=self.mind, alpha=self.alpha, beta=self.beta, gamma=self.gamma, eta=self.eta)

    def _to_traj(self, episodes):
        traj = []
        adim = int(getattr(self.mind, "action_dim", 0))
        for ep in episodes:
            node_id = ep.get("node_id")
            child = self.mind.memory.main_vectors.get(node_id)
            child = self.mind.memory.main_vectors.get(node_id)
            if child is None:
                child = ep.get("embedding")
            parents = ep.get("parent_ids") or []
            if child is None or not parents:
                # fallback: skip or synthesize from neighbors
                continue
            pv = [self.mind.memory.main_vectors.get(pid) for pid in parents if pid in self.mind.memory.main_vectors]
            if not pv: 
                continue
            s = np.mean(np.stack(pv), axis=0).astype(np.float32)
            sp = np.array(child, dtype=np.float32)
            a = np.zeros(adim, dtype=np.float32)
            r = float(ep.get("reward", ep.get("rating", 0.0)))
            traj.append((s,a,sp,r, node_id, parents))
        return traj

    def _reinforce_graph(self, traj):
        G = self.mind.memory.graph_db
        vsa = getattr(self.mind.memory, "vsa", None)
        reinforced = 0
        for (s,a,sp,r, node_id, parents) in traj:
            for pid in parents:
                try:
                    hv = None
                    if vsa is not None:
                        vec_a = self.mind.memory.main_vectors.get(pid); vec_b = self.mind.memory.main_vectors.get(node_id)
                        if vec_a is not None and vec_b is not None:
                            hv = self.mind.memory.vsa.encode_parentage(vec_a, vec_b)
                    G.increment_edge_weight(pid, node_id, delta=self.edge_eta*r, kind="consolidated")
                    if hv is not None:
                        try: G.graph[pid][node_id]['hypervector'] = hv
                        except Exception: pass
                    reinforced += 1
                except Exception:
                    pass
            # passive decay on unrelated edges could be scheduled elsewhere
        try:
            try:
                if getattr(self.mind, 'metrics', None) and hasattr(self.mind.metrics, 'increment'):
                    self.mind.metrics.increment("graph.edge_reinforce", reinforced)
            except Exception:
                pass
        except Exception:
            pass

    async def run(self):
        t0 = time.monotonic()
        wm = getattr(self.mind, "world_model", None)
        total = 0
        for _ in range(self.steps):
            episodes = self._episodes()
            if not episodes:
                break
            traj = self._to_traj(episodes)
            if not traj:
                break
            if wm and getattr(wm, "available", True):
                # train world model
                try:
                    loss = wm.train_batch([(s,a,sp,r) for (s,a,sp,r,_,_) in traj])
                    if loss and hasattr(self.mind, "metrics"):
                        try:
                            self.mind.metrics.observe("wm.loss.recon", loss.get("loss_recon"))
                            self.mind.metrics.observe("wm.loss.kl", loss.get("loss_kl"))
                        except Exception: pass
                except Exception:
                    pass
            self._reinforce_graph(traj)
            total += len(traj)
        dt = time.monotonic() - t0
        if dt > 0 and hasattr(self.mind, "metrics"):
            try: self.mind.metrics.observe("replay.samples_per_sec", total/max(1e-6, dt))
            except Exception: pass
class NarrativeStreamer:
    def __init__(self, memory_manager, llm_pool, run_id):
        self.memory = memory_manager
        self.llm_pool = llm_pool
        self.run_id = run_id
        self.narrative_file = get_path("narrative_stream.md", self.run_id)
        self.last_narrative_step = -1

    async def generate_and_add_entry(self, mind_state: 'E8Mind'):
        current_step = mind_state.step_num
        if current_step - self.last_narrative_step < 50: return
        try:
            significant_events = []
            all_nodes = list(self.memory.graph_db.graph.nodes(data=True))
            for node_id, data in all_nodes:
                if data.get("step", -1) > self.last_narrative_step:
                    event_type, rating = data.get("type"), data.get("rating", 0.0)
                    if event_type in ["dream", "blackhole_remnant", "meta_reflection"] or rating > 0.85:
                        # --- CORRECTED LOGIC ---
                        event_summary = f"- {data.get('label', 'Untitled event')} (Type: {event_type})"
                        significant_events.append(event_summary)
            
            # CORRECTED: The guard clause is now correctly placed *after* the loop.
            if len(significant_events) < 3: 
                return
            
            event_summary_text = "\n".join(significant_events[-15:])
            
            prompt = (f"You are the mind's historian. The current subconscious narrative is: '{mind_state.subconscious.narrative}'\n"
                      f"The prevailing mood feels like: {mind_state.mood.describe()}\n\n"
                      "Based on the following significant events, write a short, reflective journal entry (2-3 paragraphs) "
                      "that captures the tone and theme of this recent period. Synthesize them into a cohesive story.\n\n"
                      f"Events:\n{event_summary_text}\n\nJournal Entry for Steps {self.last_narrative_step} to {current_step}:")
            
            narrative_entry = await self.llm_pool.enqueue_and_wait(prompt, max_tokens=300, temperature=0.7)
            if narrative_entry and not narrative_entry.startswith("[LLM"):
                await self.memory.add_entry(f"## Chronicle: Steps {self.last_narrative_step}-{current_step}\n\n"
                                     f"**Mood**: {mind_state.mood.get_symbolic_weather()} | "
                                     f"**Theme**: {mind_state.synthetic_env.current_theme_region}\n\n"
                                     f"{narrative_entry}\n\n---\n\n")
                self.last_narrative_step = current_step
        except Exception as e:
            console.log(f"[NarrativeStreamer] Failed to generate entry: {e}")

class SyntheticEnvironment:
    def __init__(self, llm_caller, mind_instance):
        self.llm_caller = llm_caller
        self.mind = mind_instance
        self.current_theme_region = "The Genesis Field"
        self.region_journal: List[Dict[str, Any]] = []

    async def name_theme_region(self, seed_fragments: list[str], subconscious_narrative: str) -> str:
        if not seed_fragments: return self.current_theme_region
        prompt = (f"The mind's current subconscious narrative is: \"{subconscious_narrative}\"\n\n"
                  "Based on this narrative and these local ideas, provide a short, evocative 2-4 word name for this conceptual region.\n"
                  f"Local Ideas: {', '.join(seed_fragments)}.")
        try:
            name = await self.llm_caller.enqueue_and_wait(prompt, max_tokens=15, temperature=0.75)
            return sanitize_line(name, max_chars=36)
        except Exception as e:
            console.log(f"[bold red]Failed to name theme region: {e}[/bold red]")
            return "A Developing Region"

    def recent_triggers(self, count: int = 3) -> List[str]:
        if not self.region_journal: return []
        return self.region_journal[-1].get("triggers", [])[:count]

    async def update_from_location(self, current_node_index: int):
        blueprint_points = np.array([[p['x'], p['y']] for p in self.mind.blueprint])
        if current_node_index >= len(blueprint_points): return
        current_pos = blueprint_points[current_node_index]
        distances = np.linalg.norm(blueprint_points - current_pos, axis=1)
        neighbor_indices = set(np.argsort(distances)[:7])
        neighbor_metaphors, seen_metaphors = [], set()
        recent_nodes = list(self.mind.memory.graph_db.graph.nodes(data=True))[-250:]
        for node_id, node_data in reversed(recent_nodes):
            if len(neighbor_metaphors) > 10: break
            loc_id, metaphor = node_data.get("blueprint_location_id"), node_data.get("metaphor")
            if loc_id in neighbor_indices and metaphor and metaphor not in seen_metaphors:
                neighbor_metaphors.append(f'"{metaphor}"')
                seen_metaphors.add(metaphor)
        if not neighbor_metaphors: return
        new_name = await self.name_theme_region(neighbor_metaphors, self.mind.subconscious.narrative)
        new_name = sanitize_line(new_name, max_chars=36)
        if new_name != self.current_theme_region:
            self.current_theme_region = new_name
            self.region_journal.append({ "ts": time.time(), "name": self.current_theme_region, "triggers": neighbor_metaphors })
            console.print(Panel.fit(f"The perceived environment has shifted to:\n[bold cyan]{self.current_theme_region}[/bold cyan]",
                                    title="[bold #A020F0]ENVIRONMENT SHIFT[/]", border_style="#A020F0"))

class DomainTintEngine:
    def __init__(self, seed_domain, llm_pool):
        self.seed_domain = seed_domain
        self.llm_pool = llm_pool
        self.last_hint = seed_domain

    async def evolve(self, mood_vector):
        hint = await self.llm_pool.enqueue_and_wait(
            "ignored",
            _prompt_key="ask",
            _prompt_vars={"question": f"2-4 word domain hint for {self.seed_domain} given mood={mood_vector}."}
        )
        hint = (hint or "").strip()
        if hint and not hint.startswith("[LLM"):
            self.last_hint = hint
        return self.last_hint

@dataclass
class DecodeState:
    current_idx: int
    shadow_ids: np.ndarray
    slice_id: int
    seen_tokens: set[str]
    emap: EntropyMap
    holo: HoloEncoder

class OnlineAdapter:
    def __init__(self, state_dim: int = 8, lr: float = 1e-3, scale: float = 0.5):
        self.W = np.zeros((state_dim,), dtype=np.float32)
        self.lr, self.scale, self.state_dim = lr, scale, state_dim

    def bias_logit(self, base_logit: float, state_vec: np.ndarray) -> float:
        if state_vec.shape[0] != self.state_dim: return float(base_logit)
        return float(base_logit + self.scale * np.dot(self.W, state_vec))

    def update(self, error: float, last_state_vec: np.ndarray):
        if last_state_vec.shape[0] != self.state_dim: return
        self.W += self.lr * error * last_state_vec
        norm = np.linalg.norm(self.W)
        if norm > 1.0: self.W /= norm

class Judge:
    def __init__(self, lm_client, embed_fn, emap: "EntropyMap", holo: "HoloEncoder"):
        self.lm, self.embed, self.emap, self.holo = lm_client, embed_fn, emap, holo

    async def score(self, text: str, active_concepts_vec: np.ndarray, shadow_ids: np.ndarray, slice_id: int):
        v_text = await self.embed(text)
        fit_score = MemoryManager._cos_sim(v_text, active_concepts_vec) if np.linalg.norm(active_concepts_vec) > 0 else 0.0
        return fit_score

class ConstrainedDecoder:
    def __init__(self, lm_client, fabric: "E8BoundaryFabric"):
        self.lm, self.fabric = lm_client, fabric

    async def generate(self, prompt: str, start_node: int):
        return await self.lm.chat(messages=[{"role":"user", "content":prompt}])

class SymmetryValenceEngine:
    def __init__(self, physics, hist_len=128):
        self.physics = physics
        self.hist = deque(maxlen=int(hist_len))

    def push(self, v8):
        v = np.asarray(v8, dtype=np.float32).reshape(-1)
        if v.size != 8 or not np.isfinite(v).all(): return
        self.hist.append(v / (np.linalg.norm(v) + 1e-12))

    def _eig_entropy(self, X):
        C = np.cov(X.T) + 1e-6*np.eye(8, dtype=np.float32)
        w = np.linalg.eigvalsh(C).clip(min=1e-9)
        p = w / w.sum()
        return float(-np.sum(p * np.log(p)) / np.log(8.0))

    def score(self):
        if len(self.hist) < 8: return 0.5
        X = np.stack(list(self.hist)[-64:], axis=0).astype(np.float32)
        return 1.0 - self._eig_entropy(X)

class EgoGate:
    def __init__(self, valence_engine, min_delta=-0.02):
        self.ve, self.min_delta = valence_engine, float(min_delta)
        self._last = self.ve.score()

    def approve(self):
        cur = self.ve.score()
        ok = (cur >= self._last + self.min_delta)
        if ok: self._last = cur
        return bool(ok)

class HypothesisValidator:
    """
    A framework for classifying, planning, and (in the future) executing tests
    for hypotheses generated by the E8 Mind-Crystal. It distinguishes between
    hypotheses that can be tested computationally and those that require
    physical experimentation.
    """
    def __init__(self, mind_instance: 'E8Mind'):
        self.mind = mind_instance
        self.llm_pool = mind_instance.llm_pool
    async def validate_insight(self, insight_node_id: str):
        """
        The main entry point for validating a new insight. It orchestrates the
        classification, planning, and reporting process.
        """
        insight_data = self.mind.memory.graph_db.get_node(insight_node_id)
        if not insight_data:
            self.mind.console.log(f"[Validator] Could not find insight data for node {insight_node_id}")
            return

        hypothesis_text = insight_data.get('metaphor', insight_data.get('label', ''))
        if not hypothesis_text:
            return

        self.mind.console.print(Panel(f"Validating new insight: [bold cyan]'{insight_data.get('label')}'[/bold cyan]", title="[bold yellow]VALIDATOR[/]", border_style="yellow"))

        classification = await self._classify_hypothesis(hypothesis_text)

        node = self.mind.memory.graph_db.get_node(insight_node_id)
        if node:
            node['validation_status'] = classification

        self.mind.console.print(Panel(f"[bold]Classification:[/] {classification.get('type', 'unknown')}\n[bold]Reasoning:[/] {classification.get('reasoning', 'N/A')}", title="[bold yellow]VALIDATOR: CLASSIFICATION[/]", border_style="yellow"))

        if classification.get('type') == 'computationally_testable':
            test_plan = await self._design_test_plan(hypothesis_text)
            if node:
                node['validation_plan'] = test_plan

            plan_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(test_plan.get('steps', []))])
            self.mind.console.print(Panel(f"[bold]Required Data:[/] {test_plan.get('required_data', 'N/A')}\n\n[bold]Test Steps:[/]\n{plan_text}", title="[bold yellow]VALIDATOR: TEST PLAN[/]", border_style="yellow"))

    async def _classify_hypothesis(self, hypothesis: str) -> Dict[str, Any]:
        """
        Uses an LLM to classify a hypothesis into one of two categories:
        1. computationally_testable: Can be verified with data analysis or simulation.
        2. physically_testable: Requires a real-world physical experiment.
        """
        prompt = (
            "You are a research scientist. Classify the following hypothesis. Can it be tested and validated "
            "entirely through computational means (data analysis, simulation) or does it require a physical, "
            "real-world experiment (e.g., in a wet lab, with a particle accelerator)?\n\n"
            f"Hypothesis: \"{hypothesis}\"\n\n"
            "Respond in JSON format with two keys: 'type' (string: 'computationally_testable' or 'physically_testable') "
            "and 'reasoning' (a brief explanation for your choice)."
        )
        try:
            response = await self.llm_pool.enqueue_and_wait(prompt, max_tokens=150)
            return _parse_json_object(response)
        except Exception as e:
            self.mind.console.log(f"[Validator] Classification failed: {e}")
            return {"type": "unknown", "reasoning": "LLM classification failed."}

    async def _design_test_plan(self, hypothesis: str) -> Dict[str, Any]:
        """
        For a computationally testable hypothesis, this uses an LLM to generate a
        high-level, step-by-step plan for how to validate it.
        """
        prompt = (
            "You are a principal investigator designing an experiment. For the following computationally testable "
            "hypothesis, create a validation plan.\n\n"
            f"Hypothesis: \"{hypothesis}\"\n\n"
            "Respond in JSON format with two keys: 'required_data' (a brief description of the datasets needed, e.g., 'Historical S&P 500 price data and news sentiment scores') "
            "and 'steps' (an array of strings outlining the high-level steps for the analysis, e.g., ['Clean and align datasets by date', 'Perform time-series cross-correlation analysis', 'Check for statistical significance'])."
        )
        try:
            response = await self.llm_pool.enqueue_and_wait(prompt, max_tokens=400)
            return _parse_json_object(response)
        except Exception as e:
            self.mind.console.log(f"[Validator] Test plan design failed: {e}")
            return {"required_data": "N/A", "steps": ["LLM failed to generate a plan."]}

class DataIngestionPipeline:
    """
    A scalable pipeline to continuously ingest and process data from external sources,
    turning new information into concepts in the mind's memory.
    """
    def __init__(self, mind_instance: 'E8Mind'):
        self.mind = mind_instance
        self.console = mind_instance.console
        self.sources = {}
        self.state = {}
        self._task: Optional[asyncio.Task] = None
        self.running = False
        self.state_file = get_path("ingestion_state.json", self.mind.run_id)

    
        self._recent_ids = {}
    def add_source(self, name: str, config: Dict[str, Any]):
        """Adds a data source to be monitored."""
        self.sources[name] = config
        self.console.log(f"[Ingestion] Added data source: '{name}' (type: {config.get('type')})")

    async def start(self):
        """Starts the background ingestion process."""
        if self.running or aiohttp is None:
            return
        self.running = True
        self.state = safe_json_read(self.state_file, default={})
        self._task = asyncio.create_task(self._run())
        self.console.log("[Ingestion] Pipeline started.")

    def stop(self):
        """Stops the background ingestion process."""
        self.running = False
        if self._task:
            self._task.cancel()
        safe_json_write(self.state_file, self.state)
        self.console.log("[Ingestion] Pipeline stopped.")

    async def _run(self):
        """The main worker loop that checks sources based on their schedule."""
        while self.running:
            now = time.monotonic()
            for name, config in self.sources.items():
                last_checked = self.state.get(name, {}).get("last_checked_monotonic", 0)
                interval_seconds = config.get("schedule_minutes", 60) * 60
                if now - last_checked > interval_seconds:
                    try:
                        await self._process_source(name, config)
                    except Exception as e:
                        console.log(f"[bold red][Ingestion] Error processing source '{name}': {e}[/bold red]")
                    finally:
                        if name not in self.state: self.state[name] = {}
                        self.state[name]["last_checked_monotonic"] = now
            await asyncio.sleep(60)

    async def _process_source(self, name: str, config: Dict[str, Any]):
        """Delegates processing based on the source type."""
        source_type = config.get("type")
        self.console.log(f"[Ingestion] Checking source: '{name}'")
        if source_type == "arxiv_api":
            await self._process_arxiv(name, config)
        elif source_type == "file":
            await self._process_file(name, config)
        else:
            self.console.log(f"[yellow][Ingestion] Unknown source type '{source_type}' for '{name}'[/yellow]")

    async def _process_arxiv(self, name: str, config: Dict[str, Any]):
        """Fetches and parses new entries from an arXiv Atom feed."""
        # Get the unique ID of the last processed entry to prevent gaps
        last_processed_id = self.state.get(name, {}).get("last_processed_id", None)
        new_entries_to_add = []
        
        # Use a set for efficient lookup of recently processed articles in this session
        processed_ids_this_session = set()

        
        seen_ids = set(self.state.get(name, {}).get('seen_ids', []))
        if name not in self._recent_ids:
            from collections import deque as _deque
            self._recent_ids[name] = _deque(maxlen=4096)
        if aiohttp is None or ET is None:
            self.console.log("[yellow][Ingestion] aiohttp or ET not available; skipping arXiv fetch[/yellow]")
            return
        async with aiohttp.ClientSession() as session:
            async with session.get(config["url"]) as response:
                if response.status != 200:
                    self.console.log(f"[bold red][Ingestion] arXiv fetch failed for '{name}': HTTP {response.status}[/bold red]")
                    return
                feed_xml = await response.text()

        try:
            root = ET.fromstring(feed_xml)
        except Exception:
            self.console.log("[bold red][Ingestion] Failed to parse arXiv feed XML[/bold red]")
            return
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        # The feed is sorted newest-first, so we can stop when we see an old entry
        for entry in root.findall('atom:entry', ns):
            id_el = entry.find('atom:id', ns)
            entry_id = (id_el.text or '').strip() if id_el is not None else ''

            if entry_id in seen_ids or entry_id in set(self._recent_ids.get(name, [])):
                continue
            if entry_id == last_processed_id:
                # We've reached the last article we processed in a previous run. Stop here.
                break

            title_el = entry.find('atom:title', ns)
            summary_el = entry.find('atom:summary', ns)
            title = (title_el.text or '').strip() if title_el is not None else ''
            summary = ((summary_el.text or '').strip().replace('\n', ' ')) if summary_el is not None else ''
            new_entries_to_add.append((entry_id, f"{title}: {summary}"))
            processed_ids_this_session.add(entry_id)

        if new_entries_to_add:
            # Reverse the list so we add concepts in chronological order (oldest first)
            new_entries_to_add.reverse()
            
            for _, text in new_entries_to_add:
                await self._add_text_as_concept(text, source_name=name)

            # Update state with the ID of the newest article from this batch
            latest_id = new_entries_to_add[-1][0]
            if name not in self.state:
                self.state[name] = {}
            self.state[name]["last_processed_id"] = latest_id
            safe_json_write(self.state_file, self.state)
            self.console.log(f"[Ingestion] Added {len(new_entries_to_add)} new concepts from '{name}'.")

    async def _process_file(self, name: str, config: Dict[str, Any]):
        """Processes a local file if it has been modified."""
        filepath = str(config.get("path", ""))
        if not filepath or not os.path.exists(filepath):
            return

        last_mod_time = self.state.get(name, {}).get("last_mod_time", 0)
        current_mod_time = os.path.getmtime(filepath)

        if current_mod_time > last_mod_time:
            self.console.log(f"[Ingestion] File '{filepath}' has been updated. Processing.")
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = [p.strip() for p in content.split('\n\n') if p.strip()]
            for chunk in chunks:
                await self._add_text_as_concept(chunk, source_name=name)

            if name not in self.state:
                self.state[name] = {}
            self.state[name]["last_mod_time"] = current_mod_time
            safe_json_write(self.state_file, self.state)
            self.console.log(f"[Ingestion] Added {len(chunks)} new concepts from file '{name}'.")

    async def _add_text_as_concept(self, text: str, source_name: str):
        """Adds a chunk of text as a new concept in memory."""
        if not text: return
        rating = await self.mind.rate_concept(text)
        entry = {
            "type": "external_concept",
            "label": sanitize_line(text, 40),
            "metaphor": sanitize_block(text, 5, 500),
            "rating": rating,
            "step": self.mind.step_num,
            "source": source_name,
        }
        await self.mind.memory.add_entry(entry)

def restricted_basis(weights, hops, N, hops_limit):
    neighbors = np.where(hops <= hops_limit)[0]
    if len(neighbors) > N: return neighbors[np.argsort(-weights[neighbors])[:N]]
    return neighbors

def build_local_L_norm(W_local):
    if W_local.shape[0] == 0: return np.array([[]], dtype=np.float32)
    deg = np.sum(W_local, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-9)))
    return np.eye(W_local.shape[0]) - D_inv_sqrt @ W_local @ D_inv_sqrt

def build_joint_H(L1, V1, L2, V2, J, gamma):
    H1 = gamma * L1 + np.diag(V1)
    H2 = gamma * L2 + np.diag(V2)
    H_joint = np.kron(H1, np.eye(len(L2))) + np.kron(np.eye(len(L1)), H2)
    H_joint += J * np.eye(len(H_joint))
    return H_joint

def sample_from_2d(P_2d):
    P_flat = P_2d.flatten()
    if P_flat.sum() < 1e-9:
        return np.unravel_index(0, P_2d.shape)
    P_flat /= np.sum(P_flat)
    sample_index = np.random.choice(len(P_flat), p=P_flat)
    return np.unravel_index(sample_index, P_2d.shape)

class MetaTelemetryLogger:
    def __init__(self, mind_instance: 'E8Mind', run_id: str):
        self.mind = mind_instance
        self.diary_file = get_path("mind_diary.md", run_id)
        self.shell_tension_history = deque(maxlen=100)
        self.bh_event_steps = deque(maxlen=20)
        self.mood_history = deque(maxlen=100)

    def log_step(self):
        all_tensions = [d.get('shell_tension', 0.0) for _, d in self.mind.memory.graph_db.graph.nodes(data=True) if d.get('shell_tension') is not None]
        avg_tension = np.mean(all_tensions) if all_tensions else 0.0
        self.shell_tension_history.append(avg_tension)
        self.mood_history.append(list(self.mind.mood.mood_vector.values()))

    def log_bh_event(self, step_num: int):
        self.bh_event_steps.append(step_num)

    async def generate_diary_entry(self):
        maxlen = int(self.mood_history.maxlen or 0)
        if len(self.mood_history) < maxlen:
            return
        avg_tension = np.mean(self.shell_tension_history)
        bh_frequency = len(self.bh_event_steps) / len(self.shell_tension_history)
        mood_variance = np.mean(np.var(np.array(list(self.mood_history)), axis=0))
        metrics_summary = (f"- Average Cognitive Tension: {avg_tension:.4f}\n"
                           f"- Black Hole Event Frequency: {bh_frequency:.3f} events/step\n"
                           f"- Mood Stability (lower is more stable): {mood_variance:.4f}")
        prompt = ("You are a mind reflecting on your own internal state. Based on the following metrics from the last 100 steps, "
                  "write a short, metaphorical, first-person diary entry. Do not list the metrics; interpret their meaning.\n\n"
                  f"Internal State Metrics:\n{metrics_summary}\n\nDiary Entry:")
        try:
            entry = await asyncio.wait_for(self.mind.llm_pool.enqueue_and_wait(prompt, max_tokens=200, temperature=0.75), timeout=30)
            if entry and not entry.startswith("[LLM"):
                with open(self.diary_file, "a", encoding="utf-8") as f:
                    f.write(f"## Step {self.mind.step_num}\n\n{entry}\n\n---\n\n")
                self.mind.console.print(Panel(entry, title="[bold #FFD700]Mind's Diary[/]", border_style="#FFD700"))
        except Exception as e:
            self.mind.console.log(f"[Diary] Failed to generate entry: {e}")

class TopologyMonitor:
    """Approximate PH via epsilon-graph Betti numbers (β0, β1).
    If scipy is unavailable, returns zeros (no intrinsic from topology).
    """
    def __init__(self, eps=0.35):
        self.eps = float(eps)
        self.prev_betti = {}

    def _betti(self, X):
        try:
            import numpy as np
            from scipy.spatial.distance import pdist, squareform
        except Exception:
            return (0,0)
        if X is None or len(X)==0:
            return (0,0)
        D = squareform(pdist(X, 'euclidean'))
        A = (D <= self.eps).astype(np.int32)
        np.fill_diagonal(A, 0)
        V = A.shape[0]; E = int(A.sum()//2)
        parent = list(range(V))
        def find(a):
            while parent[a]!=a:
                parent[a]=parent[parent[a]]; a=parent[a]
            return a
        def union(a,b):
            ra,rb=find(a),find(b)
            if ra!=rb: parent[rb]=ra
        for i in range(V):
            for j in range(i+1,V):
                if A[i,j]: union(i,j)
        C = len({find(i) for i in range(V)})
        beta0 = C; beta1 = max(0, E - V + C)
        return (beta0, beta1)

    def delta_betti(self, shell):
        try:
            X, _ = shell.get_all_vectors_as_matrix()
        except Exception:
            return 0.0
        b0,b1 = self._betti(X)
        prev = self.prev_betti.get(shell.dim, (b0,b1))
        self.prev_betti[shell.dim] = (b0,b1)
        return float(abs(b0 - prev[0]) + abs(b1 - prev[1]))
        def __init__(self, layer_sizes, console=None):
            super().__init__()
            self.console = console
            self.kl_beta = 0.1
            self._trained = False
            self.K = None # For Koopman operator (future use)

            # Build Encoder
            encoder_layers = []
            for i in range(len(layer_sizes) - 2):
                encoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
                encoder_layers.append(nn.ReLU())
            self.encoder = nn.Sequential(*encoder_layers)
            self.fc_mu = nn.Linear(layer_sizes[-2], layer_sizes[-1])
            self.fc_logvar = nn.Linear(layer_sizes[-2], layer_sizes[-1])

            # Build Decoder
            decoder_layers = []
            reversed_layers = layer_sizes[::-1]
            for i in range(len(reversed_layers) - 1):
                decoder_layers.append(nn.Linear(reversed_layers[i], reversed_layers[i+1]))
                decoder_layers.append(nn.ReLU())
            # Remove the last ReLU to allow any output value
            decoder_layers.pop()
            self.decoder = nn.Sequential(*decoder_layers)

            self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            self.projection_maps = {}

        @property
        def is_trained(self):
            return self._trained

        def _get_projection_map(self, source_dim, target_dim):
            key = (source_dim, target_dim)
            if key not in self.projection_maps:
                # Simple linear projection as a fallback if specific layers aren't defined
                proj = nn.Linear(source_dim, target_dim, bias=False)
                nn.init.orthogonal_(proj.weight)
                self.projection_maps[key] = proj
            return self.projection_maps[key]

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, x):
            h = self.encoder(x)
            mu, logvar = self.fc_mu(h), self.fc_logvar(h)
            z = self.reparameterize(mu, logvar)
            recon = self.decoder(z)
            return recon, mu, logvar

        def latent(self, x):
            with torch.no_grad():
                h = self.encoder(x)
                mu, _ = self.fc_mu(h), self.fc_logvar(h)
            return mu

        def loss_function(self, recon, x, mu, logvar):
            recon_loss = F.mse_loss(recon, x, reduction='sum')
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            total_loss = recon_loss + self.kl_beta * kld_loss
            return {'recon_loss': recon_loss, 'kld_loss': kld_loss, 'total_loss': total_loss}

        def train_on_batch(self, x):
            self.train()
            self.optimizer.zero_grad()
            recon_x, mu, logvar = self.forward(x)
            losses = self.loss_function(recon_x, x, mu, logvar)
            total_loss = losses['total_loss']
            total_loss.backward()
            self.optimizer.step()
            if not self._trained: self._trained = True
            return {k: v.item() for k, v in losses.items()}

        def project_to_dim(self, x, target_dim: int):
            with torch.no_grad():
                latent_z = self.latent(x)
                if latent_z.shape[-1] == target_dim:
                    return latent_z
                proj = self._get_projection_map(latent_z.shape[-1], target_dim)
                return proj(latent_z)
        
        def project_between_dim(self, x, source_dim: int, target_dim: int):
            # This is a simplified projection; a true VAE would use partial encoders/decoders
            with torch.no_grad():
                if x.shape[-1] != source_dim:
                    pad = torch.zeros(*x.shape[:-1], source_dim - x.shape[-1], dtype=x.dtype, device=x.device)
                    x = torch.cat([x, pad], dim=-1)

                proj = self._get_projection_map(source_dim, target_dim)
                return proj(x)
            
class CognitiveScheduler:
    """
    Manages and triggers high-level, asynchronous cognitive events for the E8Mind
    without blocking the main, high-speed cognitive cycle. This acts as a
    decoupled scheduler for slower, language-based functions.
    """
    def __init__(self, mind_instance: 'E8Mind'):
        self.mind = mind_instance
        # --- Define the cadence of cognitive events ---
        self.PROXIMITY_ALERT_INTERVAL = 11
        self.INSIGHT_SYNTHESIS_INTERVAL = 23
        self.DREAM_INTERVAL = 5
        self.NARRATIVE_SUMMARY_INTERVAL = 37  # Slower for better summaries
        self.SNAPSHOT_INTERVAL = 100
        self.DECAY_INTERVAL = 24
        
        self.DREAM_REPLAY_INTERVAL = int(os.getenv('E8_DREAM_EVERY_STEPS','200'))
        # --- Teacher/Explorer Cadence from Config ---
        self.TEACHER_ASK_EVERY = TEACHER_ASK_EVERY
        self.TEACHER_OFFSET = TEACHER_OFFSET
        self.EXPLORER_OFFSET = EXPLORER_OFFSET

    def _fire(self, step: int, interval: int, offset: int) -> bool:
        """Checks if an event should be triggered at a given step."""
        return interval > 0 and step >= offset and ((step - offset) % interval == 0)

    def tick(self, step: int):

        # Backpressure & mood gating
        try:
            qdepth = self.mind.llm_pool.queue.qsize() if getattr(self.mind, 'llm_pool', None) else 0
        except Exception:
            qdepth = 0
        entropy = float(self.mind.mood.mood_vector.get('entropy', 0.0)) if getattr(self.mind, 'mood', None) else 0.0
        max_q = int(os.getenv('E8_LLM_MAX_QUEUE', '64'))
        high_entropy = float(os.getenv('E8_HIGH_ENTROPY', '0.85'))
        skip_dream = qdepth > max_q or entropy > high_entropy
        try:
            self.mind.metrics.gauge('llm.queue_depth', qdepth)
            self.mind.metrics.gauge('mood.entropy', entropy)
            if skip_dream:
                try:
                    if getattr(self.mind, 'metrics', None) and hasattr(self.mind.metrics, 'increment'):
                        self.mind.metrics.increment('dream.skipped')
                except Exception:
                    pass
        except Exception:
            pass
    
        """
        Called on every single step of the cognitive cycle. It checks its schedule
        and launches tasks in the background if their time has come.
        """
        # --- Teacher and Explorer Dialogue Logic ---
        # Check if a question should be asked (and none is pending)
        if self.mind.teacher_question is None and self._fire(step, self.TEACHER_ASK_EVERY, self.TEACHER_OFFSET):
            asyncio.create_task(self.mind._teacher_ask_new_question())
        
        # Check if a pending question should be answered
        elif self.mind.teacher_question is not None and self._fire(step, self.TEACHER_ASK_EVERY, self.EXPLORER_OFFSET):
            asyncio.create_task(self.mind._explorer_answer_pending_question()) if not skip_dream else None

        # --- Other Asynchronous Cognitive Functions ---
        if self._fire(step, self.PROXIMITY_ALERT_INTERVAL, 5):
            asyncio.create_task(self.mind._run_insight_cycle())
            
        if self._fire(step, self.INSIGHT_SYNTHESIS_INTERVAL, 13):
            asyncio.create_task(self.mind._run_proactive_insight_synthesis())
            
        if self._fire(step, self.DREAM_INTERVAL, 0):
            asyncio.create_task(self.mind.dream_engine.run_dream_sequence()) if not skip_dream else None
            
        if self._fire(step, self.NARRATIVE_SUMMARY_INTERVAL, 2):
            asyncio.create_task(self.mind._generate_subconscious_narrative())
            
        if self._fire(step, self.SNAPSHOT_INTERVAL, 0):
            asyncio.create_task(self.mind.memory.snapshot())
            
        if self._fire(step, self.DECAY_INTERVAL, 21):
            asyncio.create_task(self.mind.memory.apply_decay())

# --- M17 ADDITIONS & BUGFIX: Missing Class and Function Definitions ---
class MetricsManager:
    """
    A structured manager for logging various types of performance metrics
    (counters, gauges, timings) to a durable, machine-readable format.
    """
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.log_file = get_path("metrics.ndjson", self.run_id)
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self.lock = threading.Lock() # Use a thread-safe lock for file I/O
        self.console = console
        self._counters = defaultdict(int)

    def snapshot_counters(self):
        return dict(self._counters)

    def _log(self, metric_type: str, data: dict):
        """Internal logging method to write a structured entry to the log file."""
        try:
            log_entry = {
                "ts": (datetime.now(timezone.utc).isoformat() if (datetime is not None and timezone is not None) else __import__('datetime').datetime.utcnow().isoformat()),
                "run_id": self.run_id,
                "type": metric_type,
                **data
            }
            with self.lock:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry, cls=NumpyEncoder) + "\n")
        except Exception as e:
            self.console.log(f"[bold red][MetricsManager] Failed to log metric: {e}[/bold red]")

    def increment(self, name: str, value: int = 1, tags: Optional[Dict] = None):
        """Increments a counter metric."""
        self._log("counter", {"name": name, "value": value, "tags": tags or {}})
        try:
            self._counters[name] += int(value)
        except Exception:
            pass

    def gauge(self, name: str, value: float, tags: Optional[Dict] = None):
        """Sets a gauge metric to a specific value."""
        self._log("gauge", {"name": name, "value": value, "tags": tags or {}})

    def timing(self, name: str, duration_ms: float, tags: Optional[Dict] = None):
        """Logs a timing or duration metric in milliseconds."""
        self._log("timing", {"name": name, "duration_ms": duration_ms, "tags": tags or {}})

    # Replace the old 'log' method with a more generic event logger
    def event(self, name: str, data: dict):
        """Logs a generic event with a dictionary payload."""
        self._log("event", {"name": name, **data})
        self._log("event", {"name": name, **data})

class ContextBandit:
    """
    A true Contextual Bandit using the LinUCB (Linear Upper Confidence Bound) algorithm.
    It uses the mind's state (context) to make smarter, adaptive decisions about which
    set of parameters ("arm") to use for the cognitive cycle.
    """
    def __init__(self, arms: list, state_dim: int, path_json: str, alpha: float = 1.0):
        self.arms = arms
        self.num_arms = len(arms)
        self.state_dim = state_dim
        self.path = path_json
        self.alpha = alpha  # Exploration parameter

        # LinUCB parameters: one model per arm
        # A: (d x d) matrix for each arm (covariance)
        # b: (d x 1) vector for each arm (reward sum)
        self.A = [np.identity(state_dim) for _ in range(self.num_arms)]
        self.b = [np.zeros((state_dim, 1)) for _ in range(self.num_arms)]
        self.load()

    def load(self):
        """Loads the learned models for each arm from a file."""
        data = safe_json_read(self.path)
        if data and 'A' in data and 'b' in data:
            try:
                self.A = [np.array(arr) for arr in data['A']]
                self.b = [np.array(arr) for arr in data['b']]
                console.log("📈 [ContextBandit] Loaded learned models.")
            except Exception as e:
                console.log(f"[ContextBandit] Failed to load models, resetting: {e}")

    def save(self):
        """Saves the learned models to a file."""
        data = {
            'A': [arr.tolist() for arr in self.A],
            'b': [arr.tolist() for arr in self.b]
        }
        safe_json_write(self.path, data)

    def pull(self, context: np.ndarray) -> int:
        """
        Pulls an arm based on the current context using the LinUCB formula.
        """
        if context.shape[0] != self.state_dim:
            # Fallback for dimension mismatch
            return random.randrange(self.num_arms)
        
        x = context.reshape((self.state_dim, 1)).astype(float)
        p_t = np.zeros(self.num_arms, dtype=float)

        for i in range(self.num_arms):
            A_inv = np.linalg.inv(self.A[i])
            theta = A_inv @ self.b[i]  # Predicted reward coefficients

            # Predicted reward for this arm given the context
            pred_reward = float((theta.T @ x).squeeze())

            # Exploration bonus based on the model's uncertainty
            uncertainty = float(np.sqrt((x.T @ A_inv @ x).squeeze()))

            # UCB Score = Predicted Reward + Alpha * Uncertainty
            p_t[i] = pred_reward + self.alpha * uncertainty

        # Choose the arm with the highest UCB score
        return int(np.argmax(p_t))

    def update(self, arm_index: int, reward: float, context: np.ndarray):
        """
        Updates the model for the chosen arm with the observed reward and context.
        """
        if context.shape[0] != self.state_dim:
            return  # Cannot update if context is invalid

        x = context.reshape((self.state_dim, 1)).astype(float)
        # Update covariance matrix A
        self.A[arm_index] = self.A[arm_index] + (x @ x.T)

        # Update reward vector b
        rr = float(reward)
        # Ensure shapes are consistent (both (d,1)) and update in place
        inc = (rr * x).reshape(self.state_dim, 1)
        b_i = self.b[arm_index]
        if b_i.shape != (self.state_dim, 1):
            b_i = b_i.reshape(self.state_dim, 1)
        b_i = b_i + inc
        # Cast to Any to appease static analyzers about __setitem__ on ndarray lists
        self.b[arm_index] = cast(Any, b_i)

        # Periodically save the updated models
        if sum(a.trace() for a in self.A) % 20 < 1:
            self.save()

    # HTTP handlers are defined globally below to avoid class scoping/type issues.


class NoOpWorldModel:
    """Graceful fallback when torch is unavailable or the WM isn't ready."""
    def __init__(self):
        self.available = False
        self.ready = False
    async def imagine_with_policy(self, *args, **kwargs):
        return []
    def score_transition(self, state, action):
        return 0.0

if TORCH_AVAILABLE:
    # Create local Any-casted aliases to avoid static type checker ambiguity
    _nn_seq = cast(Any, nn.Sequential)
    _nn_linear = cast(Any, nn.Linear)
    _nn_relu = cast(Any, nn.ReLU)
    _nn_gru = cast(Any, nn.GRU)
    _nn_module = cast(Any, nn)
    _torch_any = cast(Any, torch)
    _F_any = cast(Any, F)
    class TorchStateVAEWorldModel:
        """
        A world model using a Variational Autoencoder (VAE) to learn a compressed
        latent representation of the state space, and an RNN to model state transitions.
        This allows for efficient planning through imagination.
        """
        def __init__(self, input_dim, action_dim, latent_dim=32, rnn_hidden_dim=256):
            self.input_dim = input_dim
            self.action_dim = action_dim
            self.latent_dim = latent_dim
            self.rnn_hidden_dim = rnn_hidden_dim
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.available = True
            # VAE Components
            self.encoder = _nn_seq(
                _nn_linear(input_dim, 128), _nn_relu(),
                _nn_linear(128, 64), _nn_relu()
            ).to(self.device)
            self.fc_mu = _nn_linear(64, latent_dim).to(self.device)
            self.fc_logvar = _nn_linear(64, latent_dim).to(self.device)
            self.decoder = _nn_seq(
                _nn_linear(latent_dim, 64), _nn_relu(),
                _nn_linear(64, 128), _nn_relu(),
                _nn_linear(128, input_dim)
            ).to(self.device)

            # Transition Model (RNN)
            self.rnn = _nn_gru(latent_dim + action_dim, rnn_hidden_dim, batch_first=True).to(self.device)
            self.fc_next_latent = _nn_linear(rnn_hidden_dim, latent_dim).to(self.device)

            # Optimizer
            _params = (
                list(cast(Any, self.encoder).parameters()) + list(cast(Any, self.fc_mu).parameters()) +
                list(cast(Any, self.fc_logvar).parameters()) + list(cast(Any, self.decoder).parameters()) +
                list(cast(Any, self.rnn).parameters()) + list(cast(Any, self.fc_next_latent).parameters())
            )
            self.optimizer = _torch_any.optim.Adam(_params, lr=1e-4)
            self.replay_buffer = deque(maxlen=10000)
            self.ready = False
            console.log(f"🌍 [WorldModel] Initialized with VAE and RNN on device: {self.device}.")

        def _reparameterize(self, mu, logvar):
            std = _torch_any.exp(0.5 * logvar)
            eps = _torch_any.randn_like(std)
            return mu + eps * std

        def _encode(self, state):
            h = self.encoder(state)
            return self.fc_mu(h), self.fc_logvar(h)

        def observe(self, state, action, next_state, reward):
            """Stores a transition in the replay buffer for future training."""
            self.replay_buffer.append((state, action, next_state, reward))
            if len(self.replay_buffer) > 256: # Start training when buffer has enough samples
                self._train()
                if not self.ready:
                    self.ready = True
                    console.log("🌍 [WorldModel] Model is now ready for imagination.")

        def _train(self, batch_size=64):
            """Trains the VAE and transition model on a batch of experiences."""
            if len(self.replay_buffer) < batch_size:
                return

            batch = random.sample(self.replay_buffer, batch_size)
            states, actions, next_states, _ = zip(*batch)

            t = _torch_any
            states = t.tensor(np.array(states), dtype=t.float32).to(self.device)
            actions = t.tensor(np.array(actions), dtype=t.float32).to(self.device)
            next_states = t.tensor(np.array(next_states), dtype=t.float32).to(self.device)

            # 1. Train VAE on states and next_states
            mu, logvar = self._encode(_torch_any.cat([states, next_states]))
            z = self._reparameterize(mu, logvar)
            recon_states = self.decoder(z)
            recon_loss = _F_any.mse_loss(recon_states, _torch_any.cat([states, next_states]))
            kld_loss = -0.5 * _torch_any.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            # 2. Train Transition Model
            with _torch_any.no_grad():
                mu_z, _ = self._encode(states)
                mu_next_z, _ = self._encode(next_states)
            
            rnn_input = _torch_any.cat([mu_z, actions], dim=1).unsqueeze(1)
            rnn_output, _ = self.rnn(rnn_input)
            pred_next_z = self.fc_next_latent(rnn_output.squeeze(1))
            transition_loss = _F_any.mse_loss(pred_next_z, mu_next_z)

            # Total loss and optimization step
            loss = recon_loss + 0.1 * kld_loss + transition_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        def imagine_with_policy(self, start_state, policy, horizon=10):
            """Generates a sequence of imagined future states using the agent's policy."""
            if not self.ready:
                return []
            
            with _torch_any.no_grad():
                start_state_t = _torch_any.tensor(start_state, dtype=_torch_any.float32).to(self.device).unsqueeze(0)
                mu, _ = self._encode(start_state_t)
                
                imagined_latents = []
                current_z = mu
                rnn_hidden = _torch_any.zeros(1, 1, self.rnn_hidden_dim).to(self.device)

                for _ in range(horizon):
                    current_state_recon = self.decoder(current_z).cpu().numpy().flatten()
                    action = policy.select_action(current_state_recon, deterministic=True)
                    action_t = _torch_any.tensor(action, dtype=_torch_any.float32).to(self.device).unsqueeze(0)
                    
                    rnn_input = _torch_any.cat([current_z, action_t], dim=1).unsqueeze(1)
                    rnn_output, rnn_hidden = self.rnn(rnn_input, rnn_hidden)
                    next_z = self.fc_next_latent(rnn_output.squeeze(1))
                    
                    imagined_latents.append(self.decoder(next_z))
                    current_z = next_z

            return imagined_latents

        def is_ready(self):
            return self.ready

        def train_batch(self, traj, recon_w=1.0, kl_w=0.1, trans_w=1.0):
            if not getattr(self, 'available', True):
                return None
            try:
                states, actions, next_states, rewards = [], [], [], []
                for (s,a,sp,r) in traj:
                    states.append(np.array(s, dtype=np.float32))
                    actions.append(np.array(a, dtype=np.float32))
                    next_states.append(np.array(sp, dtype=np.float32))
                    rewards.append(float(r))
                t = _torch_any
                states = t.tensor(np.array(states), dtype=t.float32).to(self.device)
                actions = t.tensor(np.array(actions), dtype=t.float32).to(self.device)
                next_states = t.tensor(np.array(next_states), dtype=t.float32).to(self.device)
                # VAE losses
                mu, logvar = self._encode(_torch_any.cat([states, next_states]))
                z = self._reparameterize(mu, logvar)
                recon_states = self.decoder(z)
                recon_loss = _F_any.mse_loss(recon_states, _torch_any.cat([states, next_states]))
                kld_loss = -0.5 * _torch_any.mean(1 + logvar - mu.pow(2) - logvar.exp())
                # Transition loss
                with _torch_any.no_grad():
                    mu_z, _ = self._encode(states)
                    mu_next_z, _ = self._encode(next_states)
                rnn_input = _torch_any.cat([mu_z, actions], dim=1).unsqueeze(1)
                rnn_output, _ = self.rnn(rnn_input)
                pred_next_z = self.fc_next_latent(rnn_output.squeeze(1))
                transition_loss = _F_any.mse_loss(pred_next_z, mu_next_z)
                # Optimize
                loss = recon_w*recon_loss + kl_w*kld_loss + trans_w*transition_loss
                self.optimizer.zero_grad()
                loss.backward()
                # Clip by summing params explicitly (avoid self.parameters resolution issues)
                _nn_module.utils.clip_grad_norm_(list(cast(Any, self.encoder).parameters()) + list(cast(Any, self.fc_mu).parameters()) +
                                                 list(cast(Any, self.fc_logvar).parameters()) + list(cast(Any, self.decoder).parameters()) +
                                                 list(cast(Any, self.rnn).parameters()) + list(cast(Any, self.fc_next_latent).parameters()), 1.0)
                self.optimizer.step()
                self.ready = True
            except Exception:
                return None
else:
    # Define a lightweight stub so code referring to StateVAEWorldModel can import
    class StateVAEWorldModelStub:
        def __init__(self, *args, **kwargs):
            self.available = False
            self.ready = False
        def observe(self, *args, **kwargs):
            return None
        def _train(self, *args, **kwargs):
            return None
        def imagine_with_policy(self, *args, **kwargs):
            return []
        def is_ready(self):
            return False
        def train_batch(self, *args, **kwargs):
            return None
    # (alias added below)
 # end if TORCH_AVAILABLE
 # Provide a single public alias for the world model class
StateVAEWorldModel = TorchStateVAEWorldModel if TORCH_AVAILABLE else StateVAEWorldModelStub  # type: ignore[name-defined]
class CausalEngine:
    """
    Infers a simplified causal graph from the stream of the mind's experiences.
    It tracks correlations in the changes of state/action variables and reward
    to build a directed graph of potential influences.
    """
    def __init__(self, console_instance):
        if nx is None:
            raise ImportError("networkx is required for the CausalEngine.")
        self.console = console_instance
        self.graph = nx.DiGraph()
        self.correlation_matrix = defaultdict(float)
        self.update_counts = defaultdict(int)
        self.learning_rate = 0.01
        self.last_state_reward = {"reward": 0.0}
        console.log("🔗 [CausalEngine] Initialized with correlation tracker.")

    def update_on_step(self, mind_state: 'E8Mind', action: np.ndarray, reward: float):
        """Updates the causal graph based on the latest transition."""
        current_state_reward = self._get_state_reward_dict(mind_state, action, reward)
        
        # Calculate deltas (changes) from the last step
        deltas = {
            key: current_state_reward.get(key, 0) - self.last_state_reward.get(key, 0)
            for key in current_state_reward
        }
        
        reward_delta = deltas.get("reward", 0)
        
        # If reward changed, update correlations with all other variables that changed
        if abs(reward_delta) > 1e-4:
            for key, delta in deltas.items():
                if key == "reward" or abs(delta) < 1e-4:
                    continue
                
                # Update correlation with EMA: if delta and reward_delta have same sign, increase correlation
                correlation_direction = 1.0 if (delta * reward_delta) > 0 else -1.0
                
                # Update the running correlation
                edge = (key, "reward")
                self.correlation_matrix[edge] = \
                    (1 - self.learning_rate) * self.correlation_matrix[edge] + self.learning_rate * correlation_direction
                self.update_counts[edge] += 1
                
                # If correlation is strong enough, add/update edge in the graph
                if abs(self.correlation_matrix[edge]) > 0.3 and self.update_counts[edge] > 50:
                    self.graph.add_edge(key, "reward", weight=self.correlation_matrix[edge])

        self.last_state_reward = current_state_reward

    def _get_state_reward_dict(self, mind: 'E8Mind', action: np.ndarray, reward: float) -> Dict[str, float]:
        """Flattens the mind's state into a dictionary of variables."""
        data = {"reward": reward}
        data.update({f"mood_{k}": v for k, v in mind.mood.mood_vector.items()})
        data.update({f"action_{i}": v for i, v in enumerate(action)})
        return data
        
    def get_strongest_influences(self, target: str = "reward", k: int = 3):
        """Returns the top k variables believed to influence a target variable."""
        if not self.graph.has_node(target):
            return []
        
        influences = sorted(
            self.graph.in_edges(target, data=True),
            key=lambda edge: abs(edge[2].get('weight', 0)),
            reverse=True
        )
        return influences[:k]

class _NullPE:
    """A fallback potential evaluator that always returns zero. This is its correct, final implementation."""
    def calculate_potential_and_get_reward(self):
        return 0.0

class HierarchicalController:
    """
    A high-level controller for Hierarchical Reinforcement Learning (HRL).
    It sets long-term goals and provides intrinsic rewards to the low-level
    policy for making progress towards them.
    """
    def __init__(self, goal_field: Any, potential_evaluator: Any, console_instance: Any):
        self.console = console_instance
        self.goal_field = goal_field
        self.potential_evaluator = potential_evaluator
        self.update_interval = 100 # How often to choose a new goal
        self.current_goal_name = "synthesis"
        self.current_goal_embedding: Optional[np.ndarray] = None
        self.last_goal_similarity = 0.0
        self.intrinsic_reward_scale = 0.1 # The weight of the bonus reward
        console.log("👑 [HRL] Initialized with goal-directed reward shaping.")

    def maybe_update(self, step: int):
        """Periodically selects a new high-level goal from the GoalField."""
        if step % self.update_interval == 0 and self.goal_field.is_initialized:
            top_goals = self.goal_field.get_top_goals(k=1)
            if top_goals:
                self.current_goal_name, _ = top_goals[0]
                self.current_goal_embedding = self.goal_field.goals[self.current_goal_name]["embedding"]
                self.console.print(Panel(f"New high-level objective set: [bold cyan]{self.current_goal_name.upper()}[/]",
                                         title="[bold yellow]HRL OBJECTIVE[/]", border_style="yellow"))

    def shape_reward(self, state: np.ndarray, next_state: np.ndarray, base_reward: float) -> float:
        """Adds an intrinsic reward based on progress towards the current goal."""
        if self.current_goal_embedding is None or not self.goal_field.is_initialized:
            return base_reward

        # The state vector is a concatenation, we need to extract the part that corresponds to the embedding
        # This is a simplification; a real system might use a dedicated state-to-embedding projection.
        # For now, we assume the goal resonates with the overall state.
        
        # We need a way to compare the state to the goal embedding.
        # Since state is not an embedding, we use the potential evaluator as a proxy.
        # The 'goal_resonance' in the potential evaluator measures alignment with the goal field.
        current_potential = self.potential_evaluator.last_potential # This was calculated for the *previous* step

        # A more direct way is to check the change in potential, which is already the base_reward
        # The base_reward *is* the intrinsic reward for goal progress in this architecture.
        # So we can add a bonus for *consistency* with the top goal.
        
        intrinsic_reward = base_reward * self.intrinsic_reward_scale
        
        return base_reward + intrinsic_reward

 
def bump_temps(memory: 'MemoryManager', node_ids: List[str], amount: float = 0.6):
    """
    Intelligently spikes the temperature of specified nodes. The boost is
    stronger for colder nodes and weaker for already hot nodes to prevent
    runaway feedback loops.
    """
    for nid in node_ids:
        node = memory.graph_db.get_node(nid)
        if node:
            current_temp = node.get('temperature', 0.1)
            # Apply less force as temperature increases (logarithmic scaling)
            boost = amount / (1 + np.log1p(current_temp))
            memory.spike_temperature(nid, boost)

def attach_to_mind(mind: 'E8Mind'):
    """
    Attaches and initializes additional, potentially optional, cognitive modules
    to the main mind instance after its core components are set up. This allows
    for a modular and extensible architecture.
    """
    console.log("🧩 Attaching auxiliary cognitive modules...")
    if getattr(mind, "action_dim", None) is None:
        mind.action_dim = int(getattr(mind, "ACTION_SIZE_NO_LOCK", 8))

    # Initialize the World Model (lazily)
    if getattr(mind, "world_model", None) is None:
        mind.world_model = None
        def _wm_lazy_init(state_dim):
            try:
                wm = StateVAEWorldModel(input_dim=int(state_dim), action_dim=int(mind.action_dim))
            except Exception as e:
                console.log(f"[Attach] World Model init failed: {e}")
                wm = None
            setattr(mind, "world_model", wm)
        setattr(mind, "_wm_lazy_init", _wm_lazy_init)

    # Initialize the Causal Engine
    if getattr(mind, "causal", None) is None:
        mind.causal = CausalEngine(getattr(mind, "console", console))

    # Initialize the Hierarchical Controller
    if getattr(mind, "hrl", None) is None:
        pe = getattr(mind, "potential_evaluator", None) or _NullPE()
        mind.hrl = HierarchicalController(getattr(mind, "goal_field", None), pe, getattr(mind, "console", console))

class E8Mind:
    # Declare commonly attached attributes for static analyzers
    world_model: Any
    causal: Any
    hrl: Any
    memory: Any
    autoencoder: Any
    dimensional_shells: Dict[int, Any]
    physics: Any
    step_num: int
    metrics: Any
    mood: Any
    subconscious: Any
    llm_pool: Any
    console: Any
    probe: Any
    society: Any
    diffusion_proposer: Any
    macro_manager: Any
    agent: Any
    latent_planner: Any
    action_dim: int
    goal_field: Any
    potential_evaluator: Any
    _wm_lazy_init: Any
    sse_clients: Any

    def __getattr__(self, name: str) -> Any:  # dynamic attributes tolerated
        return getattr(self.__dict__, name, None)

    
    def register_proximity_alert(self, distance: float):
        try:
            self.last_proximity_distance = float(distance)
            if hasattr(self, "console") and self.console:
                self.console.log(f"[PROXIMITY] Alert distance set to {self.last_proximity_distance:.4f}")
        except Exception:
            self.last_proximity_distance = float(getattr(self, 'last_proximity_distance', 0.0))

# --- inserted: seed domain if empty ---
    async def seed_domain_if_empty(self):
        """
        If memory is empty, add one bootstrap concept using the semantic domain.
        Controlled by E8_SEED_DOMAIN flag.
        """
        try:
            # Count existing vectors across dimensional shells
            total = 0
            ds = getattr(self, "dimensional_shells", {}) or {}
            for shell in ds.values():
                try:
                    mat, _ = shell.get_all_vectors_as_matrix()
                    total += (len(mat) if mat is not None else 0)
                except Exception:
                    pass
            if total > 0:
                return  # already has content
            if not E8_SEED_DOMAIN:
                return
            label = E8_SEED_LABEL.strip() or f"domain:{getattr(self, 'semantic_domain', 'experience')}"
            entry = {
                "type": "seed",
                "label": label,
                "metaphor": "bootstrap seed",
                "rating": 0.6,
                "step": getattr(self, "step_num", 0),
                "source": "seed"
            }
            try:
                await self.memory.add_entry(entry)
                self.console.log(f"[seed] Inserted initial concept: {label}")
            except Exception as _e:
                self.console.log(f"[seed] Failed to insert seed concept: {_e}")
        except Exception as e:
            try:
                self.console.log(f"[seed] Error during seed check: {e}")
            except Exception:
                pass
    # --- end inserted ---
    def _safe_compute_state_dim(self) -> int:
        """
        Calculates the total dimension of the state vector by summing the
        fixed dimensions of its constituent parts.
        """
        # Mood vector size from its class definition (known keys)
        mood_size = 6

        # Goal vector size (based on the 4 initial goals)
        goal_size = 4

        # Shell attention vector size from its class definition
        shell_att_size = ShellAttention().out_dim

        # Dynamics vector size (as constructed in _build_state_vector)
        dynamics_size = 5

        return mood_size + goal_size + shell_att_size + dynamics_size
    
    def __init__(self, semantic_domain_val, run_id, llm_client_instance, client_model, embedding_model_name, embed_adapter, embed_in_dim, console: Any, is_embed_placeholder: bool):
        self.console = console
        self.run_id = run_id
        self.is_embed_placeholder = is_embed_placeholder

        self.state_dim = self._safe_compute_state_dim()
        self.metrics = MetricsManager(self.run_id)
        self._quantizer_override = None
        self._last_cb_arm_idx = None
        self.snapshot_every = int(os.getenv('E8_SNAPSHOT_EVERY', str(CONSOLE_EXPORT_EVERY_STEPS)))
        self._last_hourly_metrics_ts = time.time()
        self.novelty_weight = 0.35
        self._coh_roll = deque(maxlen=200)

        self.bandit = ContextBandit(
            arms=[
                {"angle_vec": 0.15, "shell_mix": (0.2,0.3,0.5), "k": 24, "diffusion_sigma": 1.2, "hub_penalty": 0.0},
                {"angle_vec": 0.25, "shell_mix": (0.4,0.2,0.4), "k": 32, "diffusion_sigma": 1.5, "hub_penalty": 0.03},
                {"angle_vec": 0.10, "shell_mix": (0.1,0.2,0.7), "k": 40, "diffusion_sigma": 1.0, "hub_penalty": 0.06},
            ],
            state_dim=self.state_dim,
            path_json=get_path("bandit_state.json", self.run_id),
        )
        self._last_cb_reward = 0.0

        self.console.rule(f"[bold cyan]Initializing E8 Mind | Run ID: {run_id}[/]")
        os.makedirs(os.path.join(RUNTIME_DIR, self.run_id), exist_ok=True)

        self.proximity_log_path = get_path("logs/proximity_alerts.ndjson", self.run_id)
        os.makedirs(os.path.dirname(self.proximity_log_path), exist_ok=True)
        self._prox_lock = asyncio.Lock()

        self.market_enabled = bool(int(os.getenv("E8_MARKET_FEED_ENABLED", "0")))
        self.market_symbols = [s.strip().upper() for s in os.getenv("MARKET_SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]
        finnhub_key = os.getenv("FINNHUB_KEY", "")
        self.market = None
        if self.market_enabled and finnhub_key and 'websockets' in globals() and globals()['websockets'] is not None:
            try:
                self.market = MarketFeed(
                    symbols=self.market_symbols,
                    api_key=finnhub_key,
                    on_tick=self._on_market_tick,
                    on_bar=self._on_market_bar,
                )
            except NameError: # websockets might not be defined if import failed
                 self.market = None
        self.market_last: Dict[str, float] = {}
        self.market_last_bar: Dict[Tuple[str, str], Bar] = {}

        self.probe = Probe(run_id)
        set_asyncio_exception_logger(self.probe)

        self.llm_client = llm_client_instance
        self._recent_texts = deque(maxlen=500)
        self._recent_norms = deque(maxlen=500)
        self._anti_repeat_enabled = True

        self.local_llm_client: Optional[OllamaClient] = None
        self.local_llm_model = 'phi3:mini-4k'
        self.client_model = client_model
        self.embedding_model = embedding_model_name
        self.semantic_domain = semantic_domain_val
        self.llm_pool = AsyncLLMPool(self, worker_count=max(1, LOCAL_GEN_WORKERS))
        self.embed_adapter = embed_adapter
        self.embed_in_dim = embed_in_dim

        try:
            profile_name = os.getenv("MIND_PROFILE", "default")
            self.semantics, self.prompts = load_profile(profile_name)
            self.semantic_domain = getattr(self.semantics, "base_domain", self.semantic_domain)
            self.console.log(f"[INIT] Loaded profile: {getattr(self.semantics, 'name', profile_name)}")
        except Exception as e:
            self.console.log(f"[yellow]Profile load failed: {e}. Using defaults.[/yellow]")
            self.semantics, self.prompts = load_profile("default")

        self.console_lock = asyncio.Lock()
        self.insight_cycle_lock = asyncio.Lock()
        self._dream_lock = asyncio.Lock()
        self.teacher_explorer_lock = asyncio.Lock()
        self._teacher_question_context_ids: List[str] = []

        self.physics = E8Physics(self.console)
        self.fabric = E8BoundaryFabric(self.physics)
        self.fabric.layout_2d()
        safe_json_write(self._path("boundary_fabric.json"), self.fabric.to_json())
        self.blueprint = self.physics.generate_quasicrystal_blueprint()
        safe_json_write(self._path("quasicrystal_blueprint.json"), self.blueprint)
        self.blueprint_kdtree = KDTree([[p['x'], p['y']] for p in self.blueprint])

        if TORCH_AVAILABLE:
            import torch as _torch
            self.autoencoder = VariationalAutoencoder(layer_sizes=AUTOENCODER_LAYER_SIZES, console=self.console)
            self.shell_lattices, self.shell_kdtree_indices = {}, {}
            roots_np = np.asarray(self.physics.roots, dtype=np.float32)
            with _torch.no_grad():
                for dim in DIMENSIONAL_SHELL_SIZES:
                    if dim == 8:
                        lifted_vectors_np = roots_np
                    else:
                        projection_matrix = np.random.randn(8, dim)
                        lifted_vectors_np = roots_np @ projection_matrix
                    self.shell_lattices[dim], self.shell_kdtree_indices[dim] = lifted_vectors_np, KDTree(lifted_vectors_np)
            console.log("✅ Lifted E8 reference lattices generated for all dimensional shells.")
        else:
            self.autoencoder = VariationalAutoencoder(layer_sizes=AUTOENCODER_LAYER_SIZES, console=self.console)
            self.shell_lattices, self.shell_kdtree_indices = {}, {}
            console.log("[yellow]PyTorch not found. Autoencoder and shell lattices disabled.[/yellow]")

        self.mood = MoodEngine(self.console)
        self.subconscious = SubconsciousLayer(self.get_embedding, self.llm_pool, self.console)
        self.goal_field = GoalField(self.get_embedding, self.console)
        self.drives = DriveSystem()
        self.dimensional_shells = {dim: DimensionalShell(dim, self) for dim in DIMENSIONAL_SHELL_SIZES}
        self.proximity_engine = ProximityEngine(shell_dims=DIMENSIONAL_SHELL_SIZES, mind_instance=self, console=self.console)
        # In E8Mind.__init__
        self.memory = MemoryManager(self)
        self.novelty_scorer = NoveltyScorer(self.memory, self.llm_pool, self.console)
        self.insight_agent = InsightAgent(self.llm_pool, self.novelty_scorer, self.console)

        self.new_node_id_queue = deque(maxlen=500)
        self.shell_attention = ShellAttention(out_dim=32, keep_k=3)
        self.arbiter_gate = ArbiterGate()
        self.curriculum = AutoTaskManager(self.console)
        self.dream_engine = DreamEngine(self.memory, self)
        self.dream_replay_service = DreamReplayService(self, batch=int(os.getenv('E8_REPLAY_BATCH','32')), steps=int(os.getenv('E8_REPLAY_STEPS','20')))
        self.narrative_streamer = NarrativeStreamer(self.memory, self.llm_pool, self.run_id)
        self.synthetic_env = SyntheticEnvironment(self.llm_pool, self)
        self.domain_tint = DomainTintEngine(self.semantic_domain, self.llm_pool)
        self.validator = HypothesisValidator(self)
        self.ingestion_pipeline = DataIngestionPipeline(self)
        self.scheduler = CognitiveScheduler(self)
        self.potential_evaluator = StatePotentialEvaluator(self.dimensional_shells, self.goal_field)
        
        # --- FIX: Moved attach_to_mind() to after its dependencies are initialized ---
        try:
            attach_to_mind(self)
        except Exception as e:
            self.console.log(f"[bold red]Error attaching auxiliary modules: {e}[/bold red]")

        self.action_dim = ACTION_SIZE_NO_LOCK
        self.max_action = 0.1

        self._bh_window = deque(maxlen=50)
        self._bh_recent = deque(maxlen=100)
        self._bh_ma50 = 0.0
        self._prev_bh = 0.0
        self._low_bh_streak = 0
        self._prev_action = np.zeros(self.action_dim, dtype=np.float32)

        if TORCH_AVAILABLE:
            self.agent = SACMPOAgent(self.state_dim, self.action_dim, self.max_action, console=self.console, tau=0.002, use_per=True)
        # Society of Mind
        try:
            if os.getenv('E8_SOCIETY_ENABLED','1')=='1':
                self.society = SocietyOfMind(self, beta=float(os.getenv('E8_SOCIETY_BETA','3.0')), K=int(os.getenv('E8_SOCIETY_CANDIDATES','12')))
            else:
                self.society = None
        except Exception:
            self.society = None
            try:
                self.diffusion_proposer = LatentDiffusionProposer(self.action_dim, horizon=8, samples=16)
            except Exception:
                self.diffusion_proposer = None
            try:
                self.macro_manager = MacroManager(ACTION_LAYOUT, self.action_dim, pick_every=20)
            except Exception:
                self.macro_manager = None
        else:
            self.agent = None
            self.diffusion_proposer = None
            self.macro_manager = None

        try:
            # Accept both (layout, action_dim, ...) and (action_dim, max_action, ...) signatures in stub
            self.latent_planner = LatentCEMPlanner(ACTION_LAYOUT, self.action_dim, angle_scale=self.max_action, pop=64, elites=8, iters=3, horizon=8, sigma=0.06)
        except Exception:
            self.latent_planner = None

        self.qeng = QuantumEngine(self.physics, QuantumConfig(seed=GLOBAL_SEED), self.console)
        self.ceng = ClassicalEngine(self.physics, ClassicalConfig(seed=GLOBAL_SEED), self.console)
        self.anchors = MultiAnchorField(self.physics)
        
        self.wavey_bridge = WaveyE8Bridge(embed_dim=EMBED_DIM, seed=GLOBAL_SEED)
        self._wavey_bias_last = None
        self.valence = SymmetryValenceEngine(self.physics)
        self.ego_gate = EgoGate(self.valence, min_delta=-0.01)
        self.holo, self.emap, self.slice_stack = HoloEncoder(self.fabric, feat_dim=8), EntropyMap(self.fabric), SliceStack()
        self.judge, self.adapter, self.decoder = Judge(self.llm_client, self.get_embedding, self.emap, self.holo), OnlineAdapter(), ConstrainedDecoder(self.llm_client, self.fabric)
        self.topology_monitor = TopologyMonitor()

        self.step_num, self.max_steps, self.trace = 0, 0, []
        self.prev_node_index: Optional[int] = None
        self.visit = np.zeros(self.physics.roots.shape[0], dtype=np.int32)
        self.ego_summary, self.teacher_question, self.explorer_last_answer, self.last_teacher_question = "Nascent state.", None, "", ""
        self.current_task_embedding = np.zeros(EMBED_DIM)
        self.gravitational_lock_target: Optional[Tuple[str, str]] = None
        self.teacher_log, self.explorer_log, self.subconscious_event_log, self.black_hole_log = [], [], [], []
        self.black_hole_pressure, self._bh_cooldown_until, self._bh_inflight = 0.0, -1, False
        self._last_dream_at, self._last_dream_seed_hash, self._last_dream_step = 0.0, None, -1
        self._progress_lock, self._last_progress_step, self.sigma_q, self.last_policy_state = asyncio.Lock(), -1, 1.25, {}
        self.bardo_until, self._last_region = -1, None
        console.log("[bold green]✅ E8 Mind initialization complete. Ready for cognitive cycle.[/bold green]")

    def apply_manifold_action(self, action_vec):
        try:
            for lay in ACTION_LAYOUT:
                dim = lay["dim"]
                b0, blen, ai = lay["biv_start"], lay["biv_len"], lay["angle_idx"]
                bcoef = action_vec[b0:b0+blen]
                ang   = action_vec[ai] if ai < len(action_vec) else 0.0
                shell = self.dimensional_shells.get(dim)
                if shell is not None and hasattr(shell, "spin_with_bivector"):
                    shell.spin_with_bivector(bcoef, float(ang))

                    if hasattr(self, "proximity_engine") and hasattr(self.proximity_engine, "update_shell_index"):
                        self.proximity_engine.update_shell_index(shell.dim, shell)
                    try:
                        if hasattr(self, 'macro_manager') and self.macro_manager is not None:
                            self.macro_manager.on_action_executed(action_vec)
                    except Exception:
                        pass
        except Exception as e:
            self.console.log(f"[bold red]Error in apply_manifold_action: {e}[/bold red]")
            pass

        
    def _snap_to_lattice(self, vector: np.ndarray, dim: int) -> np.ndarray:
        # Quantizer switch: E8 (default via KD snap), cubic/random/none via env
        try:
            q = (os.getenv("E8_QUANTIZER", "e8") or "e8").lower()
            cell = float(os.getenv("E8_CELL", "0.10"))
        except Exception:
            q, cell = "e8", 0.10
        v = vector.astype(np.float32, copy=False)
        if q == "none":
            return v
        if q == "cubic":
            return (np.round(v / cell) * cell).astype(np.float32)
        if q == "random":
            try:
                seed = int(os.getenv("GLOBAL_SEED", "1337"))
            except Exception:
                seed = 1337
            rng = np.random.default_rng(seed)
            jitter = rng.normal(0.0, cell, size=v.shape).astype(np.float32)
            w = v + jitter
            return (np.round(w / cell) * cell).astype(np.float32)
        
        # default 'e8': use existing KDTree-based snap if available
        kdtree = self.shell_kdtree_indices.get(dim)
        if kdtree is None:
            return v # Fallback if no tree exists for this dimension
        
        _, nearest_index_arr = kdtree.query(vector.reshape(1, -1), k=1)
        
        try:
            # FIX: Use .item() to robustly extract the scalar index from the nested array.
            # This prevents errors if the result shape is, for example, [[42]].
            scalar_index = nearest_index_arr.item()
            return self.shell_lattices[dim][scalar_index]
        except (ValueError, IndexError):
            # Fallback if the index array is empty or malformed.
            return vector

    async def _teacher_ask_new_question(self):
        # --- HOTFIX INTEGRATED: Teacher Gating ---
        min_for_teacher = int(os.getenv("E8_MIN_FOR_TEACHER", "30"))
        current_nodes = self.memory.graph_db.graph.number_of_nodes()
        if current_nodes < min_for_teacher:
            return # Silently skip if not enough concepts
        

        async with self.teacher_explorer_lock:
                try:
                    frontier_insights = []
                    G = self.memory.graph_db.graph
                    for node_id, data in G.nodes(data=True):
                        if data.get("type") == "explorer_insight" and not any(G.get_edge_data(node_id, n, {}).get("type") == "reflection_source" for n in G.neighbors(node_id)):
                            frontier_insights.append((node_id, data))
                    frontier_insights.sort(key=lambda x: x[1].get("step", 0), reverse=True)
                    top_goal_name, top_goal_desc = self.goal_field.get_top_goals(k=1)[0]
                    self._teacher_question_context_ids = []
                    if len(frontier_insights) > 1 and random.random() > 0.4:
                        id_A, data_A = frontier_insights[1]; id_B, data_B = frontier_insights[0]
                        self._teacher_question_context_ids = [id_A, id_B]
                        # AFTER
                        prompt = (f"Goal: '{top_goal_desc}'.\nInsight A: '{data_A.get('metaphor', '')}'\nInsight B: '{data_B.get('metaphor', '')}'.\n\n"
                                "Ask one concise question (under 20 words) about the connection between A and B. "
                                "The question MUST include key terms from either Insight A or B.")
                    else:
                        recent_nodes_data = [d for _, d in self.memory.graph_db.graph.nodes(data=True) if not d.get("folded")]
                        memory_snippet = "\n".join(f"- {n.get('label','')}: {n.get('metaphor','')}" for n in recent_nodes_data[-4:])
                        prompt = (f"Goal: '{top_goal_desc}'.\nRecent thoughts:\n{memory_snippet}\n\nAsk one profound, short question (under 20 words) to advance the goal. "
                                "The question MUST include at least one key term from the 'Recent thoughts' provided.")
                    
                    question = await asyncio.wait_for(self.llm_pool.enqueue_and_wait(prompt, max_tokens=40, temperature=0.75), timeout=TEACHER_STEP_TIMEOUT)

                    # --- HOTFIX INTEGRATED: Teacher Question Filtering ---
                  
                    # CORRECTED LOGIC: Build a comprehensive set of terms from both labels and metaphors
                    all_graph_text = " ".join(
                        f"{d.get('label', '')} {d.get('metaphor', '')}"
                        for _, d in self.memory.graph_db.graph.nodes(data=True)
                    )
                    graph_terms = set(re.findall(r"[A-Za-z0-9]+", all_graph_text.lower()))

                    if not teacher_prompt_ok(question, graph_terms):
                        self.console.log(f"[TEACHER] Question '{question}' rejected: Lacks relevance to memory graph.")
                        self.teacher_question = None
                        return
                    # --- END HOTFIX ---

                    self.teacher_question = str(question).strip().replace('"', '')
                    self.teacher_log.append({"step": self.step_num, "q": self.teacher_question})
                    if self.teacher_question: self.current_task_embedding = await self.get_embedding(self.teacher_question)
                    async with self.console_lock:
                        self.console.print(Panel.fit(f"[bold white]{sanitize_block(self.teacher_question, 2, 240)}[/]", title="[bold cyan]TEACHER[/]", border_style="cyan"))
                except Exception as e:
                    self.console.log(f"[TEACHER] skipped (error): {e}")
                    self.teacher_question = None

    async def _explorer_answer_pending_question(self):
        async with self.teacher_explorer_lock:
                q = getattr(self, "teacher_question", None)
                if not q: return
                
                # --- HOTFIX INTEGRATED: Explorer Robustness ---
                answer = "[Explorer] (error)" # Default answer on failure
                try:
                    answer_prompt = f"You are the Explorer. Answer the Teacher's question plainly. Max 4 sentences. Be concrete.\n\nQuestion:\n{q}\n\nAnswer:"
                    raw_answer = str(await asyncio.wait_for(self.llm_pool.enqueue_and_wait(answer_prompt, max_tokens=150, temperature=0.8), timeout=EXPLORER_STEP_TIMEOUT)).strip()

                    # Ensure answer is valid and non-empty
                    if not raw_answer or raw_answer.startswith("[LLM"):
                        answer = "[Explorer] (no answer)"
                    else:
                        answer = raw_answer
                    
                    # Original logic proceeds with the sanitized 'answer'
                    label = "Explorer answer"
                    if not answer.startswith("[Explorer]"):
                        try:
                            label_resp = await asyncio.wait_for(self.llm_pool.enqueue_and_wait(f"Summarize in 3-6 words:\n{answer}", max_tokens=12, temperature=0.2), timeout=max(3.0, EXPLORER_STEP_TIMEOUT/2.0))
                            if label_resp and not label_resp.startswith("[LLM"):
                                label = str(label_resp).strip()
                        except Exception as e: self.console.log(f"[EXPLORER] label timeout: {e}")
                    
                    safe = rich_escape(label)
                    async with self.console_lock:
                        self.console.print(Panel.fit(f"[bold white]{sanitize_block(answer, 2, 240)}[/]", title=f"[bold green]EXPLORER[/] · {safe[:42]}{'…' if len(safe)>42 else ''}", border_style="green"))

                    # Only add to memory if it was a valid response
                    if not answer.startswith("[Explorer]"):
                        rating = await self.rate_concept(f"{label}: {answer}")
                        new_node_id = await self.memory.add_entry({"type": "explorer_insight", "label": label, "metaphor": answer, "rating": rating, "step": self.step_num}, parent_ids=self._teacher_question_context_ids)
                        await self._append_insight_log({
                            "run_id": getattr(self, "run_id", None),
                            "step": int(self.step_num),
                            "type": "explorer_insight",
                            "node_id": new_node_id,
                            "label": label,
                            "content": answer,
                            "rating": float(rating),
                            "question": q,
                            "parent_ids": list(getattr(self, "_teacher_question_context_ids", []) or []),
                        })
                        self.explorer_last_answer = answer
                        self.last_teacher_question = q
                        self.subconscious_event_log.append({'type': 'teacher_explorer', 'step': self.step_num, 'data': {'q': q, 'a': answer}})
                        self.drives.reward("curiosity", 0.15)

                except Exception as e:
                    self.console.log(f"[EXPLORER] skipped (error): {e}")
                    answer = f"[Explorer] (error: {str(e)[:100]})"
                finally:
                    # Log the final outcome, whatever it is
                    self.console.log(f"[Explorer Final Outcome] {answer}")
                    self._teacher_question_context_ids, self.teacher_question = [], None

    def _on_market_tick(self, symbol: str, tick: dict):

        self.market_last[symbol] = tick.get("p", 0.0)

    def _on_market_bar(self, symbol: str, timeframe: str, bar: Bar):

        self.market_last_bar[(symbol, timeframe)] = bar

    async def critique_and_refine(self, thought: str, goal_desc: str) -> str:
        """
        Critiques a thought against a goal and refines it.
        This is a practical application of Constitutional AI principles.
        """
        if not thought or not goal_desc:
            return thought

        try:

            critique_prompt = self.prompts.render(
                "critique_thought",
                thought=thought,
                goal=goal_desc
            )

            critique = await self.llm_pool.enqueue_and_wait(
                critique_prompt, max_tokens=150, temperature=0.4
            )

            if critique and "[NO CHANGE]" in critique:
                return thought

            refine_prompt = self.prompts.render(
                "refine_thought",
                thought=thought,
                critique=critique,
                goal=goal_desc
            )

            refined_thought = await self.llm_pool.enqueue_and_wait(
                refine_prompt, max_tokens=150, temperature=0.7
            )

            if refined_thought and not refined_thought.startswith("[LLM"):
                async with self.console_lock:
                    self.console.print(Panel(f"[dim]Original:[/] {thought}\n[bold]Refined:[/] {refined_thought}",
                        title="[bold yellow]SELF-CRITIQUE[/]", border_style="yellow"))
                return refined_thought.strip()

        except Exception as e:
            self.console.log(f"[Self-Critique] Error during refinement: {e}")

        return thought

    async def _append_proximity_log(self, record: dict):
        try:
            rec = dict(record)
            try:
                if datetime is not None and timezone is not None:
                    rec["ts"] = datetime.now(timezone.utc).isoformat()
                else:
                    from datetime import datetime as _dt
                    rec["ts"] = _dt.utcnow().isoformat()
            except Exception:
                from datetime import datetime as _dt
                rec["ts"] = _dt.utcnow().isoformat()
            line = json.dumps(rec, ensure_ascii=False)
            async with self._prox_lock:
                with open(self.proximity_log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception as e:
            self.console.log(f"[PROX-LOG] write failed: {e}")

    async def _run_insight_cycle(self):
        if self.insight_cycle_lock.locked():
            return
        async with self.insight_cycle_lock:
            source_dim, target_dim = random.sample(DIMENSIONAL_SHELL_SIZES, 2)
            source_shell = self.dimensional_shells[source_dim]
            if not source_shell.vectors:
                return
            random_node_id = random.choice(list(source_shell.vectors.keys()))
            query_vector = source_shell.get_vector(random_node_id)
            if query_vector is None:
                return
            results = self.proximity_engine.cross_dimensional_query(query_vector, source_dim, target_dim, k=1)
            if not results:
                return
            connected_node_id, distance = results[0]
            if distance < 0.5:
                self.gravitational_lock_target = (random_node_id, connected_node_id)

            A = self.memory.graph_db.get_node(random_node_id) or {}
            B = self.memory.graph_db.get_node(connected_node_id) or {}
            a_label = sanitize_line(A.get("label") or random_node_id, 60)
            b_label = sanitize_line(B.get("label") or connected_node_id, 60)
            a_meta = sanitize_line(A.get("metaphor") or "", 160)
            b_meta = sanitize_line(B.get("metaphor") or "", 160)

            hypothesis = ""
            try:
                prompt = (
                    "Write one short, plain sentence (≤24 words) that explains a possible connection between A and B"
                    "Avoid hype. Be concrete."
                    f"A (title): {a_label}\n"
                    f"A (content): {a_meta}\n"
                    f"B (title): {b_label}\n"
                    f"B (content): {b_meta}\n"
                    "Sentence:"
                )
                resp = await asyncio.wait_for(self.llm_pool.enqueue_and_wait(prompt, max_tokens=60, temperature=0.6), timeout=8)
                if isinstance(resp, str) and not resp.startswith("[LLM"):
                    hypothesis = sanitize_line(resp, 180)
            except Exception:
                pass
            if not hypothesis:
                hypothesis = f"Possible link: {a_label} ↔ {b_label}."

            await self._append_proximity_log({
                "step": int(self.step_num),
                "source_dim": int(source_dim),
                "target_dim": int(target_dim),
                "source_id": random_node_id,
                "target_id": connected_node_id,
                "source_label": a_label,
                "source": {"name": a_label},
                "target_label": b_label,
                "target": {"name": b_label},
                "distance": float(distance),
                "hypothesis": hypothesis
            })
        async with self.console_lock:
            self.console.print(Panel(
                f"Source: [cyan]{a_label}[/] ({source_dim}D) · id={random_node_id}\n"
                f"Target: [green]{b_label}[/] ({target_dim}D) · id={connected_node_id}\n"
                f"Distance: [yellow]{distance:.4f}[/]\n"
                f"[dim]Hypothesis:[/] {hypothesis}",
                title="[bold magenta]PROXIMITY ALERT[/]", border_style="magenta"
            ))
            
    def _ensure_insight_log_state(self):
        if not hasattr(self, "_insight_log_inited"):
            path = get_path("logs/insights.ndjson", getattr(self, "run_id", "run"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.insight_log_path = path
            self._insight_lock = asyncio.Lock()
            self._insight_log_inited = True

    async def _append_insight_log(self, record: dict):
        self._ensure_insight_log_state()
        try:
            try:
                if datetime is not None and timezone is not None:
                    ts = datetime.now(timezone.utc).isoformat()
                else:
                    from datetime import datetime as _dt
                    ts = _dt.utcnow().isoformat()
            except Exception:
                from datetime import datetime as _dt
                ts = _dt.utcnow().isoformat()
            line = json.dumps({**record, "ts": ts}, ensure_ascii=False)
            async with self._insight_lock:
                with open(self.insight_log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception as e:
            self.console.log(f"[INSIGHT-LOG] write failed: {e}")

    async def _generate_subconscious_narrative(self):
        all_events = self.subconscious_event_log + self.black_hole_log
        self.subconscious_event_log.clear()
        self.black_hole_log.clear()
        if not all_events:
            return
        all_events.sort(key=lambda x: x.get('step', self.step_num))
        await self.subconscious.generate_narrative_summary(all_events)

    async def _generate_internal_monologue_step(self, step_num, current_node_index, prev_node_index):
        if prev_node_index is None:
            return
        try:
            delta = self.physics.roots[current_node_index] - self.physics.roots[prev_node_index]
            themes = classify_geometry_theme(delta)

            theme_str = themes[0] if themes else "stillness"
            prompt = (f"You are the mind's inner voice, verbalizing a single, fleeting moment of thought. "
                        f"Your current subconscious narrative is: \"{self.subconscious.narrative}\"\n"
                        f"Your mood is: {self.mood.describe()}\n"
                        f"The physical sensation of this thought was a movement of '{theme_str}'.\n\n"
                        "Describe this single, instantaneous event in a single, short, first-person sentence.")
            thought_sentence = await self.llm_pool.enqueue_and_wait(prompt, max_tokens=60, temperature=0.4)
            if thought_sentence and not thought_sentence.startswith("[LLM"):
                async with self.console_lock:
                    self.console.print(Panel(f"[italic white]{thought_sentence}[/]", title=f"[bold #A9A9A9]Inner Monologue | Step {step_num}[/]", border_style="#A9A9A9"))
                if random.random() < 0.1:
                    rating = await self.rate_concept(thought_sentence)
                    if rating > 0.6:
                        await self.memory.add_entry({"type": "monologue_thought", "label": sanitize_line(thought_sentence, 25), "metaphor": thought_sentence,
                                                        "rating": rating, "step": step_num})
        except Exception as e:
            async with self.console_lock:
                self.console.log(f"[Monologue Error] Step {step_num} failed: {e}")

    async def _generate_phase_summary(self):
        """Generates a summary of the most recent phase of thought."""
        try:
            recent_nodes = [
                d.get('label', 'untitled')
                for _, d in self.memory.graph_db.graph.nodes(data=True)
                if isinstance(d, dict) and not d.get("folded") and d.get('type') == 'concept'
            ][-10:]

            if len(recent_nodes) < 3:
                return

            prompt = (f"Concepts explored recently: {', '.join(recent_nodes)}. "
                      f"Synthesize these into a one-sentence summary of this phase of thought.")
            
            summary = await self.llm_pool.enqueue_and_wait(prompt, max_tokens=150, temperature=0.6)
            if not summary or summary.startswith("[LLM"):
                return

            label_prompt = f'Create a 3-4 word title for this summary: "{summary}"'
            label = await self.llm_pool.enqueue_and_wait(label_prompt, max_tokens=15, temperature=0.5)
            if not label or label.startswith("[LLM"):
                return

            await self.memory.add_entry({
                "label": label.strip().replace('"', ''),
                "type": "phase_summary",
                "metaphor": summary,
                "rating": 0.8,
                "step": self.step_num
            })
            self.console.print(Panel(
                f"New summary created: '{label}'",
                title="[bold orange]PHASE[/bold orange]",
                border_style="dark_orange"
            ))
        except Exception as e:
            self.console.log(f"[bold red]Failed to generate phase summary: {e}[/bold red]")

    async def _generate_meta_reflection(self):
        """Reflects on recent internal monologues to find higher-level insights."""
        try:
            refl_file = self._path("reflections.txt")
            if not os.path.exists(refl_file):
                return

            with open(refl_file, "r", encoding="utf-8") as f:
                content = f.read()

            recent_egos = re.findall(r"--- Step \d+ ---\n(.*?)(?=\n--- Step|\Z)", content, re.DOTALL)[-5:]
            if not recent_egos:
                return

            prompt = ("Reflect on these recent internal monologues:\n" +
                      "\n".join(f"- {e.strip()}" for e in recent_egos) +
                      "\n\nWhat pattern or concept emerges? Synthesize a new, higher-level insight.")

            reflection = await self.llm_pool.enqueue_and_wait(prompt, temperature=0.8, max_tokens=250)
            if not reflection or reflection.startswith("[LLM"):
                return

            label_prompt = f'Summarize this insight in a 3-5 word title: "{reflection}"'
            label = await self.llm_pool.enqueue_and_wait(label_prompt, max_tokens=20)
            if not label or label.startswith("[LLM"):
                return

            await self.memory.add_entry({
                "label": label.strip().replace('"', ''),
                "type": "meta_reflection",
                "metaphor": reflection,
                "rating": 0.85,
                "step": self.step_num
            })
            self.mood.process_event("reflection")
            self.console.print(Panel(
                f"Meta-reflection '{label}' added.",
                title="[bold white]META[/bold white]",
                border_style="white"
            ))
        except Exception as e:
            self.console.log(f"[bold red]Meta-reflection failed: {e}[/bold red]")

    
    # perform_retro_relink consolidated later; duplicate removed here.

    async def _run_proactive_insight_synthesis(self):
        min_needed = int(os.getenv("E8_MIN_CONCEPTS_FOR_SYNTH", "0"))
        current_nodes = self.memory.graph_db.graph.number_of_nodes()
        if current_nodes < min_needed:
            self.console.log(f"[InsightAgent] Synthesis skipped: Not enough concepts ({current_nodes}/{min_needed}).")
            return
       
        async with self.insight_cycle_lock:
            hot_nodes = sorted(
                [(nid, data) for nid, data in self.memory.graph_db.graph.nodes(data=True)
                 if not data.get("folded") and data.get("type") in self.dream_engine.ALLOWED_TYPES],
                key=lambda item: item[1].get("temperature", 0.0), reverse=True
            )
            # Fallback 1: use DreamEngine’s eligible set (ensures embeddings exist)
            if len(hot_nodes) < 2:
                try:
                    elig = self.dream_engine._eligible_concepts()
                except Exception:
                    elig = []
                hot_nodes = sorted(list(elig), key=lambda item: item[1].get("temperature", 0.0), reverse=True)
            # Fallback 2: relax type filter; take any two recent, non-folded nodes
            if len(hot_nodes) < 2:
                any_nodes = sorted(
                    [(nid, data) for nid, data in self.memory.graph_db.graph.nodes(data=True)
                     if not data.get("folded")],
                    key=lambda item: (item[1].get("temperature", 0.0), item[1].get("step", 0)),
                    reverse=True
                )
                if len(any_nodes) >= 2:
                    hot_nodes = any_nodes[:2]
                else:
                    self.console.log("[InsightAgent] Not enough source concepts for synthesis.")
                    return
            parent_a_id, parent_a_data = hot_nodes[0]
            parent_b_id, parent_b_data = hot_nodes[1]
            new_concept_text = await self.insight_agent.create_hybrid_concept(parent_a_data, parent_b_data)
            if not new_concept_text or new_concept_text.startswith("[LLM"):
                return
            
            parent_a_label = parent_a_data.get('label', 'A')
            parent_b_label = parent_b_data.get('label', 'B')
            
            # Use a regular expression to strip any number of "Synthesis: " prefixes from the start.
            cleaned_a = re.sub(r'^(Synthesis:\s*)+', '', parent_a_label).strip()
            cleaned_b = re.sub(r'^(Synthesis:\s*)+', '', parent_b_label).strip()
            
            # M17: coherence floor + cheap resampling
            novelty_vec = await self.get_embedding(new_concept_text)

            coh = await self.novelty_scorer.calculate_coherence(new_concept_text)
            if coh < float(os.getenv("E8_COHERENCE_FLOOR", "0.45")):
                # Two low-T variants; pick best coherence
                cands = [new_concept_text]
                for _ in range(int(os.getenv("E8_RETRIES", "2"))):
                    alt = await self.insight_agent.create_hybrid_concept(parent_a_data, parent_b_data)
                    if alt and not alt.startswith("[LLM"):
                        cands.append(alt)
                scored = []
                for t in cands:
                    c = await self.novelty_scorer.calculate_coherence(t)
                    scored.append((t, c))
                new_concept_text, coh = max(scored, key=lambda x: x[1])
                if coh < float(os.getenv("E8_COHERENCE_FLOOR", "0.45")):
                    self.console.log(f"[bold yellow]Synthesis rejected: Coherence too low ({coh:.2f}) after retries.[/bold yellow]")
                    return

            nov = self.novelty_scorer.calculate_novelty(novelty_vec)

            # M17: reward shaping
            def _tokens(s: str) -> set: return set(re.findall(r"[A-Za-z0-9]+", (s or "").lower()))
            def _jaccard(a: set, b: set) -> float:
                if not a and not b: return 1.0
                return len(a & b) / max(1, len(a | b))
            def _perplexity_stub(s: str) -> float:
                w = max(1, len(s.split())); avg_len = sum(len(t) for t in s.split()) / w
                return 25.0 + 3.0 * (avg_len < 3) + 6.0 * (avg_len > 10)

            tau_ppl, beta = float(os.getenv("E8_PPL_THRESH", "35.0")), 0.01
            grammar_pen = max(0.0, _perplexity_stub(new_concept_text) - tau_ppl) * beta

            pa = (parent_a_data.get("label", "") + " " + parent_a_data.get("metaphor", "")); pb = (parent_b_data.get("label", "") + " " + parent_b_data.get("metaphor", ""))
            dup_pen = 0.1 if _jaccard(_tokens(new_concept_text), _tokens(pa + " " + pb)) > 0.8 else 0.0
            nw = float(getattr(self, "novelty_weight", float(os.getenv("E8_REWARD_ALPHA", "0.35")))); nw = float(np.clip(nw, 0.1, 0.9)); cw = 1.0 - nw
            reward = nw*nov + cw*coh - grammar_pen - dup_pen
            final_rating = 0.35*nov + 0.65*coh

            novelty_score, coherence_score = float(nov), float(coh)
            self.insight_agent.learn_from_reward(float(reward), episode_data={"type": "synthesis", "parents": [parent_a_id, parent_b_id], "text": new_concept_text})

            new_entry = {
                "type": "insight_synthesis", 
                "label": sanitize_line(f"Synthesis: {cleaned_a} + {cleaned_b}"),
                "metaphor": new_concept_text, "rating": final_rating, "step": self.step_num
            }
            new_entry["coherence"] = float(coherence_score)

            parent_vec_a = self.memory.main_vectors.get(parent_a_id); parent_vec_b = self.memory.main_vectors.get(parent_b_id)
            processed_vec = self.memory.hopfield.clean_up(novelty_vec)
            if parent_vec_a is not None and parent_vec_b is not None:
                parentage_vec = self.memory.vsa.encode_parentage(parent_vec_a, parent_vec_b)
                processed_vec = normalize_vector(0.8 * processed_vec + 0.2 * parentage_vec)
            new_entry["embedding"] = processed_vec

            if not self.memory.reranker.validate(processed_vec, [parent_a_id, parent_b_id], nov, coh): return

            new_node_id = await self.memory.add_entry(new_entry, parent_ids=[parent_a_id, parent_b_id])

            try:
                for pid in [parent_a_id, parent_b_id]:
                    try:
                        if hasattr(self.memory, "_allow_edge") and self.memory._allow_edge(pid, new_node_id):
                            self.memory.graph_db.add_edge(pid, new_node_id, kind="parent", weight=1.0)
                        else: self.memory.graph_db.add_edge(pid, new_node_id, kind="parent", weight=1.0)
                    except Exception: pass
                for _nid in [new_node_id, parent_a_id, parent_b_id]:
                    try: self.memory._trim_degree(_nid, max_deg=8)
                    except Exception: pass
            except Exception: pass

            await self._append_insight_log({
                "run_id": getattr(self, "run_id", None), "step": int(self.step_num), "type": "insight_synthesis",
                "node_id": new_node_id, "label": new_entry["label"], "content": new_concept_text,
                "rating": float(final_rating), "novelty": float(novelty_score), "coherence": float(coherence_score),
                "parent_ids": [parent_a_id, parent_b_id],
            })
            self.subconscious_event_log.append({'type': 'insight_synthesis', 'label': new_entry['label'], 'step': self.step_num})
            self.console.print(Panel(f"[bold]New Synthesis:[/bold] {new_concept_text}\n[yellow]Novelty:[/] {novelty_score:.2f} | [cyan]Coherence:[/] {coherence_score:.2f} | [green]Reward:[/] {reward:.2f}",
                                        title="[bold blue]INSIGHT SYNTHESIS[/]", border_style="blue"))

            min_rating = float(os.getenv('E8_VALIDATOR_MIN_RATING', '0.6'))
            if final_rating >= min_rating: asyncio.create_task(self.validator.validate_insight(new_node_id))

    

    def _build_telemetry_snapshot(self) -> dict:
        shells_data, shell_tensions = {}, {}
        for dim, shell in self.dimensional_shells.items():
            ori = getattr(shell, 'orientation', None)
            orientation_value = None
            try:
                if CLIFFORD_AVAILABLE and hasattr(ori, 'value'):
                    orientation_value = getattr(ori, 'value')
                elif isinstance(ori, (int, float)):
                    orientation_value = float(ori)
            except Exception:
                orientation_value = None
            shells_data[dim] = {"orientation": orientation_value}

            matrix, _ = shell.get_all_vectors_as_matrix()
            if matrix is not None and matrix.shape[0] > 1:
                dists = np.linalg.norm(matrix - matrix.mean(axis=0, keepdims=True), axis=1)
                shell_tensions[dim] = float(dists.mean())
            else:
                shell_tensions[dim] = 0.0

        rh = getattr(self.insight_agent, "reward_history", None)
        insight_reward_avg = float(np.mean(list(rh))) if rh and len(rh) > 0 else 0.0
        step = int(self.step_num)

        def _steps_to_next(current_step, every, offset):
            if every <= 0: return 0
            if current_step < offset: return offset - current_step
            mod = (current_step - offset) % every
            return 0 if mod == 0 else every - mod

        ingestion_feed_data = []
        validation_lab_data = []
        recent_nodes = list(self.memory.graph_db.graph.nodes(data=True))[-50:]
        
        for node_id, data in recent_nodes:
            if data.get("type") == "external_concept":
                ingestion_feed_data.append(f"[{data.get('source', 'ext')}] {data.get('label', '...')}")
            
            if "validation_status" in data:
                status = data["validation_status"]
                v_type = status.get('type', 'unknown').replace('_', ' ').title()
                validation_lab_data.append(f"'{data.get('label', '...')}' → {v_type}")

        insight_holocron_data = [f"BH Event: Consolidated {d.get('size', 0)} concepts (Mass: {d.get('mass', 0):.2f})" for d in self.black_hole_log]
        for _, data in recent_nodes:
            if data.get("type") == "insight_synthesis":
                insight_holocron_data.append(f"Synthesis: {data.get('label', '...')}")
        
        if not hasattr(self, '_last_node_count'):
            self._last_node_count = 0
        
        all_nodes_with_data = list(self.memory.graph_db.graph.nodes(data=True))
        new_node_count = len(all_nodes_with_data)
        # --- This entire block replaces the original logic for new_memory_nodes_data ---
        new_memory_nodes_data = []
        if hasattr(self, 'new_node_id_queue'):
            while self.new_node_id_queue:
                node_id = self.new_node_id_queue.popleft()
                data = self.memory.graph_db.get_node(node_id)
                if data and 'blueprint_location_id' in data and data['blueprint_location_id'] is not None:
                    node_info = data.copy()
                    node_info['id'] = node_id
                    # Ensure numpy arrays are converted for JSON serialization
                    if 'embedding' in node_info and isinstance(node_info['embedding'], np.ndarray):
                        node_info['embedding'] = node_info['embedding'].tolist()
                    new_memory_nodes_data.append(node_info)
        # --- End of replacement block ---
        self._last_node_count = new_node_count
        # --- BEGIN: M18.5 UI compatibility patch ---
        # Compute shell_population per dimension
        shell_population = {}
        for _dim, _shell in getattr(self, "dimensional_shells", {}).items():
            try:
                if hasattr(_shell, "get_all_vectors_as_matrix"):
                    _M, _ = _shell.get_all_vectors_as_matrix()
                    _count = int(getattr(_M, "shape", [0])[0]) if _M is not None else 0
                elif hasattr(_shell, "vectors"):
                    _count = len(_shell.vectors) if _shell.vectors is not None else 0
                else:
                    _count = 0
            except Exception:
                _count = 0
            shell_population[str(_dim)] = _count
        
        # Ensure KDTree failure counter exists
        if not hasattr(self, "kdtree_failures"):
            self.kdtree_failures = 0
        
        # psi_entropy from quantum engine probabilities (if available)
        psi_entropy = None
        qeng = getattr(self, "qeng", None)
        if qeng is not None:
            try:
                probs = None
                if hasattr(qeng, "_probs"):
                    probs = qeng._probs()
                elif hasattr(qeng, "probs"):
                    probs = qeng.probs()
                if probs is not None:
                    try:
                        arr = probs[0]
                    except Exception:
                        arr = probs
                    try:
                        import numpy as _np  # prefer numpy if available
                        p = _np.asarray(arr, dtype=float).ravel()
                        p = _np.clip(p, 1e-12, 1.0)
                        psi_entropy = float(-(_np.where(p>0, p*_np.log(p), 0.0)).sum())
                    except Exception:
                        # pure-python fallback
                        import math as _math
                        _flat = []
                        try:
                            for _x in arr: _flat.append(float(_x))
                        except TypeError:
                            _flat = [float(arr)]
                        s = 0.0
                        for _p in _flat:
                            _p = min(max(_p, 1e-12), 1.0)
                            s += _p * _math.log(_p)
                        psi_entropy = float(-s)
            except Exception:
                psi_entropy = None
        
        # Discovery metrics placeholders (use existing attributes if present)
        novelty = getattr(self, "novelty", 0.0)
        compression_gain = getattr(self, "compression_gain", 0.0)
        disagreement = getattr(self, "disagreement", 0.0)
        lam = getattr(self, "lam", 0.0)
        # --- END: M18.5 UI compatibility patch ---


        telemetry = {
            "shell_population": shell_population,
            "psi_entropy": 0.0 if psi_entropy is None else float(psi_entropy),
            "novelty": float(novelty) if isinstance(novelty, (int, float)) else 0.0,
            "compression_gain": float(compression_gain) if isinstance(compression_gain, (int, float)) else 0.0,
            "disagreement": float(disagreement) if isinstance(disagreement, (int, float)) else 0.0,
            "kdtree_failures": int(getattr(self, "kdtree_failures", 0)),
            "lam": float(lam) if isinstance(lam, (int, float)) else 0.0,
            
            "run_id": self.run_id,
            "step": step,
            "mood": self.mood.mood_vector,
            "black_hole_pressure": self.black_hole_pressure,
            "goals": {n: d.get("activation", 0.0) for n, d in self.goal_field.goals.items()} if self.goal_field.is_initialized else {},
            "shells": shells_data,
            "shell_tensions": shell_tensions,
            "global_tension": float(sum(shell_tensions.values())/len(shell_tensions)) if shell_tensions else 0.0,
            "memory_count": self.memory.graph_db.graph.number_of_nodes(),
            "teacher_in": _steps_to_next(step, TEACHER_ASK_EVERY, TEACHER_OFFSET),
            "explorer_in": _steps_to_next(step, TEACHER_ASK_EVERY, EXPLORER_OFFSET),
            "environment_theme": self.synthetic_env.current_theme_region,
            "symbolic_weather": self.mood.get_symbolic_weather(),
            "teacher_question": self.teacher_question,
            "explorer_answer": self.explorer_last_answer,
            "subconscious_narrative": self.subconscious.narrative,
            "insight_agent_avg_reward": insight_reward_avg,
            "autoencoder_trained": bool(self.autoencoder and self.autoencoder.is_trained),
            "ingestion_feed": ingestion_feed_data,
            "validation_lab": validation_lab_data,
            "insight_holocron": insight_holocron_data,
            "new_memory_nodes": new_memory_nodes_data,
        }
        if self.market:
            telemetry["market"] = {"symbols": self.market_symbols, "last": self.market_last}
        return telemetry

    async def _sse_push_telemetry(self):
        clients = getattr(self, "sse_clients", None)
        if not clients: return
        try:
            payload = json.dumps(self._build_telemetry_snapshot(), cls=NumpyEncoder, ensure_ascii=False)
            dead = set()
            for q in list(clients):
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    dead.add(q)
            for q in dead:
                clients.discard(q)
        except Exception:
            pass

    def _ensure_console_export_state(self):
        if not hasattr(self, "_console_export_inited"):
            base = get_path("logs/console", self.run_id)
            os.makedirs(base, exist_ok=True)
            self.console_export_dir = base
            self._console_last_export_len = 0
            self._console_chunk_index = 0
            self._console_export_inited = True

    def _export_console_chunk(self, end_step: int, final: bool = False) -> None:
        self._ensure_console_export_state()
        try:
            text_all = self.console.export_text()

            if len(text_all) < self._console_last_export_len:
                self._console_last_export_len = 0
                self._console_chunk_index += 1

            new_text = text_all[self._console_last_export_len:]

            if not new_text and not final:
                return

            start_step = self._console_chunk_index * CONSOLE_EXPORT_EVERY_STEPS
            end_inclusive = end_step
            base = f"console_{start_step:06d}-{end_inclusive:06d}"

            if CONSOLE_EXPORT_FORMAT in ("text", "both"):
                with open(os.path.join(self.console_export_dir, base + ".txt"), "w", encoding="utf-8") as f:
                    f.write(new_text)

            if CONSOLE_EXPORT_FORMAT in ("json", "both"):
                payload = {
                    "run_id": self.run_id,
                    "chunk_index": self._console_chunk_index,
                    "start_step": start_step,
                    "end_step": end_inclusive,
                    "timestamp": (datetime.now(timezone.utc).isoformat() if (datetime is not None and timezone is not None) else __import__('datetime').datetime.utcnow().isoformat()),
                    "content": new_text
                }
                with open(os.path.join(self.console_export_dir, base + ".json"), "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False)

            self._console_last_export_len = len(text_all)
            self._console_chunk_index += 1
        except Exception as e:
            self.console.log(f"[ConsoleExport] Failed: {e}")

    async def _project_self_into_memory(self):
        """
        One-time at start: read this script and inject sections into memory as concepts.
        """
        try:
            src_path = os.path.abspath(__file__)
            with open(src_path, "r", encoding="utf-8") as f:
                code_txt = f.read()
        except Exception as e:
            self.console.log(f"[SelfProject] failed to read source: {e}")
            return

        try:
            splitter = re.compile(r"(?m)^(class\s+\w+\s*:|def\s+\w+\s*\(|if\s+__name__\s*==\s*['\"]__main__['\"]\s*:)");
            idxs = [m.start() for m in splitter.finditer(code_txt)]
            idxs = [0] + idxs + [len(code_txt)]
            sections = []
            for a, b in zip(idxs[:-1], idxs[1:]):
                chunk = code_txt[a:b].strip()
                if not chunk:
                    continue
                first = chunk.splitlines()[0].strip()
                label = sanitize_line(first[:72]) if 'sanitize_line' in globals() else first[:72]
                excerpt = "\n".join(chunk.splitlines()[:40])
                sections.append((label, excerpt))
            if not sections:
                head = "\n".join(code_txt.splitlines()[:80])
                sections = [("source: e8_mind_server", head)]
        except Exception as e:
            self.console.log(f"[SelfProject] split failed: {e}")
            return

        try:
            root_id = await self.memory.add_entry({
                "type": "self_code",
                "label": "E8 Mind — current source",
                "metaphor": "The mind reading its own blueprint.",
                "rating": 0.9,
                "step": int(getattr(self, "step_num", 0))
            })
        except Exception as e:
            self.console.log(f"[SelfProject] root insert failed: {e}")
            return

        inserted = []
        for label, excerpt in sections[:40]:
            try:
                emb = await self.get_embedding(excerpt)
            except Exception:
                emb = None
            try:
                node_id = await self.memory.add_entry({
                    "type": "self_code_section",
                    "label": label,
                    "metaphor": excerpt,
                    "embedding": emb,
                    "rating": 0.7,
                    "temperature": 0.2,
                    "step": int(getattr(self, "step_num", 0))
                }, parent_ids=[root_id])
                inserted.append(node_id)
            except Exception as e:
                self.console.log(f"[SelfProject] section insert failed: {e}")

        try:
            if 'bump_temps' in globals():
                bump_temps(self.memory, inserted, amount=0.6)
        except Exception as e:
            self.console.log(f"[SelfProject] temp bump failed: {e}")
        self.console.log(f"[SelfProject] projected {len(inserted)} code sections into memory.")

    def _build_state_vector(self) -> np.ndarray:
        """Constructs the current state vector from all relevant cognitive modules."""
        mood_vec = np.array(list(self.mood.mood_vector.values()), dtype=np.float32)

        # Ensure goal activations are a fixed size, even if not initialized
        if self.goal_field.is_initialized and self.goal_field.goals:
            goal_activations = np.array([g["activation"] for g in self.goal_field.goals.values()], dtype=np.float32)
        else:
            goal_activations = np.zeros(4, dtype=np.float32) # Assuming 4 goals

        shell_att_vec = self.shell_attention.build(self)

        # Calculate dynamics based on the latest black hole pressure and previous action
        dynamics_vec = np.array([
            self._bh_ma50,
            (self.black_hole_pressure - self._prev_bh),
            float(np.linalg.norm(self._prev_action)),
            0.0,  # Placeholder for proximity distance
            0.0
        ], dtype=np.float32)

        return np.concatenate([
            mood_vec,
            goal_activations,
            shell_att_vec,
            dynamics_vec
        ])

    def _update_cognitive_modules(self, step: int):
        """Updates all core cognitive modules that evolve over time."""
        self.mood.update()
        self.subconscious.decay(step)
        self.goal_field.decay()
        self.goal_field.update_from_mood(self.mood.mood_vector)
        self.memory.diffuse_field()
        self._update_black_hole_pressure()
        self.memory.decay_locks()
        self.scheduler.tick(step) # The scheduler handles all timed, async events

    def _train_autoencoder_if_ready(self, autoencoder_train_buffer: list, batch_size: int) -> list:
        """Trains the VAE on a batch of new embeddings if the buffer is full."""
        if TORCH_AVAILABLE and self.autoencoder and self.memory.pending_embeddings:
            autoencoder_train_buffer.extend(self.memory.pending_embeddings)
            self.memory.pending_embeddings.clear()
            
            if len(autoencoder_train_buffer) >= batch_size:
                batch_np = np.array(autoencoder_train_buffer[:batch_size])
                autoencoder_train_buffer = autoencoder_train_buffer[batch_size:]
                
                try:
                    _t = cast(Any, torch)
                    losses = self.autoencoder.train_on_batch(_t.from_numpy(batch_np).float() if hasattr(_t, 'from_numpy') else _t.tensor(batch_np, dtype=_t.float32))
                    self.console.log(f"🧠 [VAE] Trained. Loss: {losses['total_loss']:.4f}, Recon: {losses['recon_loss']:.4f}, KLD: {losses['kld_loss']:.4f}")
                except Exception as e:
                    self.console.log(f"[bold red]VAE Training Error: {e}[/bold red]")
        return autoencoder_train_buffer

    async def run_cognitive_cycle(self, max_steps=297600, mode='quantum'):
        """
        The main operational loop of the E8Mind, upgraded to a fully integrated
        Plan -> Act -> Learn cycle using all sophisticated modules.
        """
        self._ensure_console_export_state()
        self.console.rule(f"[bold magenta]Starting Integrated Cognitive Cycle | Mode: {mode.upper()}[/bold magenta]")

        # --- Initialization ---
        await self.llm_pool.start()
        await self.goal_field.initialize_goals()
        if E8_INGEST:
            pass  # gated by E8_INGEST

        for name, config in DATA_SOURCES.items():
            self.ingestion_pipeline.add_source(name, config)

        if E8_INGEST:
            await self.ingestion_pipeline.start()
        
        if self.market:
            if globals().get("E8_MARKET_FEED_ENABLED", False):
                await self.market.start()

        self.max_steps = max_steps
        autoencoder_train_buffer, AUTOENCODER_BATCH_SIZE = [], 64

        # --- Main Loop ---
        with Progress(
            SpinnerColumn(style="green"),
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "Step", TextColumn("{task.completed}/{task.total}"),
            "Concepts:", TextColumn("[bold magenta]{task.fields[concept_count]}[/bold magenta]"),
            TimeElapsedColumn(),
            console=self.console, transient=True
        ) as progress:
            task = progress.add_task(f"Thinking ({mode})", total=max_steps, concept_count=0)
            for step in range(max_steps):
                self.step_num = step
                
                if step == 0:
                    if E8_SELF_PROJECT:
                        await self._project_self_into_memory()

                # 1. UPDATE: Evolve passive modules and fire scheduled events
                self._update_cognitive_modules(step)
                autoencoder_train_buffer = self._train_autoencoder_if_ready(autoencoder_train_buffer, AUTOENCODER_BATCH_SIZE)
                
                # 2. HIERARCHY: The HRL controller sets a high-level goal
                if hasattr(self, "hrl"):
                    self.hrl.maybe_update(step)

                # 3. PERCEIVE: Build the current state vector (context)
                current_state = self._build_state_vector()
                if hasattr(self, "_wm_lazy_init") and self.world_model is None:
                    self._wm_lazy_init(current_state.size)
                
                # 4. PLAN: Use the World Model to "imagine" the future
                if hasattr(self, "world_model") and self.world_model.is_ready():
                    # The agent can "dream" about the consequences of its policy
                    self.world_model.imagine_with_policy(current_state, self.agent, horizon=8)

                # 5. META-LEARN: The Contextual Bandit chooses the best set of parameters
                arm_index = self.bandit.pull(context=current_state)
                active_arm = self.bandit.arms[arm_index]
                self.sigma_q = active_arm.get("diffusion_sigma", 1.2) # Dynamically tune quantum diffusion

                                # 6. ACT: Society or baseline selects a low-level action
                if getattr(self, 'society', None) is not None:
                    action = await self.society.step(current_state, self)
                else:
                    action = self.agent.select_action(current_state) if self.agent else np.zeros(self.action_dim, dtype=np.float32)
                # Blend in imagination-driven bias (non-blocking, robust to WM availability)
                try:
                    wm_action = None
                    if getattr(self, 'macro_manager', None) is not None and getattr(self, 'world_model', None) is not None:
                        wm_action = await self.macro_manager.select_action(current_state, self)
                    if wm_action is not None:
                        action = 0.7 * action + 0.3 * wm_action
                except Exception:
                    pass

                clamped_action = clamp_action(action)
                self.apply_manifold_action(clamped_action)

                # 7. SIMULATE: Evolve the quantum/classical engine based on the action
                prev_idx = self.prev_node_index or random.randrange(self.physics.roots.shape[0])
                try:
                    wavey_out = integrate_one_cycle(self, self.wavey_bridge)
                    self._update_anchors_from_wavey(wavey_out)
                except Exception:
                    pass

                self.qeng.build_hamiltonian(V=self.anchors.potential())
                self.qeng.step_adaptive()
                current_node_index = self.qeng.measure_hybrid(prev_idx, sigma=self.sigma_q)[0]
                self.prev_node_index = current_node_index

                # 8. LEARN: Observe the outcome and update all models
                next_state = self._build_state_vector()
                base_reward = self.potential_evaluator.calculate_potential_and_get_reward()

                # The HRL controller shapes the final reward with intrinsic motivation
                final_reward = self.hrl.shape_reward(current_state, next_state, base_reward) if hasattr(self, "hrl") else base_reward

                # The Causal Engine learns from the transition
                if hasattr(self, "causal"):
                    self.causal.update_on_step(self, clamped_action, final_reward)

                # The World Model observes the real outcome
                if hasattr(self, "world_model"):
                    self.world_model.observe(current_state, clamped_action, next_state, base_reward)

                # The low-level agent learns from the shaped reward
                if self.agent:
                    self.agent.store(current_state, clamped_action, next_state, final_reward, (step == max_steps - 1))
                    if step > 1024:
                        self.agent.update()

                # The Contextual Bandit learns from the base reward
                self.bandit.update(arm_index, base_reward, current_state)

                # 9. MAINTAIN: Update internal trackers for the next cycle
                self._prev_action = clamped_action
                self._prev_bh = self.black_hole_pressure

                await self._sse_push_telemetry()
                await self._sse_push_telemetry()
                if (step + 1) % CONSOLE_EXPORT_EVERY_STEPS == 0:
                    self._export_console_chunk(step)

                concept_count = self.memory.graph_db.graph.number_of_nodes()
                import typing as _typing
                progress.update(_typing.cast(_typing.Any, task), advance=1, concept_count=concept_count)
                await asyncio.sleep(0.01)

        # --- Shutdown ---
        self.console.log("\nCognitive cycle complete.")
        self._export_console_chunk(self.step_num, final=True)
    def _path(self, rel: str) -> str:
        return get_path(rel, self.run_id)

    # get_embedding defined later; avoid duplicate definition

    def _norm_text(self, s: str) -> str:
        s = (s or "").lower().strip()
        s = re.sub(r'(synthesis:\s*)+', 'synthesis: ', s)
        s = re.sub(r'\s+', ' ', s)
        return s

    def _ngrams(self, s: str, n: int = 5):
        toks = re.findall(r'[a-z0-9]+', s)
        return set(tuple(toks[i:i+n]) for i in range(max(0, len(toks)-n+1)))

    def _repeat_score(self, t: str) -> float:
        if not self._recent_norms:
            return 0.0
        tn = self._norm_text(t)
        A = self._ngrams(tn, 5)
        jacc, ratio, exact = 0.0, 0.0, 0.0
        for r in self._recent_norms:
            if not r: continue
            if tn == r:
                exact = 1.0; break
            B = self._ngrams(r, 5)
            if B: jacc = max(jacc, len(A & B) / max(1, len(A | B)))
            try:
                import difflib as _df
                ratio = max(ratio, _df.SequenceMatcher(None, tn, r).ratio())
            except ImportError: pass
        return max(jacc, ratio, exact)

    def _remember_output(self, text: str):
        n = self._norm_text(text)
        self._recent_texts.append(text)
        self._recent_norms.append(n)

    # Replace the existing _async_call_llm_internal method with this robust version.
    # In E8Mind
    async def _async_call_llm_internal(self, prompt: str, **kwargs) -> str:
        """
        Calls the LLM with full context, persona, and domain hints.
        Includes a fallback to a local model to prevent repetition and increase robustness.
        """
        # --- 1. Construct the Full Prompt ---
        try:
            # FIX: Access attributes directly via `self`, not `self.mind`
            persona = self.semantics.persona_prefix(self.mood.mood_vector)
        except Exception:
            # FIX: Access attributes directly via `self`
            persona = self.mood.get_llm_persona_prefix()

        # FIX: Access attributes directly via `self`
        domain_hint = f"Domain: {getattr(self.domain_tint, 'last_hint', self.semantic_domain)}."

        _prompt_key = kwargs.pop('_prompt_key', 'ask')
        _prompt_vars = kwargs.pop('_prompt_vars', None) or {'question': prompt}
        
        # FIX: Access attribute directly via `self`
        full_prompt = self.prompts.render(
            _prompt_key,
            persona=persona,
            domain_hint=domain_hint,
            **_prompt_vars
        )

        # --- 2. Prepare and Execute LLM Calls ---
        primary_task = None
        local_task = None

        llm_kwargs = {
            'model': self.client_model,
            'max_tokens': int(kwargs.get('max_tokens', 256)),
            'temperature': float(kwargs.get('temperature', 0.7)),
        }

        try:
            messages = [{"role": "user", "content": full_prompt}]
            primary_task = asyncio.wait_for(
                self.llm_client.chat(messages=messages, **llm_kwargs),
                timeout=LLM_CALL_TIMEOUT_SEC
            )
        except Exception as e:
            self.console.log(f"[LLM] Primary client call setup failed: {e}")

        if self._anti_repeat_enabled and self.local_llm_client:
            try:
                local_kwargs = {
                    'max_tokens': llm_kwargs['max_tokens'] // 2,
                    'temperature': min(1.0, llm_kwargs['temperature'] + 0.15)
                }
                local_messages = [{"role": "user", "content": prompt}]
                local_task = asyncio.wait_for(
                    self.local_llm_client.chat(messages=local_messages, **local_kwargs),
                    timeout=LLM_CALL_TIMEOUT_SEC
                )
            except Exception as e:
                self.console.log(f"[LLM] Local client call setup failed: {e}")

        # --- 3. Await and Collect Responses ---
        tasks_to_gather = [task for task in (primary_task, local_task) if task]
        if not tasks_to_gather:
            return "[LLM ERROR] No tasks could be created."

        results = await asyncio.gather(*tasks_to_gather, return_exceptions=True)

        primary_text = results[0] if primary_task and not isinstance(results[0], BaseException) else f"[LLM ERROR] Primary: {results[0]}"

        local_text = None
        if local_task:
            result_index = 1 if primary_task else 0
            # FIX: Add a bounds check to prevent IndexError if a task failed to be created
            if len(results) > result_index and not isinstance(results[result_index], BaseException):
                local_text = results[result_index]

        # --- 4. Select the Best Candidate to Avoid Repetition ---
        candidates = []
        if isinstance(primary_text, str) and not primary_text.startswith("[LLM"):
            candidates.append(primary_text.strip())
        if isinstance(local_text, str) and not local_text.startswith("[LLM"):
            candidates.append(local_text.strip())

        if not candidates:
            return primary_text or "[LLM ERROR] No valid response from any provider."

        best_candidate = min(candidates, key=self._repeat_score)
        self._remember_output(best_candidate)
        return best_candidate

    async def get_embedding(self, text: str) -> np.ndarray:
        if self.is_embed_placeholder:
            import zlib
            seed = zlib.adler32(text.encode("utf-8")) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            v_native = rng.standard_normal(self.embed_in_dim).astype(np.float32)
            v_native = self.semantics.post_embed(v_native)
            return self.embed_adapter(v_native)

        text = self.semantics.pre_embed(text)
        raw_vec = None
        try:
            raw_vec = await asyncio.wait_for(self.llm_client.embedding(text, model=self.embedding_model), timeout=EMBEDDING_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            self.console.log("[yellow]Embedding timeout. Using fallback vector.[/yellow]")
        except Exception as e:
            self.console.log(f"[yellow]Embedding error: {e}. Using fallback vector.[/yellow]")

        if raw_vec is None:
            import zlib
            seed = zlib.adler32(text.encode("utf-8")) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            raw_vec = rng.standard_normal(self.embed_in_dim).astype(np.float32)

        raw_vec = self.semantics.post_embed(raw_vec)
        return self.embed_adapter(np.asarray(raw_vec, dtype=np.float32))

    async def rate_concept(self, concept_text: str) -> float:
        if self.is_embed_placeholder:
            return 0.6
        prompt = f'Rate the novelty and coherence of this idea on a scale from 0.0 to 1.0. Response must be only the number.\nIdea: "{concept_text}"'
        try:
            response = await self.llm_pool.enqueue_and_wait(prompt, max_tokens=10, temperature=0.1)
            num = re.findall(r"[-+]?\d*\.\d+|\d+", response)
            v = float(num[0]) if num else 0.5
            return np.clip(v / 100.0 if v > 1.0 else v, 0.0, 1.0)
        except Exception:
            return 0.5

    def _update_black_hole_pressure(self):
        hot_nodes = [nid for nid, d in self.memory.graph_db.graph.nodes(data=True) if d.get('temperature', 0) > 1.5]
        if not hot_nodes:
            self.black_hole_pressure *= 0.9
            return
        max_density = max((self.memory._local_density(nid, radius=2) for nid in hot_nodes), default=0.0)
        num_nodes = self.memory.graph_db.graph.number_of_nodes()
        saturation_factor = 0.8 * np.log1p(num_nodes / 50.0) if num_nodes > 0 else 0.0
        self.black_hole_pressure = np.clip(max_density * saturation_factor, 0.0, 1.0)
        is_ready = (self.step_num >= self._bh_cooldown_until) and (not self._bh_inflight)
        if is_ready and self.black_hole_pressure >= BH_PRESSURE_THRESHOLD:
            self._bh_inflight = True
            self._bh_cooldown_until = self.step_num + BLACK_HOLE_COOLDOWN_STEPS
            self.console.log(f"[bold red]Black hole pressure threshold exceeded ({self.black_hole_pressure:.3f}). Initiating collapse.[/bold red]")
            asyncio.create_task(self._blackhole_cycle(self.step_num))

    async def _blackhole_cycle(self, step_num: int):
        self._bh_inflight = True
        try:
            center_id, pressure = self.memory.find_event_horizon()
            if not center_id:
                self.console.log("[BH Cycle] Aborted: No event horizon found.")
                return None

            cluster = self.memory.collect_cluster(center_id)
            need = max(2, CONSOLIDATE_MIN)

            # CORRECTED PADDING LOGIC
            if not cluster or len(cluster) < need:
                self.console.log(f"[BH Cycle] Cluster for '{center_id}' too small ({len(cluster)} < {need}). Attempting local padding...")
                base = set(cluster) if cluster else {center_id}
                
                # Try to pad using only semantically similar (local) nodes
                center_vec = self.memory.main_vectors.get(center_id)
                if center_vec is not None:
                    similar_nodes = self.memory.find_similar_in_main_storage(center_vec, k=need * 3)
                    for nid, _ in similar_nodes:
                        if nid not in base and self.memory.main_vectors.get(nid) is not None:
                            base.add(nid)
                        if len(base) >= need:
                            break
                
                cluster = list(base)
                # If still too small after local padding, abort the cycle.
                if len(cluster) < need:
                    self.console.log(f"[BH Cycle] Aborted: Cluster too small ({len(cluster)}) even after local padding.")
                    return None

            remnant_data, remnant_vec, mass = await self.memory.synthesize_remnant(cluster, label_hint=f"EmergenceSeed@{step_num}")

            if not remnant_data or remnant_vec is None:
                self.console.log("[BH Cycle] Aborted: Failed to synthesize remnant.")
                return None

            self.mood.process_event("blackhole", magnitude=float(mass))
            await self.memory._cosmological_spread(remnant_vec, mass)

            remnant_data["temperature"] = 2.0
            remnant_id = await self.memory.add_entry(remnant_data)
            if not remnant_id:
                return None

            self._bh_cooldown_until = step_num + BLACK_HOLE_COOLDOWN_STEPS
            for nid in cluster:
                if nid == remnant_id: continue
                old_vec = self.memory.main_vectors.get(nid)
                if old_vec is not None:
                    self.memory.graph_db.add_edge(remnant_id, nid, type="collapse", weight=float(self.memory._cos_sim(remnant_vec, old_vec)))

            cands = sorted([(nid, self.memory._cos_sim(remnant_vec, v)) for nid, v in self.memory.main_vectors.items() if nid != remnant_id and nid not in cluster], key=lambda t: -t[1])
            for nid, s in cands[:BLACK_HOLE_K]:
                self.memory.graph_db.add_edge(remnant_id, nid, type="knn", weight=float(s))

            self.memory.fold_and_prune(cluster)

            z8 = np.zeros(8)
            if TORCH_AVAILABLE and self.autoencoder and self.autoencoder.is_trained:
                with torch.no_grad():
                    _t = cast(Any, torch)
                    remnant_t = (_t.from_numpy(remnant_vec).float().unsqueeze(0) if hasattr(_t, 'from_numpy')
                                 else _t.tensor(remnant_vec[None, :], dtype=_t.float32))
                    z8_tensor = self.autoencoder.project_to_dim(remnant_t, 8)
                    if z8_tensor is not None:
                        try:
                            z8 = z8_tensor.numpy().squeeze()  # torch tensor path
                        except Exception:
                            z8 = np.asarray(z8_tensor).squeeze()

            seed = EmergenceSeed(remnant_id=remnant_id, embedding_vector=remnant_vec, projected_vector=z8, mass=mass, absorbed_ids=cluster, step_created=step_num)
            self.console.print(Panel(f"Emergence Seed created at step {step_num} (mass={mass:.2f}) — [bold red]BLACK HOLE EVENT[/bold red]", border_style="red", expand=False))
            self.black_hole_pressure = 0.0
            self.black_hole_log.append({"type": "black_hole", "step": step_num, "size": len(cluster), "mass": float(mass)})
            return seed
        finally:
            self._bh_inflight = False

    async def perform_retro_relink(self, new_node_id, new_vec, k=12, min_age_steps=20):
        G = self.memory.graph_db.graph
        if not G.has_node(new_node_id):
            return
        candidates = []
        for nid, d in G.nodes(data=True):
            if nid != new_node_id and d.get("step", 0) <= (self.step_num - min_age_steps):
                vec = d.get("embedding")
                if vec is not None:
                    candidates.append((nid, np.asarray(vec, dtype=float)))
        if not candidates:
            return
        newv = np.asarray(new_vec, dtype=float)

        def _norm(x):
            return x / (np.linalg.norm(x) + 1e-9)

        newv = _norm(newv)
        sims = sorted([(nid, float(np.dot(newv, _norm(v)))) for nid, v in candidates], key=lambda x: x[1], reverse=True)
        top = sims[:k]
        for nid, w in top:
            try:
                self.memory.graph_db.add_edge(new_node_id, nid, kind="retrotag", weight=w)
            except Exception:
                pass

            node = G.nodes.get(nid)
            if node:
                node["temperature"] = float(node.get("temperature", 0.5) + 0.05 * w)
        self.console.log(f"[retro] linked {len(top)} prior nodes to {new_node_id}")


    # --- Wavey Integration: Engine Interface Hooks ---
    def get_focus_vector(self) -> np.ndarray:
        """Return a (EMBED_DIM,) focus vector for Wavey seeding.
        Strategy: weighted sum of GoalField embeddings (if initialized),
        blended with the mean of recent LTM vectors as context."""
        try:
            parts = []
            weights = []
            if getattr(self, "goal_field", None) and getattr(self.goal_field, "is_initialized", False):
                for name, g in self.goal_field.goals.items():
                    emb = g.get("embedding")
                    act = float(g.get("activation", 0.0))
                    if isinstance(emb, np.ndarray) and emb.size == int(1536):
                        parts.append(emb); weights.append(max(act, 1e-6))
            if getattr(self, "memory", None) and getattr(self.memory, "_main_storage_matrix", None) is not None:
                M = self.memory._main_storage_matrix
                if isinstance(M, np.ndarray) and M.size > 0:
                    parts.append(M.mean(axis=0)); weights.append(0.25)
            if parts:
                W = np.asarray(weights, dtype=np.float32); W = W / (W.sum() + 1e-12)
                v = np.sum([w*p for w, p in zip(W, parts)], axis=0).astype(np.float32)
                n = np.linalg.norm(v) + 1e-12
                return v / n
        except Exception:
            pass
        return np.zeros(int(1536), dtype=np.float32)

    def get_memory_matrix(self):
        """Return (matrix (N,D), labels) for Wavey attention mapping."""
        try:
            M = getattr(self.memory, "_main_storage_matrix", None)
            ids = getattr(self.memory, "_main_storage_ids", None)
            if isinstance(M, np.ndarray) and M.size > 0 and isinstance(ids, list) and len(ids) == M.shape[0]:
                return M.astype(np.float32), ids
        except Exception:
            pass
        return np.empty((0, int(1536)), dtype=np.float32), []

    def apply_hamiltonian_bias(self, bias: np.ndarray) -> None:
        """Store last bias; it will be folded into anchors each step."""
        try:
            if isinstance(bias, np.ndarray) and bias.size > 0:
                self._wavey_bias_last = bias.astype(np.float32).copy()
        except Exception:
            self._wavey_bias_last = None

    def apply_attention_weights(self, weights: np.ndarray, labels=None) -> None:
        """Nudge memory temperatures using Wavey attention (top-k)."""
        try:
            if not isinstance(weights, np.ndarray) or weights.ndim != 1: return
            if labels is None:
                labels = getattr(self.memory, "_main_storage_ids", [])
            k = min(8, weights.shape[0])
            if k == 0: return
            idxs = np.argsort(-weights)[:k]
            for i in idxs:
                nid = labels[i] if i < len(labels) else None
                if nid:
                    self.memory.spike_temperature(nid, amount=float(weights[i]) * 0.5 + 0.05)
        except Exception:
            pass

    def _update_anchors_from_wavey(self, wavey_out: dict):
        """Translate Wavey potentials + bias into 8D anchors and set self.anchors."""
        try:
            anchor_list = []
            pots = wavey_out.get("potentials") or []
            for pot in pots:
                try:
                    center = np.asarray(getattr(pot, "center", None), dtype=np.float32)
                except Exception:
                    center = None
                depth = float(getattr(pot, "depth", 0.0) or 0.0)
                if center is None or center.size == 0 or depth <= 0:
                    continue
                # CORRECTED: Use the TinyCompressor for encoding
                v8 = self.holo.encode(center)
                n = np.linalg.norm(v8) + 1e-12
                v8 = (v8 / n).astype(np.float32)
                anchor_list.append((v8, depth))
            if isinstance(self._wavey_bias_last, np.ndarray) and self._wavey_bias_last.size > 0:
                # CORRECTED: Use the TinyCompressor for encoding
                v8b = self.holo.encode(self._wavey_bias_last)
                nb = np.linalg.norm(v8b) + 1e-12
                v8b = (v8b / nb).astype(np.float32)
                w = float(np.linalg.norm(self._wavey_bias_last)) * 0.15
                if w > 0: anchor_list.append((v8b, w))
            if getattr(self, "goal_field", None) and getattr(self.goal_field, "is_initialized", False):
                for name, g in self.goal_field.goals.items():
                    emb = g.get("embedding")
                    act = float(g.get("activation", 0.0))
                    if isinstance(emb, np.ndarray) and emb.size > 0 and act > 0:
                        # CORRECTED: Use the TinyCompressor for encoding
                        v8g = self.holo.encode(emb)
                        v8g = (v8g / (np.linalg.norm(v8g)+1e-12)).astype(np.float32)
                        anchor_list.append((v8g, 0.12 * act))
            self.anchors.set(anchor_list)
        except Exception as e:
            try:
                self.console.log(f"[Wavey] Anchor update error: {e}")
            except Exception:
                pass

# NumpyEncoder already defined earlier; avoid duplicate class definition here.

# --- Web Handlers ---

async def shutdown_sse(app):
    clients = app.get('sse_clients')
    if not clients:
        return
    for q in list(clients):
        try:
            q.put_nowait(None)
        except Exception:
            pass

async def shutdown_market_feed(app):
    mind = app.get('mind')
    if mind and getattr(mind, "market", None):
        await mind.market.stop()

async def handle_get_graph(request):
    mind = request.app['mind']
    graph_data = export_graph(mind.memory.graph_db.graph)
    return web.Response(text=json.dumps(graph_data, cls=NumpyEncoder), content_type='application/json')

async def handle_get_qeng_telemetry(request):
    mind = request.app['mind']
    qeng = getattr(mind, "qeng", None)
    if qeng is None:
        return web.json_response({"error": "quantum engine not initialized"}, status=400)
    return web.json_response(qeng.telemetry_state())

async def handle_stream_telemetry(request):
    app = request.app
    q = asyncio.Queue(maxsize=16)
    app['sse_clients'].add(q)

    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no',
    }
    resp = web.StreamResponse(status=200, reason='OK', headers=headers)
    await resp.prepare(request)
    try:
        await resp.write(b":ok\n\n")
        while True:
            data = await q.get()
            if data is None:
                break
            chunk = f"event: telemetry\ndata: {data}\n\n".encode('utf-8')
            await resp.write(chunk)
    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
        pass
    finally:
        app['sse_clients'].discard(q)
        with contextlib.suppress(Exception):
            await resp.write_eof()
    return resp

async def handle_get_qeng_ablation(request):
    mind = request.app['mind']
    qeng = getattr(mind, "qeng", None)
    if qeng is None:
        return web.json_response({"error": "quantum engine not initialized"}, status=400)
    params = request.rel_url.query
    prev_idx = int(params.get('prev_idx', '0'))
    sigma_str = params.get('sigma')
    sigma = float(sigma_str) if sigma_str is not None else None
    window = int(params.get('window', '5'))
    trials = int(params.get('trials', '256'))
    res = qeng.measure_ablation(prev_idx=prev_idx, sigma=sigma, window=window, trials=trials)
    return web.json_response(res)

    

async def handle_get_state(request):
    mind = request.app['mind']
    try:
        snap = mind._build_telemetry_snapshot()
    except Exception:
        snap = {}
    # select key bits
    state = {
        'step': getattr(mind, 'step_num', None),
    'mood': (getattr(getattr(mind, 'mood', None), 'mood_vector', None) if getattr(mind, 'mood', None) else None),
        'insight_reward': getattr(mind, 'last_insight_reward', None),
        'goals': getattr(mind, 'active_goals', []),
        'telemetry': snap
    }
    return web.json_response(state)

async def handle_get_telemetry(request):
    mind = request.app['mind']
    try:
        telemetry_data = mind._build_telemetry_snapshot()
        if mind.market:
            telemetry_data["market"]["bars"] = {
                "1s": {s: list(mind.market.history_1s.get(s, [])) for s in mind.market_symbols},
                "1m": {s: list(mind.market.history_1m.get(s, [])) for s in mind.market_symbols},
            }
        return web.json_response(telemetry_data, dumps=lambda d: json.dumps(d, cls=NumpyEncoder))
    except Exception as e:
        console.log(f"[Telemetry Endpoint Error] {e}")
        return web.json_response({"error": "Failed to generate telemetry"}, status=500)

async def handle_get_blueprint(request):
    return web.json_response(request.app['mind'].blueprint)

async def handle_add_concept(request):
    mind = request.app['mind']
    try:
        data = await request.json()
        text = data.get("text")
        if not text: return web.json_response({"error": "Text is required"}, status=400)
        rating = await mind.rate_concept(text)
        entry = {"type": "external_concept", "label": sanitize_line(text, 25), "metaphor": text, "rating": rating, "step": mind.step_num}
        node_id = await mind.memory.add_entry(entry)
        return web.json_response({"node_id": node_id, "message": "Concept added successfully"})
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

async def handle_trigger_dream(request):
    mind = request.app['mind']
    asyncio.create_task(mind.dream_engine.run_dream_sequence())
    return web.json_response({"status": "Dream sequence initiated"})

# --- Fixed/Global Web Handlers ---
async def handle_memory_search(request):
    mind = request.app['mind']
    q = request.rel_url.query.get('q', '').strip()
    if not q:
        return web.json_response({'error': 'missing q'}, status=400)
    try:
        vec = await mind.get_embedding(q)
    except Exception:
        import zlib as _zlib
        seed = _zlib.adler32(q.encode('utf-8')) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(mind.embed_in_dim).astype(np.float32)
        vec = mind.embed_adapter(vec)
    sims = mind.memory.find_similar_in_main_storage(vec, k=5)
    results = []
    for nid, dist in sims:
        node = mind.memory.graph_db.get_node(nid) or {}
        results.append({'id': nid, 'label': node.get('label'), 'distance': float(dist)})
    return web.json_response({'q': q, 'results': results})

async def handle_get_qeng_probabilities(request):  # redeclare as final handler
    mind = request.app['mind']
    qeng = getattr(mind, 'qeng', None)
    if qeng is None:
        return web.json_response({"error": "quantum engine not initialized"}, status=400)
    try:
        P = qeng._probs()
        if isinstance(P, np.ndarray):
            probs = P[0].tolist() if P.ndim == 2 else P.tolist()
        else:
            probs = list(P)
        return web.json_response({"probs": probs})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

async def handle_get_metrics_live(request):
    import collections
    mind = request.app['mind']
    metrics_file = get_path("metrics.ndjson", mind.run_id)
    if not os.path.exists(metrics_file):
        return web.json_response({"error": "Metrics file not found."}, status=404)
    try:
        lines = []
        with open(metrics_file, 'r', encoding='utf-8') as f:
            for line in f:
                lines.append(line)
        tail = lines[-300:]
        counters = collections.defaultdict(int)
        gauges = {}
        timings = collections.defaultdict(list)
        for ln in tail:
            try:
                rec = json.loads(ln)
                t = rec.get('type')
                name = rec.get('name')
                if not name:
                    continue
                if t == 'counter':
                    counters[name] += int(rec.get('value', 0))
                elif t == 'gauge':
                    gauges[name] = rec.get('value')
                elif t == 'timing':
                    timings[name].append(float(rec.get('duration_ms', 0.0)))
            except Exception:
                continue
        timing_means = {k: (sum(v)/len(v) if v else 0.0) for k, v in timings.items()}
        return web.json_response({
            'counters': dict(counters),
            'gauges': gauges,
            'timing_means': timing_means,
        })
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)

async def handle_get_metrics_summary(request):
    mind = request.app['mind']
    metrics_file = get_path('metrics.ndjson', mind.run_id)
    summary = {
        'counters': defaultdict(int),
        'gauges': {},
        'timings': defaultdict(list),
    }
    if not os.path.exists(metrics_file):
        return web.json_response({'error': 'Metrics file not found.'}, status=404)
    try:
        with open(metrics_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    metric_type = entry.get('type')
                    name = entry.get('name')
                    if not name:
                        continue
                    if metric_type == 'counter':
                        summary['counters'][name] += entry.get('value', 1)
                    elif metric_type == 'gauge':
                        summary['gauges'][name] = entry.get('value', 0.0)
                    elif metric_type == 'timing':
                        summary['timings'][name].append(entry.get('duration_ms', 0.0))
                except json.JSONDecodeError:
                    continue
        timing_stats = {}
        for name, values in summary['timings'].items():
            if values:
                timing_stats[name] = {
                    'count': len(values),
                    'avg_ms': float(np.mean(values)),
                    'p95_ms': float(np.percentile(values, 95)),
                    'max_ms': float(np.max(values)),
                }
        summary['timings'] = timing_stats
        summary['counters'] = dict(summary['counters'])
        return web.json_response(summary)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)

async def handle_post_quantizer(request):
    mind = request.app['mind']
    try:
        data = await request.json()
        quantizer_type = data.get('type')
        if quantizer_type in ['e8', 'cubic', 'random', 'none']:
            mind._quantizer_override = quantizer_type
            console.log(f'🕹️ Quantizer override set to: {quantizer_type}')
            return web.json_response({'status': 'ok', 'message': f'Quantizer set to {quantizer_type}'})
        return web.json_response({'error': 'Invalid quantizer type. Must be one of: e8, cubic, random, none'}, status=400)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)

async def handle_post_snapshot(request):
    mind = request.app['mind']
    console.log('📸 Manual snapshot triggered via API.')
    asyncio.create_task(mind.memory.snapshot())
    return web.json_response({'status': 'ok', 'message': 'Snapshot creation initiated.'})

def _collect_config_from_user():
    print("Choose LLM provider:\n1. OpenAI\n2. Ollama (local)\n3. Gemini API")
    provider_choice = input("Enter choice (1, 2, or 3) [1]: ") or "1"
    cfg = {"provider_choice": provider_choice}
    if provider_choice == "1":
        cfg["openai_api_key"] = (input("OpenAI API Key: ") or "").strip()
        cfg["openai_model_name"] = (input("OpenAI model [gpt-5-mini-preview]: ") or "gpt-5-mini-preview").strip()
    elif provider_choice == "2":
        cfg["ollama_model_name"] = (input("Ollama model [llama3]: ") or "llama3").strip()
    elif provider_choice == "3":
        cfg["gemini_api_key"] = (input("Gemini API Key: ") or "").strip()
        cfg["gemini_model_name"] = (input("Gemini model [gemini-1.5-flash]: ") or "gemini-1.5-flash").strip()
    else:
        print("Invalid choice. Running with LLM stub.")
    use_local = (input("Augment with a local tiny-LLM via Ollama? (y/N): ") or "n").strip().lower() == "y"
    cfg["use_local_mix"] = "true" if bool(use_local) else "false"
    if use_local:
        cfg["local_model_name"] = (input("Local Ollama model [phi3:mini-4k]: ") or "phi3:mini-4k").strip()
    return cfg

# --- GLOBAL CONFIGURATION VARIABLES ---
llm_client = None
model_name = "stub"
embedding_model = "stub"
IS_EMBED_PLACEHOLDER = True
LLM_PROVIDER = "stub"
provider_native_embed_dim = 1536


async def main():
    """Main function to initialize and run the E8Mind server."""
    global llm_client, model_name, embedding_model, IS_EMBED_PLACEHOLDER, LLM_PROVIDER, provider_native_embed_dim

    run_id = get_run_id()

    try:
        seed_all(GLOBAL_SEED)
    except Exception:
        pass
    if os.getenv("E8_PROVIDER", "").strip().lower() in ("", "ask"):
        cfg = _collect_config_from_user()
        pc = str(cfg.get("provider_choice", "")).strip()
        if pc == "1":
            LLM_PROVIDER = "openai"
            api_key = cfg.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key: raise ValueError("OPENAI_API_KEY not set.")
            llm_client, model_name, embedding_model = AsyncOpenAIClient(api_key, console), cfg.get("openai_model_name") or DEFAULT_OPENAI_CHAT_MODEL, "text-embedding-3-small"
            IS_EMBED_PLACEHOLDER = False
        elif pc == "2":
            LLM_PROVIDER, model_name = "ollama", cfg.get("ollama_model_name") or os.getenv("OLLAMA_MODEL", "llama3")
            llm_client, embedding_model = OllamaClient(model_name, console), "nomic-embed-text"
            IS_EMBED_PLACEHOLDER = False
        elif pc == "3":
            LLM_PROVIDER = "gemini"
            api_key = cfg.get("gemini_api_key") or os.getenv("GEMINI_API_KEY")
            if not api_key: raise ValueError("GEMINI_API_KEY not set.")
            model_name = cfg.get("gemini_model_name") or "gemini-1.5-flash"
            llm_client, embedding_model = GeminiClient(api_key, model_name, console), "models/embedding-001"
            IS_EMBED_PLACEHOLDER = False
        else:
            LLM_PROVIDER, IS_EMBED_PLACEHOLDER = "stub", True
    else:
        LLM_PROVIDER = os.getenv("E8_PROVIDER", "stub").lower()
        if LLM_PROVIDER == "openai":
            api_key = os.getenv("OPENAI_API_KEY");
            if not api_key: raise ValueError("OPENAI_API_KEY not set.")
            llm_client, model_name, embedding_model = AsyncOpenAIClient(api_key, console), DEFAULT_OPENAI_CHAT_MODEL, "text-embedding-3-small"
            IS_EMBED_PLACEHOLDER = False
        elif LLM_PROVIDER == "ollama":
            model_name = os.getenv("OLLAMA_MODEL", "llama3")
            llm_client, embedding_model = OllamaClient(model_name, console), "nomic-embed-text"
            IS_EMBED_PLACEHOLDER = False
        elif LLM_PROVIDER == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key: raise ValueError("GEMINI_API_KEY not set.")
            model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            llm_client, embedding_model = GeminiClient(api_key, model_name, console), "models/embedding-001"
            IS_EMBED_PLACEHOLDER = False
        else:
            LLM_PROVIDER, IS_EMBED_PLACEHOLDER = "stub", True

    if IS_EMBED_PLACEHOLDER:
        class StubClient:
            def __init__(self, console): self.console = console
            async def chat(self, *a, **k): return "This is a placeholder response from a stubbed LLM."
            async def embedding(self, *a, **k): return np.random.randn(provider_native_embed_dim)
            async def batch_embedding(self, texts, *a, **k): return [np.random.randn(provider_native_embed_dim) for _ in texts]
        llm_client, model_name, embedding_model = StubClient(console), "stub", "stub"

    _test_vec = None
    try:
        if llm_client is not None and hasattr(llm_client, "embedding"):
            _test_vec = await llm_client.embedding("adapter_probe")
    except Exception as _e:
        console.log(f"[INIT] embedding probe failed: {_e}. Falling back to random.")
    if _test_vec is None:
        _test_vec = np.random.randn(provider_native_embed_dim)
    if isinstance(_test_vec, dict) and "embedding" in _test_vec: _test_vec = _test_vec["embedding"]
    if isinstance(_test_vec, list) and _test_vec and isinstance(_test_vec[0], (list, np.ndarray)): _test_vec = _test_vec[0]
    embed_in_dim = int(len(_test_vec))
    if embed_in_dim > 1: provider_native_embed_dim = embed_in_dim
    console.log(f"[INIT] Detected provider embedding dimension: {provider_native_embed_dim}")

    profile_name = os.getenv("MIND_PROFILE", "default")
    sem, _ = load_profile(profile_name)
    probe_native = np.zeros(provider_native_embed_dim, dtype=np.float32)
    
    try:
        probe_post = sem.post_embed(probe_native)
        adapter_in_dim = int(np.asarray(probe_post, dtype=np.float32).size)
        console.log(f"[INIT] post_embed output dim: {adapter_in_dim} (provider {provider_native_embed_dim})")
    except Exception as e:
        adapter_in_dim = provider_native_embed_dim
        console.log(f"[INIT] post_embed probe failed: {e}. Falling back to provider dim.")

    embed_adapter = UniversalEmbeddingAdapter(adapter_in_dim, EMBED_DIM)
    console.log(f"[INIT] Universal Embedding Adapter created: {adapter_in_dim} -> {EMBED_DIM}")

    mind = E8Mind(
        semantic_domain_val=SEMANTIC_DOMAIN,
        run_id=run_id,
        llm_client_instance=llm_client,
        client_model=model_name,
        embedding_model_name=embedding_model,
        embed_adapter=embed_adapter,
        embed_in_dim=provider_native_embed_dim,
        console=console,
        is_embed_placeholder=IS_EMBED_PLACEHOLDER
    )

    # inserted: seed domain concept if requested
    try:
        await mind.seed_domain_if_empty()
    except Exception as _e:
        console.log(f"[seed] seed_domain_if_empty error: {_e}")
    
    try:
        cfg = locals().get("cfg", {})
        if cfg.get("use_local_mix") and ollama is not None:
            local_model = cfg.get("local_model_name") or "phi3:mini-4k"
            mind.local_llm_client = OllamaClient(local_model, console)
            mind.local_llm_model = local_model
            console.log(f"[LLM MIX] Local tiny-LLM enabled via Ollama model='{local_model}'.")
        else:
            console.log("[LLM MIX] Local tiny-LLM disabled or not available.")
    except Exception as e:
        console.log(f"[LLM MIX] Failed to init local tiny-LLM: {e}")
    # Require aiohttp for the HTTP server
    if web is None:
        console.log("[bold red]aiohttp is not installed. Please `pip install aiohttp` to run the server API.[/bold red]")
        return
    app = web.Application()
    app['mind'] = mind
    app['sse_clients'] = set()
    mind.sse_clients = app['sse_clients']
    app.on_shutdown.append(shutdown_sse)
    app.on_shutdown.append(shutdown_market_feed)
    # Setup CORS if available; otherwise use a no-op shim
    if aiohttp_cors is not None:
        cors = aiohttp_cors.setup(app, defaults={"*": aiohttp_cors.ResourceOptions(allow_credentials=True, expose_headers="*", allow_headers="*")})
    else:
        class _NoCors:
            def add(self, *a, **k):
                return None
        cors = _NoCors()

    app.router.add_get("/api/graph", handle_get_graph)
    app.router.add_get("/api/memory/search", handle_memory_search)
    app.router.add_get("/api/state", handle_get_state)
    app.router.add_get("/api/telemetry", handle_get_telemetry)
    app.router.add_get("/api/telemetry/stream", handle_stream_telemetry)
    app.router.add_get("/api/blueprint", handle_get_blueprint)
    # Frontend expects this alias as well
    app.router.add_get("/api/quasicrystal_blueprint", handle_get_blueprint)
    app.router.add_post("/api/concept", handle_add_concept)
    app.router.add_post("/api/action/dream", handle_trigger_dream)
    app.router.add_get("/api/qeng/telemetry", handle_get_qeng_telemetry)
    app.router.add_get("/api/qeng/ablation", handle_get_qeng_ablation)
    app.router.add_get("/api/qeng/probabilities", handle_get_qeng_probabilities)

    app.router.add_get("/metrics/summary", handle_get_metrics_summary)
    app.router.add_get("/metrics/live", handle_get_metrics_live)
    app.router.add_post("/quantizer", handle_post_quantizer)
    app.router.add_post("/snapshot", handle_post_snapshot)

    static_path = os.path.join(BASE_DIR, 'static')
    if os.path.exists(static_path): app.router.add_static('/', static_path, show_index=True, default_filename='index.html')
    for route in list(app.router.routes()): cors.add(route)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 7870)
    await site.start()
    console.log(f"[bold green]E8 Mind Server running at http://localhost:7870[/bold green]")
    console.log(f"Run ID: {run_id}")

    cycle_task = asyncio.create_task(mind.run_cognitive_cycle())
    try:
        await cycle_task
    finally:
        await runner.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.log("\n[bold yellow]Keyboard interrupt received. Shutting down.[/bold yellow]")
    except Exception as e:
        console.log(f"[bold red]CRITICAL ERROR in main: {e}[/bold red]")
        console.print_exception()

def _deprecated_cognitive_cycle(self):
    try:
        if hasattr(self, "memory") and hasattr(self.memory, "all_embeddings_matrix"):
            mats = self.memory.all_embeddings_matrix() or []
            import numpy as _np
            if isinstance(mats, list):
                mats = _np.asarray(mats, dtype=_np.float32)
            if getattr(self, "holo", None) is not None and mats is not None and mats.size >= self.holo.out_dim:
                self.holo.train_on_memory(mats)
    except Exception:
        pass
