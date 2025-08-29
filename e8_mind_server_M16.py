# Copyright (C) 2025 Skye Malone
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations
# --- Windows asyncio policy fix (avoids Proactor self-pipe close error) ---
import sys as _sys, asyncio
if _sys.platform.startswith("win"):
    # The alias _asyncio should be replaced with asyncio here as well
    try:
        if not isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsSelectorEventLoopPolicy):
             asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass
# -------------------------------------------------------------------------
import os, sys, math, json, time, random, re, logging, tempfile, io, glob, hashlib, contextlib, traceback, threading, faulthandler, zlib
from typing import Any, Dict, List, Optional, Tuple, Union, Iterable
from collections import deque, defaultdict
from dataclasses import dataclass

import numpy as np

# --- Configuration Constants ---
# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_DIR = os.path.join(BASE_DIR, "runtime")
# Timeouts and Intervals
POOL_WORKER_TIMEOUT = int(os.getenv("POOL_WORKER_TIMEOUT", "20"))
POOL_RESULT_TIMEOUT = int(os.getenv("POOL_RESULT_TIMEOUT", "60"))
LLM_CALL_TIMEOUT_SEC = int(os.getenv("LLM_CALL_TIMEOUT_SEC", "30"))
EMBEDDING_TIMEOUT_SEC = int(os.getenv("EMBEDDING_TIMEOUT_SEC", "15"))
DREAM_MIN_INTERVAL_SEC = 30
CONSOLE_EXPORT_EVERY_STEPS = 100
CONSOLE_EXPORT_FORMAT = "both" # "text", "json", or "both"
# Model Dimensions
EMBED_DIM = int(os.getenv("E8_EMBED_DIM", "1536"))
DIMENSIONAL_SHELL_SIZES = [1, 2, 3, 4, 5, 6, 7, 8]
AUTOENCODER_LAYER_SIZES = [EMBED_DIM] + DIMENSIONAL_SHELL_SIZES
# Action Layout for RL Agent
ACTION_LAYOUT = [
    {"dim": 3, "biv_start": 0, "biv_len": 3, "angle_idx": 3},
    {"dim": 5, "biv_start": 4, "biv_len": 10, "angle_idx": 14},
    {"dim": 8, "biv_start": 15, "biv_len": 28, "angle_idx": 43},
]
ACTION_SIZE_NO_LOCK = sum(d["biv_len"] + 1 for d in ACTION_LAYOUT)
# Cognitive Cycle Timing
TEACHER_ASK_EVERY = 25
TEACHER_OFFSET = 5
EXPLORER_OFFSET = 15
TEACHER_STEP_TIMEOUT = 15.0
EXPLORER_STEP_TIMEOUT = 20.0
# Black Hole Event Parameters
BLACK_HOLE_COOLDOWN_STEPS = 50
BH_PRESSURE_THRESHOLD = 0.4
BH_SPREAD_FRAC = 0.5
BH_BG_FRAC = 0.2
BH_DIFFUSION_ETA = 0.15
BH_FIELD_LEAK = 0.02
BLACK_HOLE_K = 16 # Number of KNN links for remnant
# Memory and Temperature
CONSOLIDATE_MIN = 20
TEMP_HALF_LIFE_VIVID = 8
TEMP_HALF_LIFE_HOT = 24
TEMP_HALF_LIFE_WARM = 72
TEMP_HALF_LIFE_COLD = 240
# Misc
SEMANTIC_DOMAIN = "E8_holographic_Conscioussness"
DREAM_MODE_ENABLED = True
LOCAL_GEN_WORKERS = int(os.getenv("LOCAL_GEN_WORKERS", "1"))
GLOBAL_SEED = int(os.getenv("GLOBAL_SEED", "1337"))
# Data sources for ingestion pipeline (example)
DATA_SOURCES: Dict[str, Any] = {} # e.g. {"arxiv_cs_cl": {"type": "arxiv_api", "url": "http://export.arxiv.org/api/query?search_query=cat:cs.CL&sortBy=submittedDate&sortOrder=descending&max_results=10", "schedule_minutes": 120}}

# --- Optional Dependencies ---
try:
    import itertools as _itertools
    from itertools import combinations
except Exception:
    combinations = None
try:
    from datetime import datetime, timezone
except Exception:
    datetime = None
    timezone = None
try:
    import ollama
except Exception:
    ollama = None
try:
    import google.generativeai as genai
except Exception:
    genai = None
try:
    import clifford
    from clifford.g3 import layout as g3_layout, blades as g3_blades
    CLIFFORD_AVAILABLE = True
except Exception:
    clifford = None
    g3_layout, g3_blades = None, None
    CLIFFORD_AVAILABLE = False
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
    from rich.markup import escape
except Exception:
    class Progress:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
        def add_task(self, *args, **kwargs): return 0
        def update(self, *args, **kwargs): pass
    class Console:
        def __init__(self, record=False): pass
        def log(self, *a, **k): print(*a)
        def print(self, *a, **k): print(*a)
        def rule(self, *a, **k): print("-" * 20)
        def export_text(self): return ""
    class Panel(str):
        def __new__(cls, content, **kwargs): return str(content)
    def escape(s): return s
try:
    import networkx as nx
    from networkx.readwrite import json_graph
except Exception:
    nx = None
    class _JG:
        def node_link_data(self, g): return {"nodes": [], "links": []}
        def node_link_graph(self, d): return None
    json_graph = _JG()
try:
    from aiohttp import web
    import aiohttp_cors
    import aiohttp
    import xml.etree.ElementTree as ET
except Exception:
    web = None; aiohttp_cors = None; aiohttp = None; ET = None
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    class _NN:
        Module = object
        ModuleList = list
    nn = _NN()
    F = None
    torch = None
try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN
except Exception:
    PCA, DBSCAN = None, None
try:
    from sklearn.neighbors import KDTree as _SKKDTree
except Exception:
    _SKKDTree = None
try:
    from sklearn.metrics.pairwise import cosine_distances as _sk_cosine_distances, cosine_similarity as _sk_cosine_similarity
except Exception:
    _sk_cosine_distances, _sk_cosine_similarity = None, None
try:
    from scipy.spatial import cKDTree as _SPKDTree
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import eigsh, expm_multiply
    import scipy as sp
except Exception:
    _SPKDTree, csr_matrix, diags, eigsh, expm_multiply, sp = None, None, None, None, None, None


# --- Helper Functions and Classes ---

def get_run_id() -> str:
    """Generates a unique run ID based on the current timestamp."""
    return datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")

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

def export_graph(graph: nx.Graph) -> Dict:
    """Exports a NetworkX graph to a serializable dictionary."""
    if nx is None: return {"nodes": [], "links": []}
    return json_graph.node_link_data(graph)

def classify_geometry_theme(delta_vector: np.ndarray) -> list[str]:
    """Placeholder for classifying movement vectors into themes."""
    if np.linalg.norm(delta_vector) < 0.1:
        return ["stasis"]
    return ["integration", "growth"]

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
        if vector.shape[0] != self.in_dim:
            # Pad or truncate if there's a mismatch
            padded_vec = np.zeros(self.in_dim, dtype=np.float32)
            size_to_copy = min(vector.shape[0], self.in_dim)
            padded_vec[:size_to_copy] = vector[:size_to_copy]
            vector = padded_vec
        return vector @ self.W
# --- Main Code ---

console = Console(record=True)
LAST_INTRINSIC = {}

import sys as _sys, asyncio as _asyncio
if _sys.platform.startswith("win"):
    try:
        if not isinstance(_asyncio.get_event_loop_policy(), _asyncio.WindowsSelectorEventLoopPolicy):
             _asyncio.set_event_loop_policy(_asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

try:
    from profiles.loader import load_profile
except ImportError:
    class _FallbackPrompts:
        def render(self, key, **vars):
            q = vars.get("question") or vars.get("topic") or vars.get("text") or ""
            persona, domain_hint = vars.get("persona", ""), vars.get("domain_hint", "")
            return f"{persona}\n\n{domain_hint}\n\n{q}"
    class _FallbackSem:
        name = "default"; base_domain = "general"
        def persona_prefix(self, mood): return "You are in a balanced state."
        def pre_embed(self, t): return t
        def post_embed(self, v): return v
        def rerank(self, c): return c
    def load_profile(name):
        return _FallbackSem(), _FallbackPrompts()

class KDTree:
    """A wrapper for scikit-learn/scipy KDTree with a pure NumPy fallback."""
    def __init__(self, data):
        if _SKKDTree is not None:
            self._impl = _SKKDTree(np.asarray(data, dtype=np.float32))
            self.n = self._impl.data.shape[0]
            self._is_fallback = False
        elif _SPKDTree is not None:
            self._impl = _SPKDTree(np.asarray(data, dtype=np.float32))
            self.n = self._impl.n
            self._is_fallback = False
        else:
            # Fallback initialization
            self._impl = np.asarray(data, dtype=np.float32)
            self.n = self._impl.shape[0]
            self._is_fallback = True

    def query(self, q, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """
        Finds the k-nearest neighbors for a single vector or a batch of vectors.
        
        Args:
            q: A single vector (1D array) or a batch of vectors (2D array).
            k: The number of neighbors to find.
            
        Returns:
            A tuple of (distances, indices). For a single query, these are 1D arrays
            of shape (k,). For a batch query of N vectors, these are 2D arrays of
            shape (N, k).
        """
        q_arr = np.asarray(q, dtype=np.float32)
        is_single_query = q_arr.ndim == 1
        q_2d = np.atleast_2d(q_arr)

        if not self._is_fallback:
            # Use the efficient scikit-learn or scipy implementation
            d, i = self._impl.query(q_2d, k=k)
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
                    nearest_idx = np.argpartition(distances, k)[:k]
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
        if is_single_query:
            return d.ravel(), i.ravel()
        else:
            return d, i
        
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

MarketFeed = None # Placeholder
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

    def generate_rotor(self, shell: 'DimensionalShell', angle: float) -> clifford.MultiVector:
        """Generates a rotor that rotates by 'angle' in the plane defined by two vectors."""
        pair = self._select_dynamic_pair(shell)

        if pair is None:

            random_bivector = self._random_unit_bivector()
            return (-(random_bivector) * (angle / 2.0)).exp()

        a_vec, b_vec = pair
        a = sum(val * bv for val, bv in zip(a_vec, self.basis_vectors))
        b = sum(val * bv for val, bv in zip(b_vec, self.basis_vectors))

        a = a.normal()
        b = b.normal()

        B = (a ^ b)

        if abs(B) < 1e-9:

            return (-(self._random_unit_bivector()) * (angle / 2.0)).exp()

        B_normalized = B.normal()

        rotor = (-B_normalized * angle / 2.0).exp()
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
        self.layout, self.blades = clifford.Cl(dim)
        self.vectors: Dict[str, clifford.MultiVector] = {}
        self.basis_vectors = [self.blades[f'e{i+1}'] for i in range(dim)]
        self.rotor_generator = CliffordRotorGenerator(mind_instance, self.layout, self.blades)
        self.orientation = self.layout.scalar

        try:
            self._build_bivector_basis()
        except Exception:
            pass
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
        self.orientation = (incremental_rotor * self.orientation).normal()

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
            Bn = B.normal()
        except Exception:
            Bn = None
        if Bn is None:
            try:
                self.spin(float(angle))
            except Exception:
                pass
            return
        try:
            R = (-Bn * (float(angle) / 2.0)).exp()
            self.orientation = (R * self.orientation).normal()
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
                snapped = self.mind._snap_to_lattice(vec_np, self.dim) if hasattr(self.mind, "_snap_to_lattice") else vec_np
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

# Bind helper functions as methods on DimensionalShell
try:
    DimensionalShell._build_bivector_basis = _build_bivector_basis
    DimensionalShell.spin_with_bivector = spin_with_bivector
except Exception:
    pass

class ProximityEngine:
    """
    An engine for finding approximate nearest neighbors within and across dimensional shells.
    """
    def __init__(self, shell_dims: List[int], mind_instance: 'E8Mind', console: Console):
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
        if not TORCH_AVAILABLE or self.mind.autoencoder is None: return []
        if source_dim == target_dim:
            return self.find_similar_in_shell(query_vector, target_dim, k)

        with torch.no_grad():
            source_tensor = torch.from_numpy(query_vector).float().unsqueeze(0)

            projected_tensor = self.mind.autoencoder.project_between_dim(source_tensor, source_dim=source_dim, target_dim=target_dim)

            if projected_tensor is None:
                return []

            projected_vector = projected_tensor.squeeze(0).numpy()

        return self.find_similar_in_shell(projected_vector, target_dim, k)

    def hybrid_rerank_query(self, query_vector: np.ndarray, rerank_shell_dim: int, initial_k: int = 20, final_k: int = 5) -> List[tuple[str, float]]:
        """
        Performs a hybrid search:
        1. Fast ANN search in high-dimensional main memory to get initial candidates.
        2. Projects candidates and query into a lower-dimensional shell.
        3. Re-ranks candidates based on distance in the abstract shell space.
        """
        if rerank_shell_dim not in self.shell_dims or not TORCH_AVAILABLE or self.mind.autoencoder is None:
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

        with torch.no_grad():
            query_tensor_high_dim = torch.from_numpy(query_vector).float().unsqueeze(0)
            query_tensor_low_dim = self.mind.autoencoder.project_to_dim(query_tensor_high_dim, rerank_shell_dim)

        if query_tensor_low_dim is None:
            return initial_candidates[:final_k]

        query_vector_low_dim = query_tensor_low_dim.squeeze(0).numpy()

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

    def build(self, mind: "E8Mind", out_dim: int = None, keep_k: int = None) -> np.ndarray:
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
    def __init__(self, console: Console):
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
    def __init__(self, memory_manager: 'MemoryManager', llm_pool: 'AsyncLLMPool', console: Console):
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

class InsightAgent:
    """An agent that generates new concepts and learns from an insight-driven reward signal."""
    def __init__(self, llm_pool: 'AsyncLLMPool', novelty_scorer: NoveltyScorer, console: Console):
        self.console = console
        self.llm_pool = llm_pool
        self.novelty_scorer = novelty_scorer
        self.reward_history = deque(maxlen=100)

    async def create_hybrid_concept(self, concept_a: Dict, concept_b: Dict) -> str:
        """Uses an LLM to synthesize a new, hybrid concept from two source concepts."""
        prompt = (
            f"You are a creative synthesizer of ideas. Combine the core essence of the following two concepts "
            f"into a single, novel, and coherent hybrid concept. Describe the new idea in one or two sentences.\n\n"
            f"Concept A: '{concept_a.get('metaphor', concept_a.get('label', ''))}'\n"
            f"Concept B: '{concept_b.get('metaphor', concept_b.get('label', ''))}'\n\n"
            f"Hybrid Concept:"
        )
        new_concept_text = await self.llm_pool.enqueue_and_wait(prompt, max_tokens=100, temperature=0.85)
        return new_concept_text.strip()

    def learn_from_reward(self, reward: float):
        """
        Stores the reward and, in a full RL implementation, would update the agent's policy.
        """
        self.reward_history.append(reward)
        if len(self.reward_history) > 10:
            avg_reward = np.mean(self.reward_history)
            self.console.log(f"[InsightAgent] Average Insight Reward: {avg_reward:.3f}")

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

if TORCH_AVAILABLE:
    class GaussianActor(nn.Module):
        def __init__(self, state_dim, action_dim, max_action):
            super().__init__()
            self.l1 = nn.Linear(state_dim, 256)
            self.l2 = nn.Linear(256, 256)
            self.mu = nn.Linear(256, action_dim)
            self.log_std = nn.Linear(256, action_dim)
            self.max_action = float(max_action)
        def forward(self, state):
            h = torch.relu(self.l1(state))
            h = torch.relu(self.l2(h))
            mu = self.mu(h)
            log_std = torch.clamp(self.log_std(h), -5.0, 2.0)
            return mu, log_std
        def sample(self, state):
            mu, log_std = self.forward(state)
            std = torch.exp(log_std)
            normal = torch.distributions.Normal(mu, std)
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            action = y_t * self.max_action
            log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=1, keepdim=True)
            mu_action = torch.tanh(mu) * self.max_action
            return action, log_prob, mu_action

    class QCritic(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.l1 = nn.Linear(state_dim + action_dim, 256)
            self.l2 = nn.Linear(256, 256)
            self.l3 = nn.Linear(256, 1)
        def forward(self, state, action):
            x = torch.cat([state, action], dim=-1)
            x = torch.relu(self.l1(x))
            x = torch.relu(self.l2(x))
            return self.l3(x)

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
            self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            self.actor = GaussianActor(state_dim, action_dim, max_action).to(self.device)
            self.actor_old = GaussianActor(state_dim, action_dim, max_action).to(self.device)
            self.actor_old.load_state_dict(self.actor.state_dict())
            self.critics = nn.ModuleList([QCritic(state_dim, action_dim).to(self.device) for _ in range(4)])
            self.critics_target = nn.ModuleList([QCritic(state_dim, action_dim).to(self.device) for _ in range(4)])
            for i in range(4):
                self.critics_target[i].load_state_dict(self.critics[i].state_dict())
            self.active_critics = 2
            self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
            self.critic_opts = [torch.optim.Adam(self.critics[i].parameters(), lr=3e-4) for i in range(4)]
            self.log_alpha = torch.nn.Parameter(torch.tensor(0.0, device=self.device))
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=3e-4)
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
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                if deterministic:
                    mu, _ = self.actor.forward(s)
                    a = torch.tanh(mu) * self.max_action
                else:
                    a, _, _ = self.actor.sample(s)
            return a.squeeze(0).cpu().numpy().astype("float32")

        def store(self, state, action, next_state, reward, done):
            self.replay.add(state, action, next_state, reward, done)

        def epistemic_std(self, state, action):
            try:
                s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                a = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
                qs = []
                with torch.no_grad():
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
                state_np, action_np, next_state_np, reward_np, done_np, weights_np, indices = batch
                weights = torch.tensor(weights_np, dtype=torch.float32, device=self.device)
            else:
                state_np, action_np, next_state_np, reward_np, done_np = batch
                weights, indices = None, None

            state = torch.tensor(state_np, dtype=torch.float32, device=self.device)
            action = torch.tensor(action_np, dtype=torch.float32, device=self.device)
            next_state = torch.tensor(next_state_np, dtype=torch.float32, device=self.device)
            reward = torch.tensor(reward_np, dtype=torch.float32, device=self.device)
            done = torch.tensor(done_np, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                next_a, next_logp, _ = self.actor.sample(next_state)
                q_next = []
                for i in range(self.active_critics):
                    q_next.append(self.critics_target[i](next_state, next_a))
                q_next = torch.min(torch.stack(q_next, dim=0), dim=0).values
                target_v = q_next - self.log_alpha.exp() * next_logp
                target_q = reward + (1.0 - done) * self.gamma * target_v

            td_errors_for_buffer = []
            for i in range(self.active_critics):
                qi = self.critics[i](state, action)
                if i == 0 and use_per: # Calculate TD errors once for buffer update
                    td_errors = torch.abs(qi - target_q).detach().cpu().numpy().flatten()
                    td_errors_for_buffer = td_errors
                
                if use_per and weights is not None:
                    li = (torch.nn.functional.mse_loss(qi, target_q, reduction='none') * weights).mean()
                else:
                    li = torch.nn.functional.mse_loss(qi, target_q)
                
                self.critic_opts[i].zero_grad()
                li.backward()
                self.critic_opts[i].step()

            if use_per and len(td_errors_for_buffer) > 0:
                self.replay.update_priorities(indices, td_errors_for_buffer)

            a, logp, _ = self.actor.sample(state)
            q_pi = []
            for i in range(self.active_critics):
                q_pi.append(self.critics[i](state, a))
            q_pi = torch.min(torch.stack(q_pi, dim=0), dim=0).values
            with torch.no_grad():
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

class Probe:
    def __init__(self, run_id):
        self.path = get_path("debug/probe.ndjson", run_id)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._lock = asyncio.Lock()

    async def log(self, **kv):
        kv["ts"] = datetime.now(timezone.utc).isoformat()
        try:
            async with self._lock:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(kv, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def log_sync(self, **kv):
        kv["ts"] = datetime.now(timezone.utc).isoformat()
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(kv, ensure_ascii=False) + "\n")
        except Exception:
            pass

def set_asyncio_exception_logger(probe: Probe):
    try:
        loop = asyncio.get_running_loop()
        def _handler(loop, context):
            msg = context.get("message", "")
            exc = context.get("exception")
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)) if exc else msg
            try:
                if loop.is_running() and not loop.is_closed():
                    asyncio.create_task(probe.log(ev="loop_exception", message=msg, traceback=tb))
                else:
                    probe.log_sync(ev="loop_exception", message=msg, traceback=tb)
            except Exception:
                probe.log_sync(ev="loop_exception", message=msg, traceback=tb)
        loop.set_exception_handler(_handler)
    except RuntimeError: # No running loop
        pass

class InstrumentedLock:
    def __init__(self, name="lock", probe=None):
        self._lock = asyncio.Lock()
        self.name = name
        self.probe = probe
        self._t_acq = 0.0

    async def __aenter__(self):
        t0 = time.time()
        await self._lock.acquire()
        wait = (time.time() - t0) * 1000.0
        if self.probe:
            await self.probe.log(ev="lock_acquire", lock=self.name, wait_ms=round(wait,2))
        self._t_acq = time.time()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        held = (time.time() - self._t_acq) * 1000.0
        if self.probe:
            await self.probe.log(ev="lock_release", lock=self.name, held_ms=round(held,2))
        self._lock.release()

class AsyncOpenAIClient:
    def __init__(self, api_key: str, console: Console):
        from openai import AsyncOpenAI, BadRequestError
        self.client = AsyncOpenAI(api_key=api_key)
        self.BadRequestError = BadRequestError
        self.console = console

    async def chat(self, messages, model=None, max_tokens=None, temperature=None):
        try:
            cc = await self.client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
            if cc.choices:
                return (cc.choices[0].message.content or "").strip()
            return "[LLM ERROR] No choices returned from API."
        except self.BadRequestError as e:
            self.console.log(f"[bold red]OpenAI API Error: {e}[/bold red]")
            return f"[LLM ERROR] {e}"

    async def get_logprobs_and_tokens(self, messages, **kwargs):
        return -99.0, []

    async def embedding(self, text, model=None, dimensions=None):
        try:
            res = await self.client.embeddings.create(
                input=[text], model=model or "text-embedding-3-small", dimensions=dimensions)
            return res.data[0].embedding
        except Exception as e:
            self.console.log(f"[bold red]OpenAI Embedding Error: {e}[/bold red]")
            return np.zeros(EMBED_DIM).tolist()

    async def batch_embedding(self, texts, model=None, dimensions=None):
        try:
            res = await self.client.embeddings.create(
                input=texts, model=model or "text-embedding-3-small", dimensions=dimensions)
            return [d.embedding for d in res.data]
        except Exception as e:
            self.console.log(f"[bold red]OpenAI Batch Embedding Error: {e}[/bold red]")
            return [np.zeros(EMBED_DIM).tolist() for _ in texts]

class OllamaClient:
    def __init__(self, ollama_model: str, console: Console):
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
    def __init__(self, api_key: str, model_name: str, console: Console):
        if genai is None:
            raise RuntimeError("google-generativeai is not installed. Please `pip install google-generativeai`. ")
        if not api_key:
            raise ValueError("Gemini API key is required.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
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
            config = genai.types.GenerationConfig(max_output_tokens=max_tokens, temperature=temperature)
            response = await self.model.generate_content_async(gemini_messages, generation_config=config)

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
            result = await genai.embed_content_async(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            self.console.log(f"[bold red]Gemini Embedding Error: {e}[/bold red]")
            return np.zeros(EMBED_DIM).tolist()

    async def batch_embedding(self, texts, model="models/embedding-001", **kwargs):
        try:
            result = await genai.embed_content_async(
                model="models/embedding-001",
                content=texts,
                task_type="retrieval_document"
            )
            return result['embedding']
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
        pid = await self.submit(prompt, **kwargs)
        return await self.get_result(pid)

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
        if csr_matrix is None or diags is None: return np.eye(self.weights.shape[0])
        W = csr_matrix(self.weights, dtype=np.float32)
        deg = np.asarray(W.sum(axis=1)).ravel()
        D_inv_sqrt = diags(1.0 / np.sqrt(np.maximum(deg, 1e-9)))
        return diags(np.ones(W.shape[0])) - D_inv_sqrt @ W @ D_inv_sqrt

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
        self.dt = dt
        self.batch = batch
        self.dephase = dephase
        self.locality_sigma = locality_sigma
        self.seed = seed
        self.topk_amp = topk_amp
        self.non_linearity_strength = non_linearity_strength

class QuantumEngine:
    def __init__(self, physics, config, console: Console):
        self.console = console
        self.physics, self.config = physics, config
        self.psi = np.ones((config.batch, 240), dtype=np.complex64) / np.sqrt(240)
        self.rng = np.random.default_rng(config.seed)
        self.H: Optional[csr_matrix] = None
        self._last_H: Optional[csr_matrix] = None
        self._last_potential: Optional[np.ndarray] = None
        self._last_norm = np.nan
        self._last_energy = np.nan
        self.build_hamiltonian()
        self.console.log("[INIT] Quantum Engine online (Non-Linear Edition).")

    def build_hamiltonian(self, V: Optional[np.ndarray] = None):
        if diags is None or csr_matrix is None: return
        if V is None:
            V = np.zeros(240, dtype=np.float32)

        H = (self.config.gamma * self.physics.L_norm.astype(np.complex64)) + diags(V)
        self.H = csr_matrix(H)

        self._last_H = self.H
        self._last_potential = np.asarray(V).copy()

    def _probs(self):
        p = np.abs(self.psi)**2
        return p / (np.sum(p, axis=1, keepdims=True) + 1e-12)

    def step_adaptive(self, tv_target=0.07, dt_min=0.02, dt_max=1.2):
        if self.H is None or expm_multiply is None:
            return 0.0

        p0 = self._probs()
        H_eff = self.H.copy()
        if self.config.non_linearity_strength != 0:
            feedback = self.config.non_linearity_strength * p0[0]
            H_eff += diags(feedback.astype(np.float32), 0)

        psi_new = expm_multiply(-1j * H_eff * self.config.dt, self.psi.T).T
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

    def measure_ablation(self, prev_idx:int, sigma:float=None, window:int=5, trials:int=512):
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
    def __init__(self, physics, config, console: Console):
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
            if sp is None or eigsh is None: raise RuntimeError("scipy is required for layout_2d")
            _, vecs = eigsh(sp.csr_matrix(L), k=4, which='SM')
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

    def shadow_set(self, bulk_idx: int, pos_hint_xy: np.ndarray = None) -> np.ndarray:
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

    def usage_bits(self, store: dict, region_nodes: np.ndarray, slice_id: int = None) -> float:
        rset = set(int(i) for i in region_nodes.tolist())
        bits = 0.0
        for (nid, sid), val in store.items():
            if nid in rset and (slice_id is None or sid == int(slice_id)):
                bits += 32.0
        return float(bits)

    def deficit_ratio(self, store: dict, region_nodes: np.ndarray, slice_id: int = None) -> float:
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
        if self._use_torch:
            class AE(nn.Module):
                def __init__(self, D, C):
                    super().__init__(); self.enc = nn.Linear(D, C, bias=False); self.dec = nn.Linear(C, D, bias=False)
                def forward(self, x): return self.enc(x), self.dec(self.enc(x))
            self.net = AE(self.in_dim, self.code_dim)
            for p in self.net.parameters(): nn.init.xavier_uniform_(p.data)
            self.opt = torch.optim.Adam(self.net.parameters(), lr=3e-3)
        self.ready = self._use_torch

    def fit(self, X: np.ndarray, epochs=5, bs=64):
        if X.shape[0] < max(bs, self.code_dim + 1): return
        if self._use_torch:
            self.net.train(); loss_fn = nn.MSELoss()
            for _ in range(epochs):
                for i in range(0, X.shape[0], bs):
                    b = torch.from_numpy(X[np.random.permutation(X.shape[0])[i:i+bs]])
                    self.opt.zero_grad(); _, xh = self.net(b); loss = loss_fn(xh, b)
                    loss.backward(); self.opt.step()
        elif PCA is not None:
            self._pca = PCA(n_components=self.code_dim).fit(X)
        self.ready = True

    def encode(self, x: np.ndarray) -> np.ndarray:
        x = x.reshape(1, -1)
        if self._use_torch and self.ready:
            self.net.eval()
            with torch.no_grad(): z, _ = self.net(torch.from_numpy(x))
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
    def __init__(self, embedding_fn, console: Console):
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
        for name, goal_data in self.goals.items():
            sim = MemoryManager._cos_sim(vector, goal_data["embedding"])
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
        sorted_goals = sorted(self.goals.items(), key=lambda item: -item[1]["activation"])
        return [(name, data["description"]) for name, data in sorted_goals[:k]]

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
            goal_vec += data["activation"] * data["embedding"]
        if np.linalg.norm(goal_vec) == 0:
            return 0.0

        resonance_count = 0
        for dim, shell in self.dimensional_shells.items():
            matrix, _ = shell.get_all_vectors_as_matrix()
            if matrix is None: continue

            projected_goal_vec = np.zeros(dim)
            size_to_copy = min(EMBED_DIM, dim)
            projected_goal_vec[:size_to_copy] = goal_vec[:size_to_copy]

            similarities = cosine_similarity(matrix, projected_goal_vec.reshape(1, -1))
            shell_resonance = np.mean(similarities)
            total_resonance += shell_resonance
            resonance_count += 1

        return total_resonance / resonance_count if resonance_count > 0 else 0.0

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
    def __init__(self, console: Console, baseline=0.5, decay_rate=0.995):
        self.console = console
        self.baseline, self.decay_rate = baseline, decay_rate
        self.mood_vector = {"intensity": 0.5, "entropy": 0.5, "coherence": 0.5, "positivity": 0.5, "fluidity": 0.5, "intelligibility": 0.5}
        self.event_queue = deque()
        self._wx_last_code = None
        self._wx_repeat = 0
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
    def __init__(self, embedding_fn, llm_caller, console: Console, decay_rate=0.95, accumulation_rate=0.004):
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
            time_delta = current_step - influence["start_step"]
            if time_delta < 0: continue
            decay_factor = math.exp(-influence["decay"] * time_delta)
            oscillation_factor = math.cos(influence["frequency"] * time_delta)
            current_strength = influence["initial_strength"] * decay_factor * oscillation_factor
            if abs(current_strength) > 0.001:
                total_influence_vec += current_strength * influence["vector"]
                active_influences.append(influence)
        if np.linalg.norm(total_influence_vec) > 0:
             self.bias_vector += total_influence_vec
             self.bias_vector = normalize_vector(self.bias_vector)
        self.influences = active_influences

    async def track_concept(self, label, weight=1.0):
        vec = await self.embedding_fn(label)
        if np.linalg.norm(vec) > 0:
            if self.bias_vector is None: self.bias_vector = np.zeros_like(vec)
            if self.bias_vector.shape != vec.shape: return
            self.bias_vector += self.accumulation_rate * normalize_vector(vec) * weight
            self.bias_vector = normalize_vector(self.bias_vector)
            if len(self.bias_history) == 0 or np.linalg.norm(self.bias_history[-1] - self.bias_vector) > 0.01:
                self.bias_history.append(self.bias_vector.copy())

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

class MemoryManager:
    def __init__(self, embedding_fn, mood, subconscious, run_id, probe, llm_caller, mind_instance, **kwargs):
        self.embedding_fn, self.mood, self.subconscious, self.run_id = embedding_fn, mood, subconscious, run_id
        self.probe, self.llm_caller, self.mind = mind_instance.probe, mind_instance.llm_pool, mind_instance
        self.lock = InstrumentedLock("memory", probe=self.probe)

        self.graph_db = GraphDB()
        self.main_vectors: Dict[str, np.ndarray] = {}
        self.main_kdtree: Optional[KDTree] = None
        self._main_storage_ids: List[str] = []
        self._main_storage_matrix = np.empty((0, EMBED_DIM), dtype=np.float32)
        self.label_to_node_id: Dict[str, str] = {}
        self.autobio_index = defaultdict(list)
        self.symbol_history = defaultdict(list)
        self.core_self = np.zeros(EMBED_DIM)
        self.core_self_strength = 0.0
        self.consolidation_buffer: List[Dict] = []
        self.consolidation_task: Optional[asyncio.Task] = None
        self.memory_consolidation_min = int(kwargs.get("memory_consolidation_min", 50))
        self.max_knn_links = int(kwargs.get("max_knn_links", 4))
        self._compressor = TinyCompressor(in_dim=EMBED_DIM, code_dim=8)
        self.pending_embeddings: List[np.ndarray] = []
        self._repeat_ngrams: Dict[Tuple, int] = defaultdict(int)
        self.field: Dict[str, float] = defaultdict(float)
        self.background_temp = 0.0
        self.active_locks: Dict[Tuple[str, str], int] = {}

    def apply_soft_link(self, id_a, id_b, weight_delta=0.05, decay=0.985):
        try:
            if not self.graph_db.graph.has_edge(id_a, id_b):
                self.graph_db.graph.add_edge(id_a, id_b, weight=float(weight_delta), type="proximity")
            else:
                w = float(self.graph_db.graph[id_a][id_b].get("weight", 0.0))
                self.graph_db.graph[id_a][id_b]["weight"] = w * float(decay) + float(weight_delta)
            if hasattr(self, "spike_temperature"):
                self.spike_temperature(id_a, 0.02)
                self.spike_temperature(id_b, 0.02)
        except Exception:
            pass

    def apply_gravitational_lock(self, node_id_a: str, node_id_b: str, duration_steps: int):
        if not self.graph_db.get_node(node_id_a) or not self.graph_db.get_node(node_id_b):
            return
        expiry_step = self.mind.step_num + duration_steps
        edge_tuple = tuple(sorted((node_id_a, node_id_b)))
        self.graph_db.add_edge(node_id_a, node_id_b, type="gravitational_lock", weight=5.0)
        self.active_locks[edge_tuple] = expiry_step
        self.mind.console.print(Panel(f"Gravitational Lock engaged between [cyan]{node_id_a}[/] and [cyan]{node_id_b}[/]",
                        title="[bold yellow]ANCHOR[/]", border_style="yellow"))

    def decay_locks(self):
        current_step = self.mind.step_num
        expired_locks = [edge for edge, expiry in self.active_locks.items() if current_step >= expiry]
        for edge in expired_locks:
            node_a, node_b = edge
            if self.graph_db.graph.has_edge(node_a, node_b):
                edge_data = self.graph_db.graph.get_edge_data(node_a, node_b)
                if edge_data and edge_data.get("type") == "gravitational_lock":
                    self.graph_db.graph.remove_edge(node_a, node_b)
            del self.active_locks[edge]

    def neighbors(self, nid: str):
        return self.graph_db.get_neighbors(nid)

    def find_latest_node_at_blueprint(self, blueprint_index: int) -> Optional[str]:
        for node_id, data in reversed(list(self.graph_db.graph.nodes(data=True))):
            if data.get("blueprint_location_id") == blueprint_index:
                return node_id
        return None

    def node_vec(self, nid: str):
        v = self.main_vectors.get(nid)
        if v is None: return None
        n = float(np.linalg.norm(v)) + 1e-12
        return (v / n).astype(np.float32)

    def _rebuild_main_kdtree(self):
        if self.main_vectors:
            self._main_storage_ids = list(self.main_vectors.keys())
            self._main_storage_matrix = np.array([self.main_vectors[nid] for nid in self._main_storage_ids], dtype=np.float32)
            if self._main_storage_matrix.shape[0] > 0:
                self.main_kdtree = KDTree(self._main_storage_matrix)
            else:
                self.main_kdtree = None
        else:
            self._main_storage_ids = []
            self._main_storage_matrix = np.empty((0, EMBED_DIM), dtype=np.float32)
            self.main_kdtree = None

    def find_similar_in_main_storage(self, query_vector: np.ndarray, k: int = 5) -> List[tuple[str, float]]:
        if self.main_kdtree is None or not self._main_storage_ids:
            return []
        k = min(k, len(self._main_storage_ids))
        if k == 0:
            return []
        distances, indices = self.main_kdtree.query(query_vector, k=k)
        node_ids = self._main_storage_ids
        if isinstance(indices, (int, np.integer)):
            return [(node_ids[indices], float(distances))]
        return [(node_ids[i], d) for d, i in zip(np.atleast_1d(distances), np.atleast_1d(indices))]

    def get_average_nearest_neighbor_distance(self, sample_k: int = 256) -> float:
        if self.main_kdtree is None:
            return 1.0
        n = len(self._main_storage_ids)
        if n < 2:
            return 1.0
        rng = np.random.default_rng()
        idxs = rng.choice(n, size=min(sample_k, n), replace=False)
        pts = self._main_storage_matrix[idxs]
        dists, _ = self.main_kdtree.query(pts, k=2)
        nn_dists = dists[:, 1]
        mean_dist = float(np.mean(nn_dists))
        return mean_dist if np.isfinite(mean_dist) and mean_dist > 1e-9 else 1.0

    def _compute_rings(self, center_id: str, radius: int):
        rings = {center_id: 0}
        frontier = [center_id]
        for d in range(1, radius + 1):
            nxt = []
            for n in frontier:
                for nb in self.neighbors(n):
                    if nb not in rings:
                        rings[nb] = d
                        nxt.append(nb)
            frontier = nxt
            if not frontier: break
        return rings

    async def _cosmological_spread(self, remnant_vec: np.ndarray, mass: float):
        if mass <= 0 or remnant_vec is None:
            return
        self.mind.console.print(Panel("A cognitive aeon ends. The remnant seed will now imprint a new universe of thought.",
                                    title="[bold #D02090]COSMOLOGICAL Spread[/]", border_style="#D02090"))

        projected_vec_8d = None
        if TORCH_AVAILABLE and self.mind.autoencoder and self.mind.autoencoder.is_trained:
            with torch.no_grad():
                remnant_tensor = torch.from_numpy(remnant_vec).float().unsqueeze(0)

                projected_tensor = self.mind.autoencoder.project_to_dim(remnant_tensor, 8)
                if projected_tensor is not None:
                    projected_vec_8d = projected_tensor.squeeze(0).numpy()

        if projected_vec_8d is None:
            self.mind.console.log("[yellow]Cosmological spread skipped: Autoencoder projection to 8D failed or unavailable.[/yellow]")
            return

        e8_roots = self.mind.physics.roots_unit

        imprint_scores = e8_roots @ normalize_vector(projected_vec_8d)

        beta = 1.0
        exp_scores = np.exp(beta * imprint_scores)
        energy_distribution = exp_scores / np.sum(exp_scores)
        new_holographic_field = defaultdict(float)
        total_energy_to_distribute = BH_SPREAD_FRAC * float(mass)
        for node_index in range(len(e8_roots)):
            node_energy = total_energy_to_distribute * energy_distribution[node_index]
            node_id_at_location = self.find_latest_node_at_blueprint(node_index)
            if node_id_at_location:
                new_holographic_field[node_id_at_location] = node_energy
        self.field = new_holographic_field
        self.background_temp = BH_BG_FRAC * float(mass)

    def diffuse_field(self, eta: float = BH_DIFFUSION_ETA, leak: float = BH_FIELD_LEAK):
        if not self.field:
            self.background_temp *= (1.0 - leak)
            return
        new_field = defaultdict(float)
        for n, val in list(self.field.items()):
            if val <= 1e-12:
                continue
            stay = max(0.0, (1.0 - eta - leak)) * val
            new_field[n] += stay
            if eta > 0.0:
                nbrs = self.neighbors(n)
                deg = len(nbrs) or 1
                portion = (eta * val) / deg
                for nb in nbrs:
                    new_field[nb] += portion
        self.field = new_field
        self.background_temp *= (1.0 - leak)

    def _path(self, rel: str) -> str:
        return get_path(rel, self.run_id)

    @staticmethod
    def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        return np.dot(a, b) / (norm_a * norm_b) if norm_a > 1e-9 and norm_b > 1e-9 else 0.0

    async def add_entry(self, entry_data: dict, parent_ids: Optional[List[str]] = None, target_shells: Optional[List[int]] = None) -> str:
        pre_vec = entry_data.get("embedding")
        if entry_data.get("type") == "void":
            pre_vec = np.zeros(EMBED_DIM, dtype=np.float32)

        if not isinstance(pre_vec, np.ndarray) or np.linalg.norm(pre_vec) == 0:
            txt = f"{entry_data.get('label', '')}: {entry_data.get('metaphor', '')}"
            if txt.strip():
                raw_vec = await self.embedding_fn(txt.strip())
                if raw_vec is not None and np.linalg.norm(raw_vec) > 0:
                    mood_mod = self.mood.get_mood_modulation_vector(raw_vec.shape[0])
                    mood_blend = mood_get(self.mood.mood_vector, "intensity") * 0.15
                    pre_vec = normalize_vector(raw_vec + mood_blend * mood_mod)
                else:
                    pre_vec = np.zeros(EMBED_DIM)
            else:
                pre_vec = np.zeros(EMBED_DIM)

        if np.linalg.norm(pre_vec) > 0:
            pre_vec = normalize_vector(pre_vec)

        entry_data["embedding"] = pre_vec

        if pre_vec is not None:
            self.pending_embeddings.append(pre_vec)

        if pre_vec is not None and TORCH_AVAILABLE and self.mind.autoencoder and self.mind.autoencoder.is_trained:
            with torch.no_grad():
                if np.linalg.norm(pre_vec) > 1e-8:
                    source_tensor = torch.from_numpy(pre_vec).float().unsqueeze(0)
                    x_embed_tensor = self.mind.autoencoder.project_between_dim(source_tensor, EMBED_DIM, 8)
                    if x_embed_tensor is not None:
                        x_embed_8d = x_embed_tensor.numpy().squeeze()
                        location_id = self.mind.physics.find_nearest_root_index(x_embed_8d)
                        if location_id is not None:
                            entry_data["blueprint_location_id"] = location_id

        async with self.lock:
            entry_data.setdefault("temperature", 1.0)
            entry_data.setdefault("age", 0)
            entry_data.setdefault("last_step", self.mind.step_num)
            entry_data["mood_context"] = self.mood.mood_vector.copy()
            entry_data["timestamp"] = datetime.now(timezone.utc).isoformat()
            rating = entry_data.get("rating", 0.5)
            if rating > 0.8 and pre_vec is not None:
                current_step = entry_data.get("step", 0)
                self.subconscious.add_waveform_influence(pre_vec, rating, current_step)
            self._penalize_repeats_in_entry(entry_data)
            self.mood.process_event("new_concept", rating=rating, metaphor_themes=entry_data.get("metaphor_themes", []))
            node_id = self._add_node_to_ltm_locked(entry_data, precomputed_vec=pre_vec, target_shells=target_shells)
            if parent_ids:
                for parent_id in parent_ids:
                    if self.graph_db.get_node(parent_id):
                        self.graph_db.add_edge(node_id, parent_id, type="reflection_source", weight=0.9)
            if entry_data.get("rating", 0.5) > 0.8:
                self.spike_temperature(node_id, amount=1.5)
                node = self.graph_db.get_node(node_id)
                if node: node["vivid_until_step"] = self.mind.step_num + 24
            if entry_data.get("type") not in ["meta_reflection", "phase_summary"]:
                self.consolidation_buffer.append(entry_data)
            self.update_shell_tension(node_id)

        if self.subconscious and entry_data.get("label") and pre_vec is not None:
            await self.subconscious.track_concept(entry_data["label"], weight=entry_data.get("rating", 0.5))

        if len(self.consolidation_buffer) >= self.memory_consolidation_min and (self.consolidation_task is None or self.consolidation_task.done()):
            self.consolidation_task = asyncio.create_task(self._consolidate_memories())

        return node_id

    def _add_node_to_ltm_locked(self, node_data: Dict[str, Any], precomputed_vec: Optional[np.ndarray] = None, target_shells: Optional[List[int]] = None) -> str:
        if precomputed_vec is None: return ""
        idx_val = node_data.get("idx")
        if not idx_val:
            content_str = f"{node_data.get('label', '')}{node_data.get('metaphor', '')}{time.time()}"
            idx_val = hashlib.sha1(content_str.encode()).hexdigest()[:16]
            node_data['idx'] = idx_val
        if self.graph_db.graph.has_node(idx_val):
            self.spike_temperature(idx_val, amount=0.5)
            node = self.graph_db.get_node(idx_val)
            if node: node['last_step'] = self.mind.step_num
            return idx_val

        self.graph_db.add_node(idx_val, **node_data)
        self.main_vectors[idx_val] = precomputed_vec
        top_k = int(os.getenv("E8_KNN_ON_ADD", "8"))
        if len(self.main_vectors) > top_k:
            sims = []
            for other_id, vec in self.main_vectors.items():
                if other_id == idx_val: continue
                s = float(np.dot(precomputed_vec, vec) / (np.linalg.norm(precomputed_vec) * np.linalg.norm(vec) + 1e-9))
                sims.append((s, other_id))
            from heapq import nlargest
            for s, other_id in nlargest(top_k, sims, key=lambda t: t[0]):
                self.graph_db.add_edge(idx_val, other_id, type="knn", weight=s)

        shells_to_update = [self.mind.dimensional_shells[d] for d in target_shells] if target_shells else self.mind.dimensional_shells.values()
        for shell in shells_to_update:
            shell.add_vector(idx_val, precomputed_vec)

        if 'label' in node_data:
            self.label_to_node_id[node_data['label']] = idx_val
        return idx_val

    def update_shell_tension(self, node_id: str):
        if not TORCH_AVAILABLE or not self.mind.autoencoder or node_id not in self.main_vectors: return
        snapped_vectors_in_shells = []
        for dim, shell in self.mind.dimensional_shells.items():
            vec = shell.get_vector(node_id)
            if vec is not None:
                snapped_vectors_in_shells.append((dim, vec))
        if len(snapped_vectors_in_shells) < 2:
            node = self.graph_db.get_node(node_id)
            if node: node['shell_tension'] = 0.0
            return

        reconstructed_vectors = []
        with torch.no_grad():
            for dim, vec in snapped_vectors_in_shells:
                if dim == EMBED_DIM:
                    reconstructed_vectors.append(vec)
                    continue
                tensor_vec = torch.from_numpy(vec).float().unsqueeze(0)
                reconstructed_tensor = self.mind.autoencoder.project_between_dim(tensor_vec, dim, EMBED_DIM)
                if reconstructed_tensor is not None:
                    reconstructed_vectors.append(reconstructed_tensor.squeeze(0).numpy())

        if len(reconstructed_vectors) < 2:
            node = self.graph_db.get_node(node_id)
            if node: node['shell_tension'] = 0.0
            return

        distances = cosine_distances(np.array(reconstructed_vectors), np.array(reconstructed_vectors))
        tension_score = np.mean(distances[np.triu_indices_from(distances, k=1)])
        node = self.graph_db.get_node(node_id)
        if node: node['shell_tension'] = float(tension_score)

    def _penalize_repeats_in_entry(self, entry: Dict[str, Any]):
        text = f"{entry.get('label', '')} {entry.get('metaphor', '')}"
        words = re.findall(r'\b\w{3,}\b', text.lower())
        rating = entry.get("rating", 0.5)
        for i in range(len(words) - 2):
            trigram = tuple(words[i:i+3])
            self._repeat_ngrams[trigram] += 1
            if self._repeat_ngrams[trigram] > 2:
                penalty = 0.05 * (self._repeat_ngrams[trigram] - 2)
                rating = max(0.1, rating - penalty)
        entry["rating"] = rating

    def spike_temperature(self, node_id, amount=1.0):
        node = self.graph_db.get_node(node_id)
        if node:
            node['temperature'] = node.get('temperature', 1.0) + amount

    async def apply_decay(self):
        async with self.lock:
            decay_vivid, decay_hot, decay_warm, decay_cold = 0.5**(1.0/TEMP_HALF_LIFE_VIVID), 0.5**(1.0/TEMP_HALF_LIFE_HOT), 0.5**(1.0/TEMP_HALF_LIFE_WARM), 0.5**(1.0/TEMP_HALF_LIFE_COLD)
            nodes_to_update = list(self.graph_db.graph.nodes(data=True))
            for node_id, data in nodes_to_update:
                temp = data.get('temperature', 1.0)
                if data.get("vivid_until_step", -1) > self.mind.step_num: temp *= decay_vivid
                elif temp > 1.5: temp *= decay_hot
                elif temp > 0.5: temp *= decay_warm
                else: temp *= decay_cold
                data['temperature'] = max(0.01, temp * 0.995)
                data['age'] = data.get('age', 0) + 1

    def find_event_horizon(self, density_threshold=0.20, temp_threshold=1.05, age_threshold=1):
        graph, candidates = self.graph_db.graph, []
        for nid, d in graph.nodes(data=True):
            if d.get('age', 0) < age_threshold: continue
            temp = d.get('temperature', 0.0)
            density = self._local_density(nid)
            pressure = float(temp) * float(density)
            candidates.append((nid, pressure, (temp > temp_threshold) and (density >= density_threshold)))
        if not candidates: return None, None
        passing = [c for c in candidates if c[2]]
        pool = passing if passing else candidates
        if not pool: return None, None
        best = max(pool, key=lambda t: t[1])
        return best[0], best[1]

    def _local_density(self, center_id: str, radius: int = 4) -> float:
        try:
            nodes_in_radius = set(nx.ego_graph(self.graph_db.graph, center_id, radius=radius).nodes())
        except nx.NetworkXError:
            return 0.0

        if not nodes_in_radius: return 0.0
        subgraph = self.graph_db.graph.subgraph(nodes_in_radius)
        num_nodes, num_edges = subgraph.number_of_nodes(), subgraph.number_of_edges()
        possible_edges = num_nodes * (num_nodes - 1) / 2
        return num_edges / possible_edges if possible_edges > 0 else 0.0

    def collect_cluster(self, center_id: str, radius: int = 4) -> List[str]:
        if DBSCAN is None:
            return list(nx.ego_graph(self.graph_db.graph, center_id, radius=2).nodes())

        try:
            nodes_in_radius = list(nx.ego_graph(self.graph_db.graph, center_id, radius=radius).nodes())
        except nx.NetworkXError:
            return [center_id]

        vectors = [self.main_vectors[nid] for nid in nodes_in_radius if nid in self.main_vectors]
        if len(vectors) < 3: return nodes_in_radius
        vector_matrix = np.array(vectors)

        valid_indices = [i for i, v in enumerate(vectors) if np.any(v)]
        if len(valid_indices) < 3: return [nodes_in_radius[i] for i in valid_indices]

        valid_vector_matrix = vector_matrix[valid_indices]
        valid_node_ids = [nodes_in_radius[i] for i in valid_indices]

        clustering = DBSCAN(eps=0.85, min_samples=2, metric='cosine').fit(valid_vector_matrix)

        if center_id not in valid_node_ids: return [center_id]
        center_vector_idx_in_valid = valid_node_ids.index(center_id)

        center_cluster_label = clustering.labels_[center_vector_idx_in_valid]
        if center_cluster_label == -1: return [center_id]

        cluster_indices_in_valid = np.where(clustering.labels_ == center_cluster_label)[0]

        return [valid_node_ids[i] for i in cluster_indices_in_valid]

    async def synthesize_remnant(self, cluster_nodes: List[str], label_hint: str, is_macro: bool = False) -> Tuple[Optional[Dict], Optional[np.ndarray], float]:
        if not cluster_nodes: return None, None, 0.0
        cluster_data = [self.graph_db.get_node(nid) for nid in cluster_nodes if self.graph_db.get_node(nid)]
        cluster_vectors = [self.main_vectors.get(nid) for nid in cluster_nodes if self.main_vectors.get(nid) is not None]
        total_temp = sum(d.get('temperature', 1.0) for d in cluster_data)
        avg_rating = np.mean([d.get('rating', 0.5) for d in cluster_data]) if cluster_data else 0.5
        mass = total_temp * avg_rating
        if not cluster_vectors: return None, None, mass

        weights = np.array([self.graph_db.get_node(nid).get('temperature', 1.0) for nid in cluster_nodes if nid in self.main_vectors and self.graph_db.get_node(nid)])
        if weights.sum() < 1e-9:
            weights = np.ones(len(cluster_vectors))
        weights /= weights.sum()

        remnant_vec = np.average(np.array(cluster_vectors), axis=0, weights=weights)
        prompt_intro = "A major cognitive reset is occurring. Synthesize the core essence of these diverse ideas into a single, foundational principle for a new era of thought." if is_macro else "Synthesize the following fragmented ideas into a single, dense, core concept."
        fragments = [d.get('metaphor', d.get('label', '')) for d in cluster_data]
        prompt = (f"{prompt_intro} Provide a short, evocative label and a one-sentence metaphor for the new idea.\n\n"
                  f"Ideas: {'; '.join(fragments[:10])}\n\nRespond in JSON format with keys 'label' and 'metaphor'.")
        try:
            response = await self.llm_caller.enqueue_and_wait(prompt, max_tokens=120)
            parsed_response = _parse_json_object(response)
            new_label = parsed_response.get('label', label_hint)
            new_metaphor = parsed_response.get('metaphor', 'A consolidated memory.')
        except Exception:
            new_label, new_metaphor = label_hint, "A synthesized concept."
        remnant_data = {"type": "blackhole_remnant", "label": new_label, "metaphor": new_metaphor, "embedding": remnant_vec,
                        "rating": avg_rating, "step": self.mind.step_num, "is_macro": is_macro}
        return remnant_data, remnant_vec, mass

    def fold_and_prune(self, cluster_nodes: List[str]):
        for node_id in cluster_nodes:
            node = self.graph_db.get_node(node_id)
            if node:
                node['folded'] = True
                node['temperature'] *= 0.1
                for shell in self.mind.dimensional_shells.values():
                    if node_id in shell.vectors:
                        del shell.vectors[node_id]

    async def _consolidate_memories(self):
        async with self.lock:
            if len(self.consolidation_buffer) < CONSOLIDATE_MIN: return
            buffer = self.consolidation_buffer[:]
            self.consolidation_buffer.clear()
        self.mind.console.log(f"[Memory] Consolidating {len(buffer)} memories... (Feature stub)")

    async def snapshot(self):
        async with self.lock:
            filepath = self._path(f"snapshot_step_{self.mind.step_num:06d}.json")
            graph_data = export_graph(self.graph_db.graph)
            self._rebuild_main_kdtree()
            main_vectors_serializable = {nid: vec.tolist() for nid, vec in self.main_vectors.items()}
            snapshot_data = { "graph": graph_data, "main_vectors": main_vectors_serializable, "step": self.mind.step_num,
                              "mood": self.mind.mood.mood_vector, "subconscious_narrative": self.mind.subconscious.narrative, }
        safe_json_write(filepath, snapshot_data)
        all_snapshots = sorted(glob.glob(self._path("snapshot_step_*.json")), key=os.path.getmtime)
        while len(all_snapshots) > 10:
            os.remove(all_snapshots.pop(0))

    async def mutate_memory(self, k: int = 1):
        mood_entropy = self.mind.mood.get_entropy_level()
        num_mutations = k + int(mood_entropy * 3)
        temp_boost = 1 + mood_entropy * 0.5

        async with self.lock:
            candidates = []
            for nid, data in self.graph_db.graph.nodes(data=True):
                if data.get("type") == "concept" and not data.get("folded") and "novelty_score" in data:
                    temp = data.get("temperature", 0.5)
                    if 0.3 < temp < 1.5 and data.get("novelty_score", 0.0) > 0.6:
                        candidates.append((data["novelty_score"], nid, data))

        if not candidates:
            self.mind.console.log("[MUTATION] No high-potential seeds found for mutation.")
            return

        candidates.sort(key=lambda x: x[0], reverse=True)
        seeds, new_metaphors, parent_ids = candidates[:num_mutations], [], []

        for _, seed_id, seed_data in seeds:
            parent_ids.append(seed_id)
            seed_text = seed_data.get("metaphor", seed_data.get("label", "a thought"))
            mutation_type = random.choice(["analogy", "contrast", "merge"])
            prompt = (f"Given the idea: '{seed_text}', generate a novel variant via {mutation_type}. "
                      f"Use a creative, metaphorical style. Respond with only the new variant.")
            try:
                mutation_text = await self.llm_caller.enqueue_and_wait(prompt, max_tokens=100, temperature=0.8 * temp_boost)
                if mutation_text and not mutation_text.startswith("[LLM"):
                    new_metaphors.append(mutation_text)
            except Exception as e:
                self.mind.console.log(f"[MUTATION] LLM call failed for seed {seed_id}: {e}")

        if not new_metaphors: return

        new_entries = []
        for new_text in new_metaphors:
            new_emb = await self.embedding_fn(new_text)
            new_rating = await self.mind.rate_concept(new_text)
            new_entries.append({"type": "mutation", "label": sanitize_line(f"mutation of {seeds[0][2].get('label')}"), "metaphor": new_text,
                               "rating": new_rating, "embedding": new_emb, "temperature": 0.2, "step": self.mind.step_num})

        new_node_ids = []
        for entry in new_entries:
            node_id = await self.add_entry(entry, parent_ids=parent_ids)
            new_node_ids.append(node_id)

        self.mind.console.print(Panel(f"Generated {len(new_node_ids)} new mutated concepts (Entropy Boost: {temp_boost:.2f}x).", title="[bold yellow]MUTATION[/bold yellow]", border_style="yellow"))
        return new_node_ids

import random

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
        
        # --- CORRECTED LOGIC ---
        # Instead of always picking the single hottest concept, which is deterministic and
        # repetitive, create a small pool of top candidates and choose randomly from it.
        pool_size = min(len(elig), 5) # Take the top 5 or fewer if not enough concepts exist.
        if pool_size == 0:
            return []

        top_candidates = elig[:pool_size]
        
        # Randomly select k concepts from the top pool without replacement.
        num_to_sample = min(k, len(top_candidates))
        return random.sample(top_candidates, num_to_sample)

    async def run_dream_sequence(self, depth=1):
        """
        UPGRADED: Runs a "thought experiment" instead of a surreal dream.
        It selects a seed concept and generates a hypothetical future narrative based on it.
        """
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

            # Use the corrected picking method to get a non-deterministic seed
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
                        significant_events.append(f"- Type: {event_type}, Title: '{data.get('label', '...')}', Content: '{data.get('metaphor', '...')}'")
            if len(significant_events) < 3: return
            event_summary = "\n".join(significant_events[-15:])
            prompt = (f"You are the mind's historian. The current subconscious narrative is: '{mind_state.subconscious.narrative}'\n"
                      f"The prevailing mood feels like: {mind_state.mood.describe()}\n\n"
                      "Based on the following significant events, write a short, reflective journal entry (2-3 paragraphs) "
                      "that captures the tone and theme of this recent period. Synthesize them into a cohesive story.\n\n"
                      f"Events:\n{event_summary}\n\nJournal Entry for Steps {self.last_narrative_step} to {current_step}:")
            narrative_entry = await self.llm_pool.enqueue_and_wait(prompt, max_tokens=300, temperature=0.7)
            if narrative_entry and not narrative_entry.startswith("[LLM"):
                await self.add_entry(f"## Chronicle: Steps {self.last_narrative_step}-{current_step}\n\n"
                                     f"**Mood**: {mind_state.mood.get_symbolic_weather()} | "
                                     f"**Theme**: {mind_state.synthetic_env.current_theme_region}\n\n"
                                     f"{narrative_entry}\n\n---\n\n")
                self.last_narrative_step = current_step
        except Exception as e:
            console.log(f"[NarrativeStreamer] Failed to generate entry: {e}")

    async def add_entry(self, text: str):
        try:
            with open(self.narrative_file, "a", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            console.log(f"[NarrativeStreamer] Error writing to file: {e}")

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
        last_published_str = self.state.get(name, {}).get("last_published", "1970-01-01T00:00:00Z")
        last_published_dt = datetime.fromisoformat(last_published_str.replace("Z", "+00:00"))
        new_max_published_dt = last_published_dt
        new_entries = []

        async with aiohttp.ClientSession() as session:
            async with session.get(config["url"]) as response:
                if response.status != 200:
                    self.console.log(f"[bold red][Ingestion] arXiv fetch failed for '{name}': HTTP {response.status}[/bold red]")
                    return
                feed_xml = await response.text()

        root = ET.fromstring(feed_xml)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        for entry in root.findall('atom:entry', ns):
            published_str = entry.find('atom:published', ns).text
            published_dt = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
            if published_dt > last_published_dt:
                title = entry.find('atom:title', ns).text.strip()
                summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
                new_entries.append((published_dt, f"{title}: {summary}"))
                if published_dt > new_max_published_dt:
                    new_max_published_dt = published_dt

        if new_entries:
            new_entries.sort(key=lambda x: x[0])
            for _, text in new_entries:
                await self._add_text_as_concept(text, source_name=name)

            if name not in self.state:
                self.state[name] = {}
            self.state[name]["last_published"] = new_max_published_dt.isoformat().replace('+00:00', 'Z')
            safe_json_write(self.state_file, self.state)
            self.console.log(f"[Ingestion] Added {len(new_entries)} new concepts from '{name}'.")

    async def _process_file(self, name: str, config: Dict[str, Any]):
        """Processes a local file if it has been modified."""
        filepath = config.get("path")
        if not os.path.exists(filepath):
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
        if len(self.mood_history) < self.mood_history.maxlen: return
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
        return float(abs(b0-prev[0]) + abs(b1-prev[1]))

if TORCH_AVAILABLE:
    class VariationalAutoencoder(nn.Module):
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
        
        # --- Teacher/Explorer Cadence from Config ---
        self.TEACHER_ASK_EVERY = TEACHER_ASK_EVERY
        self.TEACHER_OFFSET = TEACHER_OFFSET
        self.EXPLORER_OFFSET = EXPLORER_OFFSET

    def _fire(self, step: int, interval: int, offset: int) -> bool:
        """Checks if an event should be triggered at a given step."""
        return interval > 0 and step >= offset and ((step - offset) % interval == 0)

    def tick(self, step: int):
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
            asyncio.create_task(self.mind._explorer_answer_pending_question())

        # --- Other Asynchronous Cognitive Functions ---
        if self._fire(step, self.PROXIMITY_ALERT_INTERVAL, 5):
            asyncio.create_task(self.mind._run_insight_cycle())
            
        if self._fire(step, self.INSIGHT_SYNTHESIS_INTERVAL, 13):
            asyncio.create_task(self.mind._run_proactive_insight_synthesis())
            
        if self._fire(step, self.DREAM_INTERVAL, 0):
            asyncio.create_task(self.mind.dream_engine.run_dream_sequence())
            
        if self._fire(step, self.NARRATIVE_SUMMARY_INTERVAL, 2):
            asyncio.create_task(self.mind._generate_subconscious_narrative())
            
        if self._fire(step, self.SNAPSHOT_INTERVAL, 0):
            asyncio.create_task(self.mind.memory.snapshot())
            
        if self._fire(step, self.DECAY_INTERVAL, 21):
            asyncio.create_task(self.mind.memory.apply_decay())

class E8Mind:
    def __init__(self, semantic_domain_val, run_id, llm_client_instance, client_model, embedding_model_name, embed_adapter, embed_in_dim, console: Console):
        self.console = console
        self.console.rule(f"[bold cyan]Initializing E8 Mind | Run ID: {run_id}[/]")
        self.run_id = run_id
        os.makedirs(os.path.join(RUNTIME_DIR, self.run_id), exist_ok=True)

        self.proximity_log_path = get_path("logs/proximity_alerts.ndjson", self.run_id)
        os.makedirs(os.path.dirname(self.proximity_log_path), exist_ok=True)
        self._prox_lock = asyncio.Lock()

        self.market_enabled = bool(int(os.getenv("MARKET_FEED_ENABLED", "0")))
        self.market_symbols = [s.strip().upper() for s in os.getenv("MARKET_SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]
        finnhub_key = os.getenv("FINNHUB_KEY", "")
        self.market = None
        if self.market_enabled and finnhub_key and MarketFeed is not None:
            self.market = MarketFeed(
                symbols=self.market_symbols,
                api_key=finnhub_key,
                on_tick=self._on_market_tick,
                on_bar=self._on_market_bar,
            )
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
        self.llm_pool = AsyncLLMPool(self, worker_count=max(4, LOCAL_GEN_WORKERS))
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

        self.console.log("[INIT] Building E8 Physics and Geometric Foundations...")
        self.physics = E8Physics(self.console)
        self.fabric = E8BoundaryFabric(self.physics)
        self.fabric.layout_2d()
        safe_json_write(self._path("boundary_fabric.json"), self.fabric.to_json())
        self.blueprint = self.physics.generate_quasicrystal_blueprint()
        safe_json_write(self._path("quasicrystal_blueprint.json"), self.blueprint)
        self.blueprint_kdtree = KDTree([[p['x'], p['y']] for p in self.blueprint])

        if TORCH_AVAILABLE:
            self.autoencoder = VariationalAutoencoder(layer_sizes=AUTOENCODER_LAYER_SIZES, console=self.console)
            self.shell_lattices, self.shell_kdtree_indices = {}, {}
            e8_roots_tensor = torch.from_numpy(self.physics.roots.astype(np.float32)).float()
            with torch.no_grad():
                for dim in DIMENSIONAL_SHELL_SIZES:
                    if dim == 8:
                        lifted_vectors_np = e8_roots_tensor.numpy()
                    else:
                        projection_matrix = np.random.randn(8, dim)
                        lifted_vectors_np = e8_roots_tensor.numpy() @ projection_matrix

                    self.shell_lattices[dim], self.shell_kdtree_indices[dim] = lifted_vectors_np, KDTree(lifted_vectors_np)
            console.log("✅ Lifted E8 reference lattices generated for all dimensional shells.")
        else:
            self.autoencoder = VariationalAutoencoder(console=self.console)
            self.shell_lattices, self.shell_kdtree_indices = {}, {}
            console.log("[yellow]PyTorch not found. Autoencoder and shell lattices disabled.[/yellow]")

        self.console.log("[INIT] Assembling Cognitive Architecture...")
        self.mood = MoodEngine(self.console)
        self.subconscious = SubconsciousLayer(self.get_embedding, self.llm_pool, self.console)
        self.goal_field = GoalField(self.get_embedding, self.console)
        self.drives = DriveSystem()
        self.dimensional_shells = {dim: DimensionalShell(dim, self) for dim in DIMENSIONAL_SHELL_SIZES}
        self.proximity_engine = ProximityEngine(shell_dims=DIMENSIONAL_SHELL_SIZES, mind_instance=self, console=self.console)
        self.memory = MemoryManager(self.get_embedding, self.mood, self.subconscious, self.run_id, self.probe, self.llm_pool, self)
        self.novelty_scorer = NoveltyScorer(self.memory, self.llm_pool, self.console)
        self.insight_agent = InsightAgent(self.llm_pool, self.novelty_scorer, self.console)

        self.shell_attention = ShellAttention(out_dim=32, keep_k=3)
        self.arbiter_gate = ArbiterGate()
        self.curriculum = AutoTaskManager(self.console)

        self.dream_engine = DreamEngine(self.memory, self)

        self.narrative_streamer = NarrativeStreamer(self.memory, self.llm_pool, self.run_id)
        self.synthetic_env = SyntheticEnvironment(self.llm_pool, self)
        self.domain_tint = DomainTintEngine(self.semantic_domain, self.llm_pool)
        self.validator = HypothesisValidator(self)
        self.ingestion_pipeline = DataIngestionPipeline(self)
        self.scheduler = CognitiveScheduler(self)
        self.console.log("[INIT] Configuring SAC-MPO Reinforcement Learning Agent...")
        self.potential_evaluator = StatePotentialEvaluator(self.dimensional_shells, self.goal_field)
        TASK_EMBED_REDUCED_DIM = 32
        self.state_dim = len(self.mood.mood_vector) + 4 + self.shell_attention.out_dim + 5 # Mood + Goals + Attention + Dynamics
        self.action_dim = ACTION_SIZE_NO_LOCK # Bivectors + Angles
        self.max_action = 0.1

        self._bh_window = deque(maxlen=50)
        self._bh_recent = deque(maxlen=100)
        self._bh_ma50 = 0.0
        self._prev_bh = 0.0
        self._low_bh_streak = 0
        self._prev_action = np.zeros(self.action_dim, dtype=np.float32)
        if TORCH_AVAILABLE:
            self.agent = SACMPOAgent(self.state_dim, self.action_dim, self.max_action, console=self.console, tau=0.002, use_per=True)

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
            self.latent_planner = LatentCEMPlanner(ACTION_LAYOUT, self.action_dim, angle_scale=self.max_action, pop=64, elites=8, iters=3, horizon=8, sigma=0.06)
        except Exception:
            self.latent_planner = None

        self.qeng = QuantumEngine(self.physics, QuantumConfig(seed=GLOBAL_SEED), self.console)
        self.ceng = ClassicalEngine(self.physics, ClassicalConfig(seed=GLOBAL_SEED), self.console)
        self.anchors = MultiAnchorField(self.physics)
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
        # ... (code to find nearest index)
        _, nearest_index_arr = kdtree.query(vector.reshape(1, -1), k=1)
        
        # --- CORRECTED LOGIC ---
        try:
            # CORRECTED: .item() robustly extracts the scalar index from the nested array (e.g., [[42]]).
            scalar_index = nearest_index_arr.item()
            # This now correctly returns a 1D vector.
            return self.shell_lattices[dim][scalar_index]
        except (ValueError, IndexError):
            # Fallback if the index array is empty or malformed.
            return vectorr

    async def _teacher_ask_new_question(self):
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
                        prompt = (f"Goal: '{top_goal_desc}'.\nInsight A: '{data_A.get('metaphor', '')}'\nInsight B: '{data_B.get('metaphor', '')}'.\n\n"
                                    "Ask one concise question (under 20 words) about the connection between A and B.")
                    else:
                        recent_nodes_data = [d for _, d in self.memory.graph_db.graph.nodes(data=True) if not d.get("folded")]
                        memory_snippet = "\n".join(f"- {n.get('label','')}: {n.get('metaphor','')}" for n in recent_nodes_data[-4:])
                        prompt = (f"Goal: '{top_goal_desc}'.\nRecent thoughts:\n{memory_snippet}\n\nAsk one profound, short question (under 20 words) to advance the goal.")
                    question = await asyncio.wait_for(self.llm_pool.enqueue_and_wait(prompt, max_tokens=40, temperature=0.75), timeout=TEACHER_STEP_TIMEOUT)
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
                try:
                    answer_prompt = f"You are the Explorer. Answer the Teacher's question plainly. Max 4 sentences. Be concrete.\n\nQuestion:\n{q}\n\nAnswer:"
                    answer = str(await asyncio.wait_for(self.llm_pool.enqueue_and_wait(answer_prompt, max_tokens=150, temperature=0.8), timeout=EXPLORER_STEP_TIMEOUT)).strip()
                    label = "Explorer answer"
                    if answer and not answer.startswith("[LLM"):
                        try:
                            label_resp = await asyncio.wait_for(self.llm_pool.enqueue_and_wait(f"Summarize in 3-6 words:\n{answer}", max_tokens=12, temperature=0.2), timeout=max(3.0, EXPLORER_STEP_TIMEOUT/2.0))
                            if label_resp and not label_resp.startswith("[LLM"):
                                label = str(label_resp).strip()
                        except Exception as e: self.console.log(f"[EXPLORER] label timeout: {e}")
                    safe = escape(label)
                    async with self.console_lock:
                        self.console.print(Panel.fit(f"[bold white]{sanitize_block(answer, 2, 240)}[/]", title=f"[bold green]EXPLORER[/] · {safe[:42]}{'…' if len(safe)>42 else ''}", border_style="green"))
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
                finally:
                    self._teacher_question_context_ids, self.teacher_question = [], None

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
            rec["ts"] = datetime.now(timezone.utc).isoformat()
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
            self.console.log("[Insight Cycle] Checking for cross-dimensional proximity...")
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

    def _on_market_tick(self, symbol: str, tick: dict):

        self.market_last[symbol] = tick.get("p", 0.0)

    def _on_market_bar(self, symbol: str, timeframe: str, bar: Bar):

        self.market_last_bar[(symbol, timeframe)] = bar

    async def _run_proactive_insight_synthesis(self):
        async with self.insight_cycle_lock:
            self.console.log("[InsightAgent] Proactively seeking novel synthesis...")
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
            novelty_vec = await self.get_embedding(new_concept_text)
            novelty_score = self.novelty_scorer.calculate_novelty(novelty_vec)
            coherence_score = await self.novelty_scorer.calculate_coherence(new_concept_text)
            final_rating = (novelty_score * 0.6) + (coherence_score * 0.4)
            reward = (novelty_score + coherence_score) / 2.0
            self.insight_agent.learn_from_reward(reward)
            new_entry = {
                "type": "insight_synthesis", "label": sanitize_line(f"Synthesis: {parent_a_data.get('label', 'A')} + {parent_b_data.get('label', 'B')}"),
                "metaphor": new_concept_text, "rating": final_rating, "step": self.step_num
            }
            new_node_id = await self.memory.add_entry(new_entry, parent_ids=[parent_a_id, parent_b_id])
            await self._append_insight_log({
                "run_id": getattr(self, "run_id", None),
                "step": int(self.step_num),
                "type": "insight_synthesis",
                "node_id": new_node_id,
                "label": new_entry["label"],
                "content": new_concept_text,
                "rating": float(final_rating),
                "novelty": float(novelty_score),
                "coherence": float(coherence_score),
                "parent_ids": [parent_a_id, parent_b_id],
            })
            self.subconscious_event_log.append({'type': 'insight_synthesis', 'label': new_entry['label'], 'step': self.step_num})
            self.console.print(Panel(f"[bold]New Synthesis:[/bold] {new_concept_text}\n[yellow]Novelty:[/] {novelty_score:.2f} | [cyan]Coherence:[/] {coherence_score:.2f} | [green]Reward:[/] {reward:.2f}",
                                        title="[bold blue]INSIGHT SYNTHESIS[/]", border_style="blue"))

            if final_rating > 0.8:
                asyncio.create_task(self.validator.validate_insight(new_node_id))

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
            line = json.dumps({**record, "ts": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False)
            async with self._insight_lock:
                with open(self.insight_log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception as e:
            self.console.log(f"[INSIGHT-LOG] write failed: {e}")

    def _build_telemetry_snapshot(self) -> dict:
        shells_data, shell_tensions = {}, {}
        for dim, shell in self.dimensional_shells.items():
            orientation_value = shell.orientation.value if CLIFFORD_AVAILABLE and hasattr(shell.orientation, 'value') else None
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

        telemetry = {
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
                    "timestamp": datetime.now(timezone.utc).isoformat(),
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
                    losses = self.autoencoder.train_on_batch(torch.from_numpy(batch_np).float())
                    self.console.log(f"🧠 [VAE] Trained. Loss: {losses['total_loss']:.4f}, Recon: {losses['recon_loss']:.4f}, KLD: {losses['kld_loss']:.4f}")
                except Exception as e:
                    self.console.log(f"[bold red]VAE Training Error: {e}[/bold red]")
                    
        return autoencoder_train_buffer

    async def run_cognitive_cycle(self, max_steps=99200, mode='quantum'):
        """The main operational loop of the E8Mind, refactored for correctness and clarity."""
        self._ensure_console_export_state()
        self.console.rule(f"[bold magenta]Starting Cognitive Cycle | Mode: {mode.upper()}[/bold magenta]")

        # --- Initialization ---
        await self.llm_pool.start()
        await self.goal_field.initialize_goals()
        for name, config in DATA_SOURCES.items():
            self.ingestion_pipeline.add_source(name, config)
        await self.ingestion_pipeline.start()
        if self.market:
            await self.market.start()

        self.max_steps = max_steps
        autoencoder_train_buffer, AUTOENCODER_BATCH_SIZE = [], 64
        
        # --- Main Loop ---
        with Progress("[progress.description]{task.description}", BarColumn(), "[progress.percentage]{task.percentage:>3.0f}%",
                    "Step", TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(), console=self.console, transient=True) as progress:
            
            task = progress.add_task(f"Thinking ({mode})", total=max_steps)
            for step in range(max_steps):
                self.step_num = step
                
                if step == 0:
                    await self._project_self_into_memory()

                # 1. Update passive cognitive modules and fire scheduled events
                self._update_cognitive_modules(step)
                autoencoder_train_buffer = self._train_autoencoder_if_ready(autoencoder_train_buffer, AUTOENCODER_BATCH_SIZE)
                
                # 2. Build the state vector for the RL agent
                current_state = self._build_state_vector()

                # 3. Select and apply an action to the dimensional shells
                action = self.agent.select_action(current_state) if self.agent else np.zeros(self.action_dim, dtype=np.float32)
                clamped_action = clamp_action(action, max_norm=getattr(self, '_action_clamp_norm', 0.04))
                self.apply_manifold_action(clamped_action)
                
                # 4. Evolve the quantum/classical engine
                prev_idx = self.prev_node_index or random.randrange(self.physics.roots.shape[0])
                self.qeng.build_hamiltonian(V=self.anchors.potential())
                self.qeng.step_adaptive()
                current_node_index = self.qeng.measure_hybrid(prev_idx, sigma=self.sigma_q)[0]
                self.prev_node_index = current_node_index
                
                # --- CORRECTED RL LOGIC ORDER ---
                # 5. Recalculate the state and THEN determine the reward for the transition.
                next_state = self._build_state_vector()
                reward = self.potential_evaluator.calculate_potential_and_get_reward()
                
                # 6. Store experience and train the RL agent.
                if self.agent:
                    # The order here is crucial: (s_t, a_t, s_{t+1}, r_t, done)
                    self.agent.store(current_state, clamped_action, next_state, reward, (step == max_steps - 1))
                    if step > 1024: # Start training after a buffer is filled
                        self.agent.update()
                
                # 7. Update internal trackers for the next cycle
                self._prev_action = clamped_action
                self._prev_bh = self.black_hole_pressure

                # 8. Handle logging and telemetry
                await self._sse_push_telemetry()
                if (step + 1) % CONSOLE_EXPORT_EVERY_STEPS == 0:
                    self._export_console_chunk(step)
                
                progress.update(task, advance=1)
                await asyncio.sleep(0.01)

        # --- Shutdown ---
        self.console.log("\nCognitive cycle complete.")
        self._export_console_chunk(self.step_num, final=True)

    def _path(self, rel: str) -> str:
        return get_path(rel, self.run_id)

    async def get_embedding(self, text: str) -> np.ndarray:
        if IS_EMBED_PLACEHOLDER:
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

    async def _async_call_llm_internal(self, prompt: str, **kwargs) -> str:
        """
        Calls the LLM with full context, persona, and domain hints.
        Includes a fallback to a local model to prevent repetition and increase robustness.
        """
        # --- 1. Construct the Full Prompt ---
        try:
            persona = self.semantics.persona_prefix(self.mood.mood_vector)
        except Exception:
            persona = self.mood.get_llm_persona_prefix()

        domain_hint = f"Domain: {getattr(self.domain_tint, 'last_hint', self.semantic_domain)}."

        _prompt_key = kwargs.pop('_prompt_key', 'ask')
        _prompt_vars = kwargs.pop('_prompt_vars', None) or {'question': prompt}
        
        full_prompt = self.prompts.render(
            _prompt_key, 
            persona=persona, 
            domain_hint=domain_hint, 
            **_prompt_vars
        )

        # --- 2. Prepare and Execute LLM Calls ---
        primary_task = None
        local_task = None
        
        # Ensure kwargs have standard Python types
        llm_kwargs = {
            'model': self.client_model,
            'max_tokens': int(kwargs.get('max_tokens', 256)),
            'temperature': float(kwargs.get('temperature', 0.7)),
        }

        # Primary LLM Call
        try:
            messages = [{"role": "user", "content": full_prompt}]
            primary_task = asyncio.wait_for(
                self.llm_client.chat(messages=messages, **llm_kwargs),
                timeout=LLM_CALL_TIMEOUT_SEC
            )
        except Exception as e:
            self.console.log(f"[LLM] Primary client call setup failed: {e}")

        # Local LLM Fallback Call (if enabled)
        if self._anti_repeat_enabled and self.local_llm_client:
            try:
                local_kwargs = {
                    'max_tokens': llm_kwargs['max_tokens'] // 2,
                    'temperature': min(1.0, llm_kwargs['temperature'] + 0.15)
                }
                local_messages = [{"role": "user", "content": prompt}] # Use simpler prompt
                local_task = asyncio.wait_for(
                    self.local_llm_client.chat(messages=local_messages, **local_kwargs),
                    timeout=LLM_CALL_TIMEOUT_SEC
                )
            except Exception as e:
                self.console.log(f"[LLM] Local client call setup failed: {e}")

        # --- 3. Await and Collect Responses ---
        results = await asyncio.gather(primary_task, local_task, return_exceptions=True)
        primary_text = results[0] if not isinstance(results[0], BaseException) else f"[LLM ERROR] {results[0]}"
        local_text = results[1] if local_task and not isinstance(results[1], BaseException) else None

        # --- 4. Select the Best Candidate to Avoid Repetition ---
        candidates = []
        if isinstance(primary_text, str) and primary_text.strip() and not primary_text.startswith("[LLM"):
            candidates.append(primary_text.strip())
        if isinstance(local_text, str) and local_text.strip() and not local_text.startswith("[LLM"):
            candidates.append(local_text.strip())

        if not candidates:
            return primary_text or "[LLM ERROR] No valid response from any provider."

        # Helper function to score candidates against recent outputs
        def get_repetition_score(text: str) -> float:
            norm_text = self._norm_text(text)
            if not norm_text: return 1.0
            
            # Use a simpler text similarity check for efficiency
            for recent_norm in self._recent_norms:
                if norm_text == recent_norm:
                    return 1.0 # Exact match is highest penalty
            
            # Fallback to ngram similarity for non-exact matches
            return self._repeat_score(text)

        best_candidate = min(candidates, key=get_repetition_score)
        
        self._remember_output(best_candidate)
        return best_candidate

    async def rate_concept(self, concept_text: str) -> float:
        if IS_EMBED_PLACEHOLDER:
            return 0.6
        prompt = f'Rate the novelty and coherence of this idea on a scale from 0.0 to 1.0. Response must be only the number.\nIdea: "{concept_text}"'
        try:
            response = await self.llm_pool.enqueue_and_wait(prompt, max_tokens=10, temperature=0.1)
            num = re.findall(r"[-+]?\d*\.\d+|\d+", response)
            v = float(num[0]) if num else 0.5
            return np.clip(v / 100.0 if v > 1.0 else v, 0.0, 1.0)
        except Exception:
            return 0.5

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
        self.console.print(Panel.fit("Generating phase summary...", title="[bold orange]PHASE[/bold orange]", border_style="dark_orange"))
        recent_nodes = [d.get('label', 'untitled') for _, d in self.memory.graph_db.graph.nodes(data=True) if isinstance(d, dict) and not d.get("folded") and d.get('type') == 'concept'][-10:]
        if len(recent_nodes) < 3:
            return
        prompt = (f"Concepts explored recently: {', '.join(recent_nodes)}. Synthesize these into a one-sentence summary of this phase of thought.")
        try:
            summary = await self.llm_pool.enqueue_and_wait(prompt, max_tokens=150, temperature=0.6)
            if not summary or summary.startswith("[LLM"): return
            label = await self.llm_pool.enqueue_and_wait(f'Create a 3-4 word title for this summary: "{summary}"', max_tokens=15, temperature=0.5)
            if not label or label.startswith("[LLM"): return
            await self.memory.add_entry({"label": label.strip().replace('"', ''), "type": "phase_summary", "metaphor": summary, "rating": 0.8, "step": self.step_num})
            self.console.print(Panel(f"New summary created: '{label}'", title="[bold orange]PHASE[/bold orange]", border_style="dark_orange"))
        except Exception as e:
            self.console.log(f"[bold red]Failed to generate phase summary: {e}[/bold red]")

    async def _generate_meta_reflection(self):
        self.console.print(Panel.fit("Generating meta-reflection...", title="[bold white]META[/bold white]", border_style="white"))
        refl_file = self._path("reflections.txt")
        if not os.path.exists(refl_file):
            return
        with open(refl_file, "r", encoding="utf-8") as f:
            content = f.read()
        recent_egos = re.findall(r"--- Step \d+ ---\n(.*?)(?=\n--- Step|\Z)", content, re.DOTALL)[-5:]
        if not recent_egos:
            return
        prompt = ("Reflect on these recent internal monologues:\n" + "\n".join(f"- {e.strip()}" for e in recent_egos) +
                    "\n\nWhat pattern or concept emerges? Synthesize a new, higher-level insight.")
        try:
            reflection = await self.llm_pool.enqueue_and_wait(prompt, temperature=0.8, max_tokens=250)
            if not reflection or reflection.startswith("[LLM"): return
            label = await self.llm_pool.enqueue_and_wait(f'Summarize this insight in a 3-5 word title: "{reflection}"', max_tokens=20)
            if not label or label.startswith("[LLM"): return
            await self.memory.add_entry({"label": label.strip().replace('"', ''), "type": "meta_reflection", "metaphor": reflection, "rating": 0.85, "step": self.step_num})
            self.mood.process_event("reflection")
            self.console.print(Panel(f"Meta-reflection '{label}' added.", title="[bold white]META[/bold white]", border_style="white"))
        except Exception as e:
            self.console.log(f"[bold red]Meta-reflection failed: {e}[/bold red]")

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
        if is_ready and self.black_hole_pressure > BH_PRESSURE_THRESHOLD:
            self._bh_inflight = True
            self._bh_cooldown_until = self.step_num + BLACK_HOLE_COOLDOWN_STEPS
            self.console.log(f"[bold red]Black hole pressure threshold exceeded ({self.black_hole_pressure:.3f}). Initiating collapse.[/bold red]")
            asyncio.create_task(self._blackhole_cycle(self.step_num))

    async def _blackhole_cycle(self, step_num: int) -> Optional[EmergenceSeed]:
        self._bh_inflight = True
        try:
            center_id, pressure = self.memory.find_event_horizon()
            if not center_id:
                self.console.log("[BH Cycle] Aborted: No event horizon.")
                return None

            cluster = self.memory.collect_cluster(center_id)
            need = max(2, CONSOLIDATE_MIN)

            if not cluster or len(cluster) < need:
                base = set(cluster) if cluster else {center_id}
                cvec = self.memory.main_vectors.get(center_id)
                if cvec is not None:
                    for nid, _ in self.memory.find_similar_in_main_storage(cvec, k=need * 2):
                        if nid not in base and nid in self.memory.main_vectors:
                            base.add(nid)
                            if len(base) >= need:
                                break
                cluster = list(base)

            if not cluster or len(cluster) < need:
                self.console.log(f"[BH Cycle] Aborted: Cluster for '{center_id}' too small ({len(cluster)} < {need}).")
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
                    z8_tensor = self.autoencoder.project_to_dim(torch.from_numpy(remnant_vec).float().unsqueeze(0), 8)
                    if z8_tensor is not None: z8 = z8_tensor.numpy().squeeze()

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
            except Exception: pass

            node = G.nodes.get(nid)
            if node:
                node["temperature"] = float(node.get("temperature", 0.5) + 0.05 * w)
        self.console.log(f"[retro] linked {len(top)} prior nodes to {new_node_id}")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

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

async def handle_get_qeng_probabilities(request):
    mind = request.app['mind']
    qeng = getattr(mind, "qeng", None)
    if qeng is None:
        return web.json_response({"error": "quantum engine not initialized"}, status=400)

    # The qeng._probs() method returns probabilities for the entire batch.
    # We'll return the probabilities for the first instance in the batch.
    probabilities = qeng._probs()[0].tolist() 

    return web.json_response({"probabilities": probabilities})

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

def _collect_config_from_user():
    print("Choose LLM provider:\n1. OpenAI\n2. Ollama (local)\n3. Gemini API")
    provider_choice = input("Enter choice (1, 2, or 3) [1]: ") or "1"
    cfg = {"provider_choice": provider_choice}
    if provider_choice == "1":
        cfg["openai_api_key"] = (input("OpenAI API Key: ") or "").strip()
        cfg["openai_model_name"] = (input("OpenAI model [gpt-4-turbo-preview]: ") or "gpt-4-turbo-preview").strip()
    elif provider_choice == "2":
        cfg["ollama_model_name"] = (input("Ollama model [llama3]: ") or "llama3").strip()
    elif provider_choice == "3":
        cfg["gemini_api_key"] = (input("Gemini API Key: ") or "").strip()
        cfg["gemini_model_name"] = (input("Gemini model [gemini-1.5-flash]: ") or "gemini-1.5-flash").strip()
    else:
        print("Invalid choice. Running with LLM stub.")
    if provider_choice == "3":
        use_local = (input("Augment with a local tiny-LLM via Ollama? (y/N): ") or "n").strip().lower() == "y"
        cfg["use_local_mix"] = bool(use_local)
        if use_local:
            cfg["local_model_name"] = (input("Local Ollama model [phi3:mini-4k]: ") or "phi3:mini-4k").strip()

    return cfg

async def main():
    run_id = get_run_id()
    global llm_client, model_name, embedding_model, IS_EMBED_PLACEHOLDER, LLM_PROVIDER
    provider_native_embed_dim = 1536
    IS_EMBED_PLACEHOLDER = False

    if os.getenv("E8_PROVIDER", "").strip().lower() in ("", "ask"):
        cfg = _collect_config_from_user()
        pc = str(cfg.get("provider_choice", "")).strip()
        if pc == "1":
            LLM_PROVIDER = "openai"
            api_key = cfg.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key: raise ValueError("OPENAI_API_KEY not set.")
            llm_client, model_name, embedding_model = AsyncOpenAIClient(api_key, console), cfg.get("openai_model_name") or "gpt-4-turbo-preview", "text-embedding-3-small"
        elif pc == "2":
            LLM_PROVIDER, model_name = "ollama", cfg.get("ollama_model_name") or os.getenv("OLLAMA_MODEL", "llama3")
            llm_client, embedding_model = OllamaClient(model_name, console), "nomic-embed-text"
        elif pc == "3":
            LLM_PROVIDER = "gemini"
            api_key = cfg.get("gemini_api_key") or os.getenv("GEMINI_API_KEY")
            if not api_key: raise ValueError("GEMINI_API_KEY not set.")
            model_name = cfg.get("gemini_model_name") or "gemini-1.5-flash"
            llm_client, embedding_model = GeminiClient(api_key, model_name, console), "models/embedding-001"
        else: LLM_PROVIDER, IS_EMBED_PLACEHOLDER = "stub", True
    else:
        LLM_PROVIDER = os.getenv("E8_PROVIDER", "stub").lower()
        if LLM_PROVIDER == "openai":
            api_key = os.getenv("OPENAI_API_KEY");
            if not api_key: raise ValueError("OPENAI_API_KEY not set.")
            llm_client, model_name, embedding_model = AsyncOpenAIClient(api_key, console), os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"), "text-embedding-3-small"
        elif LLM_PROVIDER == "ollama":
            model_name = os.getenv("OLLAMA_MODEL", "llama3")
            llm_client, embedding_model = OllamaClient(model_name, console), "nomic-embed-text"
        elif LLM_PROVIDER == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key: raise ValueError("GEMINI_API_KEY not set.")
            model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            llm_client, embedding_model = GeminiClient(api_key, model_name, console), "models/embedding-001"
        else: LLM_PROVIDER, IS_EMBED_PLACEHOLDER = "stub", True

    if IS_EMBED_PLACEHOLDER:
        class StubClient:
            def __init__(self, console): self.console = console
            async def chat(self, *a, **k): return "This is a placeholder response from a stubbed LLM."
            async def embedding(self, *a, **k): return np.random.randn(provider_native_embed_dim)
            async def batch_embedding(self, texts, *a, **k): return [np.random.randn(provider_native_embed_dim) for _ in texts]
        llm_client, model_name, embedding_model = StubClient(console), "stub", "stub"

    console.log("[INIT] Probing embedding dimension from provider...")
    _test_vec = await llm_client.embedding("adapter_probe")
    if isinstance(_test_vec, dict) and "embedding" in _test_vec: _test_vec = _test_vec["embedding"]
    if isinstance(_test_vec, list) and _test_vec and isinstance(_test_vec[0], (list, np.ndarray)): _test_vec = _test_vec[0]
    embed_in_dim = int(len(_test_vec))
    if embed_in_dim > 1: provider_native_embed_dim = embed_in_dim
    console.log(f"[INIT] Detected provider embedding dimension: {provider_native_embed_dim}")

    try:
        profile_name = os.getenv("MIND_PROFILE", "default")
        sem, _ = load_profile(profile_name)
        probe_native = np.zeros(provider_native_embed_dim, dtype=np.float32)
        probe_post = sem.post_embed(probe_native)
        adapter_in_dim = int(np.asarray(probe_post, dtype=np.float32).size)
        console.log(f"[INIT] post_embed output dim: {adapter_in_dim} (provider {provider_native_embed_dim})")
    except Exception as e:
        adapter_in_dim = provider_native_embed_dim
        console.log(f"[INIT] post_embed probe failed: {e}. Falling back to provider dim.")

    embed_adapter = UniversalEmbeddingAdapter(adapter_in_dim, EMBED_DIM)
    console.log(f"[INIT] Universal Embedding Adapter created: {adapter_in_dim} -> {EMBED_DIM}")

    mind = E8Mind(
        semantic_domain_val=SEMANTIC_DOMAIN, run_id=run_id,
        llm_client_instance=llm_client, client_model=model_name,
        embedding_model_name=embedding_model,
        embed_adapter=embed_adapter,
        embed_in_dim=provider_native_embed_dim,
        console=console
    )

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

    app = web.Application()
    app['mind'] = mind
    app['sse_clients'] = set()
    mind.sse_clients = app['sse_clients']
    app.on_shutdown.append(shutdown_market_feed)
    cors = aiohttp_cors.setup(app, defaults={"*": aiohttp_cors.ResourceOptions(allow_credentials=True, expose_headers="*", allow_headers="*")})
    app.router.add_get("/api/graph", handle_get_graph)
    app.router.add_get("/api/telemetry", handle_get_telemetry)
    app.router.add_get("/api/telemetry/stream", handle_stream_telemetry)
    app.router.add_get("/api/blueprint", handle_get_blueprint)
    app.router.add_post("/api/concept", handle_add_concept)
    app.router.add_post("/api/action/dream", handle_trigger_dream)
    app.router.add_get("/api/qeng/telemetry", handle_get_qeng_telemetry)
    app.router.add_get("/api/qeng/ablation", handle_get_qeng_ablation)
    app.router.add_get("/api/qeng/probabilities", handle_get_qeng_probabilities)

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
        console.log("\n[bold cyan]Shutting down E8 Mind...[/bold cyan]")
    except Exception as e:
        console.log(f"[bold red]CRITICAL ERROR in main: {e}[/bold red]")
        console.print_exception()

class LatentCEMPlanner:
    """Plans angles using linear latent: z_{t+1} = K z_t + B a_t. Only angle entries are optimized."""
    def __init__(self, action_layout, action_dim, angle_scale, pop=64, elites=8, iters=3, horizon=5, sigma=0.06, seed=1337):
        self.lay = action_layout
        self.action_dim = action_dim
        self.angle_idx = [L["angle_idx"] for L in self.lay]
        self.pop, self.elites, self.iters, self.horizon = pop, elites, iters, horizon
        self.sigma = float(sigma)
        self.rng = np.random.default_rng(seed)
        self.B = None
        self.angle_scale = float(angle_scale)
        self.hist_z, self.hist_bh = deque(maxlen=256), deque(maxlen=256)
        self.w = None

    def _fit_bh_head(self):
        if len(self.hist_z) < 16:
            return
        Z = np.stack(self.hist_z).astype(np.float32)
        y = np.array(self.hist_bh, dtype=np.float32).reshape(-1,1)
        lam = 1e-2
        A = Z.T @ Z + lam*np.eye(Z.shape[1], dtype=np.float32)
        b = Z.T @ y
        try:
            self.w = (np.linalg.solve(A, b)).reshape(-1)
        except Exception:
            self.w = None

    def observe(self, z, bh):
        try:
            self.hist_z.append(np.array(z, dtype=np.float32))
            self.hist_bh.append(float(bh))
            if (len(self.hist_z) % 16) == 0:
                self._fit_bh_head()
        except Exception:
            pass

    def _predict_bh(self, z):
        if self.w is None:
            return float(np.linalg.norm(z))
        return float(np.dot(self.w, z))

    def plan(self, z0, actor_action, K, max_action):
        mu = actor_action[self.angle_idx].copy()
        std = np.ones_like(mu) * self.sigma
        if self.B is None:
            self.B = np.random.randn(z0.size, self.action_dim).astype(np.float32) * 0.02
        pop = self.pop
        scores = np.zeros((pop,), dtype=np.float32)
        Aseq = np.zeros((pop, self.horizon, mu.size), dtype=np.float32)
        for it in range(self.iters):
            for p in range(pop):
                Aseq[p] = self.rng.normal(mu, std)
                Aseq[p] = np.clip(Aseq[p], -max_action, max_action)
                z = z0.copy()
                score = 0.0
                for t in range(self.horizon):
                    a_full = actor_action.copy()
                    a_full[self.angle_idx] = Aseq[p, t]
                    z = (K @ z) + (self.B @ a_full)
                    score += - self._predict_bh(z)
                scores[p] = score
            elite_idx = np.argsort(scores)[-self.elites:]
            elite = Aseq[elite_idx]
            mu = elite.mean(axis=0)
            std = elite.std(axis=0) + 1e-6
        best_first = mu[0]
        return best_first

def _safety_from_mood(mood_vec: dict):
    try:
        e = float(mood_vec.get("entropy", 0.5))
        i = float(mood_vec.get("intensity", 0.5))
        c = float(mood_vec.get("coherence", 0.5))
    except Exception:
        e, i, c = 0.5, 0.5, 0.5
    if (i >= 0.55 and e >= 0.55 and c <= 0.55):
        bucket = "storm"
    elif (c >= 0.62 and e <= 0.55):
        bucket = "flow"
    elif (e <= 0.48 and i <= 0.50):
        bucket = "calm"
    else:
        bucket = "turbulent"
    if bucket in ("storm", "turbulent"):
        clamp, critics, horizon = 0.030, 4, 5
    elif bucket == "flow":
        clamp, critics, horizon = 0.060, 2, 12
    else:
        clamp, critics, horizon = 0.040, 2, 8
    return {"bucket": bucket, "clamp": clamp, "critics": critics, "horizon": horizon}