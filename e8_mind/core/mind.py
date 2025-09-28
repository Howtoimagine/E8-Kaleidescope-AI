"""Orchestrator entry: E8Mind facade.

This module establishes the canonical import location for the E8Mind orchestrator
as part of the modularization effort. For now, it safely aliases the legacy
implementation from the M24.x server files by importing them directly from disk.

Priority order:
- e8_mind_server_M24.2.py
- e8_mind_server_M24.1.py

If neither is present or importable, a small placeholder E8Mind is provided
that raises a helpful error on use. This keeps imports stable while we migrate
the full class into this module incrementally.
"""

from __future__ import annotations

import os
import sys
import importlib.util
from pathlib import Path
from typing import Type, Optional, Any

# Optionally pull in common config/utilities to keep typical side effects
# and environment flags available when importing E8Mind from here.
try:
    from .config import *  # noqa: F401,F403
    from .utils import *   # noqa: F401,F403
except Exception:
    # Utilities are optional for aliasing; failures here shouldn't block import
    pass


def _load_legacy_e8mind() -> Optional[Type[Any]]:
    """Attempt to load E8Mind class from legacy server files by path.

    Returns the E8Mind type if found, else None.
    """
    # Resolve project root (two levels up from this file: e8_mind/core/mind.py)
    here = Path(__file__).resolve()
    project_root = here.parent.parent.parent

    # Candidate legacy files in order of preference
    candidates = [
        project_root / "e8_mind_server_M24.2.py",
        project_root / "e8_mind_server_M24.1.py",
    ]

    for path in candidates:
        try:
            if not path.exists():
                continue
            module_name = f"_legacy_{path.stem.replace('.', '_')}"
            spec = importlib.util.spec_from_file_location(module_name, str(path))
            if spec is None or spec.loader is None:  # pragma: no cover
                continue
            mod = importlib.util.module_from_spec(spec)
            # Register in sys.modules so intra-module relative imports (if any) can work
            sys.modules[module_name] = mod
            # Patch builtins.input during import to avoid interactive prompts
            import builtins as _bi
            _orig_input = getattr(_bi, 'input', None)
            try:
                os.environ.setdefault("E8_NON_INTERACTIVE", "1")
                def _no_input(prompt: str = ""):
                    # Return empty/default to any unexpected prompt
                    return ""
                setattr(_bi, 'input', _no_input)
                spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            finally:
                try:
                    if _orig_input is not None:
                        setattr(_bi, 'input', _orig_input)
                except Exception:
                    pass
            E8Cls = getattr(mod, "E8Mind", None)
            if E8Cls is not None:
                return E8Cls
        except Exception:
            # On any failure, try next candidate
            continue
    return None

# Lazy proxy to avoid executing heavy legacy module top-level code on import.
class _E8MindProxy:
    """Deferred loader for the legacy E8Mind implementation.

    Instantiating this class will locate and import the legacy E8Mind, then
    return an instance of the real class. This prevents side effects during
    mere import of e8_mind.core.mind.
    """

    __doc__ = "E8Mind orchestrator (lazy-loaded from legacy until migration completes)."

    def __new__(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
        E8Cls = _load_legacy_e8mind()
        if E8Cls is None:
            raise RuntimeError(
                "E8Mind is not available: legacy server files (M24.2/M24.1) were not found "
                "or failed to import. Ensure the repository includes e8_mind_server_M24.2.py "
                "or e8_mind_server_M24.1.py, or migrate the orchestrator implementation "
                "into e8_mind.core.mind.E8Mind."
            )
        # Instantiate and return the real object; caller receives underlying type
        obj = E8Cls(*args, **kwargs)
        return obj


# Public alias used by callers
E8Mind = _E8MindProxy  # type: ignore

__all__ = ["E8Mind"]


def new_default_mind(
    semantic_domain: Optional[str] = None,
    run_id: Optional[str] = None,
    console=None,
    embedding_model_name: Optional[str] = None,
    embed_in_dim: Optional[int] = None,
):
    """Construct a legacy E8Mind with safe default wiring.

    - Builds a Console if none provided (quiet SafeConsole if rich unavailable)
    - Uses a passthrough embed adapter if none provided (in_dim -> out_dim mapping)
    - Leaves llm_client_instance/client_model None (legacy code tolerates None)
    - Picks run_id if not provided via core.utils.get_run_id()
    """
    # Resolve E8Mind class
    E8Cls = _load_legacy_e8mind()
    if E8Cls is None:
        raise RuntimeError("Legacy E8Mind not available; cannot build default mind.")

    # Console
    if console is None:
        try:
            from rich.console import Console  # type: ignore
            from .utils import SafeConsole as _Safe
            console = _Safe(Console(record=True))
        except Exception:
            class _C:
                def print(self, *a, **k): print(*a)
                def log(self, *a, **k): print(*a)
            console = _C()

    # Defaults
    semantic_domain_val = semantic_domain or os.getenv("SEMANTIC_DOMAIN", "General Inquiry")
    try:
        from .utils import get_run_id as _grid
        # In modular utils, get_path signature differs (runtime_dir param), but get_run_id is consistent
        rid = run_id or _grid()
    except Exception:
        import time as _t
        rid = run_id or f"run_{int(_t.time())}"

    llm_client_instance = None
    client_model = None
    embedding_model_name = embedding_model_name or os.getenv("E8_EMBED_MODEL", "text-embedding-3-small")

    # Adapter: passthrough linear adapter to required EMBED_DIM if needed
    try:
        from .config import EMBED_DIM
    except Exception:
        EMBED_DIM = 1536

    adapter_in_dim = int(embed_in_dim or EMBED_DIM)
    out_dim = EMBED_DIM

    def _np_eye(n):
        try:
            import numpy as np
            return np.eye(n, dtype=np.float32)
        except Exception:  # pragma: no cover
            return [[1.0 if i==j else 0.0 for j in range(n)] for i in range(n)]

    class _Adapter:
        def __init__(self, in_dim, out_dim):
            self.in_dim = int(in_dim)
            self.out_dim = int(out_dim)
            self.W = _np_eye(self.out_dim)
        def __call__(self, v):
            try:
                import numpy as np
                arr = np.asarray(v, dtype=np.float32).reshape(-1)
                if arr.shape[0] == self.out_dim:
                    return arr
                if arr.shape[0] < self.out_dim:
                    pad = np.zeros(self.out_dim - arr.shape[0], dtype=np.float32)
                    return np.concatenate([arr, pad])
                return arr[: self.out_dim]
            except Exception:
                # Fallback: list ops
                try:
                    seq = list(float(x) for x in v)
                except Exception:
                    seq = []
                if len(seq) < self.out_dim:
                    seq = seq + [0.0] * (self.out_dim - len(seq))
                return seq[: self.out_dim]

    embed_adapter = _Adapter(adapter_in_dim, out_dim)

    is_embed_placeholder = True  # signals adapter is generic

    # Positional call to tolerate legacy signature (avoids keyword mismatch if aliased)
    # Type-ignore: legacy constructor signature is dynamic and not known to static checker
    return E8Cls(  # type: ignore[misc,call-arg]
        semantic_domain_val,
        rid,
        llm_client_instance,
        client_model,
        embedding_model_name,
        embed_adapter,
        adapter_in_dim,
        console,
        is_embed_placeholder,
    )
