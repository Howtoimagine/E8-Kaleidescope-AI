# Core utility functions and helper classes
"""
General utility functions used throughout the E8Mind system.
Contains text processing, data validation, and common helper functions.
"""

import os
import time
import json
import tempfile
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple


def sanitize_line(text: str, max_chars: int = 80) -> str:
    """Cleans a string to be a single, sanitized line."""
    if not isinstance(text, str): return ""
    text = text.replace('\n', ' ').replace('\r', '').strip()
    return text[:max_chars]


def sanitize_block(text: str, max_sentences: int = 5, max_chars: int = 500) -> str:
    """Cleans and truncates a block of text."""
    if not isinstance(text, str): return ""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    truncated_text = " ".join(sentences[:max_sentences])
    return truncated_text[:max_chars]


def normalize_vector(v):
    """Helper function to ensure vectors have unit length."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-9 else v


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


def get_run_id() -> str:
    """Generates a unique run ID based on the current timestamp."""
    try:
        from datetime import datetime, timezone
        if datetime is not None and timezone is not None:
            return datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    except Exception:
        pass
    from datetime import datetime as _dt
    return _dt.utcnow().strftime("run_%Y%m%d_%H%M%S")


def get_path(rel: str, run_id: str) -> str:
    """Constructs an absolute path within the current run's directory."""
    from ..core.config import RUNTIME_DIR
    base = os.path.join(RUNTIME_DIR, str(run_id)) if run_id else RUNTIME_DIR
    path = os.path.join(base, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def mood_get(mood_vector: dict, key: str, default: float = 0.5) -> float:
    """Safely retrieves a float value from the mood vector dictionary."""
    return float(mood_vector.get(key, default))


def seed_all(seed: int):
    """Seeds all random number generators for reproducible results."""
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


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        return super(NumpyEncoder, self).default(obj)
