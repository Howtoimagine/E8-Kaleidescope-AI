import json
import time
import re
import os
import sys
import logging
import tempfile
import unicodedata
from datetime import datetime, timezone

import numpy as np

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markup import escape
except ImportError:
    class Console:
        def print(self, *args, **kwargs): print(*args)
        def log(self, *args, **kwargs): print(*args)
    class Panel(str):
        def __new__(cls, content, **kwargs): return str(content)
    def escape(s): return s

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        return super(NumpyEncoder, self).default(obj)

def _safe_number(x, default: float = 0.0, min_val: float | None = None, max_val: float | None = None) -> float:
    try:
        v = float(x)
    except (ValueError, TypeError):
        return float(default)
    if not np.isfinite(v):
        return float(default)
    if min_val is not None:
        v = max(min_val, v)
    if max_val is not None:
        v = min(max_val, v)
    return float(v)

def get_run_id() -> str:
    try:
        return datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    except Exception:
        return time.strftime("run_%Y%m%d_%H%M%S", time.gmtime())

def get_path(rel: str, run_id: str, runtime_dir: str) -> str:
    base = os.path.join(runtime_dir, str(run_id)) if run_id else runtime_dir
    path = os.path.join(base, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

def safe_json_write(filepath: str, data: any):
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with tempfile.NamedTemporaryFile('w', delete=False, dir=os.path.dirname(filepath), encoding='utf-8') as tf:
            json.dump(data, tf, ensure_ascii=False, indent=2, cls=NumpyEncoder)
            tempname = tf.name
        os.replace(tempname, filepath)
    except Exception as e:
        logging.warning(f"Failed to write JSON to {filepath}: {e}")

def safe_json_read(filepath: str, default: any = None) -> any:
    if not os.path.exists(filepath): return default
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logging.warning(f"Failed to read JSON from {filepath}: {e}")
        return default

def _now_ts():
    return datetime.now(timezone.utc).isoformat()

def metrics_log(event_name: str, payload: dict | None = None):
    try:
        payload = dict(payload or {})
        payload['event'] = payload.get('event', event_name)
        if not payload.get('event'): return
        payload.setdefault('ts', _now_ts())
        path = os.getenv('E8_METRICS_PATH', 'metrics.ndjson')
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=False) + '\n')
    except Exception:
        pass

def sanitize_line(text: str, max_chars: int = 80) -> str:
    if not isinstance(text, str): return ""
    return text.replace('\n', ' ').replace('\r', '').strip()[:max_chars]

def sanitize_block(text: str, max_sentences: int = 5, max_chars: int = 500) -> str:
    if not isinstance(text, str): return ""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return " ".join(sentences[:max_sentences])[:max_chars]

def _parse_json_object(text: str) -> dict:
    if not text: return {}
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}

def _fix_mojibake(text: str) -> str:
    if not isinstance(text, str): return str(text)
    try:
        return text.encode('latin1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text

class SafeConsole:
    def __init__(self, console):
        self._console = console
    def _sanitize_arg(self, a):
        return _fix_mojibake(a) if isinstance(a, str) else a
    def log(self, *args, **kwargs):
        safe_args = tuple(self._sanitize_arg(a) for a in args)
        return self._console.log(*safe_args, **kwargs)
    def print(self, *args, **kwargs):
        safe_args = tuple(self._sanitize_arg(a) for a in args)
        return self._console.print(*safe_args, **kwargs)
    def __getattr__(self, name):
        return getattr(self._console, name)

class DimmedLoggerConsole:
    def __init__(self, console_instance):
        self._original_console = console_instance
    def log(self, *args, **kwargs):
        if os.getenv("E8_HIDE_DIMMED_LOGS", "1") == "1":
            return
        dimmed_args = tuple((f"[dim]{arg}[/dim]" if isinstance(arg, str) else arg) for arg in args)
        return self._original_console.log(*dimmed_args, **kwargs)
    def __getattr__(self, name):
        return getattr(self._original_console, name)

def seed_all(seed: int):
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
