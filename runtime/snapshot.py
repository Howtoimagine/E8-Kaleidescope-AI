from __future__ import annotations

"""
Runtime snapshot utilities for E8Mind.

- SnapshotWriter: creates a timestamped run directory under config.runtime_dir
  and writes JSON snapshots like snapshot_step_000100.json
- SnapshotReader: minimal helpers to load prior snapshots if needed
"""

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List, cast


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _jsonify(obj: Any) -> Any:
    """Best-effort conversion to JSON-serializable types."""
    # Guard against classes (including dataclass types)
    if isinstance(obj, type):
        return str(obj)
    # Only call asdict on dataclass instances, not dataclass classes
    if is_dataclass(obj):
        # Pylance type narrowing doesn't recognize dataclass instance; cast to Any
        return asdict(cast(Any, obj))
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    if isinstance(obj, (set,)):
        return list(obj)
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8", errors="ignore")
        except Exception:
            return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


class SnapshotWriter:
    """
    Manages a single run directory and writes snapshots as JSON files.

    Structure:
      {runtime_dir}/{run_name}/
        - bandit_state.json         (optional global state)
        - boundary_fabric.json      (optional aux)
        - quasicrystal_blueprint.json (optional aux)
        - snapshot_step_000100.json
        - logs/
        - debug/
    """

    def __init__(self, runtime_dir: str) -> None:
        self.runtime_dir = runtime_dir
        _ensure_dir(self.runtime_dir)
        self.run_dir: Optional[str] = None

    def start_new_run(self, run_name: Optional[str] = None) -> str:
        if not run_name:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{ts}"
        self.run_dir = os.path.join(self.runtime_dir, run_name)
        _ensure_dir(self.run_dir)
        _ensure_dir(os.path.join(self.run_dir, "logs"))
        _ensure_dir(os.path.join(self.run_dir, "logs", "console"))
        _ensure_dir(os.path.join(self.run_dir, "debug"))
        return self.run_dir

    def ensure_started(self) -> str:
        if not self.run_dir:
            return self.start_new_run()
        return self.run_dir

    def write_json(self, filename: str, payload: Dict[str, Any]) -> str:
        run_dir = self.ensure_started()
        path = os.path.join(run_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_jsonify(payload), f, ensure_ascii=False, indent=2)
        return path

    def write_snapshot(self, step: int, payload: Dict[str, Any]) -> str:
        fname = f"snapshot_step_{step:06d}.json"
        return self.write_json(fname, payload)

    def write_state(self, payload: Dict[str, Any], filename: str = "bandit_state.json") -> str:
        return self.write_json(filename, payload)

    def run_path(self, *parts: str) -> str:
        run_dir = self.ensure_started()
        return os.path.join(run_dir, *parts)


class SnapshotReader:
    """Utility to list runs and load latest snapshot."""

    @staticmethod
    def list_runs(runtime_dir: str) -> List[str]:
        if not os.path.isdir(runtime_dir):
            return []
        items = [d for d in os.listdir(runtime_dir) if os.path.isdir(os.path.join(runtime_dir, d))]
        # Prefer run_YYYYMMDD_HHMMSS lexicographic order
        items.sort()
        return items

    @staticmethod
    def latest_run(runtime_dir: str) -> Optional[str]:
        runs = SnapshotReader.list_runs(runtime_dir)
        if not runs:
            return None
        return os.path.join(runtime_dir, runs[-1])

    @staticmethod
    def latest_snapshot(run_dir: str) -> Optional[str]:
        if not os.path.isdir(run_dir):
            return None
        snaps = [f for f in os.listdir(run_dir) if f.startswith("snapshot_step_") and f.endswith(".json")]
        if not snaps:
            return None
        snaps.sort()
        return os.path.join(run_dir, snaps[-1])

    @staticmethod
    def load_json(path: str) -> Optional[Dict[str, Any]]:
        if not path or not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
