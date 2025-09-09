from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, Optional, List

from core.config import AppConfig
from runtime.snapshot import SnapshotWriter, SnapshotReader

ISO_FMT = "%Y-%m-%dT%H:%M:%S"

@dataclass
class RuntimeState:
    """Lightweight, JSON-safe runtime state."""
    step: int = 0
    profile: str = "default"
    rng_seed: int = 42
    created_at: str = field(default_factory=lambda: datetime.now().strftime(ISO_FMT))
    updated_at: str = field(default_factory=lambda: datetime.now().strftime(ISO_FMT))
    curiosity_visits: Optional[List[float]] = None  # optional tracking vector
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuntimeState":
        # Be resilient to missing keys and only accept iterable curiosity_visits
        cv_raw = data.get("curiosity_visits")
        if isinstance(cv_raw, (list, tuple)):
            try:
                cv: Optional[List[float]] = [float(x) for x in cv_raw]
            except Exception:
                cv = None
        else:
            cv = None

        return cls(
            step=int(data.get("step", 0)),
            profile=str(data.get("profile", "default")),
            rng_seed=int(data.get("rng_seed", 42)),
            created_at=str(data.get("created_at", datetime.now().strftime(ISO_FMT))),
            updated_at=str(data.get("updated_at", datetime.now().strftime(ISO_FMT))),
            curiosity_visits=cv,
            extras=dict(data.get("extras", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Ensure JSON-safe types
        if self.curiosity_visits is not None:
            d["curiosity_visits"] = list(self.curiosity_visits)
        return d

class StateManager:
    """Owns the runtime state and persistence lifecycle."""
    def __init__(self, cfg: AppConfig, writer: SnapshotWriter, console: Optional[Any] = None) -> None:
        self.cfg = cfg
        self.writer = writer
        self.console = console
        self.state: RuntimeState = RuntimeState(
            step=0,
            profile=cfg.default_profile,
            rng_seed=cfg.global_seed,
        )

    def log(self, *args: Any) -> None:
        try:
            if self.console and hasattr(self.console, "log"):
                self.console.log("[StateManager]", *args)
        except Exception:
            pass

    def start(self, run_name: Optional[str] = None, resume: bool = False) -> str:
        if resume:
            latest = SnapshotReader.latest_run(self.cfg.runtime_dir)
            if latest:
                # Attach to existing run and attempt to load state
                self.writer.run_dir = latest
                st = self._load_state_file()
                if st:
                    self.state = st
                    self.log("Resumed run:", latest, "at step", self.state.step)
                    return latest
        # Else fresh run
        run_dir = self.writer.start_new_run(run_name=run_name)
        self.state = RuntimeState(
            step=0,
            profile=self.cfg.default_profile,
            rng_seed=self.cfg.global_seed,
        )
        self.save_state()
        self.log("Started new run:", run_dir)
        return run_dir

    def _load_state_file(self) -> Optional[RuntimeState]:
        try:
            path = self.writer.run_path("bandit_state.json")
            import json, os
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return RuntimeState.from_dict(data)
        except Exception as e:
            self.log("load_state_error:", e)
        return None

    def save_state(self) -> str:
        self.state.updated_at = datetime.now().strftime(ISO_FMT)
        return self.writer.write_state(self.state.to_dict())

    def next_step(self) -> int:
        self.state.step += 1
        self.state.updated_at = datetime.now().strftime(ISO_FMT)
        return self.state.step

    def apply_tick_result(self, result: Dict[str, Any]) -> None:
        """
        Optionally integrate tick outputs into global state.
        For now, store minima count and potential stats under extras.
        """
        try:
            extras = self.state.extras
            extras["last_minima"] = result.get("minima", [])
            extras["last_created"] = result.get("created", [])
            extras["last_potential_stats"] = result.get("potential_stats")
            self.state.extras = extras
        except Exception:
            pass
