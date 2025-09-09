from __future__ import annotations

"""
Runtime Orchestrator for E8Mind.

Coordinates:
- Configuration loading
- Services (Physics + Engines + Memory)
- SubconsciousLayer tick loop
- State persistence and periodic snapshots
"""

from typing import Any, Dict, Optional
from dataclasses import asdict
import numpy as np

from core.config import AppConfig
from physics.e8_lattice import E8Physics
from physics.engines import QuantumEngine, ClassicalEngine
from memory.manager import MemoryManager
from cognitive.subconscious import SubconsciousLayer, SubconsciousConfig

from runtime.snapshot import SnapshotWriter
from runtime.state_manager import StateManager


class Console:
    def log(self, *args: Any, **kwargs: Any) -> None:
        print(*args)


class Orchestrator:
    def __init__(self, cfg: AppConfig, console: Optional[Any] = None) -> None:
        self.cfg = cfg
        self.console = console or Console()

        # Core services
        self.physics = E8Physics(self.console)
        self.qengine = QuantumEngine(self.physics, config={"kernel": "cosine", "rbf_sigma": 0.8}, console=self.console)
        self.cengine = ClassicalEngine(self.physics, console=self.console)
        self.memory = MemoryManager(embed_dim=cfg.embed_dim, seed=cfg.global_seed)

        # Subconscious layer
        scfg = SubconsciousConfig()  # keep defaults configurable later
        self.subcon = SubconsciousLayer(
            physics=self.physics,
            qengine=self.qengine,
            cengine=self.cengine,
            memory=self.memory,
            console=self.console,
            config=scfg,
        )

        # Persistence
        self.writer = SnapshotWriter(cfg.runtime_dir)
        self.state_mgr = StateManager(cfg, self.writer, console=self.console)

        self._started: bool = False

    def log(self, *args: Any) -> None:
        try:
            if self.console and hasattr(self.console, "log"):
                self.console.log("[Orchestrator]", *args)
        except Exception:
            pass

    def start(self, run_name: Optional[str] = None, resume: bool = False) -> str:
        run_dir = self.state_mgr.start(run_name=run_name, resume=resume)
        self._started = True

        # Precompute and store blueprint for visualization tools
        try:
            pts = self.cengine.blueprint(seed=None)
            self.writer.write_json("quasicrystal_blueprint.json", {"points": pts})
        except Exception as e:
            self.log("blueprint_error:", e)

        return run_dir

    def step(self, label_hint: Optional[str] = None) -> Dict[str, Any]:
        if not self._started:
            self.start()

        # Advance global step
        step_id = self.state_mgr.next_step()

        # Prepare inputs (anchors/visits can evolve over time; keep minimal for now)
        curiosity_visits = None
        if self.state_mgr.state.curiosity_visits is not None:
            try:
                curiosity_visits = np.asarray(self.state_mgr.state.curiosity_visits, dtype=np.float32)
            except Exception:
                curiosity_visits = None

        # Perform subconscious tick
        result = self.subcon.tick(
            anchors=None,  # extend later with curriculum anchors
            curiosity_visits=curiosity_visits,
            label_hint=label_hint,
        )

        # Integrate results to state
        self.state_mgr.apply_tick_result(result)

        # Snapshot current step
        payload = {
            "state": self.state_mgr.state.to_dict(),
            "tick": result,
        }
        self.writer.write_snapshot(step_id, payload)

        # Persist state
        self.state_mgr.save_state()

        # Log concise step summary
        try:
            mins = result.get("minima", [])
            stats = result.get("potential_stats") or {}
            self.log(f"step={step_id} minima={len(mins)} V[min]={stats.get('min')} V[max]={stats.get('max')}")
        except Exception:
            pass

        return result

    def run(self, max_steps: int = 100, label_hint: Optional[str] = None) -> None:
        if not self._started:
            self.start()
        for _ in range(max_steps):
            self.step(label_hint=label_hint)


def run_orchestrator(cfg: AppConfig, steps: int = 100, resume: bool = False, run_name: Optional[str] = None,
                     label_hint: Optional[str] = None, console: Optional[Any] = None) -> None:
    orch = Orchestrator(cfg, console=console)
    orch.start(run_name=run_name, resume=resume)
    orch.run(max_steps=int(steps), label_hint=label_hint)
