from __future__ import annotations

from typing import Any, Dict, List, Optional
import asyncio
import os
import re
import time

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    class _NP:
        def stack(self, arr):
            return arr
        def mean(self, arr, axis=0):
            try:
                # simple average over list of lists
                if not arr:
                    return []
                n = len(arr)
                m = len(arr[0]) if isinstance(arr[0], (list, tuple)) else 0
                if not m:
                    return arr[0]
                return [sum(row[i] for row in arr)/n for i in range(m)]
            except Exception:
                return arr[0] if arr else []
        def array(self, x, dtype=None):
            return x
        def zeros(self, n, dtype=None):
            return [0.0]*int(n)
    np = _NP()  # type: ignore

try:
    from rich.panel import Panel  # type: ignore
except Exception:  # pragma: no cover
    Panel = None  # type: ignore


DREAM_MODE_ENABLED = True
DREAM_MIN_INTERVAL_SEC = float(os.getenv("E8_DREAM_COOLDOWN", "30"))
DREAM_SEQUENCE_TIMEOUT = float(os.getenv("E8_DREAM_SEQUENCE_TIMEOUT", "30"))


class DreamEngine:
    """
    Generates synthetic memories by running thought experiments about future possibilities,
    allowing the AI to learn from events that haven't happened.
    """
    ALLOWED_TYPES = (
        "explorer_insight", "insight_synthesis", "meta_reflection", "phase_summary",
        "concept", "external_concept", "mutation", "synthetic_memory", "self_code", "self_code_section"
    )

    def __init__(self, memory, mind_instance):
        self.memory = memory
        self.mind = mind_instance
        self.console = mind_instance.console

    def _eligible_concepts(self):
        G = self.memory.graph_db.graph
        out = []
        for nid, d in G.nodes(data=True):
            if d.get("folded"):
                continue
            if d.get("type") not in self.ALLOWED_TYPES:
                continue
            if self.memory.main_vectors.get(nid) is None:
                continue
            out.append((nid, d))
        return out

    def _pick_from_tension(self, elig, k=1):
        if not elig:
            return []
        tension_candidates = sorted(elig, key=lambda item: item[1].get('shell_tension', 0.0), reverse=True)
        high_tension_seeds = [item for item in tension_candidates if item[1].get('shell_tension', 0.0) > 0.1]
        if high_tension_seeds:
            return high_tension_seeds[:k]
        else:
            return self._pick_neutral(elig, k)

    def _pick_neutral(self, elig, k=1):
        if not elig:
            return []
        elig.sort(key=lambda item: (item[1].get("temperature", 0.0), item[1].get("step", 0)), reverse=True)
        pool_size = min(len(elig), 5)
        if pool_size == 0:
            return []
        top_candidates = elig[:pool_size]
        import random
        num_to_sample = min(k, len(top_candidates))
        return random.sample(top_candidates, num_to_sample)

    async def run_dream_sequence(self, depth=1):
        min_for_thought_exp = int(os.getenv("E8_MIN_FOR_THOUGHT_EXP", "40"))
        current_nodes = self.memory.graph_db.graph.number_of_nodes()
        if current_nodes < min_for_thought_exp:
            return
        if not hasattr(self.mind, "_dream_lock"):
            self.mind._dream_lock = asyncio.Lock()
        if not hasattr(self.mind, "_last_dream_at"):
            self.mind._last_dream_at = 0.0

        if not DREAM_MODE_ENABLED:
            return
        now = time.monotonic()
        if self.mind._dream_lock.locked() or (now - self.mind._last_dream_at < DREAM_MIN_INTERVAL_SEC):
            return

        async with self.mind._dream_lock:
            if time.monotonic() - self.mind._last_dream_at < DREAM_MIN_INTERVAL_SEC:
                return

            self.mind._last_dream_at = time.monotonic()

            elig = self._eligible_concepts()
            if not elig:
                try:
                    self.console.log("[Thought Experiment] No suitable concepts found.")
                except Exception:
                    pass
                return

            seed = self._pick_neutral(elig, k=1)
            if not seed:
                try:
                    self.console.log("[Thought Experiment] Seed picking failed.")
                except Exception:
                    pass
                return

            seed_node_id, seed_node_data = seed[0]

            _dream_t0 = time.perf_counter()
            try:
                _, top_goal_desc = self.mind.goal_field.get_top_goals(k=1)[0]
            except (IndexError, TypeError, AttributeError):
                top_goal_desc = "achieve a greater understanding"

            try:
                experiment_prompt = self.mind.prompts.render(
                    "thought_experiment",
                    concept=seed_node_data.get('label', 'a concept'),
                    details=seed_node_data.get('metaphor', ''),
                    goal=top_goal_desc
                )

                narrative = await asyncio.wait_for(self.mind.llm_pool.enqueue_and_wait(
                    experiment_prompt, max_tokens=600, temperature=0.85
                ), timeout=DREAM_SEQUENCE_TIMEOUT)

                if not narrative or (isinstance(narrative, str) and narrative.startswith("[LLM")):
                    try:
                        self.console.log("[Thought Experiment] LLM failed, using synthetic fallback.")
                    except Exception:
                        pass
                    narrative = f"A hypothetical exploration of {seed_node_data.get('label', 'this concept')} reveals potential connections to broader patterns in the system."

            except asyncio.TimeoutError:
                try:
                    self.console.log(f"[Thought Experiment] LLM request timed out after {DREAM_SEQUENCE_TIMEOUT}s, using synthetic fallback.")
                except Exception:
                    pass
                narrative = f"A hypothetical exploration of {seed_node_data.get('label', 'this concept')} reveals potential connections to broader patterns in the system."
            except asyncio.CancelledError:
                try:
                    self.console.log("[Thought Experiment] LLM request was cancelled, using synthetic fallback.")
                except Exception:
                    pass
                narrative = f"A hypothetical exploration of {seed_node_data.get('label', 'this concept')} reveals potential connections to broader patterns in the system."
            except Exception as llm_e:
                try:
                    self.console.log(f"[Thought Experiment] LLM request failed: {llm_e}, using synthetic fallback.")
                except Exception:
                    pass
                narrative = f"A hypothetical exploration of {seed_node_data.get('label', 'this concept')} reveals potential connections to broader patterns in the system."

            # Process narrative
            try:
                if narrative and not (isinstance(narrative, str) and narrative.startswith("[LLM")):
                    if len(str(narrative).strip()) < 10:
                        try:
                            self.console.log("[Thought Experiment] Generated narrative too short, using enhanced fallback.")
                        except Exception:
                            pass
                        narrative = f"A hypothetical exploration of {seed_node_data.get('label', 'this concept')} suggests it may interact with related systems through emergent patterns, potentially revealing new connections and implications for our understanding."

                    try:
                        seed_neighbors = self.memory.graph_db.get_neighbors(seed_node_id)
                        neighbor_data = [self.memory.graph_db.get_node(n) for n in seed_neighbors if self.memory.graph_db.get_node(n)]
                        neighbor_labels = [d.get('label', '') for d in neighbor_data if d]
                        local_terms = set(re.findall(r"[A-Za-z0-9]+", " ".join(neighbor_labels).lower()))
                        if local_terms and not any(term in str(narrative).lower() for term in local_terms):
                            narrative = f"(loose) {narrative}"
                    except Exception:
                        pass

                    experiment_label = f"Experiment: {seed_node_data.get('label', 'Unknown Concept')}"
                    if len(experiment_label) > 100:
                        experiment_label = experiment_label[:97] + "..."

                    new_node_id = await self.mind.memory.add_entry({
                        "type": "synthetic_memory",
                        "label": experiment_label,
                        "metaphor": narrative,
                        "rating": 0.75,
                        "is_synthetic": True,
                        "step": getattr(self.mind, 'step_num', None)
                    }, parent_ids=[seed_node_id])

                    try:
                        latency_ms = (time.perf_counter() - _dream_t0) * 1000.0
                        if hasattr(self.mind, 'metrics'):
                            self.mind.metrics.increment('dream.performed')
                            self.mind.metrics.timing('dream.latency_ms', latency_ms)
                    except Exception:
                        pass

                    if Panel is not None:
                        try:
                            self.console.print(Panel(
                                f"[bold]Seed Concept:[/] {seed_node_data.get('label', 'Unknown')}\n[bold]Hypothetical Narrative:[/] {narrative}",
                                title="[bold blue]THOUGHT EXPERIMENT[/]", border_style="blue"
                            ))
                        except Exception:
                            pass

                    try:
                        if hasattr(self.mind, 'subconscious_event_log') and isinstance(self.mind.subconscious_event_log, list):
                            self.mind.subconscious_event_log.append({
                                'type': 'thought_experiment',
                                'label': experiment_label,
                                'step': getattr(self.mind, 'step_num', None),
                                'data': {'summary': str(narrative)}
                            })
                    except Exception:
                        pass
                else:
                    try:
                        self.console.log("[Thought Experiment] No valid narrative generated, skipping.")
                    except Exception:
                        pass
            except Exception as e:
                import traceback
                tb = traceback.format_exc(limit=6)
                try:
                    self.console.log(f"[Thought Experiment] Failed to run experiment: {e}\n[trace]\n{tb}")
                except Exception:
                    pass
                try:
                    if hasattr(self.mind, 'metrics'):
                        self.mind.metrics.increment('dream.experiment_fail')
                except Exception:
                    pass


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
            if child is None:
                child = ep.get("embedding")
            parents = ep.get("parent_ids") or []
            if child is None or not parents:
                continue
            pv = [self.mind.memory.main_vectors.get(pid) for pid in parents if pid in self.mind.memory.main_vectors]
            if not pv:
                continue
            if hasattr(np, 'stack'):
                s_arr = np.mean(np.stack(pv), axis=0)
                try:
                    s = s_arr.astype('float32')  # type: ignore[attr-defined]
                except Exception:
                    s = s_arr
            else:
                s = np.mean(pv, axis=0)
            sp = np.array(child, dtype='float32') if hasattr(np, 'array') else child
            a = np.zeros(adim, dtype='float32') if hasattr(np, 'zeros') else [0.0]*adim
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
                        try:
                            G.graph[pid][node_id]['hypervector'] = hv
                        except Exception:
                            pass
                    reinforced += 1
                except Exception:
                    pass
        try:
            if hasattr(self.mind, 'metrics'):
                self.mind.metrics.increment("graph.edge_reinforce", reinforced)
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
                try:
                    loss = wm.train_batch([(s,a,sp,r) for (s,a,sp,r,_,_) in traj])
                    if loss and hasattr(self.mind, "metrics"):
                        try:
                            self.mind.metrics.observe("wm.loss.recon", loss.get("loss_recon"))
                            self.mind.metrics.observe("wm.loss.kl", loss.get("loss_kl"))
                        except Exception:
                            pass
                except Exception:
                    pass
            self._reinforce_graph(traj)
            total += len(traj)
        dt = time.monotonic() - t0
        if dt > 0 and hasattr(self.mind, "metrics"):
            try:
                self.mind.metrics.observe("replay.samples_per_sec", total/max(1e-6, dt))
            except Exception:
                pass
