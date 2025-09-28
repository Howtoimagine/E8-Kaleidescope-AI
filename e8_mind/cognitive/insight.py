from __future__ import annotations

# Modularized cognitive insight components, ported from legacy M24.2 implementation
# with cautious dependency guards and graceful fallbacks.

from typing import Any, Dict, List, Optional
import asyncio
import json
import os
import re
import threading
import time
import hashlib
import heapq
from collections import deque

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - minimal fallback
    # Tiny shim to avoid crashes if numpy is temporarily unavailable
    class _NP:
        def clip(self, x, lo, hi):
            try:
                return max(lo, min(hi, float(x)))
            except Exception:
                return lo
        def exp(self, x):
            try:
                import math
                return math.e ** float(x)
            except Exception:
                return 1.0
        def mean(self, arr):
            try:
                return sum(arr) / max(1, len(arr))
            except Exception:
                return 0.0
        def array(self, x):
            return x
        def isfinite(self, x):
            try:
                import math
                return math.isfinite(float(x))
            except Exception:
                return False
    np = _NP()  # type: ignore

try:
    # Optional rich console types for nicer output if available
    from rich.console import Console  # type: ignore
    from rich.panel import Panel  # type: ignore
    from rich.markup import escape  # type: ignore
except Exception:  # pragma: no cover
    Console = object  # type: ignore
    Panel = None  # type: ignore
    def escape(x: str) -> str:  # type: ignore
        return x


# Fallback metrics logger
def metrics_log(event: str, data: Dict[str, Any]):  # pragma: no cover - no-op default
    try:
        # Hook for external metrics systems if injected elsewhere
        _ = (event, data)
    except Exception:
        pass


# Conservative defaults for timeouts and missing constants
HYPOTHESIS_TIMEOUT: float = float(os.getenv("E8_HYPOTHESIS_TIMEOUT", "20"))
DIMENSIONAL_SHELL_SIZES: List[int] = [8, 16, 32, 64, 128]


def _now_ts() -> int:
    """Return a monotonic-ish timestamp in milliseconds."""
    return int(time.time() * 1000)


def _parse_json_object(s: Any) -> Any:
    """Best-effort JSON object parser: accepts plain objects or fenced blocks.

    Returns parsed Python object on success; raises on hard failure to match legacy behavior.
    """
    if isinstance(s, (dict, list)):
        return s
    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    t = s.strip()
    # Remove Markdown code fences if present
    if t.startswith("```json"):
        t = t[7:]
    if t.startswith("```"):
        t = t[3:]
    if t.endswith("```"):
        t = t[:-3]
    t = t.strip()
    try:
        return json.loads(t)
    except Exception:
        # Try to extract the first balanced {...} block
        start = t.find('{')
        if start != -1:
            depth = 0
            for i in range(start, len(t)):
                ch = t[i]
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        frag = t[start:i+1]
                        try:
                            return json.loads(frag)
                        except Exception:
                            break
        # Last resort: return empty dict to avoid crashes in callers that expect dict
        return {}


def module_fallback_get_recent_insights(mind: Any, n: int = 5) -> List[Dict[str, Any]]:
    """Fallback accessor used by sandbox if mind lacks helper.

    Attempts to call mind.get_recent_insights; otherwise returns an empty list.
    """
    try:
        fn = getattr(mind, 'get_recent_insights', None)
        if callable(fn):
            res = fn(n)
            return res if isinstance(res, list) else []
    except Exception:
        pass
    return []


class NoveltyScorer:
    def __init__(self, memory_manager: Any, llm_pool: Any, console: Any):
        self.console = console
        self.memory_manager = memory_manager
        self.llm_pool = llm_pool

    def calculate_novelty(self, new_vector: Any) -> float:
        """Calculate novelty score in consistent 0.0-1.0 range"""
        try:
            similar_nodes = self.memory_manager.find_similar_in_main_storage(new_vector, k=1)
        except Exception:
            similar_nodes = []
        if not similar_nodes:
            return 1.0  # Maximum novelty for completely unique vectors

        try:
            distance_to_nearest = similar_nodes[0][1]
        except Exception:
            distance_to_nearest = 0.0
        try:
            avg_distance = self.memory_manager.get_average_nearest_neighbor_distance()
        except Exception:
            avg_distance = 1.0

        if avg_distance < 1e-6:
            return 1.0

        # Calculate raw novelty ratio
        novelty_ratio = distance_to_nearest / avg_distance

        # Map to 0.0-1.0 range using sigmoid-like function
        novelty_score = 2.0 / (1.0 + np.exp(-novelty_ratio)) - 1.0  # Maps [0,inf) -> [0,1)

        try:
            return float(np.clip(novelty_score, 0.0, 1.0))
        except Exception:
            return float(novelty_score)

    async def calculate_coherence(self, new_concept_text: str) -> float:
        if not new_concept_text:
            return 0.0
        prompt = (
            'On a scale from 0.0 to 1.0, how coherent and meaningful is the following idea? '
            'A coherent idea is well-formed, logical, and potentially useful. '
            'Respond with ONLY the numeric score.\n\n'
            f'Idea: "{new_concept_text}"\n\n'
            'Coherence Score:'
        )
        try:
            response = await self.llm_pool.enqueue_and_wait(prompt, max_tokens=10, temperature=0.1)
            if response is None:
                return 0.5  # failure fallback
            match = re.search(r"[-+]?\d*\.\d+|\d+", response)
            if match:
                return float(np.clip(float(match.group()), 0.0, 1.0))
            return 0.5
        except Exception as e:  # pragma: no cover - external system
            try:
                self.console.log(f"[NoveltyScorer] Coherence check failed: {e}")
            except Exception:
                pass
            return 0.5

    def score_text_novelty(self, text: str) -> float:
        """Heuristic novelty score from text when vector not available.

        Uses fraction of unique tokens and length scaling as a lightweight proxy.
        Returns value in [0,1].
        """
        try:
            if not isinstance(text, str) or not text.strip():
                return 0.0
            toks = re.findall(r"[A-Za-z0-9']+", text.lower())
            if not toks:
                return 0.0
            unique = len(set(toks)) / len(toks)
            length_boost = min(1.0, len(toks) / 50.0)
            return float(max(0.0, min(1.0, 0.6 * unique + 0.4 * length_boost)))
        except Exception:
            return 0.5


class EpisodicMemory:
    def __init__(self, max_size: int = 500):
        self.max_size = int(max_size)
        self.heap: List[tuple[float, Dict[str, Any]]] = []

    def add_episode(self, episode_data: Dict[str, Any], reward: float):
        # Store reward explicitly; do not conflate with any rating field that may exist
        try:
            episode_data['reward'] = float(reward)
        except Exception:
            episode_data['reward'] = 0.0
        priority = -float(episode_data['reward'])
        heapq.heappush(self.heap, (priority, episode_data))
        if len(self.heap) > self.max_size:
            heapq.heappop(self.heap)

    def get_top_episodes(self, k: int = 1) -> List[Dict[str, Any]]:
        if not self.heap:
            return []
        top_k = heapq.nsmallest(int(k), self.heap)
        return [data for _priority, data in top_k]

    def sample_prioritized(self, k: int = 32, mind: Any = None, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0, eta: float = 0.6) -> List[Dict[str, Any]]:
        if not self.heap:
            return []
        episodes = [ep for (_prio, ep) in self.heap]
        rewards: List[float] = []
        surprises: List[float] = []
        novelties: List[float] = []
        for ep in episodes:
            # Only use reward. If missing, treat as neutral (0.0). Ratings are evaluative scores, not reinforcement.
            try:
                rewards.append(float(ep.get('reward', 0.0)))
            except Exception:
                rewards.append(0.0)
            try:
                if mind is not None and 'node_id' in ep and 'parent_ids' in ep:
                    child_vec = mind.memory.main_vectors.get(ep['node_id']) or ep.get('embedding')
                    parents = [mind.memory.main_vectors.get(pid) for pid in ep['parent_ids'] if pid in mind.memory.main_vectors]
                    if child_vec is not None and parents:
                        # Surprise: distance from parent mean
                        pmean = np.mean([p for p in parents if p is not None])  # type: ignore[arg-type]
                        try:
                            # If numpy present, prefer vector ops
                            import numpy as _np2  # type: ignore
                            pmean = _np2.mean(_np2.stack(parents), axis=0)
                            surprises.append(float(_np2.linalg.norm(_np2.array(child_vec) - pmean)))
                        except Exception:
                            surprises.append(0.0)
                    else:
                        surprises.append(0.0)
                else:
                    surprises.append(float(ep.get('surprise', 0.0)))
            except Exception:
                surprises.append(0.0)
            try:
                nov = float(ep.get('novelty', 0.0))
                if nov == 0.0 and mind is not None and 'node_id' in ep:
                    try:
                        sim = mind.memory.find_similar_in_main_storage(mind.memory.main_vectors.get(ep['node_id']), k=2)
                        d = 1.0 - (sim[0][1] if sim else 0.0)
                        nov = float(d)
                    except Exception:
                        pass
                novelties.append(nov)
            except Exception:
                novelties.append(0.0)

        # Convert to arrays if possible
        try:
            rewards_arr = np.asarray(rewards, dtype=float)  # type: ignore[attr-defined]
            surprises_arr = np.asarray(surprises, dtype=float)  # type: ignore[attr-defined]
            novelties_arr = np.asarray(novelties, dtype=float)  # type: ignore[attr-defined]
            pri = alpha * rewards_arr + beta * surprises_arr + gamma * novelties_arr
            pri = np.maximum(pri, 1e-8)  # type: ignore[attr-defined]
            w = np.power(pri, eta)  # type: ignore[attr-defined]
            w = w / (w.sum() + 1e-12)  # type: ignore[attr-defined]
            # Weighted sample without replacement (approx using choice)
            import numpy as _np3  # type: ignore
            idxs = _np3.random.choice(_np3.arange(len(episodes)), size=min(int(k), len(episodes)), replace=False, p=w)
            return [episodes[int(i)] for i in idxs]
        except Exception:
            # Fallback: uniform sample
            return episodes[: min(int(k), len(episodes))]


class InsightAgent:
    def __init__(self, llm_pool: Any, novelty_scorer: NoveltyScorer, console: Any):
        self.console = console
        self.llm_pool = llm_pool
        self.novelty_scorer = novelty_scorer
        self.reward_history: deque[float] = deque(maxlen=100)
        self.episodic_memory = EpisodicMemory()
        self.recent_insights: deque[Dict[str, Any]] = deque(maxlen=200)

    class _InsightResult:
        def __init__(self, content: str, rating: float):
            self.content = content
            self.rating = rating

    def generate_insight_from_prompt(self, prompt: str):
        """Synchronous helper used by RecursiveArchitectureEngine.

        If in an active event loop, schedule the task and return a placeholder.
        Otherwise, block briefly to get the completion and score it.
        """
        content: str = ""
        rating: float = 0.0
        try:
            can_call = hasattr(self.llm_pool, 'enqueue_and_wait')
            if can_call:
                try:
                    loop = asyncio.get_running_loop()
                    # Loop is running; schedule task but do not block
                    try:
                        loop.create_task(self.llm_pool.enqueue_and_wait(prompt, max_tokens=160, temperature=0.75))
                        content = "[deferred insight generation: async loop active]"
                        rating = 0.0
                    except Exception:
                        content = ""
                except RuntimeError:
                    # No running loop: safe to block
                    try:
                        content = asyncio.run(self.llm_pool.enqueue_and_wait(prompt, max_tokens=160, temperature=0.75))
                    except Exception as exc:
                        try:
                            self.console.log(f"[InsightAgent] LLM generation failed: {exc}")
                        except Exception:
                            pass
        except Exception as exc:
            try:
                self.console.log(f"[InsightAgent] generate_insight_from_prompt error: {exc}")
            except Exception:
                pass

        # Rate with novelty scorer if available
        if content and not content.startswith('[deferred'):
            try:
                if hasattr(self.novelty_scorer, 'score_text_novelty'):
                    rating = float(self.novelty_scorer.score_text_novelty(content))
                    if not np.isfinite(rating):  # type: ignore[attr-defined]
                        rating = 0.0
                    rating = max(0.0, min(1.0, rating))
            except Exception:
                rating = 0.0

        normalized_content = content if isinstance(content, str) else ""
        safe_rating = float(rating) if getattr(np, 'isfinite', lambda x: True)(rating) else 0.0
        self._record_insight(
            normalized_content,
            safe_rating,
            source="prompt",
            metadata={'prompt': prompt} if isinstance(prompt, str) else None,
        )
        return self._InsightResult(content=normalized_content, rating=safe_rating)

    async def create_hybrid_concept(self, concept_a: Dict, concept_b: Dict) -> str:
        mind_instance = getattr(self.novelty_scorer, 'memory_manager', None)
        mind_instance = getattr(mind_instance, 'mind', None)
        top_goal_desc = "achieve greater understanding"
        try:
            if mind_instance is not None and hasattr(mind_instance, 'goal_field'):
                _, top_goal_desc = mind_instance.goal_field.get_top_goals(k=1)[0]
        except Exception:
            pass
        subconscious_narrative = getattr(getattr(mind_instance, 'subconscious', object()), 'narrative', '')
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
        if not isinstance(new_concept_text, str) or not new_concept_text.strip():
            try:
                self.console.log("[InsightAgent] LLM failed to generate hybrid concept (None), using synthetic fallback.")
            except Exception:
                pass
            # Generate a simple synthetic hybrid concept
            label_a = concept_a.get('metaphor', concept_a.get('label', 'unknown concept A'))
            label_b = concept_b.get('metaphor', concept_b.get('label', 'unknown concept B'))
            fallback = f"A synthesis combining aspects of {label_a} with {label_b} to advance understanding."
            self._record_insight(
                fallback,
                rating=0.0,
                source="hybrid_concept",
                metadata={'concept_a': concept_a, 'concept_b': concept_b, 'fallback': True},
            )
            return fallback

        stripped = new_concept_text.strip()
        rating = 0.0
        try:
            if hasattr(self.novelty_scorer, 'score_text_novelty'):
                rating = float(self.novelty_scorer.score_text_novelty(stripped))
                if not np.isfinite(rating):  # type: ignore[attr-defined]
                    rating = 0.0
                rating = max(0.0, min(1.0, rating))
        except Exception:
            rating = 0.0
        self._record_insight(
            stripped,
            rating=rating,
            source="hybrid_concept",
            metadata={'concept_a': concept_a, 'concept_b': concept_b, 'fallback': False},
        )
        return stripped

    def _record_insight(self, content: str, rating: float, source: str = "unknown", metadata: Optional[Dict[str, Any]] = None):
        try:
            entry: Dict[str, Any] = {
                'content': content if isinstance(content, str) else "",
                'rating': float(rating) if getattr(np, 'isfinite', lambda x: True)(rating) else 0.0,
                'source': source,
                'timestamp': time.time(),
                'uncertainty': 0.5,  # Default uncertainty, will be updated by validation
                'tags': self._extract_semantic_tags(content) if content else [],
            }
            if metadata:
                entry['metadata'] = metadata
            self.recent_insights.append(entry)

            # Validator hook: Auto-validate high-scoring insights
            try:
                if rating > 0.8 and hasattr(self, 'novelty_scorer') and self.novelty_scorer:
                    mind = getattr(self.novelty_scorer, 'memory_manager', None)
                    mind = getattr(mind, 'mind', None) if mind else None
                    if mind and hasattr(mind, 'validator'):
                        try:
                            loop = asyncio.get_running_loop()
                            loop.create_task(self._schedule_insight_validation(mind, entry))
                        except RuntimeError:
                            # No event loop, skip validation
                            try:
                                self.console.log(f"[InsightAgent] High-score insight ({rating:.3f}) recorded but validation skipped (no event loop)")
                            except Exception:
                                pass
                        except Exception as sched_err:
                            try:
                                self.console.log(f"[InsightAgent] Validation scheduling failed: {sched_err}")
                            except Exception:
                                pass
            except Exception as hook_err:  # pragma: no cover
                try:
                    self.console.log(f"[InsightAgent] Validator hook failed: {hook_err}")
                except Exception:
                    pass
        except Exception:
            try:
                # Last resort: keep structure consistent
                self.recent_insights.append({'content': str(content), 'rating': 0.0, 'source': source})
            except Exception:
                pass

    def _extract_semantic_tags(self, content: str) -> List[str]:
        """Extract semantic tags from insight content for clustering and prioritization."""
        if not content or not isinstance(content, str):
            return []

        tag_patterns: Dict[str, List[str]] = {
            'quantum': ['quantum', 'entanglement', 'superposition', 'decoherence', 'eigenstate', 'hamiltonian'],
            'physics': ['energy', 'momentum', 'force', 'field', 'particle', 'wave', 'relativity'],
            'complexity': ['complex', 'emergence', 'nonlinear', 'chaos', 'attractor', 'bifurcation'],
            'information': ['information', 'entropy', 'bit', 'encoding', 'compression', 'channel'],
            'mathematical': ['matrix', 'vector', 'algebra', 'calculus', 'topology', 'geometry'],
            'cognitive': ['learning', 'memory', 'attention', 'consciousness', 'perception'],
            'computational': ['algorithm', 'computation', 'simulation', 'optimization', 'recursive'],
            'predictive': ['predict', 'forecast', 'model', 'pattern', 'correlation', 'regression'],
            'experimental': ['experiment', 'test', 'validate', 'measure', 'observe', 'empirical'],
            'theoretical': ['theory', 'hypothesis', 'conjecture', 'proof', 'axiom', 'proposition'],
            'gamma': ['gamma'],
        }

        content_lower = content.lower()
        detected_tags: List[str] = []

        for tag_name, keywords in tag_patterns.items():
            if any(keyword in content_lower for keyword in keywords):
                detected_tags.append(tag_name)

        # Add error-related tag
        if any(word in content_lower for word in ['error', 'mistake', 'wrong', 'incorrect', 'fail']):
            detected_tags.append('error_analysis')

        return detected_tags[:5]

    async def _schedule_insight_validation(self, mind: Any, insight_entry: Dict[str, Any]):
        """Schedule validation for a high-scoring insight."""
        try:
            memory_entry = {
                "type": "insight",
                "label": insight_entry['content'][:50] + "..." if len(insight_entry['content']) > 50 else insight_entry['content'],
                "metaphor": insight_entry['content'],
                "rating": insight_entry['rating'],
                "source": "insight_agent",
                "step": getattr(mind, 'step_num', 0),
                "uncertainty": insight_entry.get('uncertainty', 0.5),
                "tags": insight_entry.get('tags', []),
            }

            # Add to memory and get node ID
            node_id = await mind.memory.add_entry(memory_entry)

            if node_id and hasattr(mind, 'validator'):
                await mind.validator.validate_insight(node_id)
                try:
                    self.console.log(f"[InsightAgent] Validated high-score insight: {insight_entry['rating']:.3f}")
                except Exception:
                    pass
        except Exception as e:  # pragma: no cover - external integration
            try:
                self.console.log(f"[InsightAgent] Insight validation failed: {e}")
            except Exception:
                pass

    def learn_from_reward(self, reward: float, episode_data: Optional[Dict[str, Any]] = None):
        self.reward_history.append(float(reward))
        if len(self.reward_history) > 10:
            try:
                avg_reward = float(np.mean(list(self.reward_history)))  # type: ignore[attr-defined]
            except Exception:
                avg_reward = sum(self.reward_history) / len(self.reward_history)
            try:
                self.console.log(f"[InsightAgent] Average Insight Reward: {avg_reward:.3f}")
            except Exception:
                pass
        if episode_data:
            self.episodic_memory.add_episode(episode_data, reward)

        # Metaphysical Compiler funnel: proposals on very-high-rated insights
        try:
            rating = float(reward)
        except Exception:
            rating = 0.0
        try:
            if rating >= 0.95:
                testable = bool(episode_data.get('computationally_testable', False)) if isinstance(episode_data, dict) else False
                proposal = self._make_proposal_from_episode(episode_data or {}, testable=testable)
                prop_dir = os.getenv('E8_PROPOSAL_PATH', '/mnt/data/proposals')
                try:
                    os.makedirs(prop_dir, exist_ok=True)
                except Exception:
                    prop_dir = '.'
                fname = os.path.join(prop_dir, f"proposal_{int(time.time())}_{hashlib.sha1(json.dumps(proposal).encode()).hexdigest()[:8]}.json")
                try:
                    with open(fname, 'w', encoding='utf-8') as f:
                        json.dump(proposal, f, ensure_ascii=False, indent=2)
                    try:
                        self.console.log(f"[MetaphysicalCompiler] Wrote proposal: {fname}")
                    except Exception:
                        pass
                except Exception as e:
                    try:
                        self.console.log(f"[MetaphysicalCompiler] Failed to write proposal: {e}")
                    except Exception:
                        pass

                # Run a short monitor-only sandbox experiment (non-destructive)
                try:
                    mind = getattr(self.novelty_scorer, 'memory_manager', None)
                    mind = getattr(mind, 'mind', None) if mind is not None else None
                    if mind is None:
                        mind = globals().get('MIND')
                    if mind is not None:
                        threading.Thread(target=lambda: self._run_sandbox_monitor_only(mind, proposal), daemon=True).start()
                except Exception:
                    pass
        except Exception:
            pass

    def _make_proposal_from_episode(self, episode: Dict[str, Any], testable: bool = False) -> Dict[str, Any]:
        # Minimal conservative proposal format
        proposal: Dict[str, Any] = {
            'ts': _now_ts(),
            'type': 'param' if testable else 'source',
            'episode_summary': episode.get('summary') if isinstance(episode, dict) else str(episode),
            'computationally_testable': bool(testable),
            'meta': {
                'reward_history_mean': float(np.mean(list(self.reward_history))) if self.reward_history else 0.0,  # type: ignore[attr-defined]
            },
        }
        return proposal

    def _run_sandbox_monitor_only(self, mind: Any, proposal: Dict[str, Any]):
        """Run a short monitor-only thought experiment (no geometry writes)."""
        try:
            steps = int(os.getenv('E8_SANDBOX_STEPS', '300'))
            prop_dir = os.getenv('E8_PROPOSAL_PATH', '/mnt/data/proposals')
            log_lines: List[Dict[str, Any]] = []
            prev_flag = getattr(mind, '_monitor_only', False)
            try:
                setattr(mind, '_monitor_only', True)
            except Exception:
                pass
            try:
                for s in range(steps):
                    recent = []
                    try:
                        recent = getattr(mind, 'get_recent_insights', lambda n=5: module_fallback_get_recent_insights(mind, n))(5)
                    except Exception:
                        recent = module_fallback_get_recent_insights(mind, 5)
                    try:
                        avg_rating = float(np.mean([ri.get('rating', 0.0) for ri in recent])) if recent else 0.0  # type: ignore[attr-defined]
                    except Exception:
                        avg_rating = 0.0
                    log_lines.append({'step': s, 'avg_rating': avg_rating})
                    time.sleep(0.005)
            finally:
                try:
                    setattr(mind, '_monitor_only', prev_flag)
                except Exception:
                    pass

            # Write sandbox telemetry
            try:
                os.makedirs(prop_dir, exist_ok=True)
                sname = os.path.join(prop_dir, f"sandbox_{int(time.time())}_{hashlib.sha1(json.dumps(proposal).encode()).hexdigest()[:8]}.ndjson")
                with open(sname, 'w', encoding='utf-8') as sf:
                    for ln in log_lines:
                        sf.write(json.dumps(ln) + '\n')
                try:
                    self.console.log(f"[MetaphysicalCompiler] Sandbox completed: {sname}")
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            pass


class HypothesisValidator:
    def __init__(self, mind_instance: Any):
        self.mind = mind_instance
        self.llm_pool = mind_instance.llm_pool
        # Provide direct console reference
        try:
            self.console: Any = mind_instance.console
            if hasattr(mind_instance.console, '_original_console'):
                self.undimmed_console: Any = mind_instance.console._original_console
            else:
                self.undimmed_console: Any = mind_instance.console
        except Exception:
            self.console = None  # type: ignore[assignment]
            self.undimmed_console = None  # type: ignore[assignment]

    async def validate_insight(self, insight_node_id: str, prompt_vars: Dict[str, Any] | None = None):
        insight_data = self.mind.memory.graph_db.get_node(insight_node_id)
        if not insight_data:
            try:
                self.mind.console.log(f"[Validator] Could not find insight data for node {insight_node_id}")
            except Exception:
                pass
            return

        hypothesis_text = insight_data.get('metaphor', insight_data.get('label', ''))
        if not hypothesis_text:
            return

        # Sanitize text using semantics if available
        if hasattr(self.mind, "semantics") and hasattr(self.mind.semantics, "sanitize_for_validation"):
            safe_text = self.mind.semantics.sanitize_for_validation(hypothesis_text)
        else:
            safe_text = hypothesis_text

        # Build prompt variables if not provided
        if prompt_vars is None:
            rating = insight_data.get('rating', 0.0)
            epoch = getattr(self.mind, 'epoch', 0)
            prompt_vars = {
                "insight_text": safe_text,
                "insight_rating": f"{rating:.3f}" if rating is not None else "null",
                "node_id": insight_node_id,
                "epoch": epoch,
            }

        # Try to use the new validator prompt from prompts.yaml
        try:
            prompts = getattr(self.mind, 'prompts', {})
            _pd = getattr(prompts, '_d', None)
            if (isinstance(_pd, dict) and 'validator' in _pd) or ('validator' in prompts if hasattr(prompts, 'keys') else False):
                validation_result = await self._validate_with_new_prompt(safe_text, prompt_vars)
                if validation_result:
                    await self._write_validation_to_node(insight_node_id, validation_result)
                    return
        except Exception as e:
            try:
                self.mind.console.log(f"[Validator] New prompt validation failed, falling back: {e}")
            except Exception:
                pass

        # Fallback to old validation system
        started = time.time()
        validation_obj: Dict[str, Any] = {
            "run_id": getattr(self.mind, 'run_id', None),
            "node_id": insight_node_id,
            "label": insight_data.get('label'),
            "hypothesis": hypothesis_text,
            "method": None,
            "confidence": None,
            "inputs": {"source_type": insight_data.get('type'), "rating": insight_data.get('rating')},
            "verdict": "unknown",
            "deltas": {},
            "next_action": None,
            "reasoning": None,
            "step": getattr(self.mind, 'step_num', None),
        }

        # Use undimmed console for key validation messages
        validation_console = self.undimmed_console if self.undimmed_console else self.mind.console
        try:
            hypothesis_preview = hypothesis_text[:120] + "..." if len(hypothesis_text) > 120 else hypothesis_text
            validation_console.print(f"\n[bold magenta]🔬 VALIDATING HYPOTHESIS[/]: [cyan]{insight_data.get('label','')[:64]}[/]")
            validation_console.print(f"[dim]   Hypothesis:[/] {hypothesis_preview}")
            validation_console.print(f"[dim]   Node:[/] {insight_node_id[:12]}...")
        except Exception:
            try:
                validation_console.log(f"[Validator] ⇒ validating '{insight_data.get('label','')[:64]}' …")
            except Exception:
                pass

        # Metric-causal attention: compute a local context (guarded)
        try:
            G = self.mind.memory.graph_db.graph
            neighbor_ids: List[str] = []
            try:
                neighbor_ids = list(G.neighbors(insight_node_id))[:32]
            except Exception:
                neighbor_ids = []
            origin_id = insight_node_id
            shell_dim: Optional[int] = None
            if hasattr(self.mind, 'dimensional_shells'):
                for d in DIMENSIONAL_SHELL_SIZES:
                    shell = self.mind.dimensional_shells.get(d)
                    if not shell or not getattr(shell, 'vectors', None):
                        continue
                    if origin_id in shell.vectors:
                        any_nb_here = any((nb in shell.vectors) for nb in neighbor_ids)
                        if any_nb_here:
                            shell_dim = d
                            break
            gated_neighbors = neighbor_ids
            if origin_id and shell_dim and hasattr(self.mind, 'proximity_engine') and neighbor_ids:
                gated_neighbors = self.mind.proximity_engine.filter_by_light_cone(origin_id, neighbor_ids, shell_dim)
            validation_obj["context_nodes"] = gated_neighbors[:10]
            try:
                metrics_log("validator.light_cone", {"event": "validator.cone", "origin": origin_id, "dim": shell_dim, "candidates": len(neighbor_ids or []), "kept": len(gated_neighbors or [])})
            except Exception:
                pass
        except Exception:
            pass

        try:
            classification = await self._classify_hypothesis(hypothesis_text)
            if not isinstance(classification, dict):
                raise ValueError("classification not dict")
        except Exception as e:
            classification = {"type": "unknown", "reasoning": f"Classifier failed: {e}"}
            try:
                self.mind.console.log(f"[Validator] Classification failure for {insight_node_id}: {e}")
            except Exception:
                pass

        node = self.mind.memory.graph_db.get_node(insight_node_id)
        if node is not None:
            node['validation_status'] = classification

        # Populate structured fields
        validation_obj["method"] = "classify_then_plan"
        validation_obj["confidence"] = 0.65 if classification.get('type') == 'computationally_testable' else 0.50
        validation_obj["reasoning"] = classification.get('reasoning')

        # Determine verdict
        verdict = "pass" if classification.get('type') == 'computationally_testable' else "unknown"
        validation_obj["verdict"] = verdict

        # Attempt test plan if computationally testable
        test_plan: Dict[str, Any] = {}
        if verdict == "pass":
            try:
                test_plan = await self._design_test_plan(hypothesis_text)
            except Exception as e:
                test_plan = {"required_data": None, "steps": [], "error": str(e)}
                try:
                    self.mind.console.log(f"[Validator] Test plan generation failed: {e}")
                except Exception:
                    pass
        if node is not None and test_plan:
            node['validation_plan'] = test_plan
        validation_obj["next_action"] = (test_plan.get('steps', [None])[0] if test_plan else None)
        validation_obj["test_plan"] = test_plan

        # Perform writeback adjustments & capture deltas (simplified: only rating change)
        before_rating = node.get('rating') if node else None
        try:
            bump = getattr(self.mind, '_bump_edge_weights', None)
            if callable(bump):
                bump(insight_node_id, verdict)
        except Exception as e:
            try:
                self.mind.console.log(f"[Validator] Edge weight bump failed: {e}")
            except Exception:
                pass
        after_rating = node.get('rating') if node else None
        if before_rating is not None and after_rating is not None and after_rating != before_rating:
            validation_obj['deltas']['rating'] = after_rating - before_rating

        latency_ms = int((time.time() - started) * 1000)
        validation_obj['latency_ms'] = latency_ms

        # Update uncertainty based on validation outcome (confidence learning)
        try:
            self._update_uncertainty_from_validation(insight_node_id, verdict, validation_obj['confidence'])
        except Exception as e:
            try:
                self.mind.console.log(f"[Validator] Uncertainty update failed: {e}")
            except Exception:
                pass

        # Emit concise console panel
        try:
            summary_line = (
                f"VERDICT={verdict} method={validation_obj['method']} conf={validation_obj['confidence']:.2f} "
                f"Δrating={validation_obj['deltas'].get('rating',0):+.3f} latency={latency_ms}ms"
            )
            verdict_colors = {"pass": "green", "fail": "red", "unknown": "yellow"}
            verdict_color = verdict_colors.get(verdict, "white")
            try:
                validation_console.print(f"\n[bold {verdict_color}]🎯 HYPOTHESIS VALIDATION RESULT[/]:")
                validation_console.print(f"   [bold]Verdict:[/] [{verdict_color}]{verdict.upper()}[/]")
                validation_console.print(f"   [dim]Confidence:[/] {validation_obj['confidence']:.2f}")
                validation_console.print(f"   [dim]Method:[/] {validation_obj['method']}")
                if validation_obj['deltas'].get('rating'):
                    rating_change = validation_obj['deltas']['rating']
                    rating_color = "green" if rating_change > 0 else "red" if rating_change < 0 else "white"
                    validation_console.print(f"   [dim]Rating Change:[/] [{rating_color}]{rating_change:+.3f}[/]")
                validation_console.print(f"   [dim]Processing Time:[/] {latency_ms}ms")
                if test_plan and verdict == 'pass':
                    try:
                        steps_preview = ', '.join(test_plan.get('steps', [])[:3])
                    except Exception:
                        steps_preview = ''
                    validation_console.print(f"   [dim]Next Steps:[/] {steps_preview}")
                validation_console.log(f"[Validator] {summary_line}")
            except Exception:
                pass
        except Exception:
            pass

        # Write structured JSON line to insights file
        try:
            insights_path = os.getenv("E8_INSIGHTS_PATH", "insights.ndjson")
            with open(insights_path, 'a', encoding='utf-8') as fobj:
                fobj.write(json.dumps({**validation_obj, "event": "validator.outcome"}, ensure_ascii=False) + "\n")
        except Exception:
            pass

        # Metrics counters
        try:
            outcome_key = f"validation_outcome_{verdict}"
            metrics_log("validator.outcome", {"event": "validator", "run_id": validation_obj['run_id'], "step": validation_obj['step'], "node_id": insight_node_id, "verdict": verdict, "latency_ms": latency_ms, outcome_key: 1})
        except Exception:
            pass

        # Legacy panels (minimal)
        if verdict == 'pass' and test_plan:
            try:
                plan_text = " | ".join(step for step in test_plan.get('steps', [])[:5])
                validation_console.log(f"[ValidatorPlan] data={test_plan.get('required_data')} steps={plan_text}")
            except Exception:
                pass

    async def _classify_hypothesis(self, hypothesis: str) -> Dict[str, Any]:
        """
        Return one of:
          {"type": "computationally_testable" | "physically_testable", "reasoning": str}
        Never returns 'unknown'.
        """

        def _balanced_object(s: str) -> dict:
            if not isinstance(s, str) or '{' not in s:
                return {}
            start = s.find('{')
            if start == -1:
                return {}
            depth = 0
            for i in range(start, len(s)):
                ch = s[i]
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        frag = s[start:i+1]
                        try:
                            obj = _parse_json_object(frag)
                            return obj if isinstance(obj, dict) else {}
                        except Exception:
                            return {}
            return {}

        def _extract_json(s: str) -> dict:
            if not isinstance(s, str) or not s:
                return {}
            try:
                whole = _parse_json_object(s)
                if isinstance(whole, dict) and whole:
                    return whole
            except Exception:
                pass
            d = _balanced_object(s)
            if d:
                return d
            try:
                m = re.search(r"\{[^{}]*\}", s)
                if m:
                    frag = m.group(0)
                    obj = _parse_json_object(frag)
                    if isinstance(obj, dict):
                        return obj
            except Exception:
                pass
            try:
                self.mind.console.log("[VALIDATOR] JSON extraction failed; using deterministic defaults.")
            except Exception:
                pass
            return {}

        def _normalize_type(t: str, raw: str) -> str:
            t = (t or "").strip().lower()
            if re.search(r"comput", t) or re.search(r"\b(simulate|model|algorithm|code|program|mining|embedding)\b", raw, re.I):
                return "computationally_testable"
            if re.search(r"physic", t) or re.search(r"\b(experiment|sensor|measure|lab|hardware|device)\b", raw, re.I):
                return "physically_testable"
            return "computationally_testable"

        base_prompt = (
            "You are a precise classifier.\n"
            "Goal: classify the hypothesis as either computationally_testable or physically_testable.\n"
            "Definitions:\n"
            "  - computationally_testable: can be tested fully in-silico (algorithms, simulations, embeddings, code).\n"
            "  - physically_testable: needs real-world measurement, sensors, lab hardware, or experiments.\n\n"
            "Hypothesis:\n"
            f"{hypothesis}\n\n"
            "Respond with ONLY this JSON object (no prose, no code fences):\n"
            '{{"type":"computationally_testable|physically_testable","reasoning":"<brief why>"}}'
        )

        raw = await self.llm_pool.enqueue_and_wait(base_prompt, max_tokens=96, temperature=0.0)
        if not isinstance(raw, str):
            raw = str(raw) if raw is not None else ""
        data = _extract_json(raw)

        if not isinstance(data, dict) or "type" not in data:
            strict_prompt = base_prompt + "\nSTRICT: Output only the JSON object, nothing else."
            raw2 = await self.llm_pool.enqueue_and_wait(strict_prompt, max_tokens=96, temperature=0.0)
            if not isinstance(raw2, str):
                raw2 = str(raw2) if raw2 is not None else ""
            data = _extract_json(raw2) or {}

        ctype = _normalize_type(str(data.get("type") or ""), raw if isinstance(raw, str) else "")
        reasoning = (data.get("reasoning") or "").strip()
        if not reasoning:
            if ctype == "computationally_testable":
                reasoning = "Mentions compute/simulation/algorithmic evaluation or lacks physical measurement cues."
            else:
                reasoning = "Requires real-world measurement, sensors, or lab conditions."

        # Pretty console panel
        try:
            lock = getattr(self.mind, 'console_lock', None)
            if lock is not None:
                async with lock:  # type: ignore[attr-defined]
                    if Panel is not None:
                        try:
                            self.console.print(
                                Panel(
                                    f"[bold]Classification:[/bold] {ctype}\n"
                                    f"[bold]Reasoning:[/bold] {escape(reasoning)}",
                                    title="[bold yellow]VALIDATOR: CLASSIFICATION[/]",
                                    border_style="red",
                                )
                            )
                        except Exception as e:
                            self.console.log(f"[VALIDATOR] Rich Panel formatting failed: {e}")
                            self.console.print(f"VALIDATOR: {ctype} - {reasoning}")
            else:
                try:
                    self.console.print(f"VALIDATOR: {ctype} - {reasoning}")
                except Exception:
                    pass
        except Exception as e:
            try:
                self.console.log(f"[VALIDATOR] Console lock failed: {e}")
                self.console.print(f"VALIDATOR: {ctype} - {reasoning}")
            except Exception:
                pass

        return {"type": ctype, "reasoning": reasoning}

    async def _design_test_plan(self, hypothesis: str) -> Dict[str, Any]:
        prompt = (
            "You are a principal investigator designing an experiment. For the following computationally testable "
            "hypothesis, create a validation plan.\n\n"
            f"Hypothesis: \"{hypothesis}\"\n\n"
            "Respond in JSON format with two keys: 'required_data' (a brief description of the datasets needed, e.g., 'Historical S&P 500 price data and news sentiment scores') "
            "and 'steps' (an array of strings outlining the high-level steps for the analysis, e.g., ['Clean and align datasets by date', 'Perform time-series cross-correlation analysis', 'Check for statistical significance'])."
        )
        if not hypothesis or not str(hypothesis).strip():
            return {"required_data": None, "steps": [], "reason": "missing_hypothesis"}
        try:
            response = await asyncio.wait_for(self.llm_pool.enqueue_and_wait(prompt, max_tokens=400), timeout=HYPOTHESIS_TIMEOUT)
            parsed = _parse_json_object(response)
            if not isinstance(parsed, dict):
                raise ValueError("non-dict response")
            if 'steps' not in parsed or not isinstance(parsed.get('steps'), (list, tuple)):
                parsed['steps'] = []
            return parsed
        except asyncio.TimeoutError:
            try:
                self.mind.console.log("[Validator] Test plan timeout")
            except Exception:
                pass
            return {"required_data": None, "steps": [], "reason": "timeout"}
        except Exception as e:
            try:
                self.mind.console.log(f"[Validator] Test plan design failed: {e}")
            except Exception:
                pass
            return {"required_data": None, "steps": [], "reason": f"error:{type(e).__name__}"}

    async def _validate_with_new_prompt(self, insight_text: str, prompt_vars: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        validation_console = self.undimmed_console if self.undimmed_console else self.mind.console
        try:
            hypothesis_preview = insight_text[:120] + "..." if len(insight_text) > 120 else insight_text
            node_id = prompt_vars.get("node_id", "unknown")
            label = self.mind.memory.graph_db.get_node(node_id).get('label', '') if node_id != "unknown" else ""
            validation_console.print(f"\n[bold magenta]🔬 VALIDATING HYPOTHESIS[/]: [cyan]{label[:64]}[/]")
            validation_console.print(f"[dim]   Hypothesis:[/] {hypothesis_preview}")
            validation_console.print(f"[dim]   Node:[/] {node_id[:12]}...")
        except Exception:
            try:
                validation_console.log(f"[Validator] ⇒ validating '{prompt_vars.get('node_id', 'unknown')}'…")
            except Exception:
                pass

        try:
            prompts = getattr(self.mind, 'prompts', {})
            if hasattr(prompts, '_d') and 'validator' in prompts._d:
                validator_prompts = prompts._d['validator']
            elif hasattr(prompts, 'get'):
                validator_prompts = prompts.get('validator', {})
            else:
                validator_prompts = {}
            if not validator_prompts:
                raise ValueError("Validator prompts not found")

            sys_prompt = validator_prompts.get('system', '')

            safe_prompt_vars: Dict[str, Any] = {}
            for key, value in (prompt_vars or {}).items():
                if isinstance(value, str):
                    safe_prompt_vars[key] = value.replace('{', '{{').replace('}', '}}')
                else:
                    safe_prompt_vars[key] = value

            user_template = validator_prompts.get('user', '')
            try:
                user_prompt = user_template.format(**safe_prompt_vars)
            except KeyError as fmt_err:
                missing_key = getattr(fmt_err, 'args', ['?'])[0]
                raise ValueError(f"Validator prompt missing variable: {missing_key}") from fmt_err
            except Exception as fmt_err:
                raise ValueError(f"Validator prompt formatting failed: {fmt_err}") from fmt_err

            # Add validator persona if available
            if hasattr(self.mind, 'semantics') and hasattr(self.mind.semantics, 'validator_persona'):
                persona = self.mind.semantics.validator_persona()
                if persona:
                    sys_prompt = persona + "\n\n" + sys_prompt

            full_prompt = sys_prompt + "\n\n" + user_prompt

            response = await asyncio.wait_for(
                self.llm_pool.enqueue_and_wait(full_prompt, max_tokens=200, temperature=0.0),
                timeout=HYPOTHESIS_TIMEOUT,
            )

            if not isinstance(response, str):
                response = str(response) if response is not None else ""
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()

            try:
                result = _parse_json_object(response)
                if not isinstance(result, dict):
                    raise ValueError("Response is not a valid JSON object")

                required_fields = ["validation_status", "validation_score", "validator_notes", "evidence_refs", "checked_at"]
                for field in required_fields:
                    if field not in result:
                        if field == "validation_status":
                            result[field] = "inconclusive"
                        elif field == "validation_score":
                            result[field] = 0.0
                        elif field == "validator_notes":
                            result[field] = "incomplete response"
                        elif field == "evidence_refs":
                            result[field] = []
                        elif field == "checked_at":
                            from datetime import datetime, timezone
                            result[field] = datetime.now(timezone.utc).isoformat()

                if result["validation_status"] not in ["pass", "fail", "inconclusive"]:
                    result["validation_status"] = "inconclusive"
                try:
                    result["validation_score"] = float(result["validation_score"])
                    if not (0.0 <= result["validation_score"] <= 1.0):
                        result["validation_score"] = 0.0
                except (ValueError, TypeError):
                    result["validation_score"] = 0.0

                result["validator_notes"] = str(result["validator_notes"])[:280]
                if not isinstance(result["evidence_refs"], list):
                    result["evidence_refs"] = []
                result["evidence_refs"] = result["evidence_refs"][:5]
                return result
            except Exception as parse_error:
                try:
                    self.mind.console.log(f"[Validator] JSON parse failed: {parse_error}")
                except Exception:
                    pass
                from datetime import datetime, timezone
                return {
                    "validation_status": "inconclusive",
                    "validation_score": 0.0,
                    "validator_notes": "invalid JSON from model",
                    "evidence_refs": [],
                    "checked_at": datetime.now(timezone.utc).isoformat(),
                }
        except Exception as e:
            try:
                self.mind.console.log(f"[Validator] New prompt validation failed: {e}")
            except Exception:
                pass
            return None

    async def _write_validation_to_node(self, node_id: str, validation_result: Dict[str, Any]):
        try:
            node = self.mind.memory.graph_db.get_node(node_id)
            if node is not None:
                node["validation_status"] = validation_result["validation_status"]
                node["validation_score"] = float(validation_result["validation_score"])
                node["validator_notes"] = validation_result["validator_notes"][:280]
                node["validator_refs"] = validation_result["evidence_refs"][:5]
                node["validated_at"] = validation_result["checked_at"]
                try:
                    self.mind.console.log(
                        f"[Validator] {node_id}: status={validation_result['validation_status']} "
                        f"score={validation_result['validation_score']:.3f} "
                        f"notes={validation_result['validator_notes'][:50]}{'...' if len(validation_result['validator_notes']) > 50 else ''}"
                    )
                except Exception:
                    pass
                try:
                    metrics_log("validator.new_format", {
                        "event": "validator.new_format",
                        "node_id": node_id,
                        "status": validation_result["validation_status"],
                        "score": validation_result["validation_score"],
                    })
                except Exception:
                    pass
        except Exception as e:
            try:
                self.mind.console.log(f"[Validator] Failed to write validation to node {node_id}: {e}")
            except Exception:
                pass

    def _update_uncertainty_from_validation(self, node_id: str, verdict: str, confidence: float):
        """Update uncertainty based on validation outcome (confidence learning)."""
        try:
            node = self.mind.memory.graph_db.get_node(node_id)
            if not node:
                return

            current_uncertainty = node.get('uncertainty', 0.5)
            learning_rate = 0.1
            if verdict == 'pass':
                new_uncertainty = current_uncertainty * (1.0 - learning_rate * confidence)
            elif verdict == 'fail':
                new_uncertainty = current_uncertainty + learning_rate * (1.0 - current_uncertainty) * confidence
            else:
                new_uncertainty = current_uncertainty + learning_rate * 0.1
            new_uncertainty = max(0.01, min(0.99, new_uncertainty))
            node['uncertainty'] = float(new_uncertainty)
            node['uncertainty_updated_at'] = time.time()
            if abs(new_uncertainty - current_uncertainty) > 0.05:
                try:
                    self.mind.console.log(
                        f"[Validator] Uncertainty updated for {node_id[:12]}: "
                        f"{current_uncertainty:.3f} → {new_uncertainty:.3f} (verdict: {verdict})"
                    )
                except Exception:
                    pass
        except Exception as e:
            try:
                self.mind.console.log(f"[Validator] Uncertainty update error: {e}")
            except Exception:
                pass
