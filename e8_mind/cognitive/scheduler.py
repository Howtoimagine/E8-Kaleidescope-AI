from __future__ import annotations

import os
import asyncio


def _scale_steps(n: int) -> int:
    try:
        scale = float(os.getenv('E8_CADENCE_SCALE', '1.0'))
    except Exception:
        scale = 1.0
    try:
        return max(1, int(round(n * scale)))
    except Exception:
        return n


class CognitiveScheduler:
    def __init__(self, mind_instance):
        self.mind = mind_instance
        # Base intervals (scaled)
        self.PROXIMITY_ALERT_INTERVAL = _scale_steps(int(os.getenv("E8_PROXIMITY_INTERVAL", "7")))
        self.INSIGHT_SYNTHESIS_INTERVAL = _scale_steps(23)
        self.DREAM_INTERVAL = _scale_steps(5)
        self.NARRATIVE_SUMMARY_INTERVAL = _scale_steps(37)
        self.SNAPSHOT_INTERVAL = _scale_steps(10000)
        self.DECAY_INTERVAL = _scale_steps(24)
        self.DREAM_REPLAY_INTERVAL = _scale_steps(int(os.getenv('E8_DREAM_EVERY_STEPS','200')))
        self.PROXIMITY_MIN_GAP = _scale_steps(int(os.getenv('E8_PROXIMITY_MIN_GAP', '4')))
        self.ENABLE_EXTRA_PROXIMITY = os.getenv('E8_PROXIMITY_EXTRA', '1') != '0'
        self._dream_skip_streak = 0

    def _fire(self, step: int, interval: int, offset: int) -> bool:
        return interval > 0 and step >= offset and ((step - offset) % interval == 0)

    def tick(self, step: int):
        # LLM queue and entropy gating
        try:
            qdepth = self.mind.llm_pool.queue.qsize() if getattr(self.mind, 'llm_pool', None) else 0
        except Exception:
            qdepth = 0
        entropy = float(getattr(getattr(self.mind, 'mood', None), 'mood_vector', {}).get('entropy', 0.0))
        max_q = int(os.getenv('E8_LLM_MAX_QUEUE', '64'))
        high_entropy = float(os.getenv('E8_HIGH_ENTROPY', '0.90'))
        skip_dream = qdepth > max_q or entropy > high_entropy
        dream_forced = False

        # Proximity insights
        if self._fire(step, self.PROXIMITY_ALERT_INTERVAL, 5):
            try:
                if not getattr(self.mind, "_insight_cycle_pending", False):
                    self.mind._insight_cycle_pending = True
                    async def _runner():
                        try:
                            await self.mind._run_insight_cycle()
                        finally:
                            self.mind._insight_cycle_pending = False
                            self.mind._last_insight_cycle_step = step
                    asyncio.create_task(_runner())
            except Exception:
                pass
        elif self.ENABLE_EXTRA_PROXIMITY and step % 3 == 0:
            try:
                g = self.mind.memory.graph_db.graph
                if g.number_of_nodes() > 10:
                    if not getattr(self.mind, "_insight_cycle_pending", False):
                        self.mind._insight_cycle_pending = True
                        asyncio.create_task(self.mind._run_insight_cycle())
            except Exception:
                pass

        # Insight synthesis
        if self._fire(step, self.INSIGHT_SYNTHESIS_INTERVAL, 13):
            try:
                asyncio.create_task(self.mind._run_proactive_insight_synthesis())
            except Exception:
                pass

        # Dreams (with skip streak and force-on-max)
        if self._fire(step, self.DREAM_INTERVAL, 0):
            if skip_dream:
                self._dream_skip_streak += 1
                if self._dream_skip_streak >= int(os.getenv('E8_DREAM_MAX_SKIPS', '10')):
                    skip_dream = False
                    dream_forced = True
            else:
                self._dream_skip_streak = 0
            if not skip_dream:
                try:
                    self.mind.slots.dream.start(self.mind.dream_engine.run_dream_sequence())
                except Exception:
                    pass

        # Subconscious narrative
        if self._fire(step, self.NARRATIVE_SUMMARY_INTERVAL, 2):
            try:
                self.mind.slots.subnarr.start(self.mind._generate_subconscious_narrative())
            except Exception:
                pass

        # Memory snapshot
        if self._fire(step, self.SNAPSHOT_INTERVAL, 0):
            try:
                self.mind.slots.snapshot.start(self.mind.memory.snapshot())
            except Exception:
                pass

        # Decay
        if self._fire(step, self.DECAY_INTERVAL, 21):
            try:
                self.mind.slots.decay.start(self.mind.memory.apply_decay())
            except Exception:
                pass
