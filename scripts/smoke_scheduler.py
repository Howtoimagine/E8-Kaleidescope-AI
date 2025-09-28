import sys, os, asyncio
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from e8_mind.cognitive.scheduler import CognitiveScheduler

class Console:
    def log(self, *args, **kwargs):
        print(*args)

class DummySlots:
    def __init__(self):
        self.teacher = type('S', (), {'start': lambda *a, **k: None, 'running': lambda *a, **k: False})()
        self.explorer = type('S', (), {'start': lambda *a, **k: None, 'running': lambda *a, **k: False})()
        self.dream = type('S', (), {'start': lambda *a, **k: None, 'running': lambda *a, **k: False})()
        self.insight = type('S', (), {'start': lambda *a, **k: None, 'running': lambda *a, **k: False})()
        self.subnarr = type('S', (), {'start': lambda *a, **k: None})()
        self.snapshot = type('S', (), {'start': lambda *a, **k: None})()
        self.decay = type('S', (), {'start': lambda *a, **k: None})()

class DummyLLMPool:
    class Q:
        def qsize(self):
            return 0
    def __init__(self):
        self.queue = self.Q()

class DummyMemory:
    class G:
        class Graph:
            def number_of_nodes(self):
                return 11
        def __init__(self):
            self.graph = self.Graph()
    def __init__(self):
        self.graph_db = self.G()
    async def snapshot(self):
        pass
    async def apply_decay(self):
        pass

class DummyMood:
    def __init__(self):
        self.mood_vector = {'entropy': 0.2}

class DummyMind:
    def __init__(self):
        self.console = Console()
        self.slots = DummySlots()
        self.llm_pool = DummyLLMPool()
        self.memory = DummyMemory()
        self.mood = DummyMood()
        async def _dream_seq():
            await asyncio.sleep(0)
        self.dream_engine = type('E', (), {'run_dream_sequence': _dream_seq})()
        # methods invoked by scheduler
        async def _run_insight_cycle():
            await asyncio.sleep(0)
        async def _run_proactive_insight_synthesis():
            await asyncio.sleep(0)
        async def _generate_subconscious_narrative():
            await asyncio.sleep(0)
        self._run_insight_cycle = _run_insight_cycle
        self._run_proactive_insight_synthesis = _run_proactive_insight_synthesis
        self._generate_subconscious_narrative = _generate_subconscious_narrative

async def main():
    mind = DummyMind()
    sched = CognitiveScheduler(mind)
    # run through 45 ticks to trigger various intervals
    for step in range(0, 46):
        sched.tick(step)
        await asyncio.sleep(0)  # allow any scheduled tasks to progress
    print("scheduler smoke ok")

if __name__ == '__main__':
    asyncio.run(main())
