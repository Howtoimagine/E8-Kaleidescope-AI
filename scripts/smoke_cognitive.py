import asyncio
import os
import sys

# Make repo root importable
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from e8_mind.cognitive.insight import HypothesisValidator


class FakeLLMPool:
    async def enqueue_and_wait(self, prompt: str, max_tokens: int = 96, temperature: float = 0.0):
        # Always return a valid JSON classification
        return '{"type":"computationally_testable","reasoning":"synthetic test response"}'


class FakeConsole:
    def print(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass


class FakeMind:
    def __init__(self):
        self.llm_pool = FakeLLMPool()
        self.console = FakeConsole()
        self.prompts = {}
        self.memory = type("Mem", (), {"graph_db": type("DB", (), {"get_node": lambda *_: None, "graph": {}})()})()


async def main():
    mind = FakeMind()
    hv = HypothesisValidator(mind)
    result = await hv._classify_hypothesis("We can validate this hypothesis purely via simulation.")
    print("classification:", result)


if __name__ == "__main__":
    asyncio.run(main())
