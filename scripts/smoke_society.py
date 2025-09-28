import sys, os, asyncio
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from e8_mind.cognitive.agents import SocietyOfMind

class Console:
    def log(self, *args, **kwargs):
        print(*args)

class DummySamplerMind:
    def __init__(self, action_dim=4):
        self.action_dim = action_dim
        self.agent = None
        self.metrics = type('M', (), {'increment': lambda *a, **k: None})()
        self.console = Console()
        self.drives = type('D', (), {
            'novelty_need': lambda self, s: 0.5,
            'synthesis_need': lambda self, s: 0.4,
            'stability_need': lambda self, s: 0.3,
            'exploit_need': lambda self, s: 0.2,
        })()

async def main():
    mind = DummySamplerMind(action_dim=4)
    som = SocietyOfMind(mind)
    state = [0.0, 0.0, 0.0, 0.0]
    action = await som.step(state, mind)
    print("society.action_len:", len(action) if hasattr(action, '__len__') else 'scalar')
    print("society smoke ok")

if __name__ == '__main__':
    asyncio.run(main())
