import sys, os, asyncio
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from e8_mind.cognitive.affect import MoodEngine, SubconsciousLayer, GoalField

class Console:
    def log(self, *args, **kwargs):
        print(*args)

async def fake_embed(text: str):
    # deterministic toy embedding
    import zlib
    seed = zlib.adler32(text.encode())
    import random
    r = random.Random(seed)
    return [r.random() for _ in range(16)]

class FakeLLM:
    async def enqueue_and_wait(self, prompt: str, max_tokens=256, temperature=0.7):
        return "From dreams and questions, a thread was woven into a brief awakening."

async def main():
    console = Console()

    # MoodEngine test
    mood = MoodEngine(console)
    mood.process_event("movement", magnitude=2.0, themes=["integration"])  # increase coherence
    mood.process_event("new_concept", rating=0.8)
    mood.process_event("weather_tick", step=5, bh=0.0)
    mood.update()
    print("mood.describe:", mood.describe())
    print("mood.weather:", mood.get_symbolic_weather())
    prefix = mood.get_llm_persona_prefix()
    print("mood.persona_prefix:", prefix[:60], "...")
    mod = mood.get_mood_modulation_vector(8)
    print("mood.mod_vector_len:", len(mod) if hasattr(mod, "__len__") else "?" )

    # GoalField test
    gf = GoalField(fake_embed, console)
    await gf.initialize_goals()
    gf.update_from_mood(mood.mood_vector)
    top_goals = gf.get_top_goals()
    print("goal.top:", top_goals)

    # SubconsciousLayer test
    sub = SubconsciousLayer(fake_embed, FakeLLM(), console)
    await sub.track_concept("unified field", weight=1.0)
    sub.decay(current_step=1)
    await sub.generate_narrative_summary([
        {"type": "dream", "label": "river of light"},
        {"type": "insight_synthesis", "label": "coherent resonance"},
    ])
    b = sub.get_bias()
    print("sub.bias_type:", type(b).__name__)
    print("sub.narrative_len:", len(sub.narrative))

    print("affect smoke ok")

if __name__ == "__main__":
    asyncio.run(main())
