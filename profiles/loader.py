
import importlib.util, os, yaml
from .base_interfaces import SemanticsPlugin, PromptPack
from typing import Mapping, List, Tuple, Any

def _load_py(path: str):
    spec = importlib.util.spec_from_file_location("plugin_mod", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(mod)  # type: ignore
    return mod

class YamlPromptPack:
    def __init__(self, dct): self._d = dct or {}
    def render(self, key: str, **vars):
        tpl = self._d.get(key, "")
        try:
            return tpl.format(**vars)
        except Exception:
            # be robust if vars missing
            return tpl

def load_profile(profile_name: str):
    base = os.path.join(os.path.dirname(__file__), profile_name)
    sem_path = os.path.join(base, "semantics.py")
    prm_path = os.path.join(base, "prompts.yaml")
    if not os.path.exists(sem_path):
        raise FileNotFoundError(f"Semantics file not found: {sem_path}")
    mod = _load_py(sem_path)
    # New flexible loading: allow either PLUGIN (old style) or SEMANTIC_CATEGORIES / get_semantic_categories (new lightweight style)
    if hasattr(mod, 'PLUGIN'):
        sem = mod.PLUGIN
    else:
        # build a minimal adapter implementing SemanticsPlugin
        cats = None
        if hasattr(mod, 'get_semantic_categories'):
            try:
                cats = mod.get_semantic_categories()
            except Exception:
                cats = None
        if cats is None and hasattr(mod, 'SEMANTIC_CATEGORIES'):
            cats = getattr(mod, 'SEMANTIC_CATEGORIES')
        if cats is None:
            raise AttributeError("Loaded semantics module missing PLUGIN or SEMANTIC_CATEGORIES definition")

        class _MinimalSemantics:
            def __init__(self, name: str, categories):
                self.name = name
                self.base_domain = 'general'
                self._categories = categories or {}
            # Basic passthrough behaviors
            def persona_prefix(self, mood_vector: Mapping[str, float]) -> str:
                return ''
            def pre_text(self, text: str) -> str: return text
            def post_text(self, text: str) -> str: return text
            def pre_embed(self, text: str) -> str: return text
            def post_embed(self, vec) -> Any: return vec
            def rerank(self, candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]: return candidates

        sem = _MinimalSemantics(profile_name, cats)
    pack = {}
    if os.path.exists(prm_path):
        with open(prm_path, "r", encoding="utf-8") as f:
            import yaml as _yaml
            pack = _yaml.safe_load(f) or {}
    return sem, YamlPromptPack(pack)
