
import importlib.util, os, yaml
from .base_interfaces import SemanticsPlugin, PromptPack

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
    sem = _load_py(sem_path).PLUGIN  # each plugin file defines PLUGIN
    pack = {}
    if os.path.exists(prm_path):
        with open(prm_path, "r", encoding="utf-8") as f:
            import yaml as _yaml
            pack = _yaml.safe_load(f) or {}
    return sem, YamlPromptPack(pack)
