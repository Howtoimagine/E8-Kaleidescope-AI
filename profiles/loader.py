

import importlib.util, os
from .base_interfaces import SemanticsPlugin, PromptPack
from typing import Mapping, List, Tuple, Any, Optional, Dict
import logging

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    yaml = None
    YAML_AVAILABLE = False

# Set up logger
logger = logging.getLogger(__name__)

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

# Profile cache for performance
_profile_cache: Dict[str, Tuple[Any, Any]] = {}

def clear_profile_cache():
    """Clear the profile cache. Useful for development."""
    global _profile_cache
    _profile_cache.clear()
    logger.info("Profile cache cleared")

def list_available_profiles() -> List[str]:
    """List all available profile names."""
    profiles_dir = os.path.dirname(__file__)
    profiles = []
    for item in os.listdir(profiles_dir):
        item_path = os.path.join(profiles_dir, item)
        if os.path.isdir(item_path) and not item.startswith('__'):
            # Check if it has the required semantics.py file
            sem_path = os.path.join(item_path, "semantics.py")
            if os.path.exists(sem_path):
                profiles.append(item)
    return sorted(profiles)

def validate_profile_structure(profile_name: str) -> Tuple[bool, str]:
    """Validate that a profile has the required structure."""
    base = os.path.join(os.path.dirname(__file__), profile_name)
    sem_path = os.path.join(base, "semantics.py")

    if not os.path.exists(base):
        return False, f"Profile directory does not exist: {base}"

    if not os.path.exists(sem_path):
        return False, f"Semantics file not found: {sem_path}"

    try:
        mod = _load_py(sem_path)
        # Check for either old PLUGIN or new SEMANTIC_CATEGORIES
        has_plugin = hasattr(mod, 'PLUGIN')
        has_categories = hasattr(mod, 'SEMANTIC_CATEGORIES') or hasattr(mod, 'get_semantic_categories')

        if not (has_plugin or has_categories):
            return False, "Profile must have either PLUGIN or SEMANTIC_CATEGORIES/get_semantic_categories"

        return True, "Profile structure is valid"
    except Exception as e:
        return False, f"Error loading profile: {e}"

def load_profile(profile_name: str, use_cache: bool = True, inherit_from: Optional[str] = None):
    """Load a profile with optional caching and inheritance."""
    cache_key = f"{profile_name}:{inherit_from}"

    if use_cache and cache_key in _profile_cache:
        logger.debug(f"Loading profile '{profile_name}' from cache")
        return _profile_cache[cache_key]

    try:
        # Validate profile first
        is_valid, error_msg = validate_profile_structure(profile_name)
        if not is_valid:
            raise ValueError(f"Invalid profile '{profile_name}': {error_msg}")

        base = os.path.join(os.path.dirname(__file__), profile_name)
        sem_path = os.path.join(base, "semantics.py")
        prm_path = os.path.join(base, "prompts.yaml")

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

        # Load prompts
        pack_dict = {}
        if os.path.exists(prm_path):
            if YAML_AVAILABLE and yaml is not None:
                try:
                    with open(prm_path, "r", encoding="utf-8") as f:
                        pack_dict = yaml.safe_load(f) or {}
                except Exception as e:
                    logger.warning(f"Failed to load prompts from {prm_path}: {e}")
                    pack_dict = {}
            else:
                logger.warning(f"PyYAML not available, skipping prompts from {prm_path}")
                pack_dict = {}

        # Handle inheritance if specified
        if inherit_from:
            logger.debug(f"Loading parent profile '{inherit_from}' for inheritance")
            parent_sem, parent_pack = load_profile(inherit_from, use_cache=use_cache)

            # Merge prompts (child overrides parent)
            merged_pack = dict(parent_pack._d) if hasattr(parent_pack, '_d') else {}
            merged_pack.update(pack_dict)

            pack = YamlPromptPack(merged_pack)

            # For semantics, we keep the child semantics but could add merging logic here if needed
            logger.debug(f"Profile '{profile_name}' inherited from '{inherit_from}'")

        else:
            pack = YamlPromptPack(pack_dict)

        result = (sem, pack)

        if use_cache:
            _profile_cache[cache_key] = result

        logger.info(f"Successfully loaded profile '{profile_name}'")
        return result

    except Exception as e:
        logger.error(f"Failed to load profile '{profile_name}': {e}")
        raise
