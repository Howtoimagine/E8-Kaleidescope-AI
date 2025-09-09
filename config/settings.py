from __future__ import annotations

"""
Centralized configuration management for E8Mind.

- Loads defaults from core.config.AppConfig
- Allows environment variable overrides (E8_* already handled by AppConfig)
- Optionally merges a JSON config file pointed to by E8_CONFIG_FILE or explicit path
"""

import json
import os
from typing import Any, Dict, Optional

from core.config import AppConfig


def _load_json_file(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"[settings] Failed to load JSON config '{path}': {e}")
        return {}


def load_settings(config_file: Optional[str] = None) -> AppConfig:
    """
    Build an AppConfig using:
    1) AppConfig.from_env() defaults (already reading E8_* envs)
    2) Optional JSON file overrides (E8_CONFIG_FILE or provided path)
    """
    cfg = AppConfig.from_env()

    # Optional file-based overrides (JSON only to keep deps minimal)
    file_path = config_file or os.getenv("E8_CONFIG_FILE")
    if file_path:
        data = _load_json_file(file_path)
        if data:
            # Only apply known fields
            for field in vars(cfg).keys():
                if field in data:
                    try:
                        setattr(cfg, field, data[field])
                    except Exception:
                        pass

    return cfg


def dump_settings(cfg: Optional[AppConfig] = None) -> Dict[str, Any]:
    """Return a JSON-serializable dictionary of current settings."""
    c = cfg or AppConfig.from_env()
    return {
        "global_seed": c.global_seed,
        "embed_dim": c.embed_dim,
        "runtime_dir": c.runtime_dir,
        "llm_provider": c.llm_provider,
        "llm_model": c.llm_model,
        "llm_embed_model": c.llm_embed_model,
        "e8_quantizer": c.e8_quantizer,
        "memory_maintenance_interval": c.memory_maintenance_interval,
        "blackhole_threshold": c.blackhole_threshold,
        "web_host": c.web_host,
        "web_port": c.web_port,
        "default_profile": c.default_profile,
    }
