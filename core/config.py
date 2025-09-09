"""
Core configuration for E8Mind system.
Centralizes paths, seeds, and environment variables.
"""

import os
from dataclasses import dataclass
from typing import Optional

# Environment variable defaults
GLOBAL_SEED = int(os.getenv("E8_SEED", "42"))
EMBED_DIM = int(os.getenv("E8_EMBED_DIM", "1536"))
RUNTIME_DIR = os.getenv("E8_RUNTIME_DIR", "runtime")

@dataclass
class AppConfig:
    """Application configuration container."""
    global_seed: int = GLOBAL_SEED
    embed_dim: int = EMBED_DIM
    runtime_dir: str = RUNTIME_DIR
    
    # LLM settings
    llm_provider: str = os.getenv("E8_LLM_PROVIDER", "openai")
    llm_model: str = os.getenv("E8_LLM_MODEL", "gpt-4")
    llm_embed_model: str = os.getenv("E8_LLM_EMBED_MODEL", "text-embedding-3-large")
    
    # Physics settings
    e8_quantizer: str = os.getenv("E8_QUANTIZER", "e8")
    
    # Memory settings
    memory_maintenance_interval: int = int(os.getenv("E8_MEMORY_MAINTENANCE", "500"))
    blackhole_threshold: float = float(os.getenv("E8_BLACKHOLE_THRESHOLD", "0.95"))
    
    # Web server settings
    web_host: str = os.getenv("E8_WEB_HOST", "localhost")
    web_port: int = int(os.getenv("E8_WEB_PORT", "8080"))
    
    # Profile settings
    default_profile: str = os.getenv("E8_PROFILE", "science")
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create configuration from environment variables."""
        return cls()
