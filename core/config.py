"""
Core configuration for E8Mind system.
Centralizes paths, seeds, and environment variables.
"""

import os
from dataclasses import dataclass
from typing import Optional

def _env_true(name: str, default: bool = False) -> bool:
    val = os.getenv(name, str(int(default))).strip().lower()
    return val in ("1", "true", "yes", "on")

# Environment variable defaults
GLOBAL_SEED = int(os.getenv("E8_SEED", "42"))
EMBED_DIM = int(os.getenv("E8_EMBED_DIM", "1536"))
RUNTIME_DIR = os.getenv("E8_RUNTIME_DIR", "runtime")

# Field Mantle logging flags (physics subsystem)
E8_LOG_FIELDMANTLE_THERMOSTAT = _env_true("E8_LOG_FIELDMANTLE_THERMOSTAT", default=False)
E8_LOG_FIELDMANTLE_METRIC = _env_true("E8_LOG_FIELDMANTLE_METRIC", default=False)
E8_LOG_FIELDMANTLE_INV = _env_true("E8_LOG_FIELDMANTLE_INV", default=False)

# VAE configuration knobs
E8_VAE_ENABLE = _env_true("E8_VAE_ENABLE", default=True)
E8_VAE_LR = float(os.getenv("E8_VAE_LR", "1e-3"))
E8_VAE_BETA = float(os.getenv("E8_VAE_BETA", "1.0"))
E8_VAE_LAYERS = os.getenv("E8_VAE_LAYERS", f"{EMBED_DIM},48,32,16,8")
E8_VAE_LATENT = int(os.getenv("E8_VAE_LATENT", "8"))
E8_VAE_BUFFER_SIZE = int(os.getenv("E8_VAE_BUFFER_SIZE", "4096"))
E8_VAE_BATCH = int(os.getenv("E8_VAE_BATCH", "64"))
E8_VAE_MIN_BUFFER = int(os.getenv("E8_VAE_MIN_BUFFER", "256"))
E8_VAE_TRAIN_EVERY = int(os.getenv("E8_VAE_TRAIN_EVERY", "1"))
E8_VAE_TRAIN_PARTIAL = _env_true("E8_VAE_TRAIN_PARTIAL", default=False)
E8_VAE_KL_WARMUP_STEPS = int(os.getenv("E8_VAE_KL_WARMUP_STEPS", "1000"))
E8_VAE_KL_TARGET_BETA = float(os.getenv("E8_VAE_KL_TARGET_BETA", "0.1"))
E8_VAE_FREE_BITS = float(os.getenv("E8_VAE_FREE_BITS", "0.25"))
E8_VAE_GRAD_CLIP = float(os.getenv("E8_VAE_GRAD_CLIP", "1.0"))
E8_VAE_ENHANCED_LOGGING = _env_true("E8_VAE_ENHANCED_LOGGING", default=True)
E8_VAE_USE_FOR_PROJECTION = _env_true("E8_VAE_USE_FOR_PROJECTION", default=True)
E8_VAE_USE_FOR_QUERY = _env_true("E8_VAE_USE_FOR_QUERY", default=True)

# LLM pool and embedding timings
LOCAL_GEN_WORKERS = int(os.getenv("LOCAL_GEN_WORKERS", "1"))
POOL_WORKER_TIMEOUT = int(os.getenv("POOL_WORKER_TIMEOUT", "120"))
POOL_RESULT_TIMEOUT = int(os.getenv("POOL_RESULT_TIMEOUT", "180"))
LLM_CALL_TIMEOUT_SEC = int(os.getenv("LLM_CALL_TIMEOUT_SEC", "90"))
EMBEDDING_TIMEOUT_SEC = int(os.getenv("EMBEDDING_TIMEOUT_SEC", "60"))
E8_MAX_INFLIGHT = int(os.getenv("E8_MAX_INFLIGHT", "32"))
E8_LLM_TIMEOUT = float(os.getenv("E8_LLM_TIMEOUT", "45"))
E8_LLM_TIMEOUT_ALIGN = _env_true("E8_LLM_TIMEOUT_ALIGN", default=True)
E8_LLM_MAX_RETRIES = int(os.getenv("E8_LLM_MAX_RETRIES", "1"))
E8_LLM_RETRY_BACKOFF = float(os.getenv("E8_LLM_RETRY_BACKOFF", "0.4"))

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
    
    # Validator settings
    VALIDATOR_WRITEBACK_ENABLED: bool = _env_true("E8_VALIDATOR_WRITEBACK_ENABLED", default=True)
    if "E8_DISABLE_VALIDATOR_WRITEBACK" in os.environ:
        VALIDATOR_WRITEBACK_ENABLED = not _env_true("E8_DISABLE_VALIDATOR_WRITEBACK", default=False)
    
    # Additional validator defaults
    auto_validate_insights: bool = _env_true("E8_AUTO_VALIDATE_INSIGHTS", default=True)
    validator_min_rating: float = float(os.getenv("E8_VALIDATOR_MIN_RATING", "0.6"))
    
    # Profile settings
    default_profile: str = os.getenv("E8_PROFILE", "science")
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create configuration from environment variables."""
        return cls()
    
    def log_validator_config(self, console=None):
        """Log validator configuration for startup visibility."""
        if console:
            console.log(f"[VALIDATOR CONFIG] Writeback: {self.VALIDATOR_WRITEBACK_ENABLED}")
            console.log(f"[VALIDATOR CONFIG] Auto-validate: {self.auto_validate_insights}")  
            console.log(f"[VALIDATOR CONFIG] Min rating: {self.validator_min_rating}")
            console.log(f"[VALIDATOR CONFIG] Profile: {self.default_profile}")
        else:
            print(f"[VALIDATOR CONFIG] Writeback: {self.VALIDATOR_WRITEBACK_ENABLED}")
            print(f"[VALIDATOR CONFIG] Auto-validate: {self.auto_validate_insights}")
            print(f"[VALIDATOR CONFIG] Min rating: {self.validator_min_rating}")
            print(f"[VALIDATOR CONFIG] Profile: {self.default_profile}")
