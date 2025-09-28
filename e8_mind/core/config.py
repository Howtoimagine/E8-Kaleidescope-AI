import os, math

# Global Flags
# Enable self-projection and ingestion by default. These can still be overridden
# by environment variables or debug tools if needed.
E8_SELF_PROJECT = os.getenv("E8_SELF_PROJECT", "1") == "0"
# Enable ingestion by default so self-projection and configured data sources run
# unless an environment overrides it. Use E8_INGEST=0 to disable in testing.
E8_INGEST = os.getenv("E8_INGEST", "1") == "1"

# Optional Wavey Integration flags are handled dynamically in monolith.

# Configuration Constants
BASE_DIR = os.path.dirname(os.path.abspath(globals().get('__file__', 'e8_mind_server_M20.py')))
RUNTIME_DIR = os.path.join(BASE_DIR, "runtime")
POOL_WORKER_TIMEOUT = int(os.getenv("POOL_WORKER_TIMEOUT", "120"))  # Increased from 40s for slower models like phi3
POOL_RESULT_TIMEOUT = int(os.getenv("POOL_RESULT_TIMEOUT", "180"))  # Increased from 120s
LLM_CALL_TIMEOUT_SEC = int(os.getenv("LLM_CALL_TIMEOUT_SEC", "90"))  # Increased from 60s for phi3 model
EMBEDDING_TIMEOUT_SEC = int(os.getenv("EMBEDDING_TIMEOUT_SEC", "60"))  # Increased from 30s
DREAM_MIN_INTERVAL_SEC = 30
CONSOLE_EXPORT_EVERY_STEPS = 5000
CONSOLE_EXPORT_FORMAT = "both"
EMBED_DIM = int(os.getenv("E8_EMBED_DIM", "1536"))
DIMENSIONAL_SHELL_SIZES = [8, 16, 32, 64]
AUTOENCODER_LAYER_SIZES = [EMBED_DIM, max(256, EMBED_DIM//3), 128, 64] + DIMENSIONAL_SHELL_SIZES
ACTION_LAYOUT = [
    {"dim": 3, "biv_start": 0, "biv_len": 3, "angle_idx": 3},
    {"dim": 5, "biv_start": 4, "biv_len": 10, "angle_idx": 14},
    {"dim": 8, "biv_start": 15, "biv_len": 28, "angle_idx": 43},
]
ACTION_SIZE_NO_LOCK = sum(d["biv_len"] + 1 for d in ACTION_LAYOUT)
TEACHER_ASK_EVERY = 20  # more frequent teacher prompts (was 25)
TEACHER_OFFSET = 3      # ask slightly earlier after start (was 5)
EXPLORER_OFFSET = 10    # explorer responds sooner after teacher (was 15)
BLACK_HOLE_COOLDOWN_STEPS = 50
# ---- Cadence Profile & Scaling -----------------------------------------------
E8_CADENCE_PROFILE = os.getenv("E8_CADENCE_PROFILE", "custom").strip().lower()
try:
    E8_CADENCE_SCALE = float(os.getenv("E8_CADENCE_SCALE", "1.0"))
except Exception:
    E8_CADENCE_SCALE = 1.0

# Allow overriding teacher cadence via env (pre-scale)
TEACHER_ASK_EVERY = int(os.getenv("E8_TEACHER_ASK_EVERY", str(TEACHER_ASK_EVERY)))
TEACHER_OFFSET = int(os.getenv("E8_TEACHER_OFFSET", str(TEACHER_OFFSET)))
EXPLORER_OFFSET = int(os.getenv("E8_EXPLORER_OFFSET", str(EXPLORER_OFFSET)))

def _scale_steps(v: int) -> int:
    try:
        return max(1, int(round(v * E8_CADENCE_SCALE)))
    except Exception:
        return v

# Apply M20 presets if chosen (pre-scale)
if E8_CADENCE_PROFILE == "m20":
    try:
        BH_PRESSURE_THRESHOLD = float(os.getenv("E8_BH_PRESSURE_THRESHOLD", "0.25"))
    except Exception:
        pass
    try:
        BLACK_HOLE_COOLDOWN_STEPS = int(os.getenv("E8_BLACK_HOLE_COOLDOWN_STEPS", "50"))
    except Exception:
        pass
    TEACHER_ASK_EVERY = int(os.getenv("E8_TEACHER_ASK_EVERY", "25"))

# ---- Console Logging Flags ---------------------------------------------------
E8_LOG_FIELDMANTLE_THERMOSTAT = os.getenv("E8_LOG_FIELDMANTLE_THERMOSTAT", "0") == "1"
E8_LOG_FIELDMANTLE_METRIC = os.getenv("E8_LOG_FIELDMANTLE_METRIC", "0") == "1"
E8_LOG_FIELDMANTLE_INV = os.getenv("E8_LOG_FIELDMANTLE_INV", "0") == "1"
E8_LOG_SCHEDULER = os.getenv("E8_LOG_SCHEDULER", "0") == "1"
BLACK_HOLE_COOLDOWN_STEPS = _scale_steps(BLACK_HOLE_COOLDOWN_STEPS)
TEACHER_ASK_EVERY = _scale_steps(TEACHER_ASK_EVERY)
TEACHER_OFFSET = _scale_steps(TEACHER_OFFSET)
EXPLORER_OFFSET = _scale_steps(EXPLORER_OFFSET)

TEACHER_STEP_TIMEOUT = float(os.getenv("E8_TEACHER_STEP_TIMEOUT", "45.0"))
EXPLORER_STEP_TIMEOUT = float(os.getenv("E8_EXPLORER_STEP_TIMEOUT", "45.0"))
DREAM_SEQUENCE_TIMEOUT = float(os.getenv("E8_DREAM_SEQUENCE_TIMEOUT", "240.0"))
DIARY_ENTRY_TIMEOUT = float(os.getenv("E8_DIARY_ENTRY_TIMEOUT", "180.0"))
HYPOTHESIS_TIMEOUT = float(os.getenv("E8_HYPOTHESIS_TIMEOUT", "90.0"))
HYPOTHESIS_DASHBOARD_INTERVAL = int(os.getenv("E8_HYPOTHESIS_DASHBOARD_INTERVAL", "50"))
BH_PRESSURE_THRESHOLD = float(os.getenv("E8_BH_PRESSURE_THRESHOLD", "0.25"))
BH_SPREAD_FRAC = 0.5
BH_DIFFUSION_ETA = 0.15
BLACK_HOLE_K = 16
CONSOLIDATE_MIN = 20

BH_SIGNAL_WINDOW      = int(os.getenv("E8_BH_WINDOW",            "64"))
BH_TARGET_INTERVAL    = int(os.getenv("E8_BH_TARGET_INTERVAL",    "120"))
BH_COOLDOWN_MIN       = int(os.getenv("E8_BH_COOLDOWN_MIN",       "10"))
BH_COOLDOWN_MAX       = int(os.getenv("E8_BH_COOLDOWN_MAX",       "600"))
BH_COOLDOWN_INIT      = int(os.getenv("E8_BH_COOLDOWN_INIT",      str(BLACK_HOLE_COOLDOWN_STEPS)))
BH_W_DENSITY          = float(os.getenv("E8_BH_W_DENSITY",        "0.55"))
BH_W_UNKNOWN          = float(os.getenv("E8_BH_W_UNKNOWN",        "0.25"))
BH_W_TIMEOUT          = float(os.getenv("E8_BH_W_TIMEOUT",        "0.20"))
BH_JITTER_FRAC        = float(os.getenv("E8_BH_JITTER_FRAC",      "0.10"))
BH_THRESH_MIN         = float(os.getenv("E8_BH_THRESH_MIN",       "0.60"))
BH_THRESH_MAX         = float(os.getenv("E8_BH_THRESH_MAX",       "0.95"))
BH_THRESH_STEP_UP     = float(os.getenv("E8_BH_THRESH_STEP_UP",   "0.01"))
BH_THRESH_STEP_DOWN   = float(os.getenv("E8_BH_THRESH_STEP_DOWN", "0.03"))
POTENTIAL_SUCCESS_THRESH = float(os.getenv("E8_POTENTIAL_SUCCESS_THRESH", "0.6"))

E8_Q_SCALE = float(os.getenv("E8_Q_SCALE", "0.25"))
E8_Q_TAU = float(os.getenv("E8_Q_TAU", "0.40"))
E8_Q_ALPHA = float(os.getenv("E8_Q_ALPHA", "0.65"))
E8_Q_INFINITY = float(os.getenv("E8_Q_INFINITY", "1.0"))
E8_Q_HEADROOM_MIN = float(os.getenv("E8_Q_HEADROOM_MIN", "0.75"))
E8_Q_HEADROOM_MAX = float(os.getenv("E8_Q_HEADROOM_MAX", "0.99"))
E8_Q_HEADROOM_TARGET = float(os.getenv("E8_Q_HEADROOM_TARGET", "0.87"))

HOLOGRAPHIC_COMPRESSION_ENABLED = bool(int(os.getenv("E8_HOLOGRAPHIC_COMPRESSION", "1")))
INFORMATION_CONSERVATION_CHECK = bool(int(os.getenv("E8_INFO_CONSERVATION_CHECK", "1")))
SPACETIME_CURVATURE_ENABLED = bool(int(os.getenv("E8_SPACETIME_CURVATURE", "1")))
E8_LATTICE_QUANTIZATION = bool(int(os.getenv("E8_LATTICE_QUANTIZATION", "1")))
HOLOGRAPHIC_FIDELITY_THRESHOLD = float(os.getenv("E8_HOLOGRAPHIC_FIDELITY_THRESHOLD", "0.8"))
POTENTIAL_REWARD        = float(os.getenv("E8_POTENTIAL_REWARD", "0.12"))
POTENTIAL_DECAY         = float(os.getenv("E8_POTENTIAL_DECAY", "0.004"))

E8_UNIFIED_RATING_ENABLED = bool(int(os.getenv("E8_UNIFIED_RATING_ENABLED", "1")))
E8_RATING_VALIDATION_ENABLED = bool(int(os.getenv("E8_RATING_VALIDATION_ENABLED", "1")))
E8_RATING_CALIBRATION_SAMPLES = int(os.getenv("E8_RATING_CALIBRATION_SAMPLES", "20"))
E8_MAX_SPIN_ANGLE = float(os.getenv('E8_MAX_SPIN_ANGLE', str(math.pi)))

E8_VAE_ENABLE = bool(int(os.getenv("E8_VAE_ENABLE", "1")))
E8_VAE_LR = float(os.getenv("E8_VAE_LR", "1e-3"))
E8_VAE_BETA = float(os.getenv("E8_VAE_BETA", "1.0"))
E8_VAE_LAYERS = os.getenv("E8_VAE_LAYERS", f"{EMBED_DIM},48,32,16,8")
E8_VAE_LATENT = int(os.getenv("E8_VAE_LATENT", "8"))
E8_VAE_BUFFER_SIZE = int(os.getenv("E8_VAE_BUFFER_SIZE", "4096"))
E8_VAE_BATCH = int(os.getenv("E8_VAE_BATCH", "64"))
E8_VAE_MIN_BUFFER = int(os.getenv("E8_VAE_MIN_BUFFER", "256"))
E8_VAE_TRAIN_EVERY = int(os.getenv("E8_VAE_TRAIN_EVERY", "1"))
E8_VAE_TRAIN_PARTIAL = bool(int(os.getenv("E8_VAE_TRAIN_PARTIAL", "0")))
E8_VAE_KL_WARMUP_STEPS = int(os.getenv("E8_VAE_KL_WARMUP_STEPS", "1000"))
E8_VAE_KL_TARGET_BETA = float(os.getenv("E8_VAE_KL_TARGET_BETA", "0.1"))
E8_VAE_FREE_BITS = float(os.getenv("E8_VAE_FREE_BITS", "0.25"))
E8_VAE_GRAD_CLIP = float(os.getenv("E8_VAE_GRAD_CLIP", "1.0"))
E8_VAE_ENHANCED_LOGGING = bool(int(os.getenv("E8_VAE_ENHANCED_LOGGING", "1")))
E8_BANDIT_SYMMETRIZE = bool(int(os.getenv("E8_BANDIT_SYMMETRIZE", "1")))
E8_BANDIT_ROW_NORMALIZE = bool(int(os.getenv("E8_BANDIT_ROW_NORMALIZE", "1")))
E8_BANDIT_CLIP_EXPLORATION = bool(int(os.getenv("E8_BANDIT_CLIP_EXPLORATION", "1")))
E8_BANDIT_CLIP_PERCENTILE = float(os.getenv("E8_BANDIT_CLIP_PERCENTILE", "95.0"))
E8_JOURNEY_LOGGER_ENABLE = bool(int(os.getenv("E8_JOURNEY_LOGGER_ENABLE", "1")))
E8_JOURNEY_LOGGER_FILE = os.getenv("E8_JOURNEY_LOGGER_FILE", "journey.ndjson")
E8_JOURNEY_LOGGER_BUFFER_SIZE = int(os.getenv("E8_JOURNEY_LOGGER_BUFFER_SIZE", "100"))
E8_VAE_USE_FOR_PROJECTION = bool(int(os.getenv("E8_VAE_USE_FOR_PROJECTION", "1")))
E8_VAE_USE_FOR_QUERY = bool(int(os.getenv("E8_VAE_USE_FOR_QUERY", "1")))
E8_VAE_TELEM = bool(int(os.getenv("E8_VAE_TELEM", "1")))
E8_POTENTIAL_MAPPING = os.getenv("E8_POTENTIAL_MAPPING", "linear").strip().lower()
E8_POTENTIAL_SIGMOID_CENTER = float(os.getenv("E8_POTENTIAL_SIGMOID_CENTER", "0.5"))
E8_POTENTIAL_SIGMOID_GAIN = float(os.getenv("E8_POTENTIAL_SIGMOID_GAIN", "8.0"))
E8_POTENTIAL_SIGMOID_SPREAD = float(os.getenv("E8_POTENTIAL_SIGMOID_SPREAD", "0.85"))
E8_USE_POTENTIAL_IN_RETROLINK = bool(int(os.getenv("E8_USE_POTENTIAL_IN_RETROLINK", "1")))
E8_USE_POTENTIAL_EDGE_WEIGHT = bool(int(os.getenv("E8_USE_POTENTIAL_EDGE_WEIGHT", "1")))
E8_USE_POTENTIAL_IN_KDTREE = bool(int(os.getenv("E8_USE_POTENTIAL_IN_KDTREE", "1")))
E8_POTENTIAL_KDTREE_FACTOR = float(os.getenv("E8_POTENTIAL_KDTREE_FACTOR", "0.25"))
