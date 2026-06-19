"""
SleepAI Configuration
All settings can be overridden via .env file or environment variables
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = Path(os.getenv("SCM_DATA_DIR", str(PROJECT_ROOT / "data"))).expanduser()
CONCEPTS_DIR = DATA_DIR / "concepts"
EPISODES_DIR = DATA_DIR / "episodes"
SESSIONS_DIR = DATA_DIR / "sessions"
MODELS_DIR = PROJECT_ROOT / "models"

# Database configuration
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_DB = os.getenv("POSTGRES_DB", "sleepai")
DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# LLM Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:latest")  # Ollama model name
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "openai"

# Hierarchical extraction (Zhong-inspired recursive chunking)
# When enabled, MeaningEncoder first chunks text into K semantically coherent
# segments, then extracts concepts from each segment independently.
# Produces 2-4x more granular concepts for multi-topic inputs.
# Only effective with cloud LLM providers (deepseek/openai); falls back to
# flat extraction for Ollama/heuristic modes.
HIERARCHICAL_EXTRACTION = os.getenv("HIERARCHICAL_EXTRACTION", "true").lower() == "true"
HIERARCHICAL_K = int(os.getenv("HIERARCHICAL_K", "4"))  # max branching factor

# LLaMA configuration (legacy, for direct llama.cpp usage)
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", str(MODELS_DIR / "llama-7b.q4.bin"))
LLAMA_N_CTX = int(os.getenv("LLAMA_N_CTX", "2048"))
LLAMA_N_THREADS = int(os.getenv("LLAMA_N_THREADS", "4"))

# Memory configuration
WORKING_MEMORY_CAPACITY = int(os.getenv("WORKING_MEMORY_CAPACITY", "7"))
IMPORTANCE_THRESHOLD = float(os.getenv("IMPORTANCE_THRESHOLD", "0.3"))
NOVELTY_THRESHOLD = float(os.getenv("NOVELTY_THRESHOLD", "0.7"))
EMOTIONAL_POSITIVE_THRESHOLD = float(os.getenv("EMOTIONAL_POSITIVE_THRESHOLD", "0.5"))
EMOTIONAL_NEGATIVE_THRESHOLD = float(os.getenv("EMOTIONAL_NEGATIVE_THRESHOLD", "-0.5"))

# Sleep configuration
SLEEP_ENTROPY_THRESHOLD = float(os.getenv("SLEEP_ENTROPY_THRESHOLD", "0.9"))
SLEEP_CONFLICT_THRESHOLD = float(os.getenv("SLEEP_CONFLICT_THRESHOLD", "0.3"))
SLEEP_INTERVAL_MAX = int(os.getenv("SLEEP_INTERVAL_MAX", "3600"))
NREM_DOWNSCALE_FACTOR = float(os.getenv("NREM_DOWNSCALE_FACTOR", "0.8"))
REM_DREAM_COUNT = int(os.getenv("REM_DREAM_COUNT", "5"))

# Auto-sleep during chat
AUTO_SLEEP_ENABLED = os.getenv("AUTO_SLEEP_ENABLED", "true").lower() == "true"
AUTO_SLEEP_INTERVAL = int(os.getenv("AUTO_SLEEP_INTERVAL", "5"))  # Messages between sleep checks

# Embedding configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

# API configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Session configuration
DEFAULT_SESSION_ID = os.getenv("DEFAULT_SESSION_ID", "default")
ENABLE_SESSION_PERSISTENCE = os.getenv("ENABLE_SESSION_PERSISTENCE", "true").lower() == "true"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_LLM_RAW = os.getenv("LOG_LLM_RAW", "false").lower() == "true"

# HME Phase 1: Selective Encoding & Fast Grasp
HME_ENABLED = os.getenv("HME_ENABLED", "true").lower() == "true"

# AttentionGate thresholds
SALIENT_ENCODE_THRESHOLD = float(os.getenv("SALIENT_ENCODE_THRESHOLD", "0.70"))
NORMAL_ENCODE_THRESHOLD = float(os.getenv("NORMAL_ENCODE_THRESHOLD", "0.40"))
SKIP_ENCODE_THRESHOLD = float(os.getenv("SKIP_ENCODE_THRESHOLD", "0.15"))

# Salience weights
SALIENCE_WEIGHT_NOVELTY = float(os.getenv("SALIENCE_WEIGHT_NOVELTY", "0.30"))
SALIENCE_WEIGHT_TASK = float(os.getenv("SALIENCE_WEIGHT_TASK", "0.35"))
SALIENCE_WEIGHT_EMOTIONAL = float(os.getenv("SALIENCE_WEIGHT_EMOTIONAL", "0.15"))
SALIENCE_WEIGHT_REPETITION = float(os.getenv("SALIENCE_WEIGHT_REPETITION", "0.10"))
SALIENCE_WEIGHT_PREDICTION = float(os.getenv("SALIENCE_WEIGHT_PREDICTION", "0.10"))

# Grasp score weights
GRASP_WEIGHT_SALIENCE = float(os.getenv("GRASP_WEIGHT_SALIENCE", "0.40"))
GRASP_WEIGHT_SCHEMA = float(os.getenv("GRASP_WEIGHT_SCHEMA", "0.30"))
GRASP_WEIGHT_CLARITY = float(os.getenv("GRASP_WEIGHT_CLARITY", "0.20"))
GRASP_WEIGHT_COGNITIVE_LOAD = float(os.getenv("GRASP_WEIGHT_COGNITIVE_LOAD", "0.10"))

# One-shot learning
ONE_SHOT_GRASP_THRESHOLD = float(os.getenv("ONE_SHOT_GRASP_THRESHOLD", "0.75"))

# Prediction error
PREDICTION_ERROR_WINDOW = int(os.getenv("PREDICTION_ERROR_WINDOW", "5"))
PREDICTION_ERROR_DECAY = float(os.getenv("PREDICTION_ERROR_DECAY", "0.85"))

# Noise estimation
NOISE_ESTIMATE_THRESHOLD = float(os.getenv("NOISE_ESTIMATE_THRESHOLD", "0.80"))

# HME Phase 2: Event + Association Binding
ASSOCIATION_LEARNING_RATE = float(os.getenv("ASSOCIATION_LEARNING_RATE", "0.30"))
ASSOCIATION_SEMANTIC_THRESHOLD = float(os.getenv("ASSOCIATION_SEMANTIC_THRESHOLD", "0.72"))
ASSOCIATION_MAX_EDGES_PER_CONCEPT = int(os.getenv("ASSOCIATION_MAX_EDGES_PER_CONCEPT", "10"))
ASSOCIATION_MIN_EDGE_STRENGTH = float(os.getenv("ASSOCIATION_MIN_EDGE_STRENGTH", "0.12"))
ASSOCIATION_AGING_DECAY = float(os.getenv("ASSOCIATION_AGING_DECAY", "0.92"))
ASSOCIATION_AGING_STALE_STEPS = int(os.getenv("ASSOCIATION_AGING_STALE_STEPS", "3"))
ASSOCIATION_AGING_THRESHOLD = float(os.getenv("ASSOCIATION_AGING_THRESHOLD", "0.35"))

# HME Phase 3: Spreading Activation Retrieval
SPREADING_ACTIVATION_STEPS = int(os.getenv("SPREADING_ACTIVATION_STEPS", "3"))
SPREADING_ACTIVATION_DECAY = float(os.getenv("SPREADING_ACTIVATION_DECAY", "0.45"))
SPREADING_ACTIVATION_THRESHOLD = float(os.getenv("SPREADING_ACTIVATION_THRESHOLD", "0.05"))
SPREADING_ACTIVATION_MAX_CANDIDATES = int(os.getenv("SPREADING_ACTIVATION_MAX_CANDIDATES", "50"))
SPREADING_CONTEXT_GATE_WEIGHT = float(os.getenv("SPREADING_CONTEXT_GATE_WEIGHT", "0.30"))

# Hypothesis Ranker
HYPOTHESIS_ACTIVATION_WEIGHT = float(os.getenv("HYPOTHESIS_ACTIVATION_WEIGHT", "0.30"))
HYPOTHESIS_RECENCY_WEIGHT = float(os.getenv("HYPOTHESIS_RECENCY_WEIGHT", "0.15"))
HYPOTHESIS_DENSITY_WEIGHT = float(os.getenv("HYPOTHESIS_DENSITY_WEIGHT", "0.15"))
HYPOTHESIS_IMPORTANCE_WEIGHT = float(os.getenv("HYPOTHESIS_IMPORTANCE_WEIGHT", "0.25"))
HYPOTHESIS_REHEARSAL_WEIGHT = float(os.getenv("HYPOTHESIS_REHEARSAL_WEIGHT", "0.10"))
HYPOTHESIS_CONTRADICTION_PENALTY = float(os.getenv("HYPOTHESIS_CONTRADICTION_PENALTY", "0.40"))
HYPOTHESIS_HIGH_CONFIDENCE = float(os.getenv("HYPOTHESIS_HIGH_CONFIDENCE", "0.70"))
HYPOTHESIS_MEDIUM_CONFIDENCE = float(os.getenv("HYPOTHESIS_MEDIUM_CONFIDENCE", "0.40"))
HYPOTHESIS_LOW_CONFIDENCE = float(os.getenv("HYPOTHESIS_LOW_CONFIDENCE", "0.15"))
HYPOTHESIS_MAX = int(os.getenv("HYPOTHESIS_MAX", "10"))

# HME Phase 4: SleepKernelV2 (Micro + Deep)
MICRO_SLEEP_ENABLED = os.getenv("MICRO_SLEEP_ENABLED", "true").lower() == "true"
MICRO_SLEEP_INTERVAL_TURNS = int(os.getenv("MICRO_SLEEP_INTERVAL_TURNS", "4"))
MICRO_SLEEP_ENTROPY_THRESHOLD = float(os.getenv("MICRO_SLEEP_ENTROPY_THRESHOLD", "0.82"))
MICRO_SLEEP_LIGHT_DECAY_FACTOR = float(os.getenv("MICRO_SLEEP_LIGHT_DECAY_FACTOR", "0.97"))
MICRO_SLEEP_REPLAY_TOP_K = int(os.getenv("MICRO_SLEEP_REPLAY_TOP_K", "12"))

DEEP_SLEEP_MIN_IDLE_SECONDS = int(os.getenv("DEEP_SLEEP_MIN_IDLE_SECONDS", "900"))
DEEP_SLEEP_SESSION_TURNS = int(os.getenv("DEEP_SLEEP_SESSION_TURNS", "24"))
DEEP_SLEEP_PRESSURE_THRESHOLD = float(os.getenv("DEEP_SLEEP_PRESSURE_THRESHOLD", "0.93"))
DEEP_SLEEP_GLOBAL_DOWNSCALE_FACTOR = float(os.getenv("DEEP_SLEEP_GLOBAL_DOWNSCALE_FACTOR", "0.90"))
DEEP_SLEEP_ENABLE_SYNTHESIS = os.getenv("DEEP_SLEEP_ENABLE_SYNTHESIS", "true").lower() == "true"
DREAM_STATE_ENABLED = os.getenv("DREAM_STATE_ENABLED", "true").lower() == "true"
# HME Phase 6 fix: cooldown prevents auto-sleep from firing on every turn when
# the entropy distribution saturates (e.g. with a uniform-salience encoder).
DEEP_SLEEP_COOLDOWN_TURNS = int(os.getenv("DEEP_SLEEP_COOLDOWN_TURNS", "12"))
MICRO_SLEEP_COOLDOWN_TURNS = int(os.getenv("MICRO_SLEEP_COOLDOWN_TURNS", "2"))
# HME Phase 6 fix: enable sleep-time paraphrase. Rewrites consolidated concept
# descriptions into clean fact form, dramatically improving token-overlap
# retrieval scores at zero ingestion-time cost.
DEEP_SLEEP_ENABLE_PARAPHRASE = os.getenv("DEEP_SLEEP_ENABLE_PARAPHRASE", "false").lower() == "true"
# Phase 7: schema extraction during deep-sleep. Detects recurring patterns
# (entities, co-occurrences, trajectories, cadences) across episodes and
# emits ABSTRACT-type "schema" concepts. The wake-summary endpoint uses
# these to tell the user what the agent noticed while they were away.
DEEP_SLEEP_ENABLE_SCHEMAS = os.getenv("DEEP_SLEEP_ENABLE_SCHEMAS", "false").lower() == "true"
SCHEMA_MIN_REPETITIONS = int(os.getenv("SCHEMA_MIN_REPETITIONS", "3"))
SCHEMA_COOCCURRENCE_MIN = int(os.getenv("SCHEMA_COOCCURRENCE_MIN", "2"))
SCHEMA_MAX_PER_CYCLE = int(os.getenv("SCHEMA_MAX_PER_CYCLE", "12"))
SCHEMA_MAX_EPISODES_WINDOW = int(os.getenv("SCHEMA_MAX_EPISODES_WINDOW", "200"))
SCHEMA_TEMPORAL_WINDOW_HOURS = float(os.getenv("SCHEMA_TEMPORAL_WINDOW_HOURS", "24"))

# Phase 7 (de-hardcoding): central path to the linguistic-resources JSON.
# Empty = use the default English bundle at src/core/locales/en.json.
# Override to point at a localized or domain-specific file. Loaded once
# per process by src/core/linguistic_resources.py.
LINGUISTIC_CONFIG_PATH = os.getenv("LINGUISTIC_CONFIG_PATH", "")

# Phase 7: Curiosity engine. Detects entity gaps and ingests external
# knowledge during sleep. Default off; opt-in by setting source paths.
CURIOSITY_ENGINE_ENABLED = os.getenv("CURIOSITY_ENGINE_ENABLED", "false").lower() == "true"
CURIOSITY_MAX_GAPS_PER_CYCLE = int(os.getenv("CURIOSITY_MAX_GAPS_PER_CYCLE", "3"))
CURIOSITY_MIN_OCCURRENCES = int(os.getenv("CURIOSITY_MIN_OCCURRENCES", "2"))
CURIOSITY_MAX_BRIEF_CHARS = int(os.getenv("CURIOSITY_MAX_BRIEF_CHARS", "280"))
# Source: local docs folder. Empty = disabled.
CURIOSITY_LOCAL_DOCS_FOLDER = os.getenv("CURIOSITY_LOCAL_DOCS_FOLDER", "")
# Source: path to a JSON dictionary file (entity → brief). Empty = disabled.
CURIOSITY_DICTIONARY_PATH = os.getenv("CURIOSITY_DICTIONARY_PATH", "")
# Source: LLM-backed brief generation. Uses the project's configured LLM
# (LLM_PROVIDER + LLM_MODEL). Default off — opt-in for cost reasons.
CURIOSITY_LLM_SOURCE_ENABLED = os.getenv("CURIOSITY_LLM_SOURCE_ENABLED", "false").lower() == "true"

# Phase 7 (autonomous learning): IdleLearner daemon configuration.
# When enabled, a background thread fires sleep cycles on idle sessions so
# the agent can consolidate, abstract, and prune while the user is away.
IDLE_LEARNER_ENABLED = os.getenv("IDLE_LEARNER_ENABLED", "false").lower() == "true"
IDLE_LEARNER_IDLE_THRESHOLD_SECONDS = float(os.getenv("IDLE_LEARNER_IDLE_THRESHOLD_SECONDS", "600"))
IDLE_LEARNER_MIN_SLEEP_INTERVAL_SECONDS = float(os.getenv("IDLE_LEARNER_MIN_SLEEP_INTERVAL_SECONDS", "1800"))
IDLE_LEARNER_TICK_INTERVAL_SECONDS = float(os.getenv("IDLE_LEARNER_TICK_INTERVAL_SECONDS", "60"))
IDLE_LEARNER_MAX_SLEEP_DURATION_SECONDS = float(os.getenv("IDLE_LEARNER_MAX_SLEEP_DURATION_SECONDS", "120"))
IDLE_LEARNER_SLEEP_MODE = os.getenv("IDLE_LEARNER_SLEEP_MODE", "deep")
# Phase 7 M6: state persistence so the daemon survives API restarts.
IDLE_LEARNER_STATE_PATH = os.getenv("IDLE_LEARNER_STATE_PATH", "data/idle_learner_state.json")
IDLE_LEARNER_PERSIST_ENABLED = os.getenv("IDLE_LEARNER_PERSIST_ENABLED", "true").lower() == "true"
IDLE_LEARNER_PERSIST_EVERY_N_TICKS = int(os.getenv("IDLE_LEARNER_PERSIST_EVERY_N_TICKS", "5"))

# Phase 7 M6: Lifecycle policy (power/CPU gating)
LIFECYCLE_POLICY_ENABLED = os.getenv("LIFECYCLE_POLICY_ENABLED", "true").lower() == "true"
LIFECYCLE_POLICY_MIN_BATTERY_PERCENT = float(os.getenv("LIFECYCLE_POLICY_MIN_BATTERY_PERCENT", "30"))
LIFECYCLE_POLICY_ALLOW_WHEN_PLUGGED_IN = os.getenv("LIFECYCLE_POLICY_ALLOW_WHEN_PLUGGED_IN", "true").lower() == "true"
LIFECYCLE_POLICY_MAX_CPU_PERCENT = float(os.getenv("LIFECYCLE_POLICY_MAX_CPU_PERCENT", "80"))
LIFECYCLE_POLICY_CPU_SAMPLE_SECONDS = float(os.getenv("LIFECYCLE_POLICY_CPU_SAMPLE_SECONDS", "0.2"))
LIFECYCLE_POLICY_REQUIRE_PSUTIL = os.getenv("LIFECYCLE_POLICY_REQUIRE_PSUTIL", "false").lower() == "true"

# Phase 7 (autonomous learning): Cross-Session Memory Pool.
# When enabled, sleep cycles see recent prior-session episodes (not just the
# current session's working memory). This is what gives the agent continuity
# of self across days.
CROSS_SESSION_POOL_ENABLED = os.getenv("CROSS_SESSION_POOL_ENABLED", "false").lower() == "true"
CROSS_SESSION_POOL_MAX_SESSIONS = int(os.getenv("CROSS_SESSION_POOL_MAX_SESSIONS", "5"))
CROSS_SESSION_POOL_LOOKBACK_HOURS = float(os.getenv("CROSS_SESSION_POOL_LOOKBACK_HOURS", "168"))  # 7 days
CROSS_SESSION_POOL_MAX_EPISODES_PER_SESSION = int(os.getenv("CROSS_SESSION_POOL_MAX_EPISODES_PER_SESSION", "30"))
CROSS_SESSION_POOL_MAX_TOTAL_BORROWED = int(os.getenv("CROSS_SESSION_POOL_MAX_TOTAL_BORROWED", "100"))

# HME Phase 5: Forgetting Dynamics + Contradiction Versioning
FORGETTING_WEIGHT_GRASP = float(os.getenv("FORGETTING_WEIGHT_GRASP", "0.22"))
FORGETTING_WEIGHT_SALIENCE = float(os.getenv("FORGETTING_WEIGHT_SALIENCE", "0.24"))
FORGETTING_WEIGHT_REHEARSAL = float(os.getenv("FORGETTING_WEIGHT_REHEARSAL", "0.18"))
FORGETTING_WEIGHT_ASSOCIATION = float(os.getenv("FORGETTING_WEIGHT_ASSOCIATION", "0.18"))
FORGETTING_WEIGHT_RECENCY = float(os.getenv("FORGETTING_WEIGHT_RECENCY", "0.12"))
FORGETTING_WEIGHT_INTERFERENCE = float(os.getenv("FORGETTING_WEIGHT_INTERFERENCE", "0.22"))
FORGETTING_SUPPRESS_THRESHOLD = float(os.getenv("FORGETTING_SUPPRESS_THRESHOLD", "0.32"))
FORGETTING_ARCHIVE_THRESHOLD = float(os.getenv("FORGETTING_ARCHIVE_THRESHOLD", "0.16"))
FORGETTING_BASE_DECAY = float(os.getenv("FORGETTING_BASE_DECAY", "0.04"))
CONTRADICTION_VERSION_SIMILARITY = float(os.getenv("CONTRADICTION_VERSION_SIMILARITY", "0.70"))
CONTRADICTION_VERSION_CONTEXT_BONUS = float(os.getenv("CONTRADICTION_VERSION_CONTEXT_BONUS", "0.12"))
# HME Phase 6 fix: protect concepts above this salience floor from being
# archived even under high entropy pressure. Prevents the "uniform encoder
# saturates entropy → forgetting prunes everything" failure mode AND the
# brutal-harness Tier-6 failure where freshly-ingested user facts (e.g.
# "I'm allergic to seafood") get archived on the very first sleep cycle.
#
# Default raised to 0.5 in v0.7.1 so the protection is ON by default —
# legacy aggressive-forgetting behavior is restored by setting
# FORGETTING_PROTECT_SALIENCE=0.0 in the environment.
FORGETTING_PROTECT_SALIENCE = float(os.getenv("FORGETTING_PROTECT_SALIENCE", "0.5"))
# Min rehearsal count before a concept can be archived. New concepts that
# haven't survived a sleep cycle are protected from immediate eviction.
# Default raised to 1 in v0.7.1 — concepts must survive at least one
# sleep cycle before they're eligible for archiving.
FORGETTING_MIN_REHEARSAL_BEFORE_ARCHIVE = int(os.getenv("FORGETTING_MIN_REHEARSAL_BEFORE_ARCHIVE", "1"))

current_time: float = 0.0


def get_config_summary():
    """Get a summary of current configuration (safe for display)"""
    return {
        "llm_model": LLM_MODEL,
        "llm_provider": LLM_PROVIDER,
        "working_memory_capacity": WORKING_MEMORY_CAPACITY,
        "importance_threshold": IMPORTANCE_THRESHOLD,
        "sleep_entropy_threshold": SLEEP_ENTROPY_THRESHOLD,
        "sleep_conflict_threshold": SLEEP_CONFLICT_THRESHOLD,
        "auto_sleep_enabled": AUTO_SLEEP_ENABLED,
        "auto_sleep_interval": AUTO_SLEEP_INTERVAL,
        "micro_sleep_enabled": MICRO_SLEEP_ENABLED,
        "micro_sleep_interval_turns": MICRO_SLEEP_INTERVAL_TURNS,
        "deep_sleep_min_idle_seconds": DEEP_SLEEP_MIN_IDLE_SECONDS,
        "embedding_model": EMBEDDING_MODEL,
        "api_host": API_HOST,
        "api_port": API_PORT,
        "session_persistence": ENABLE_SESSION_PERSISTENCE,
    }
