import os
from dotenv import load_dotenv

load_dotenv()

# ── Model Configuration ──────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Optional hard override, e.g. GROQ_MODEL=llama-3.1-8b-instant. Ignored if
# Groq is no longer serving it.
LLM_MODEL = os.getenv("GROQ_MODEL", "").strip() or None

# Preference order, best first. Groq retires model IDs on a rolling basis, so
# llm_wrapper checks these against the live /models list and takes the first
# one still available instead of trusting any single name.
LLM_MODEL_CANDIDATES = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3-32b",
    "moonshotai/kimi-k2-instruct",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
]

LLM_TEMPERATURE = 0.6
LLM_MAX_TOKENS = 4096

# ── Domain Classification ────────────────────────────────────────────
# Keyword-based by default, so the detected domain reflects what the corpus
# actually covers instead of the LLM's world knowledge. Set
# USE_LLM_DOMAIN_CLASSIFIER=true to hand classification to the LLM instead.
USE_LLM_DOMAIN_CLASSIFIER = os.getenv(
    "USE_LLM_DOMAIN_CLASSIFIER", ""
).strip().lower() in ("1", "true", "yes", "on")

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# ── Chunking ──────────────────────────────────────────────────────────
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


# ── API Keys ──────────────────────────────────────────────────────────
def _read_api_key():
    """Read GROQ_API_KEY from the environment, then Streamlit app secrets."""
    key = os.getenv("GROQ_API_KEY", "").strip()
    if key:
        return key

    # Streamlit Cloud deployments set the key under app secrets rather than in
    # a .env file.
    try:
        import streamlit as st

        return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        return None


GROQ_API_KEY = _read_api_key()


def validate_config():
    """Validate that all required configuration is present."""
    errors = []
    if not GROQ_API_KEY:
        errors.append(
            "GROQ_API_KEY is not set. Add it to your .env file locally, or to "
            "app secrets on Streamlit Cloud."
        )
    if errors:
        return False, errors
    return True, []
