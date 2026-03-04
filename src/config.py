import os
from dotenv import load_dotenv

load_dotenv()

# ── Model Configuration ──────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.6
LLM_MAX_TOKENS = 4096

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# ── Chunking ──────────────────────────────────────────────────────────
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# ── API Keys ──────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def validate_config():
    """Validate that all required configuration is present."""
    errors = []
    if not GROQ_API_KEY:
        errors.append("GROQ_API_KEY is not set. Add it to your .env file.")
    if errors:
        return False, errors
    return True, []
