from groq import Groq, NotFoundError
from langchain_groq import ChatGroq
from src.config import (
    GROQ_API_KEY,
    LLM_MODEL,
    LLM_MODEL_CANDIDATES,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)

# Substrings marking models that can't serve chat completions.
_NON_CHAT_HINTS = ("whisper", "tts", "playai", "embed", "guard", "prompt-guard")

# Lazy-initialized client
_client = None
_active_model = None


def _available_chat_models():
    """Ask Groq which chat models this API key can actually use right now."""
    try:
        listing = Groq(api_key=GROQ_API_KEY).models.list()
    except Exception:
        # Network hiccup or an unexpected payload — caller falls back to the
        # static candidate list.
        return []

    models = []
    for entry in getattr(listing, "data", None) or []:
        model_id = getattr(entry, "id", None)
        if not model_id:
            continue
        if any(hint in model_id.lower() for hint in _NON_CHAT_HINTS):
            continue
        models.append(model_id)
    return models


def _resolve_model():
    """Pick a model Groq is actually serving.

    Groq retires model IDs on a rolling basis, so any hard-coded name
    eventually starts returning 404 (groq.NotFoundError). Resolving against the
    live list keeps the app working across those retirements.
    """
    available = _available_chat_models()

    # Couldn't reach the models endpoint — fall back to the static preference
    # list and let the chat call report any real problem.
    if not available:
        return LLM_MODEL or LLM_MODEL_CANDIDATES[0]

    # An explicit GROQ_MODEL override wins, but only if it's still served.
    if LLM_MODEL and LLM_MODEL in available:
        return LLM_MODEL

    for candidate in LLM_MODEL_CANDIDATES:
        if candidate in available:
            return candidate

    # Nothing from the preference list survived; take whatever is on offer.
    return available[0]


def _get_client():
    """Lazily initialize the ChatGroq client."""
    global _client, _active_model
    if _client is None:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Please add it to your .env file "
                "(or to Streamlit Cloud app secrets)."
            )
        _active_model = _resolve_model()
        _client = ChatGroq(
            model=_active_model,
            api_key=GROQ_API_KEY,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
    return _client


def _reset_client():
    """Drop the cached client so the next call re-resolves the model."""
    global _client, _active_model
    _client = None
    _active_model = None


def get_active_model():
    """Return the model currently in use, initializing the client if needed."""
    _get_client()
    return _active_model


def _stream_chunks(prompt):
    """Stream response text, re-resolving the model once if it 404s."""
    messages = [{"role": "user", "content": prompt}]
    started = False
    try:
        for chunk in _get_client().stream(messages):
            if chunk.content:
                started = True
                yield chunk.content
        return
    except NotFoundError:
        # The model was retired since we resolved it. Retry once against a
        # freshly resolved model — but only if nothing was emitted yet, so we
        # never duplicate partial output.
        if started:
            raise

    _reset_client()
    for chunk in _get_client().stream(messages):
        if chunk.content:
            yield chunk.content


def generate(prompt):
    """Generate a complete response (non-streaming)."""
    return "".join(_stream_chunks(prompt))


def stream_generate(prompt):
    """Yield response chunks for streaming display in Streamlit."""
    return _stream_chunks(prompt)
