from langchain_groq import ChatGroq
from src.config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

# Lazy-initialized client
_client = None


def _get_client():
    """Lazily initialize the ChatGroq client."""
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Please add it to your .env file."
            )
        _client = ChatGroq(
            model=LLM_MODEL,
            api_key=GROQ_API_KEY,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
    return _client


def generate(prompt):
    """Generate a complete response (non-streaming)."""
    client = _get_client()
    response = ""
    for chunk in client.stream([{"role": "user", "content": prompt}]):
        if chunk.content:
            response += chunk.content
    return response


def stream_generate(prompt):
    """Yield response chunks for streaming display in Streamlit."""
    client = _get_client()
    for chunk in client.stream([{"role": "user", "content": prompt}]):
        if chunk.content:
            yield chunk.content