from src.llm_wrapper import generate

# Domains the system can classify into
SUPPORTED_DOMAINS = ["legal", "medical", "academic", "general"]

# Quick keyword-based fallback (used if LLM classification fails)
DOMAIN_KEYWORDS = {
    "legal": ["law", "legal", "court", "judge", "copyright", "contract", "statute",
              "plaintiff", "defendant", "attorney", "lawsuit", "regulation", "rights",
              "amendment", "jurisdiction", "verdict", "prosecution", "litigation"],
    "medical": ["disease", "symptom", "treatment", "medical", "patient", "diabetes",
                "diagnosis", "clinical", "therapy", "health", "doctor", "medicine",
                "surgery", "prescription", "hospital", "chronic", "acute", "pathology"],
    "academic": ["theory", "research", "study", "academic", "university", "paper",
                 "hypothesis", "experiment", "scholarly", "thesis", "professor",
                 "lecture", "curriculum", "peer-review", "citation", "methodology"],
}


def classify_domain_keyword(query):
    """Fast keyword-based domain classification (fallback)."""
    query_lower = query.lower()
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw in query_lower)

    best_domain = max(scores, key=scores.get)
    if scores[best_domain] > 0:
        return best_domain
    return "general"


def classify_domain(query):
    """
    Classify the domain of a user query using the LLM.
    Falls back to keyword-based classification on error.
    """
    try:
        prompt = (
            "You are a domain classifier. Classify the following user question into "
            "exactly ONE of these domains: legal, medical, academic, general.\n\n"
            "Rules:\n"
            "- Respond with ONLY the domain name (one word, lowercase).\n"
            "- 'legal' = questions about law, regulations, courts, contracts, rights.\n"
            "- 'medical' = questions about health, diseases, treatments, anatomy.\n"
            "- 'academic' = questions about scientific theories, research, education.\n"
            "- 'general' = anything that doesn't clearly fit the above categories.\n\n"
            f"Question: {query}\n\n"
            "Domain:"
        )
        result = generate(prompt).strip().lower()

        # Extract just the domain word from the response
        for domain in SUPPORTED_DOMAINS:
            if domain in result:
                return domain

        # If LLM gave an unexpected response, fall back
        return classify_domain_keyword(query)

    except Exception:
        # If LLM fails, fall back to keyword-based
        return classify_domain_keyword(query)
