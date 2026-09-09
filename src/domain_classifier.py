from src.config import USE_LLM_DOMAIN_CLASSIFIER

# Domains the system can classify into
SUPPORTED_DOMAINS = ["legal", "medical", "academic", "general"]

# Keyword vocabulary per domain. This is deliberately scoped to the kind of
# material actually indexed in data/ — a query that matches nothing here is
# treated as "general", which is the signal that the knowledge base has no
# grounding for it. Widen a list only when you index documents to match.
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
    """Fast keyword-based domain classification.

    This is the default classifier. It is deterministic and bounded by the
    vocabulary above, so it can only claim a domain the knowledge base is
    actually meant to cover.
    """
    query_lower = query.lower()
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw in query_lower)

    best_domain = max(scores, key=scores.get)
    if scores[best_domain] > 0:
        return best_domain
    return "general"


def classify_domain_llm(query):
    """Classify using the LLM's own world knowledge.

    Off by default: the LLM recognises topics the corpus knows nothing about
    (it labels "typhoid" medical even though only diabetes is indexed), which
    routes the query into a domain-filtered retrieval that cannot support it.
    Enable with USE_LLM_DOMAIN_CLASSIFIER=true only if the knowledge base is
    broad enough to back it up.
    """
    from src.llm_wrapper import generate

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

    # If the LLM gave an unexpected response, fall back
    return classify_domain_keyword(query)


def classify_domain(query):
    """Classify the domain of a user query.

    Keyword-based by default so that domain detection stays tied to the
    indexed corpus rather than to the LLM's general knowledge.
    """
    if not USE_LLM_DOMAIN_CLASSIFIER:
        return classify_domain_keyword(query)

    try:
        return classify_domain_llm(query)
    except Exception:
        # If the LLM is unavailable, fall back to keywords rather than failing.
        return classify_domain_keyword(query)
