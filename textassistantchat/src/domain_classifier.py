def classify_domain(query):
    query_lower = query.lower()
    if any(word in query_lower for word in ["law", "legal", "court", "judge", "copyright", "contract"]):
        return "legal"
    elif any(word in query_lower for word in ["disease", "symptom", "treatment", "medical", "patient", "diabetes"]):
        return "medical"
    elif any(word in query_lower for word in ["theory", "research", "study", "academic", "university", "paper"]):
        return "academic"
    else:
        return "general"
