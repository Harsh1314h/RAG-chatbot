import streamlit as st
from src.llm_wrapper import stream_generate
from src.retrieval import Retriever
from src.domain_classifier import classify_domain
from src.config import validate_config

# ── Page Config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Text Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main header */
.main-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem 0;
}
.main-header h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.4rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.main-header p {
    color: #9ca3af;
    font-size: 1rem;
    font-weight: 300;
}

/* Domain badge */
.domain-badge {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}
.domain-legal { background: rgba(59,130,246,0.15); color: #60a5fa; border: 1px solid rgba(59,130,246,0.3); }
.domain-medical { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.domain-academic { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
.domain-general { background: rgba(156,163,175,0.15); color: #9ca3af; border: 1px solid rgba(156,163,175,0.3); }

/* Source cards */
.source-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.7rem;
    backdrop-filter: blur(10px);
}
.source-card h4 {
    color: #a78bfa;
    font-size: 0.85rem;
    margin-bottom: 0.4rem;
    font-weight: 600;
}
.source-card p {
    color: #d1d5db;
    font-size: 0.8rem;
    line-height: 1.5;
}

/* Sidebar style */
section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}
.sidebar-stat {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 0.8rem;
    margin-bottom: 0.5rem;
    text-align: center;
}
.sidebar-stat h3 { color: #a78bfa; font-size: 1.3rem; margin: 0; }
.sidebar-stat p { color: #9ca3af; font-size: 0.75rem; margin: 0; }
</style>
""", unsafe_allow_html=True)

# ── Validation ────────────────────────────────────────────────────────
config_ok, config_errors = validate_config()
if not config_ok:
    st.error("⚠️ Configuration Error")
    for err in config_errors:
        st.error(err)
    st.stop()

# ── Initialize session state ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = Retriever()
if "last_domain" not in st.session_state:
    st.session_state.last_domain = None
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

retriever = st.session_state.retriever

# ── Auto-index if ChromaDB is empty ──────────────────────────────────
if retriever.document_count == 0:
    with st.spinner("📦 First run detected — indexing documents into ChromaDB..."):
        from src.data_processing import process_and_index_documents
        count = process_and_index_documents()
        # Reload retriever after indexing
        st.session_state.retriever = Retriever()
        retriever = st.session_state.retriever
        if count and count > 0:
            st.success(f"✅ Indexed {count} document chunks. Ready to answer questions!")
        else:
            st.warning("⚠️ No documents found to index. Add .txt files to the `data/` folder.")


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Text Assistant")
    st.markdown("---")

    # Stats
    doc_count = retriever.document_count
    st.markdown(f"""
    <div class="sidebar-stat">
        <h3>{doc_count}</h3>
        <p>Document Chunks Indexed</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.last_domain:
        domain = st.session_state.last_domain
        st.markdown(f"""
        <div class="sidebar-stat">
            <h3>{domain.capitalize()}</h3>
            <p>Last Detected Domain</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📚 Supported Domains")
    st.markdown("""
    - ⚖️ **Legal** — Law, courts, contracts
    - 🏥 **Medical** — Health, diseases, treatments
    - 🎓 **Academic** — Research, theories, studies
    - 🌐 **General** — Everything else
    """)

    st.markdown("---")

    # Sources panel
    if st.session_state.last_sources:
        st.markdown("### 📄 Last Retrieved Sources")
        for i, doc in enumerate(st.session_state.last_sources, 1):
            source_name = doc.metadata.get("source", "Unknown")
            domain_tag = doc.metadata.get("domain", "general")
            preview = doc.page_content[:150].replace("\n", " ")
            st.markdown(f"""
            <div class="source-card">
                <h4>Source {i}: {source_name} ({domain_tag})</h4>
                <p>{preview}...</p>
            </div>
            """, unsafe_allow_html=True)

    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_domain = None
        st.session_state.last_sources = []
        st.rerun()

# ── Main Area ─────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧠 Text Assistant</h1>
    <p>RAG-powered Q&A across Legal, Medical & Academic domains</p>
</div>
""", unsafe_allow_html=True)

# ── Chat history ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("domain"):
            domain = msg["domain"]
            st.markdown(
                f'<span class="domain-badge domain-{domain}">📌 {domain}</span>',
                unsafe_allow_html=True,
            )
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────
if question := st.chat_input("Ask a question about law, medicine, or academics..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate response
    with st.chat_message("assistant"):
        # Step 1: Classify domain
        with st.spinner("🔍 Detecting domain..."):
            domain = classify_domain(question)
            st.session_state.last_domain = domain

        st.markdown(
            f'<span class="domain-badge domain-{domain}">📌 {domain}</span>',
            unsafe_allow_html=True,
        )

        # Step 2: Retrieve relevant context
        with st.spinner("📚 Retrieving relevant documents..."):
            retrieved_docs = retriever.retrieve(question, k=4, domain_filter=domain)
            st.session_state.last_sources = retrieved_docs

        # Step 3: Build RAG prompt
        if retrieved_docs:
            context = "\n\n---\n\n".join(
                [f"[Source: {doc.metadata.get('source', 'N/A')}]\n{doc.page_content}"
                 for doc in retrieved_docs]
            )
            prompt = (
                f"You are a knowledgeable assistant specialized in {domain} topics.\n"
                f"Use the following retrieved context to answer the user's question. "
                f"If the context doesn't contain enough information, say so and provide "
                f"what you know from your general knowledge.\n\n"
                f"--- CONTEXT ---\n{context}\n--- END CONTEXT ---\n\n"
                f"User Question: {question}\n\n"
                f"Provide a clear, well-structured answer:"
            )
        else:
            prompt = (
                f"You are a knowledgeable assistant. The user asked a question in the "
                f"'{domain}' domain, but no relevant documents were found in the knowledge base.\n"
                f"Answer the question using your general knowledge.\n\n"
                f"User Question: {question}\n\n"
                f"Answer:"
            )

        # Step 4: Stream the response
        response = st.write_stream(stream_generate(prompt))

        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "domain": domain,
        })