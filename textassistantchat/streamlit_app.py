import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
from src.llm_wrapper import generate
from src.retrieval import Retriever
from src.domain_classifier import classify_domain



st.set_page_config(page_title="Free Llama 3 Assistant", layout="wide")

st.title("TEXT ASSISTANT")
st.write("Ask any question in legal, medical, or academic domains.")

# Add a unique key to the text_input widget
question = st.text_input("Enter your question:", key="user_question_input")

if "retriever" not in st.session_state:
    st.session_state.retriever = Retriever()

if question:
    domain = classify_domain(question)
    st.write(f"**Domain detected:** {domain.capitalize()}")
    retrieved_docs = st.session_state.retriever.retrieve(question, k=4)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt = (
        f"You are a helpful assistant specialized in {domain} topics. "
        f"Use the following context to answer the user's question. "
        f"Context:\n{context}\n\nUser Question: {question}\n\nAnswer:"
    )
    with st.spinner("Generating answer..."):
        answer = generate(prompt)
    st.markdown("### Answer")
    st.write(answer)
    st.markdown("### Sources")
    for i, doc in enumerate(retrieved_docs, 1):
        st.write(f"**Source {i}:** {doc.metadata.get('source', 'N/A')}")
        st.write(doc.page_content[:200] + "...")