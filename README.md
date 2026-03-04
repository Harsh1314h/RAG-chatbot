# 🧠 Text Assistant

A RAG-powered Q&A chatbot that detects question domains (Legal, Medical, Academic) and retrieves relevant context from a knowledge base to generate accurate, well-structured answers.

## Features

- **Domain Detection** — LLM-based classification of questions into Legal, Medical, Academic, or General
- **RAG Pipeline** — Retrieves relevant documents from ChromaDB vector store before answering
- **Streaming Responses** — Real-time word-by-word response generation
- **Chat Interface** — Full conversational UI with chat history
- **Auto-Indexing** — Automatically indexes knowledge base documents on first run

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: Groq (Llama 3.3 70B)
- **Vector Store**: ChromaDB
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
- **Framework**: LangChain

## Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/textassistantchat.git
cd textassistantchat

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Create .env file with your Groq API key
echo GROQ_API_KEY="your_groq_api_key_here" > .env

# Run the app
streamlit run streamlit_app.py
```

## Live Demo

🔗 [Text Assistant on Streamlit Cloud](https://your-app-url.streamlit.app)

## License

MIT
