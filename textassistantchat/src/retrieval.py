from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from src.config import CHROMA_DB_DIR

class Retriever:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.db = Chroma(persist_directory=CHROMA_DB_DIR)

    def retrieve(self, query, k=4):
        query_vec = self.embedder.encode([query])[0]
        results = self.db.similarity_search_by_vector(query_vec, k=k)
        return results
