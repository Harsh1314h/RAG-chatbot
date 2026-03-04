from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import CHROMA_DB_DIR, EMBEDDING_MODEL


class Retriever:
    """RAG retriever using ChromaDB with HuggingFace embeddings."""

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.db = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=self.embeddings,
        )

    def retrieve(self, query, k=4, domain_filter=None):
        """
        Retrieve relevant document chunks for a query.

        Args:
            query: The user's question.
            k: Number of results to return.
            domain_filter: Optional domain string to filter results (e.g. "legal").

        Returns:
            List of Document objects with page_content and metadata.
        """
        search_kwargs = {"k": k}
        if domain_filter and domain_filter != "general":
            search_kwargs["filter"] = {"domain": domain_filter}

        results = self.db.similarity_search(query, **search_kwargs)
        return results

    @property
    def document_count(self):
        """Return the number of documents in the vector store."""
        try:
            return self.db._collection.count()
        except Exception:
            return 0
