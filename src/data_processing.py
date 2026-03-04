import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import CHROMA_DB_DIR, DATA_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


def get_embeddings():
    """Return the shared embedding function used across the project."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def process_and_index_documents(data_dir=None):
    """Load .txt files from data_dir, split them, and index into ChromaDB."""
    data_dir = data_dir or DATA_DIR

    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            # Determine the domain from filename (e.g. sample_legal.txt → legal)
            domain = "general"
            for d in ["legal", "medical", "academic"]:
                if d in filename.lower():
                    domain = d
                    break

            loader = TextLoader(filepath, encoding="utf-8")
            docs = loader.load_and_split(text_splitter=text_splitter)

            # Tag each chunk with domain metadata
            for doc in docs:
                doc.metadata["domain"] = domain
                doc.metadata["source"] = filename

            documents.extend(docs)

    if not documents:
        print("⚠ No documents found for indexing.")
        return 0

    # Clear existing data to avoid duplicates on re-index
    try:
        existing_db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=get_embeddings())
        existing_db.delete_collection()
    except Exception:
        pass  # DB doesn't exist yet, that's fine

    embeddings = get_embeddings()
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    db = Chroma.from_texts(
        texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=CHROMA_DB_DIR,
    )

    count = db._collection.count()
    print(f"✅ Indexed {count} document chunks into ChromaDB at '{CHROMA_DB_DIR}'")
    return count


if __name__ == "__main__":
    process_and_index_documents()
