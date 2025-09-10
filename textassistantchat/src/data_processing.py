import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import CHROMA_DB_DIR

def process_and_index_documents(data_dir="data"):
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(data_dir, filename))
            docs = loader.load_and_split(text_splitter=text_splitter)
            documents.extend(docs)
    if not documents:
        print("No documents found for indexing.")
        return
    # Use the updated HuggingFaceEmbeddings class
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    db = Chroma.from_texts(
        texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=CHROMA_DB_DIR
    )
    print(f"Indexed {len(documents)} documents.")

if __name__ == "__main__":
    process_and_index_documents()
