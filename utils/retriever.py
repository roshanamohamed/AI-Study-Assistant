import os
from typing import List, Optional

from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "data"
NOTES_PATH = os.path.join(DATA_DIR, "notes.txt")

PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "internship-knowledge-hub")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

# Global vector store + embeddings
_embeddings = VertexAIEmbeddings(
    model_name="textembedding-gecko@003",
    project=PROJECT,
    location=LOCATION,
)
_vectorstore: Optional[FAISS] = None


def _load_corpus() -> List[str]:
    """Load initial corpus from notes.txt and any extra .txt files in data/."""
    texts: List[str] = []

    # Base notes file
    if os.path.exists(NOTES_PATH):
        with open(NOTES_PATH, "r", encoding="utf-8") as f:
            texts.append(f.read())

    # Any extra .txt files in data (excluding notes.txt)
    if os.path.exists(DATA_DIR):
        for fname in os.listdir(DATA_DIR):
            path = os.path.join(DATA_DIR, fname)
            if (
                os.path.isfile(path)
                and fname.endswith(".txt")
                and os.path.abspath(path) != os.path.abspath(NOTES_PATH)
            ):
                with open(path, "r", encoding="utf-8") as f:
                    texts.append(f.read())

    return [t for t in texts if t.strip()]


def init_vectorstore() -> None:
    """
    Initialize FAISS vector store from existing corpus.
    Call this once at startup.
    """
    global _vectorstore

    texts = _load_corpus()
    if not texts:
        _vectorstore = None
        return

    _vectorstore = FAISS.from_texts(texts=texts, embedding=_embeddings)


def add_text_document(text: str) -> None:
    """
    Add new text to the vector store (e.g., from uploaded notes or PDFs).
    """
    global _vectorstore

    cleaned = text.strip()
    if not cleaned:
        return

    if _vectorstore is None:
        _vectorstore = FAISS.from_texts(texts=[cleaned], embedding=_embeddings)
    else:
        _vectorstore.add_texts([cleaned])


def get_relevant_context(query: str, k: int = 4, max_chars: int = 1200) -> str:
    """
    Retrieve the most relevant chunks from the vector store using semantic similarity.
    """
    if _vectorstore is None:
        return "No study documents found yet. Try adding some notes or PDFs."

    docs = _vectorstore.similarity_search(query, k=k)
    combined = "\n\n".join(doc.page_content for doc in docs)

    if not combined.strip():
        return "No matching context found. I will answer from general knowledge."

    return combined[:max_chars]
