"""Vector store builders. In-memory store (no external DB)."""
from langchain_core.vectorstores import InMemoryVectorStore


def build_inmemory_vectorstore(documents, embeddings, collection_name: str = "health_benefits_docs"):
    """
    Build in-memory vector store from documents and embeddings.
    Uses InMemoryVectorStore. collection_name kept for API compatibility.
    """
    return InMemoryVectorStore.from_documents(documents, embeddings)
