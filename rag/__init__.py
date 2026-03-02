"""
RAG package: loaders, splitters, vectorstores, retrievers, pipeline.
Aligned with Health Insurance Policy Explainer.ipynb.
Entry points: get_retrieval_chain (ensemble), get_naive_retrieval_chain, load_pdf, split_docs.
"""
from rag.loaders import load_pdf
from rag.splitters import (
    split_docs,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    get_parent_splitter,
    get_child_splitter,
    PARENT_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
)
from rag.vectorstores import build_inmemory_vectorstore
from rag.retrievers import (
    make_vector_retriever,
    RETRIEVER_K,
    build_bm25_retriever,
    build_compression_retriever,
    build_multi_query_retriever,
    build_parent_document_retriever,
    build_ensemble_retriever,
)
from rag.pipeline import (
    get_retrieval_chain,
    get_naive_retrieval_chain,
    get_retriever,
    get_naive_retriever,
    get_ensemble_retriever,
    rag_prompt,
    RAG_TEMPLATE,
)

# Backward compatibility: build_inmemory_qdrant_vectorstore alias
def build_inmemory_qdrant_vectorstore(documents, embeddings, collection_name: str = "health_benefits_docs"):
    return build_inmemory_vectorstore(documents, embeddings, collection_name)

__all__ = [
    "load_pdf",
    "split_docs",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "get_parent_splitter",
    "get_child_splitter",
    "PARENT_CHUNK_SIZE",
    "PARENT_CHUNK_OVERLAP",
    "CHILD_CHUNK_SIZE",
    "CHILD_CHUNK_OVERLAP",
    "build_inmemory_vectorstore",
    "build_inmemory_qdrant_vectorstore",
    "make_vector_retriever",
    "RETRIEVER_K",
    "build_bm25_retriever",
    "build_compression_retriever",
    "build_multi_query_retriever",
    "build_parent_document_retriever",
    "build_ensemble_retriever",
    "get_retriever",
    "get_retrieval_chain",
    "get_naive_retrieval_chain",
    "get_naive_retriever",
    "get_ensemble_retriever",
    "rag_prompt",
    "RAG_TEMPLATE",
]
