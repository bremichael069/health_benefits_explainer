"""
RAG pipeline aligned with Health Insurance Policy Explainer.ipynb.
Load → split (health_benefits_docs) → vectorstore → naive_retriever, BM25, compression, multi_query,
parent_document → EnsembleRetriever → ensemble_retrieval_chain (default).
"""
from __future__ import annotations

from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

import lib
from rag.loaders import load_pdf
from rag.splitters import split_docs, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
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

# ---- RAG prompt (notebook: RAG_TEMPLATE, rag_prompt) ----
RAG_TEMPLATE = """\
You are a helpful and kind assistant. Use the context provided below to answer the question.

If you do not know the answer, or are unsure, say you don't know.

Query:
{question}

Context:
{context}
"""
rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

# Cached state (notebook naming: raw_docs, health_benefits_docs, vectorstore, naive_retriever, etc.)
_raw_docs = None
_health_benefits_docs = None
_vectorstore = None
_naive_retriever = None
_bm25_retriever = None
_compression_retriever = None
_multi_query_retriever = None
_parent_document_retriever = None
_ensemble_retriever = None
_ensemble_retrieval_chain = None
_naive_retrieval_chain = None


def _ensure_data():
    """Load PDF and create health_benefits_docs (notebook Task 2)."""
    global _raw_docs, _health_benefits_docs
    if _raw_docs is not None:
        return _raw_docs, _health_benefits_docs
    s = lib.get_settings()
    raw_docs = load_pdf(s.policy_pdf_path)
    health_benefits_docs = split_docs(
        raw_docs,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )
    _raw_docs = raw_docs
    _health_benefits_docs = health_benefits_docs
    return _raw_docs, _health_benefits_docs


def _get_vectorstore():
    """Build vectorstore from health_benefits_docs (notebook: vectorstore = QdrantVectorStore.from_documents(...))."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore
    _, health_benefits_docs = _ensure_data()
    embeddings = lib.get_embeddings()
    _vectorstore = build_inmemory_vectorstore(
        documents=health_benefits_docs,
        embeddings=embeddings,
        collection_name="health_benefits_docs",
    )
    return _vectorstore


def get_naive_retriever():
    """Notebook: naive_retriever = vectorstore.as_retriever(search_kwargs={\"k\": 10})."""
    global _naive_retriever
    if _naive_retriever is not None:
        return _naive_retriever
    vs = _get_vectorstore()
    _naive_retriever = make_vector_retriever(vs, k=RETRIEVER_K)
    return _naive_retriever


def get_bm25_retriever():
    """Notebook: bm25_retriever = BM25Retriever.from_documents(raw_docs). Returns None if rank_bm25 not installed."""
    global _bm25_retriever
    if _bm25_retriever is not None:
        return _bm25_retriever
    try:
        raw_docs, _ = _ensure_data()
        _bm25_retriever = build_bm25_retriever(raw_docs)
        return _bm25_retriever
    except Exception:
        return None


def get_compression_retriever():
    """Notebook: compression_retriever (Cohere rerank on naive_retriever). Returns None if Cohere not configured."""
    global _compression_retriever
    if _compression_retriever is not None:
        return _compression_retriever
    if not lib.is_cohere_available():
        return None
    try:
        _compression_retriever = build_compression_retriever(get_naive_retriever())
        return _compression_retriever
    except Exception:
        return None


def get_multi_query_retriever():
    """Notebook: multi_query_retriever = MultiQueryRetriever.from_llm(...). Returns None if not available."""
    global _multi_query_retriever
    if _multi_query_retriever is not None:
        return _multi_query_retriever
    try:
        chat_model = lib.get_rag_llm()
        _multi_query_retriever = build_multi_query_retriever(get_naive_retriever(), chat_model)
        return _multi_query_retriever
    except Exception:
        return None


def get_parent_document_retriever():
    """Notebook: parent_document_retriever. Returns None if not available."""
    global _parent_document_retriever
    if _parent_document_retriever is not None:
        return _parent_document_retriever
    try:
        raw_docs, _ = _ensure_data()
        embeddings = lib.get_embeddings()
        _parent_document_retriever = build_parent_document_retriever(raw_docs, embeddings)
        return _parent_document_retriever
    except Exception:
        return None


def get_ensemble_retriever():
    """Notebook: ensemble_retriever = EnsembleRetriever(retrievers=[...], weights=equal_weighting)."""
    global _ensemble_retriever
    if _ensemble_retriever is not None:
        return _ensemble_retriever
    retrievers = []
    bm25 = get_bm25_retriever()
    if bm25 is not None:
        retrievers.append(bm25)
    retrievers.append(get_naive_retriever())
    parent = get_parent_document_retriever()
    if parent is not None:
        retrievers.append(parent)
    compression = get_compression_retriever()
    if compression is not None:
        retrievers.append(compression)
    multi_query = get_multi_query_retriever()
    if multi_query is not None:
        retrievers.append(multi_query)
    if not retrievers:
        raise RuntimeError("No retrievers available. Install rank_bm25 and ensure PDF is loaded.")
    _ensemble_retriever = build_ensemble_retriever(retrievers)
    return _ensemble_retriever


def _make_retrieval_chain(retriever):
    """Build LCEL chain (notebook: naive_retrieval_chain / ensemble_retrieval_chain)."""
    chat_model = lib.get_rag_llm()
    return (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
    )


def get_retrieval_chain():
    """Return ensemble_retrieval_chain — used by the health benefits agent.

    Ensemble combines retrievers suited to policy Q&A: BM25 (keyword), semantic (naive),
    parent_document (broader context), optional Cohere rerank, optional multi-query expansion.
    """
    global _ensemble_retrieval_chain
    if _ensemble_retrieval_chain is not None:
        return _ensemble_retrieval_chain
    _ensemble_retrieval_chain = _make_retrieval_chain(get_ensemble_retriever())
    return _ensemble_retrieval_chain


def get_naive_retrieval_chain():
    """Return naive_retrieval_chain (notebook Task 4). Used for eval or fallback."""
    global _naive_retrieval_chain
    if _naive_retrieval_chain is not None:
        return _naive_retrieval_chain
    _naive_retrieval_chain = _make_retrieval_chain(get_naive_retriever())
    return _naive_retrieval_chain


def get_retriever():
    """Return the ensemble retriever (for callers that need the retriever only)."""
    return get_ensemble_retriever()
