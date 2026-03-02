"""Retrievers (notebook-aligned: naive, BM25, compression, multi_query, parent_document, ensemble)."""
from __future__ import annotations

from typing import List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

RETRIEVER_K = 10


def make_vector_retriever(vectorstore, k: int = RETRIEVER_K):
    """Return a retriever from a vector store with top-k."""
    return vectorstore.as_retriever(search_kwargs={"k": k})


def build_bm25_retriever(documents):
    """Build BM25Retriever from documents (notebook: bm25_retriever)."""
    from langchain_community.retrievers import BM25Retriever
    return BM25Retriever.from_documents(documents)


def build_compression_retriever(base_retriever, model: str = "rerank-v3.5"):
    """Wrap base retriever with CohereRerank (notebook: compression_retriever)."""
    try:
        from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
    except ImportError:
        try:
            from langchain_community.retrievers import ContextualCompressionRetriever
        except ImportError:
            raise ImportError(
                "ContextualCompressionRetriever not found. Install langchain or langchain_community with retrievers support."
            )
    from langchain_cohere import CohereRerank
    compressor = CohereRerank(model=model)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )


def build_multi_query_retriever(retriever, llm):
    """Build MultiQueryRetriever (notebook: multi_query_retriever). Raises ImportError if not available."""
    try:
        from langchain.retrievers.multi_query import MultiQueryRetriever
    except ImportError:
        from langchain_community.retrievers import MultiQueryRetriever
    return MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)


def build_parent_document_retriever(raw_docs, embeddings):
    """
    Build ParentDocumentRetriever (notebook: parent_document_retriever).
    Raises ImportError if ParentDocumentRetriever/InMemoryStore not available.
    """
    from langchain_core.vectorstores import InMemoryVectorStore
    from rag.splitters import get_parent_splitter, get_child_splitter
    try:
        from langchain.retrievers import ParentDocumentRetriever
        from langchain.storage import InMemoryStore
    except ImportError:
        from langchain_community.retrievers import ParentDocumentRetriever
        from langchain_community.storage import InMemoryStore
    parent_splitter = get_parent_splitter()
    child_splitter = get_child_splitter()
    vectorstore = InMemoryVectorStore(embedding=embeddings)
    docstore = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    retriever.add_documents(raw_docs, ids=None)
    return retriever


def _merge_documents_reciprocal_rank(
    all_docs: List[List[Document]], weights: Optional[List[float]] = None
) -> List[Document]:
    """Merge and rank documents using reciprocal rank fusion (RRF). Dedupe by page_content."""
    if weights is None:
        weights = [1.0 / len(all_docs)] * len(all_docs)
    k = 60
    doc_scores: List[tuple] = []
    for retriever_docs, w in zip(all_docs, weights):
        for rank, doc in enumerate(retriever_docs, start=1):
            rrf = w * (1.0 / (k + rank))
            doc_scores.append((doc, rrf))
    agg: dict = {}
    for doc, rrf in doc_scores:
        key = doc.page_content
        if key not in agg:
            agg[key] = (doc, 0.0)
        agg[key] = (agg[key][0], agg[key][1] + rrf)
    return [d for d, _ in sorted(agg.values(), key=lambda x: -x[1])]


class _FallbackEnsembleRetriever(BaseRetriever):
    """Combines multiple retrievers with optional weights (RRF)."""

    retrievers: List[BaseRetriever]
    weights: Optional[List[float]] = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        all_docs = []
        for r in self.retrievers:
            docs = r.invoke(query)
            if isinstance(docs, list):
                all_docs.append(docs)
            else:
                all_docs.append(list(docs) if hasattr(docs, "__iter__") else [])
        return _merge_documents_reciprocal_rank(all_docs, self.weights)


def build_ensemble_retriever(retrievers: List[BaseRetriever], weights=None) -> BaseRetriever:
    """Build EnsembleRetriever (notebook: ensemble_retriever). Uses LangChain or fallback RRF."""
    if weights is None:
        weights = [1.0 / len(retrievers)] * len(retrievers)
    try:
        from langchain.retrievers import EnsembleRetriever
        return EnsembleRetriever(retrievers=retrievers, weights=weights)
    except ImportError:
        try:
            from langchain_community.retrievers import EnsembleRetriever
            return EnsembleRetriever(retrievers=retrievers, weights=weights)
        except ImportError:
            return _FallbackEnsembleRetriever(retrievers=retrievers, weights=weights)
