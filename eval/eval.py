"""
Eval: dataset, retriever chain builders, and RAGAS run.
Run with: python -m eval.eval
Set LANGCHAIN_API_KEY (and LANGCHAIN_PROJECT) to see traces in LangSmith.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from operator import itemgetter
from typing import List, Optional, Tuple

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import lib
import rag

# ---- Dataset (notebook evaluation questions) ----
EVAL_QUESTIONS = [
    "Is MRI covered under my plan?",
    "What documents are required to file a claim?",
    "What happens if I go out of network?",
    "Why was claim X denied?",
    "What is the overall deductible?",
]
EVAL_GROUND_TRUTH = [
    "Imaging services such as MRIs are generally covered with copayment and coinsurance as described in the plan.",
    "Complete information as part of your claim, appeal, or grievance; consult plan instructions or contact information.",
    "Out-of-network care costs more; higher coinsurance, possible balance billing; check plan for urgent care preauthorization.",
    "Denials can be due to missing information, non-covered services, or not following plan procedures; see plan documents for appeal.",
    "The overall deductible is specified in the Summary of Benefits and Coverage for the coverage period.",
]


def get_eval_dataset():
    return [{"question": q, "ground_truth": g} for q, g in zip(EVAL_QUESTIONS, EVAL_GROUND_TRUTH)]


# ---- Chain building (all retrievers from notebook) ----
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

RAG_TEMPLATE = """\
You are a helpful and kind assistant. Use the context provided below to answer the question.

If you do not know the answer, or are unsure, say you don't know.

Query:
{question}

Context:
{context}
"""
_rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
_state: Optional[dict] = None


def _get_state() -> dict:
    global _state
    if _state is not None:
        return _state
    s = lib.get_settings()
    pdf_path = s.policy_pdf_path
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Policy PDF not found at '{pdf_path}'. Set POLICY_PDF_PATH or add the PDF.")
    raw_docs = rag.load_pdf(pdf_path)
    chunks = rag.split_docs(raw_docs, chunk_size=500, chunk_overlap=50)
    embeddings = lib.get_embeddings()
    chat_model = lib.get_rag_llm()

    vectorstore = rag.build_inmemory_qdrant_vectorstore(
        documents=chunks, embeddings=embeddings, collection_name="health_benefits_eval"
    )
    naive_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    from langchain_community.retrievers import BM25Retriever
    bm25_retriever = BM25Retriever.from_documents(chunks)

    try:
        from langchain.retrievers.multi_query import MultiQueryRetriever
    except ImportError:
        from langchain_community.retrievers import MultiQueryRetriever
    multi_query_retriever = MultiQueryRetriever.from_llm(retriever=naive_retriever, llm=chat_model)

    try:
        from langchain.retrievers import ParentDocumentRetriever
        from langchain.storage import InMemoryStore
    except ImportError:
        from langchain_community.retrievers import ParentDocumentRetriever
        from langchain_community.storage import InMemoryStore
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_qdrant import QdrantVectorStore as QVS
    from qdrant_client import QdrantClient, models

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    client = QdrantClient(location=":memory:")
    emb_dim = 3072 if "large" in (os.environ.get("OPENAI_EMBEDDING_MODEL") or "").lower() else 1536
    client.create_collection(
        collection_name="eval_parent_child",
        vectors_config=models.VectorParams(size=emb_dim, distance=models.Distance.COSINE),
    )
    parent_vectorstore = QVS(collection_name="eval_parent_child", embedding=embeddings, client=client)
    store = InMemoryStore()
    parent_document_retriever = ParentDocumentRetriever(
        vectorstore=parent_vectorstore, docstore=store,
        child_splitter=child_splitter, parent_splitter=parent_splitter,
    )
    parent_document_retriever.add_documents(raw_docs, ids=None)

    compression_retriever = None
    if os.environ.get("COHERE_API_KEY"):
        try:
            try:
                from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
            except ImportError:
                from langchain_community.retrievers import ContextualCompressionRetriever
            from langchain_cohere import CohereRerank
            compressor = CohereRerank(model="rerank-v3.5")
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=naive_retriever
            )
        except Exception:
            pass

    ensemble_retriever = None
    if compression_retriever is not None:
        try:
            from langchain.retrievers import EnsembleRetriever
        except ImportError:
            from langchain_community.retrievers import EnsembleRetriever
        retriever_list = [bm25_retriever, naive_retriever, parent_document_retriever, compression_retriever, multi_query_retriever]
        ensemble_retriever = EnsembleRetriever(retrievers=retriever_list, weights=[1.0 / len(retriever_list)] * len(retriever_list))

    semantic_retriever = None
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        semantic_chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
        semantic_docs = semantic_chunker.split_documents(raw_docs)
        semantic_vectorstore = QVS.from_documents(
            semantic_docs, embeddings, location=":memory:", collection_name="eval_semantic"
        )
        semantic_retriever = semantic_vectorstore.as_retriever(search_kwargs={"k": 10})
    except Exception:
        pass

    _state = {
        "chat_model": chat_model,
        "naive_retriever": naive_retriever,
        "bm25_retriever": bm25_retriever,
        "multi_query_retriever": multi_query_retriever,
        "parent_document_retriever": parent_document_retriever,
        "compression_retriever": compression_retriever,
        "ensemble_retriever": ensemble_retriever,
        "semantic_retriever": semantic_retriever,
    }
    return _state


def _make_chain(retriever):
    state = _get_state()
    chain = (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": _rag_prompt | state["chat_model"], "context": itemgetter("context")}
    )
    return chain


def get_retriever_names() -> List[str]:
    state = _get_state()
    names = ["naive", "bm25", "multi_query", "parent_document"]
    if state.get("compression_retriever") is not None:
        names.append("compression")
    if state.get("ensemble_retriever") is not None:
        names.append("ensemble")
    if state.get("semantic_retriever") is not None:
        names.append("semantic")
    return names


def get_chain_for_retriever(name: str):
    state = _get_state()
    m = {
        "naive": state["naive_retriever"],
        "bm25": state["bm25_retriever"],
        "multi_query": state["multi_query_retriever"],
        "parent_document": state["parent_document_retriever"],
        "compression": state.get("compression_retriever"),
        "ensemble": state.get("ensemble_retriever"),
        "semantic": state.get("semantic_retriever"),
    }
    retriever = m.get(name)
    if retriever is None:
        raise ValueError(f"Unknown or unavailable retriever: {name}. Available: {get_retriever_names()}")
    return _make_chain(retriever), retriever


def invoke_and_collect(chain, question: str) -> Tuple[List[str], str]:
    out = chain.invoke({"question": question})
    context_docs = out.get("context") or []
    contexts = [
        getattr(d, "page_content", "") or str(d)
        for d in context_docs
        if (getattr(d, "page_content", "") or str(d)).strip()
    ]
    resp = out.get("response")
    answer = resp.content if hasattr(resp, "content") else str(resp)
    return contexts, answer


# ---- Run RAGAS ----
def _setup_langsmith():
    api_key = os.environ.get("LANGCHAIN_API_KEY", "").strip()
    if api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        if not os.environ.get("LANGCHAIN_PROJECT"):
            os.environ["LANGCHAIN_PROJECT"] = "Health-Policy-Explainer-Eval"
        return True
    return False


def main():
    langsmith_on = _setup_langsmith()
    if not langsmith_on:
        print("LANGCHAIN_API_KEY not set. Set it to see eval runs in LangSmith.")

    try:
        from datasets import Dataset
        from ragas import evaluate
        _ragas_metrics = None
        try:
            from ragas.metrics import faithfulness, context_precision, context_recall
            _ragas_metrics = [faithfulness, context_precision, context_recall]
        except ImportError:
            pass
    except ImportError as e:
        print("Install eval deps: pip install ragas datasets")
        raise SystemExit(1) from e

    eval_data = get_eval_dataset()
    questions = [d["question"] for d in eval_data]
    ground_truths = [d["ground_truth"] for d in eval_data]
    retriever_names = get_retriever_names()

    print(f"Evaluating {len(retriever_names)} retrievers on {len(questions)} questions.")
    print("Retrievers:", retriever_names)
    if langsmith_on:
        print("LangSmith tracing: ON (project:", os.environ.get("LANGCHAIN_PROJECT", ""), ")")

    all_results = {}
    for retriever_name in retriever_names:
        print(f"\n--- Retriever: {retriever_name} ---")
        try:
            chain, _ = get_chain_for_retriever(retriever_name)
        except Exception as e:
            print(f"  Skip: {e}")
            continue

        rows_question, rows_contexts, rows_answer, rows_ground_truth = [], [], [], []
        for i, (q, gt) in enumerate(zip(questions, ground_truths)):
            try:
                contexts, answer = invoke_and_collect(chain, q)
                rows_question.append(q)
                rows_contexts.append(contexts)
                rows_answer.append(answer)
                rows_ground_truth.append(gt)
            except Exception as e:
                print(f"  Error on Q{i+1}: {e}")
                rows_question.append(q)
                rows_contexts.append([])
                rows_answer.append("")
                rows_ground_truth.append(gt)

        dataset = Dataset.from_dict({
            "question": rows_question,
            "contexts": rows_contexts,
            "answer": rows_answer,
            "ground_truth": rows_ground_truth,
        })
        experiment_name = f"retriever_eval_{retriever_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"
        try:
            result = evaluate(dataset, metrics=_ragas_metrics, experiment_name=experiment_name, show_progress=True)
        except Exception as e:
            print(f"  RAGAS evaluate failed: {e}")
            all_results[retriever_name] = {"error": str(e)}
            continue

        if hasattr(result, "to_pandas"):
            try:
                print(result.to_pandas().to_string())
            except Exception:
                pass
        scores = {}
        if hasattr(result, "scores"):
            scores = dict(result.scores)
        elif hasattr(result, "__iter__") and not isinstance(result, (str, list)):
            try:
                scores = dict(result)
            except Exception:
                pass
        all_results[retriever_name] = scores
        for k, v in scores.items():
            if hasattr(v, "item"):
                v = v.item()
            print(f"  {k}: {v}")

    print("\n========== Summary ==========")
    for name, res in all_results.items():
        if "error" in res:
            print(f"{name}: ERROR - {res['error']}")
        else:
            print(f"{name}: {res}")
    return all_results


if __name__ == "__main__":
    main()
