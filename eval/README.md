# Eval: Retriever evaluation with RAGAS and LangSmith

Evaluates each retriever from the **Health Insurance Policy Explainer** notebook and sends traces to LangSmith.

## Retrievers evaluated

- **naive** – Dense vector (Qdrant, k=10)
- **bm25** – BM25 sparse
- **multi_query** – Multi-query expansion over naive retriever
- **parent_document** – Parent-document retriever (child 400/50, parent 2000/200)
- **compression** – Contextual compression with Cohere rerank (optional; needs `COHERE_API_KEY`)
- **ensemble** – Ensemble of the above (optional; only when compression is available)
- **semantic** – Semantic chunker + vector store (optional; needs `langchain-experimental`)

## Setup

1. **Env** (from project root): copy `.env.sample` to `.env` and set at least:
   - `OPENAI_API_KEY` (required for RAG and RAGAS)
   - `LANGCHAIN_API_KEY` (required to see runs in LangSmith)
   - `POLICY_PDF_PATH` or place `data/health_benefits_and_coverage.pdf`
   - Optional: `COHERE_API_KEY` for compression + ensemble

2. **Install**:
   ```bash
   pip install -r requirements.txt
   ```

## Run

From project root:

```bash
python -m eval.eval
```

- Each retriever is run on the same eval questions; each chain invoke is **traced to LangSmith** when `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` are set.
- RAGAS computes **faithfulness**, **context_precision**, and **context_recall** per retriever; results are printed and (where supported) associated with the experiment name in the tracing tool.

## LangSmith

- Set `LANGCHAIN_API_KEY` and optionally `LANGCHAIN_PROJECT` (default: `Health-Policy-Explainer-Eval`).
- In LangSmith you will see:
  - One trace per chain invocation (per question × per retriever).
  - RAGAS evaluation runs when using the experiment name (e.g. `retriever_eval_naive_...`).

## Dataset

- Questions and optional ground truth live in `eval/eval.py` (EVAL_QUESTIONS, EVAL_GROUND_TRUTH).
- You can edit those lists in `eval/eval.py` to add or change eval examples.
