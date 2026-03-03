# Certification Challenge — Agentic RAG (Vercel-ready)

Runnable code based on **Health Insurance Policy Explainer.ipynb**: LangGraph RAG → optional Tavily → answer. Data: `data/health_benefits_and_coverage.pdf`. Deployable on Vercel.

## Call sequence (which file calls which)

Request flow for a `/chat` request:

1. **`main.py`** (entrypoint)
   - Imports **`lib`** → `lib.enable_langsmith_if_configured()` at startup.
   - On `POST /chat`: imports **`agent`** → calls `health_agent.invoke(state)`.

2. **`agent.py`** (LangGraph agent)
   - Imports **`lib`**, **`rag`**, **`tools`**.
   - Graph runs in order:
     - **rag_node** → calls `rag.get_retrieval_chain()` then `chain.invoke(...)`.
     - **should_continue_node** → calls `lib.get_chat_llm()`.
     - **tavily_node** (if needed) → calls `tools.public_insurance_search.invoke(...)`.
     - **respond_node** → calls `lib.get_chat_llm()` again.

3. **`rag`** package (RAG chain)
   - **`rag/pipeline.py`** → `get_retrieval_chain()` uses:
     - **`rag/loaders.py`** → `load_pdf(path)` (PDF → documents).
     - **`rag/splitters.py`** → `split_docs()` (documents → chunks).
     - **`rag/vectorstores.py`** → `build_inmemory_vectorstore()` (chunks + embeddings → vector store).
     - **`rag/retrievers.py`** → BM25, naive retriever, optional parent_document, compression, multi_query → **`build_ensemble_retriever()`**.
   - Pipeline also calls **`lib.get_settings()`**, **`lib.get_embeddings()`**, **`lib.get_rag_llm()`**, **`lib.is_cohere_available()`**.

4. **`lib.py`**
   - Used by `main.py`, `agent.py`, `rag`, and `tools.py`.
   - Provides: `get_settings()`, `get_chat_llm()`, `get_rag_llm()`, `get_embeddings()`, `enable_langsmith_if_configured()`, `is_cohere_available()`.

5. **`tools.py`**
   - Used by **`agent.py`** (tavily_node).
   - Uses **`lib.get_settings()`** for Tavily API key.
   - Exposes `public_insurance_search` (Tavily search).

**Summary:** `main.py` → `agent.py` → `rag` (loaders → splitters → vectorstores → retrievers → pipeline) and `lib`, `tools`. All app code goes through this path; **`eval/`** and **`scripts/local_chat.py`** use the same `agent` and `rag` when run separately.

## Folder structure

```
├── main.py              # FastAPI app: /health, /chat (entrypoint for local & Vercel)
├── agent.py              # LangGraph: RAG → decide → [Tavily] → respond
├── rag/
│   ├── __init__.py       # Public API
│   ├── loaders.py        # PDF loader
│   ├── splitters.py      # RecursiveCharacterTextSplitter
│   ├── vectorstores.py   # In-memory vector store
│   ├── retrievers.py     # BM25, compression, multi-query, parent-doc, ensemble
│   └── pipeline.py       # Ensemble RAG chain
├── lib.py                # Settings, LLM, embeddings (env-only)
├── tools.py              # Tavily public_insurance_search
├── eval/
│   ├── eval.py           # RAGAS eval + LangSmith (python -m eval.eval)
│   └── README.md
├── notebooks/            # Health Insurance Policy Explainer.ipynb
├── data/                 # Policy PDF
├── scripts/local_chat.py
├── .env.sample
├── vercel.json
└── requirements.txt
```

## Secure deployment (no secrets in code)

- **Secrets** (OpenAI, Tavily, Cohere, LangSmith) come **only from environment variables**.
- **Local:** Copy `.env.sample` to `.env` and fill in keys. **Do not commit `.env`** (add it to `.gitignore`).
- **Vercel:** Set all keys in **Project → Settings → Environment Variables**. They are encrypted and not visible in the repo.
- Never put API keys in source code or in committed files.

## Credentials you need

| Env var | Required | Where to get |
|--------|----------|---------------|
| `OPENAI_API_KEY` | Yes (for /chat) | [OpenAI API keys](https://platform.openai.com/api-keys) |
| `TAVILY_API_KEY` | Yes (for /chat) | [Tavily](https://tavily.com/) |
| `COHERE_API_KEY` | No (rerank) | [Cohere](https://dashboard.cohere.com/) |
| `LANGCHAIN_API_KEY` | No (tracing) | [LangSmith](https://smith.langchain.com/) |

Optional: `LANGCHAIN_PROJECT`, `OPENAI_CHAT_MODEL`, `OPENAI_RAG_MODEL`, `OPENAI_EMBEDDING_MODEL`, `POLICY_PDF_PATH`.

## Dependencies

- **Vercel / production:** `pyproject.toml` lists a **minimal** set (FastAPI, LangChain/LangGraph, OpenAI, Tavily, pypdf) so the serverless bundle stays under the 500 MB limit. The app runs with the **naive (semantic) retriever** only; BM25, Cohere rerank, parent-document, and multi-query are omitted and the ensemble falls back to the single retriever.
- **Local notebook & eval:** For the full notebook and `python -m eval.eval` (RAGAS, all retrievers), install the extra deps:  
  `pip install ragas datasets rank-bm25 langchain-cohere qdrant-client langchain-qdrant langsmith`  
  (or use a separate venv with a full `requirements.txt` if you maintain one).

## Quick start (local)

1. Copy `.env.sample` to `.env` and set `OPENAI_API_KEY` and `TAVILY_API_KEY`.
2. PDF is under `data/health_benefits_and_coverage.pdf` (or set `POLICY_PDF_PATH`).
3. Install and run:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

4. Test:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"question\":\"What does the plan cover for preventive care?\"}"
```

## Deploy on Vercel

1. **Push your code** to a Git repo (GitHub, GitLab, or Bitbucket) if you haven’t already.
2. Go to [vercel.com](https://vercel.com) → **Add New Project** → import your repo.
3. **Environment variables** (Project → Settings → Environment Variables): add  
   `OPENAI_API_KEY`, `TAVILY_API_KEY`. Optional: `COHERE_API_KEY`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`, `POLICY_PDF_PATH`.
4. Ensure **`data/health_benefits_and_coverage.pdf`** is committed so the build has the PDF (or set `POLICY_PDF_PATH` to a path that exists in the build).
5. **Deploy.** Vercel uses `vercel.json` and runs `main.py` as the serverless function.

**Endpoints after deploy** (replace `YOUR_PROJECT` with your Vercel project URL, e.g. `certification-challenge-abc123`):

- **Health check:**  
  `GET https://YOUR_PROJECT.vercel.app/health`  
  Example: `curl https://YOUR_PROJECT.vercel.app/health`

- **Chat (health benefits agent with ensemble retriever):**  
  `POST https://YOUR_PROJECT.vercel.app/chat`  
  Body (JSON): `{"question": "Is MRI covered under my plan?"}`  
  Example:
  ```bash
  curl -X POST https://YOUR_PROJECT.vercel.app/chat \
    -H "Content-Type: application/json" \
    -d "{\"question\":\"Is MRI covered under my plan?\"}"
  ```

You’ll see your exact base URL (e.g. `https://certification-challenge-xxx.vercel.app`) in the Vercel dashboard after the first successful deploy.

## Components

- **RAG (health benefits agent):** Uses the **ensemble retriever** (not a single retriever). Combination tuned for policy Q&A:
  - **BM25** — keyword match (e.g. deductible, copay, MRI).
  - **Naive (semantic)** — vector similarity, k=10.
  - **Parent document** — when available, returns broader context around a hit.
  - **Cohere compression** — when `COHERE_API_KEY` is set, reranks results.
  - **Multi-query** — when available, expands the question into multiple phrasings.
  Results are merged with equal-weight reciprocal-rank fusion.
- **Notebook alignment:** Matches **Health Insurance Policy Explainer.ipynb**: loaders (PyPDFLoader), splitters (RecursiveCharacterTextSplitter 500/50; parent/child 2000/200, 400/50).
- **Data loading:** `rag.load_pdf` → PDF from `data/` or `POLICY_PDF_PATH`.
- **Splitting:** `RecursiveCharacterTextSplitter` (chunk_size=500, chunk_overlap=50).
- **Vector store:** In-memory (no external DB).
- **Agent:** LangGraph (RAG → decide → optional Tavily → respond).
- **Tools:** Tavily `public_insurance_search`.
- **Eval:** `python -m eval.eval` — RAGAS (faithfulness, context precision/recall); LangSmith when `LANGCHAIN_API_KEY` is set.

**Optional dependencies (local only):** `rank_bm25`, `langchain-cohere`, `qdrant-client`, `langchain-qdrant`, `ragas`, `datasets`, `langsmith` — used for BM25, Cohere rerank, parent-document, multi-query, and eval. Not included in the minimal Vercel bundle.

## Optional env vars

- `OPENAI_CHAT_MODEL` (default: `gpt-4o`)
- `OPENAI_RAG_MODEL` (default: `gpt-4o-mini`)
- `OPENAI_EMBEDDING_MODEL` (default: `text-embedding-3-large`)
- `POLICY_PDF_PATH` (default: `data/health_benefits_and_coverage.pdf`)
- `COHERE_API_KEY` — enables rerank in RAG
- `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` — LangSmith tracing
