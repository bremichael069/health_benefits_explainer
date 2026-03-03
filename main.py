"""
FastAPI app: /health, /chat. Entrypoint for local (uvicorn) and Vercel.
LangSmith tracing enabled at startup when LANGCHAIN_API_KEY is set.
Secrets from environment only (Vercel Environment Variables in production).
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import lib


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: enable LangSmith if configured. No secrets in code."""
    lib.enable_langsmith_if_configured()
    yield


app = FastAPI(title="Certification Challenge - Agentic RAG", lifespan=lifespan)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    used_public_search: bool = False


@app.get("/")
def root():
    """Root: point to health and chat."""
    return {"message": "Certification Challenge - Agentic RAG", "health": "/health", "chat": "POST /chat"}


@app.get("/health")
def health():
    """No keys required. Use for Vercel health checks."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    try:
        from agent import health_agent
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Agent not available. Check OPENAI_API_KEY, TAVILY_API_KEY, and POLICY_PDF_PATH.",
        )

    state = {
        "question": question,
        "rag_context": "",
        "rag_response": "",
        "tavily_result": "",
        "messages": [],
    }

    try:
        result = health_agent.invoke(state)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail="Policy PDF not found. Add data/health_benefits_and_coverage.pdf or set POLICY_PDF_PATH.",
        ) from e
    except RuntimeError as e:
        if "API_KEY" in str(e) or "TAVILY" in str(e):
            raise HTTPException(
                status_code=503,
                detail="Missing API keys. Set OPENAI_API_KEY and TAVILY_API_KEY in Vercel.",
            ) from e
        raise HTTPException(status_code=503, detail=str(e)) from e

    answer = result.get("final")
    if not answer:
        msgs = result.get("messages") or []
        answer = getattr(msgs[-1], "content", "") if msgs else ""

    return ChatResponse(
        answer=answer or "",
        used_public_search=bool(result.get("tavily_result")),
    )
