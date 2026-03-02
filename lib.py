"""
Configuration and LLM/embedding helpers.
All secrets from environment only — never hardcode keys (safe for public/Vercel deploy).
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """All config from env; used at runtime, not at import."""
    openai_api_key: str
    tavily_api_key: str
    openai_chat_model: str
    openai_rag_model: str
    openai_embedding_model: str
    policy_pdf_path: str
    cohere_api_key: str
    langsmith_api_key: str
    langsmith_project: str


def get_settings() -> Settings:
    """Load settings from environment. Required: OPENAI_API_KEY, TAVILY_API_KEY."""
    openai_api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    tavily_api_key = (os.environ.get("TAVILY_API_KEY") or "").strip()
    if not openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Set it in .env (local) or Vercel Environment Variables.")
    if not tavily_api_key:
        raise RuntimeError("Missing TAVILY_API_KEY. Set it in .env (local) or Vercel Environment Variables.")

    return Settings(
        openai_api_key=openai_api_key,
        tavily_api_key=tavily_api_key,
        openai_chat_model=(os.environ.get("OPENAI_CHAT_MODEL") or "gpt-4o").strip() or "gpt-4o",
        openai_rag_model=(os.environ.get("OPENAI_RAG_MODEL") or "gpt-4o-mini").strip() or "gpt-4o-mini",
        openai_embedding_model=(os.environ.get("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-large").strip() or "text-embedding-3-large",
        policy_pdf_path=(os.environ.get("POLICY_PDF_PATH") or "data/health_benefits_and_coverage.pdf").strip() or "data/health_benefits_and_coverage.pdf",
        cohere_api_key=(os.environ.get("COHERE_API_KEY") or "").strip(),
        langsmith_api_key=(os.environ.get("LANGCHAIN_API_KEY") or "").strip(),
        langsmith_project=(os.environ.get("LANGCHAIN_PROJECT") or "health-policy-explainer").strip() or "health-policy-explainer",
    )


def is_cohere_available() -> bool:
    return bool((os.environ.get("COHERE_API_KEY") or "").strip())


def enable_langsmith_if_configured() -> bool:
    """Turn on LangSmith tracing when LANGCHAIN_API_KEY is set. Call once at app startup."""
    api_key = (os.environ.get("LANGCHAIN_API_KEY") or "").strip()
    if not api_key:
        return False
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if not os.environ.get("LANGCHAIN_PROJECT"):
        os.environ["LANGCHAIN_PROJECT"] = os.environ.get("LANGCHAIN_PROJECT", "health-policy-explainer")
    return True


def get_chat_llm():
    from langchain_openai import ChatOpenAI
    s = get_settings()
    return ChatOpenAI(model=s.openai_chat_model)


def get_rag_llm():
    from langchain_openai import ChatOpenAI
    s = get_settings()
    return ChatOpenAI(model=s.openai_rag_model)


def get_embeddings():
    from langchain_openai import OpenAIEmbeddings
    s = get_settings()
    return OpenAIEmbeddings(model=s.openai_embedding_model)
