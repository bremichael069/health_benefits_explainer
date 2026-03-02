"""Tavily public insurance search tool."""
import os
from typing import List, TypedDict

from tavily import TavilyClient
from langchain_core.tools import tool

import lib


class PublicSearchResult(TypedDict):
    query: str
    answer: str
    sources: List[str]
    confidence: str


def _get_tavily_client() -> TavilyClient:
    return TavilyClient(api_key=lib.get_settings().tavily_api_key)


@tool
def public_insurance_search(
    query: str,
    location: str = "Texas",
    year: int = 2026,
) -> PublicSearchResult:
    """Fetch up-to-date, publicly available insurance information when internal RAG knowledge is insufficient."""
    tavily_client = _get_tavily_client()
    enriched_query = (
        f"{query} public insurance programs {location} {year} "
        f"official government or trusted sources"
    )
    results = tavily_client.search(
        query=enriched_query, max_results=5, include_answer=True, include_sources=True
    )
    return {
        "query": enriched_query,
        "answer": (results.get("answer", "") or "").strip(),
        "sources": results.get("sources", []) or [],
        "confidence": "medium" if results.get("answer") else "low",
    }


tools = [public_insurance_search]
