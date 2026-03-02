"""LangGraph agent: RAG → decide → optional Tavily → respond."""
from typing import Literal, List, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

import lib
import rag
import tools

# ---- State ----
class AgentState(TypedDict, total=False):
    question: str
    rag_context: str
    rag_response: str
    tavily_result: str
    messages: List[BaseMessage]
    final: str

# ---- Prompts ----
CONTINUE_PROMPT = """Given the user question and the RAG answer from internal documents, decide if we need to search the public web (Tavily).
Use Tavily only if: the RAG answer says it doesn't know, is unsure, or the question clearly needs current/public info (e.g. latest rates, recent policy changes).
Reply with exactly one word: continue (need Tavily) or end (RAG answer is sufficient).

User question: {question}
RAG answer: {rag_response}

Your one-word reply:"""

RESPOND_PROMPT = """You are an insurance and health benefits assistant. Be faithful to sources.

User question: {question}

Answer from internal policy documents (RAG):
{rag_response}
{tavily_section}

Provide a clear, helpful final answer. If you used both RAG and web search, say so and combine only when consistent."""

# ---- Nodes ----
def rag_node(state: AgentState) -> dict:
    chain = rag.get_retrieval_chain()
    out = chain.invoke({"question": state["question"]})
    context_docs = out.get("context", []) or []
    context = "\n\n".join(
        getattr(d, "page_content", "") for d in context_docs if getattr(d, "page_content", "").strip()
    )
    resp = out.get("response")
    response_text = resp.content if hasattr(resp, "content") else str(resp)
    return {"rag_context": context, "rag_response": response_text}


def should_continue_node(state: AgentState) -> dict:
    llm = lib.get_chat_llm()
    prompt = CONTINUE_PROMPT.format(
        question=state["question"], rag_response=state.get("rag_response", "")
    )
    msg = llm.invoke([HumanMessage(content=prompt)])
    text = (msg.content or "").strip().lower() if hasattr(msg, "content") else str(msg).lower()
    decision = "continue" if "continue" in text else "end"
    messages = (state.get("messages") or []) + [AIMessage(content=decision)]
    return {"messages": messages}


def should_continue_route(state: AgentState) -> Literal["tavily", "respond"]:
    last = (state.get("messages") or [])[-1] if state.get("messages") else None
    if last and hasattr(last, "content") and "continue" in ((last.content or "").lower()):
        return "tavily"
    return "respond"


def tavily_node(state: AgentState) -> dict:
    result = tools.public_insurance_search.invoke({"query": state["question"]})
    text = result.get("answer", "") or str(result)
    return {"tavily_result": text}


def respond_node(state: AgentState) -> dict:
    llm = lib.get_chat_llm()
    tavily_section = ""
    if state.get("tavily_result"):
        tavily_section = "\n\nAdditional public/web information:\n" + state["tavily_result"]
    prompt = RESPOND_PROMPT.format(
        question=state["question"],
        rag_response=state.get("rag_response", ""),
        tavily_section=tavily_section,
    )
    msg = llm.invoke([HumanMessage(content=prompt)])
    final = msg.content if hasattr(msg, "content") else str(msg)
    messages = (state.get("messages") or []) + [AIMessage(content=final)]
    return {"messages": messages, "final": final}


# ---- Graph ----
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("rag", rag_node)
    graph.add_node("should_continue", should_continue_node)
    graph.add_node("tavily", tavily_node)
    graph.add_node("respond", respond_node)
    graph.add_edge(START, "rag")
    graph.add_edge("rag", "should_continue")
    graph.add_conditional_edges(
        "should_continue", should_continue_route, {"tavily": "tavily", "respond": "respond"}
    )
    graph.add_edge("tavily", "respond")
    graph.add_edge("respond", END)
    return graph.compile()


health_agent = build_graph()
