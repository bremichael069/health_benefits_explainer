"""Interactive chat. Loads .env; requires OPENAI_API_KEY, TAVILY_API_KEY, and data PDF."""
from dotenv import load_dotenv
load_dotenv()

from agent import health_agent

def main():
    print("Local Agentic RAG Chat (type 'exit' to quit)")
    while True:
        q = input("\nQuestion: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        state = {
            "question": q,
            "rag_context": "",
            "rag_response": "",
            "tavily_result": "",
            "messages": [],
        }
        result = health_agent.invoke(state)
        answer = result.get("final")
        if not answer:
            msgs = result.get("messages") or []
            answer = getattr(msgs[-1], "content", "") if msgs else ""
        print("\nAnswer:\n", answer or "(no response)")

if __name__ == "__main__":
    main()
