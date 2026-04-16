# scripts/test_rag_smoke.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backend.retrieval.hybrid_retriever import HybridRetriever
from src.backend.gemini_client import GeminiClient


def main():
    retriever = HybridRetriever()
    client    = GeminiClient()

    question = "What are the week 10/11 milestones?"

    print(f"Question: {question}\n")

    # Step 1 — retrieve
    results = retriever.retrieve(question, top_k=5)
    print(f"Retrieved {len(results)} chunks")

    # Step 2 — build context string
    context_parts = []
    for r in results:
        title = r["metadata"]["title"]
        url   = r["metadata"].get("url", "")
        text  = r["text"]
        context_parts.append(f"[Source: {title}]\n{url}\n{text}")
    context = "\n\n".join(context_parts)

    # Step 3 — generate answer
    print("\nGenerating answer...\n")
    answer = client.ask_workmate(context, question)
    print(answer)


if __name__ == "__main__":
    main()