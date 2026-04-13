import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backend.chroma_manager import ChromaManager
from src.backend.load.bm25_manager import BM25Manager

BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "workmate_db/bm25_index.pkl")
TOP_K = 5


def run_query(query: str, chroma: ChromaManager, bm25: BM25Manager) -> None:
    print(f"\n{'='*65}")
    print(f"Query: '{query}'")
    print(f"{'='*65}")

    # ── ChromaDB vector search ─────────────────────────────────
    vector_results = chroma.query(query, n_results=TOP_K)

    print(f"\n--- ChromaDB (vector)  top {TOP_K} ---")
    if not vector_results:
        print("  No results returned")
    for r in vector_results:
        print(f"  [score={r['score']}]  '{r['metadata']['title']}'")
        print(f"    {r['text'][:100].strip()}...")

    # ── BM25 keyword search ────────────────────────────────────
    bm25_results = bm25.search(query, n_results=TOP_K)

    print(f"\n--- BM25 (keyword)     top {TOP_K} ---")
    if not bm25_results:
        print("  No results — zero keyword overlap with any chunk")
    for r in bm25_results:
        print(f"  [score={r['score']}]  '{r['metadata']['title']}'")
        print(f"    {r['text'][:100].strip()}...")

    # ── Agreement analysis ─────────────────────────────────────
    vector_titles = {r["metadata"]["title"] for r in vector_results}
    bm25_titles   = {r["metadata"]["title"] for r in bm25_results}
    both  = vector_titles & bm25_titles
    v_only = vector_titles - bm25_titles
    b_only = bm25_titles - vector_titles

    print(f"\n--- Agreement ---")
    print(f"  In both        : {both if both else 'none'}")
    print(f"  Vector only    : {v_only if v_only else 'none'}")
    print(f"  BM25 only      : {b_only if b_only else 'none'}")


def main():
    parser = argparse.ArgumentParser(description="Test end-to-end retrieval")
    parser.add_argument("query", nargs="?", default=None, help="Optional single query to test")
    args = parser.parse_args()

    # ── Load services ──────────────────────────────────────────
    print("Loading ChromaDB...")
    chroma = ChromaManager()
    print(f"  {chroma.count()} chunks loaded")

    print("Loading BM25 index...")
    bm25 = BM25Manager()
    bm25.load(BM25_INDEX_PATH)
    print(f"  {bm25.count()} documents loaded")

    # ── Run queries ────────────────────────────────────────────
    if args.query:
        run_query(args.query, chroma, bm25)
    else:
        queries = [
            # Semantic — no exact keyword match expected
            "What are the goals we are working towards this week?",
            # Keyword — exact page name or acronym
            "RAG pipeline BM25 accuracy",
            # Mixed — both semantic and keyword signals
            "What milestones are due for the Q4 RAG improvements?",
        ]
        for q in queries:
            run_query(q, chroma, bm25)

    print(f"\n{'='*65}")
    print("Retrieval test complete.")
    print("Note: BM25 scores and cosine scores are on different scales.")
    print("Only the ranking matters — not the absolute numbers.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()