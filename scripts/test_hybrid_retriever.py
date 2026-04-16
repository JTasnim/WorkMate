import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backend.retrieval.hybrid_retriever import HybridRetriever
from src.backend.chroma_manager import ChromaManager
from src.backend.load.bm25_manager import BM25Manager


def run_comparison(query: str, retriever: HybridRetriever,
                   chroma: ChromaManager, bm25: BM25Manager) -> None:
    print(f"\n{'='*60}")
    print(f"Query: '{query}'")
    print(f"{'='*60}")

    # Individual results
    vector_results = chroma.query(query, n_results=5)
    bm25_results   = bm25.search(query, n_results=5)
    hybrid_results = retriever.retrieve(query, top_k=5)

    print("\n-- Vector only (top 5) --")
    for r in vector_results:
        print(f"  [score={r['score']}]  '{r['metadata']['title']}'")

    print("\n-- BM25 only (top 5) --")
    if not bm25_results:
        print("  No keyword matches")
    for r in bm25_results:
        print(f"  [score={r['score']}]  '{r['metadata']['title']}'")

    print("\n-- Hybrid RRF (top 5) --")
    for r in hybrid_results:
        v = f"v={r['vector_rank']}" if r['vector_rank'] else "v=--"
        b = f"b={r['bm25_rank']}"   if r['bm25_rank']   else "b=--"
        both = " ★" if r['vector_rank'] and r['bm25_rank'] else ""
        print(f"  [rrf={r['rrf_score']:.5f}  {v}  {b}]{both}  '{r['metadata']['title']}'")

    # Agreement analysis
    v_titles = {r["metadata"]["title"] for r in vector_results}
    b_titles  = {r["metadata"]["title"] for r in bm25_results}
    h_titles  = {r["metadata"]["title"] for r in hybrid_results}
    promoted  = h_titles - v_titles - b_titles

    print(f"\n-- Agreement --")
    print(f"  In both        : {v_titles & b_titles or 'none'}")
    print(f"  Hybrid promoted: {promoted or 'none'}")


def main():
    BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "workmate_db/bm25_index.pkl")

    print("Loading services...")
    retriever = HybridRetriever()
    chroma    = ChromaManager()
    bm25      = BM25Manager()
    bm25.load(BM25_INDEX_PATH)

    # ── Test 1: semantic query — BM25 should miss ──────────────
    run_comparison(
        "What are the goals we are working towards this week?",
        retriever, chroma, bm25
    )

    # ── Test 2: keyword query — vector may miss ────────────────
    run_comparison(
        "RAG pipeline BM25 accuracy",
        retriever, chroma, bm25
    )

    # ── Test 3: mixed — both should agree ─────────────────────
    run_comparison(
        "What milestones are due for the Q4 RAG improvements?",
        retriever, chroma, bm25
    )

    # ── Test 4: RRF math check ─────────────────────────────────
    print(f"\n{'='*60}")
    print("RRF math check (k=60)")
    print(f"{'='*60}")
    print(f"  Rank 1 only      : {1/(60+1):.5f}")
    print(f"  Rank 1 + Rank 1  : {1/(60+1) + 1/(60+1):.5f}  ← both lists agree")
    print(f"  Rank 1 + Rank 5  : {1/(60+1) + 1/(60+5):.5f}")
    print(f"  Rank 5 + Rank 5  : {1/(60+5) + 1/(60+5):.5f}  ← still beats rank 1 alone")
    print(f"\nConclusion: agreement across methods always beats single-method dominance")

    print("\nAll tests complete ✓")


if __name__ == "__main__":
    main()