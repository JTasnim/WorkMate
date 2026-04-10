import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backend.chroma_manager import ChromaManager


def main():
    manager = ChromaManager()

    # ─────────────────────────────────────────
    # Reset first so the test is always clean
    # ─────────────────────────────────────────
    manager.reset()
    assert manager.count() == 0, "Collection should be empty after reset"
    print("reset() ✓  count=0")

    # ─────────────────────────────────────────
    # add_documents — insert 5 test chunks
    # ─────────────────────────────────────────
    test_chunks = [
        {
            "id": "page_001_chunk_0",
            "text": "The RAG pipeline retrieves Notion pages before generating an answer.",
            "metadata": {"parent_id": "page_001", "title": "RAG Pipeline Overview"},
        },
        {
            "id": "page_001_chunk_1",
            "text": "Hybrid retrieval combines vector search with BM25 keyword matching.",
            "metadata": {"parent_id": "page_001", "title": "RAG Pipeline Overview"},
        },
        {
            "id": "page_002_chunk_0",
            "text": "Week 10/11 milestones: RAG accuracy target set for Q4.",
            "metadata": {"parent_id": "page_002", "title": "Week 10/11 Milestones"},
        },
        {
            "id": "page_002_chunk_1",
            "text": "The BM25 index is saved to workmate_db/bm25_index.pkl after ingestion.",
            "metadata": {"parent_id": "page_002", "title": "Week 10/11 Milestones"},
        },
        {
            "id": "page_003_chunk_0",
            "text": "Google Gemini generates the final answer from the retrieved context.",
            "metadata": {"parent_id": "page_003", "title": "Gemini Integration"},
        },
    ]

    manager.add_documents(test_chunks)
    assert manager.count() == 5, f"Expected 5 chunks, got {manager.count()}"
    print(f"add_documents() ✓  count={manager.count()}")

    # ─────────────────────────────────────────
    # query — semantic search
    # ─────────────────────────────────────────
    results = manager.query("How does retrieval work in the pipeline?", n_results=3)

    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    assert all("id" in r for r in results), "Each result must have an id"
    assert all("text" in r for r in results), "Each result must have text"
    assert all("score" in r for r in results), "Each result must have a score"
    assert results[0]["score"] >= results[-1]["score"], "Results must be sorted by score descending"

    print(f"\nquery() ✓  top {len(results)} results:")
    for r in results:
        print(f"  [score={r['score']}]  {r['metadata']['title']}  —  '{r['text'][:60]}...'")

    # ─────────────────────────────────────────
    # get_by_parent — fetch siblings
    # ─────────────────────────────────────────
    siblings = manager.get_by_parent("page_001")

    assert len(siblings) == 2, f"Expected 2 chunks for page_001, got {len(siblings)}"
    assert all(s["metadata"]["parent_id"] == "page_001" for s in siblings)

    print(f"\nget_by_parent('page_001') ✓  {len(siblings)} chunks:")
    for s in siblings:
        print(f"  [{s['id']}]  '{s['text'][:60]}...'")

    # ─────────────────────────────────────────
    # upsert — re-adding the same id updates, not duplicates
    # ─────────────────────────────────────────
    updated_chunk = [{
        "id": "page_001_chunk_0",
        "text": "UPDATED: The RAG pipeline retrieves Notion pages before generating.",
        "metadata": {"parent_id": "page_001", "title": "RAG Pipeline Overview"},
    }]
    manager.add_documents(updated_chunk)
    assert manager.count() == 5, f"Upsert should not increase count, got {manager.count()}"

    re_fetched = manager.get_by_parent("page_001")
    updated_texts = [c["text"] for c in re_fetched]
    assert any("UPDATED" in t for t in updated_texts), "Upsert should update existing chunk text"
    print(f"\nupsert ✓  count still={manager.count()} (no duplicate created)")

    # ─────────────────────────────────────────
    # reset — wipes everything
    # ─────────────────────────────────────────
    manager.reset()
    assert manager.count() == 0, "Collection should be empty after reset"
    print(f"\nreset() ✓  count=0")

    print("\nAll assertions passed ✓")


if __name__ == "__main__":
    main()