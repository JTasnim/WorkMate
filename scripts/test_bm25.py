import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backend.load.bm25_manager import BM25Manager

BM25_TEST_PATH = "workmate_db/bm25_test_index.pkl"


def main():
    manager = BM25Manager()

    # ─────────────────────────────────────────
    # Test 1 — build_index
    # ─────────────────────────────────────────
    test_chunks = [
        {
            "id": "page_001_chunk_0",
            "text": "Week 10/11 Milestones\nRAG Pipeline Accuracy targeted for Q4.",
            "metadata": {"parent_id": "page_001", "title": "Week 10/11 Milestones", "url": ""},
        },
        {
            "id": "page_001_chunk_1",
            "text": "Week 10/11 Milestones\nHybrid retrieval combines vector search with BM25.",
            "metadata": {"parent_id": "page_001", "title": "Week 10/11 Milestones", "url": ""},
        },
        {
            "id": "page_002_chunk_0",
            "text": "Findings & Updates\nThe BM25 index catches exact keyword matches.",
            "metadata": {"parent_id": "page_002", "title": "Findings & Updates", "url": ""},
        },
        {
            "id": "page_003_chunk_0",
            "text": "RAG Pipeline Overview\nGemini generates answers from retrieved context.",
            "metadata": {"parent_id": "page_003", "title": "RAG Pipeline Overview", "url": ""},
        },
        {
            "id": "page_004_chunk_0",
            "text": "Gemini Integration\nGoogle Gemini API called with temperature 0.2.",
            "metadata": {"parent_id": "page_004", "title": "Gemini Integration", "url": ""},
        },
    ]

    manager.build_index(test_chunks)
    assert manager.count() == 5, f"Expected 5 docs, got {manager.count()}"
    print(f"build_index() ✓  count={manager.count()}")

    # ─────────────────────────────────────────
    # Test 2 — search keyword match
    # ─────────────────────────────────────────
    results = manager.search("milestones RAG pipeline", n_results=3)

    assert len(results) > 0, "Expected at least 1 result"
    assert results[0]["score"] >= results[-1]["score"], "Results must be sorted by score descending"
    assert all("id" in r for r in results), "Each result must have an id"
    assert all("text" in r for r in results), "Each result must have text"
    assert all("score" in r for r in results), "Each result must have a score"

    print(f"\nsearch('milestones RAG pipeline') ✓  top {len(results)} results:")
    for r in results:
        print(f"  [score={r['score']}]  '{r['metadata']['title']}'  —  '{r['text'][:60]}...'")

    # ─────────────────────────────────────────
    # Test 3 — zero score chunks excluded
    # ─────────────────────────────────────────
    results_narrow = manager.search("temperature gemini", n_results=15)
    titles = [r["metadata"]["title"] for r in results_narrow]

    assert "Gemini Integration" in titles, "Gemini chunk should match 'temperature gemini' query"
    assert "Week 10/11 Milestones" not in titles, "Milestones chunk should not match 'temperature gemini'"
    print(f"\nzero-score exclusion ✓  matched: {titles}")

    # ─────────────────────────────────────────
    # Test 4 — save and load
    # ─────────────────────────────────────────
    manager.save(BM25_TEST_PATH)
    assert os.path.exists(BM25_TEST_PATH), "Pickle file should exist after save()"
    print(f"\nsave() ✓  file exists at {BM25_TEST_PATH}")

    manager2 = BM25Manager()
    manager2.load(BM25_TEST_PATH)
    assert manager2.count() == 5, f"Expected 5 docs after load, got {manager2.count()}"

    results2 = manager2.search("milestones RAG pipeline", n_results=3)
    assert len(results2) > 0, "Search should work after load"
    assert results2[0]["score"] == results[0]["score"], "Scores must be identical after reload"
    print(f"load() ✓  count={manager2.count()}  scores match after reload")

    # ─────────────────────────────────────────
    # Test 5 — rebuild_index
    # ─────────────────────────────────────────
    new_chunk = {
    "id": "page_005_chunk_0",
    "text": "New Upload\nThis is a newly uploaded document about kubernetes deployment.",
    "metadata": {"parent_id": "page_005", "title": "New Upload", "url": ""},
    }
    updated_chunks = test_chunks + [new_chunk]
    manager.rebuild_index(updated_chunks, BM25_TEST_PATH)
    assert manager.count() == 6, f"Expected 6 after rebuild, got {manager.count()}"

    results3 = manager.search("kubernetes", n_results=5)
    assert len(results3) > 0, "Expected at least 1 result for 'kubernetes'"
    assert results3[0]["metadata"]["title"] == "New Upload", \
        f"Expected 'New Upload' as top result, got '{results3[0]['metadata']['title']}'"
    print(f"\nrebuild_index() ✓  count={manager.count()}  new chunk searchable")

    # ─────────────────────────────────────────
    # Cleanup test pickle
    # ─────────────────────────────────────────
    os.remove(BM25_TEST_PATH)
    print(f"\nCleaned up test index ✓")
    print("\nAll tests passed ✓")


if __name__ == "__main__":
    main()