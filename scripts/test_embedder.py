import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backend.embedder import GoogleEmbedder


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


def main():
    embedder = GoogleEmbedder()
    print("GoogleEmbedder initialised ✓")

    # --- Test embed_text ---
    vec = embedder.embed_text("The RAG pipeline retrieves context before generating.")
    assert isinstance(vec, list), "embed_text must return a list"
    assert len(vec) == 3072, f"Expected 3072 dims, got {len(vec)}"
    assert isinstance(vec[0], float), "Embedding values must be floats"
    print(f"embed_text ✓  dims={len(vec)}  first_value={vec[0]:.6f}")

    # --- Test embed_query ---
    qvec = embedder.embed_query("What does the RAG pipeline do?")
    assert len(qvec) == 3072
    print(f"embed_query ✓  dims={len(qvec)}")

    # --- Test cosine similarity pairs ---
    pairs = [
        (
            "The RAG pipeline retrieves context before generating.",
            "Retrieval-augmented generation fetches documents first.",
            0.75,   # expect high similarity
        ),
        (
            "The RAG pipeline retrieves context before generating.",
            "The weather in Paris is warm in summer.",
            0.0,    # expect low similarity
        ),
    ]

    print("\nSimilarity pairs:")
    for text_a, text_b, min_expected in pairs:
        vec_a = embedder.embed_text(text_a)
        vec_b = embedder.embed_text(text_b)
        score = cosine_similarity(vec_a, vec_b)
        status = "✓" if score > min_expected else "✗"
        print(f"  {status}  score={score:.4f}  (min={min_expected})  '{text_a[:40]}...'")
        assert score > min_expected, f"Expected similarity > {min_expected}, got {score:.4f}"

    # --- Test embed_batch ---
    texts = [
        "Week 10 milestones",
        "RAG pipeline accuracy",
        "Google Gemini embedding model",
    ]
    batch_vecs = embedder.embed_batch(texts)
    assert len(batch_vecs) == 3, f"Expected 3 embeddings, got {len(batch_vecs)}"
    assert all(len(v) == 3072 for v in batch_vecs), "All batch embeddings must be 768 dims"
    print(f"\nembed_batch ✓  count={len(batch_vecs)}  dims={len(batch_vecs[0])}")

    print("\nAll tests passed ✓")


if __name__ == "__main__":
    main()