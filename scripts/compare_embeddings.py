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

    # ─────────────────────────────────────────
    # embed_text — stores a document passage
    # ─────────────────────────────────────────
    doc = "The RAG pipeline retrieves context from Notion before generating an answer."
    text_vec = embedder.embed_text(doc)

    print("=== embed_text ===")
    print(f"Input  : '{doc}'")
    print(f"Type   : {type(text_vec)}")
    print(f"Dims   : {len(text_vec)}")
    print(f"First 5: {[round(v, 6) for v in text_vec[:5]]}")
    print(f"Last 5 : {[round(v, 6) for v in text_vec[-5:]]}")

    # ─────────────────────────────────────────
    # embed_query — embeds a search question
    # ─────────────────────────────────────────
    query = "What does the RAG pipeline do?"
    query_vec = embedder.embed_query(query)

    print("\n=== embed_query ===")
    print(f"Input  : '{query}'")
    print(f"Type   : {type(query_vec)}")
    print(f"Dims   : {len(query_vec)}")
    print(f"First 5: {[round(v, 6) for v in query_vec[:5]]}")
    print(f"Last 5 : {[round(v, 6) for v in query_vec[-5:]]}")

    # ─────────────────────────────────────────
    # compare query vs text — should be high
    # ─────────────────────────────────────────
    score = cosine_similarity(query_vec, text_vec)
    print(f"\n=== query vs text similarity ===")
    print(f"Score  : {score:.4f}  ← high = good match")

    # ─────────────────────────────────────────
    # embed_batch — embeds multiple passages
    # ─────────────────────────────────────────
    passages = [
        "Week 10/11 milestones: RAG pipeline accuracy target set for Q4.",
        "The BM25 index catches exact keyword matches that embeddings miss.",
        "The weather in Paris is warm in the summer months.",
    ]
    batch_vecs = embedder.embed_batch(passages)

    print("\n=== embed_batch ===")
    for i, (passage, vec) in enumerate(zip(passages, batch_vecs)):
        sim = cosine_similarity(query_vec, vec)
        print(f"[{i}] dims={len(vec)}  sim_to_query={sim:.4f}  '{passage[:55]}...'")

    # ─────────────────────────────────────────
    # key difference: same text, different task_type
    # ─────────────────────────────────────────
    same_text = "The RAG pipeline retrieves Notion pages."
    as_doc   = embedder.embed_text(same_text)   # task_type = RETRIEVAL_DOCUMENT
    as_query = embedder.embed_query(same_text)  # task_type = RETRIEVAL_QUERY

    diff_score = cosine_similarity(as_doc, as_query)
    print(f"\n=== same text, different task_type ===")
    print(f"Input      : '{same_text}'")
    print(f"doc vs query similarity: {diff_score:.4f}  ← not 1.0, proves task_type changes the vector")


if __name__ == "__main__":
    main()