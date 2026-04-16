import os
from dotenv import load_dotenv

from src.backend.chroma_manager import ChromaManager
from src.backend.embedder import GoogleEmbedder
from src.backend.load.bm25_manager import BM25Manager

load_dotenv()

BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "workmate_db/bm25_index.pkl")


class HybridRetriever:
    """
    Combines ChromaDB vector search and BM25 keyword search
    using Reciprocal Rank Fusion (RRF) to produce a single ranked list.

    Neither method alone is sufficient:
    - Vector search misses exact keyword matches (acronyms, version numbers, page titles)
    - BM25 misses semantic similarity (paraphrases, intent-based queries)
    RRF merges both by rank position — scale differences don't matter.
    """

    def __init__(self):
        self.embedder = GoogleEmbedder()
        self.chroma = ChromaManager()
        self.bm25 = BM25Manager()
        self.bm25.load(BM25_INDEX_PATH)
        print("HybridRetriever ready ✓")

    def retrieve(self, query: str, top_k: int = 15) -> list[dict]:
        """
        Run hybrid retrieval for a query.

        Steps:
        1. Embed the query with GoogleEmbedder
        2. Run ChromaDB vector search (top 30)
        3. Run BM25 keyword search (top 30)
        4. Merge both ranked lists with RRF (k=60)
        5. Return top_k results sorted by RRF score descending

        Each result dict has keys:
            id, text, metadata, rrf_score,
            vector_rank, bm25_rank (None if not in that list)
        """
        # Step 1 — vector search
        vector_results = self.chroma.query(query, n_results=30)

        # Step 2 — keyword search
        bm25_results = self.bm25.search(query, n_results=30)

        # Step 3 — merge with RRF
        merged = self._rrf_merge(vector_results, bm25_results, k=60)

        return merged[:top_k]

    def _rrf_merge(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion.

        For each document, compute:
            rrf_score = 1/(k + rank_in_vector) + 1/(k + rank_in_bm25)

        Documents only in one list contribute only one term.
        Documents in both lists get both terms added — they rise to the top.
        Results sorted by rrf_score descending.

        k=60 is the standard dampening constant — prevents rank-1 from
        dominating too heavily over rank-2 or rank-3.
        """
        scores: dict[str, dict] = {}

        # Add vector ranks
        for rank, result in enumerate(vector_results, start=1):
            doc_id = result["id"]
            scores[doc_id] = {
                "id":           doc_id,
                "text":         result["text"],
                "metadata":     result["metadata"],
                "rrf_score":    1 / (k + rank),
                "vector_rank":  rank,
                "bm25_rank":    None,
            }

        # Add BM25 ranks — sum scores if document already seen from vector
        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result["id"]
            if doc_id in scores:
                scores[doc_id]["rrf_score"] += 1 / (k + rank)
                scores[doc_id]["bm25_rank"] = rank
            else:
                scores[doc_id] = {
                    "id":           doc_id,
                    "text":         result["text"],
                    "metadata":     result["metadata"],
                    "rrf_score":    1 / (k + rank),
                    "vector_rank":  None,
                    "bm25_rank":    rank,
                }

        # Sort by RRF score descending
        merged = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)
        return merged