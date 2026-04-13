import os
import pickle
from rank_bm25 import BM25Okapi


class BM25Manager:
    """
    Standalone BM25 keyword search index.
    Built from chunked Notion documents, saved to disk as a pickle file.
    Used by HybridRetriever (Week 2) alongside ChromaDB vector search.
    """

    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunks: list[dict] = []

    # ─────────────────────────────────────────
    # Build
    # ─────────────────────────────────────────

    def build_index(self, chunks: list[dict]) -> None:
        """
        Build a BM25 index from a list of chunk dicts.

        Tokenises each chunk as '{title} {text}'.lower().split()
        so title keywords appear in every chunk from that page,
        boosting keyword search for page name lookups.

        Each chunk dict must have:
            text     — the chunk content
            metadata — dict with at least 'title'
        """
        if not chunks:
            raise ValueError("Cannot build BM25 index from empty chunk list")

        self.chunks = chunks

        tokenised = [
            f"{c['metadata']['title']} {c['text']}".lower().split()
            for c in chunks
        ]
        self.bm25 = BM25Okapi(tokenised)
        print(f"BM25 index built with {len(chunks)} documents ✓")

    # ─────────────────────────────────────────
    # Search
    # ─────────────────────────────────────────

    def search(self, query: str, n_results: int = 15) -> list[dict]:
        """
        Search the BM25 index for the top-n matching chunks.

        Returns a list of dicts with keys:
            id       — chunk id
            text     — chunk text
            metadata — chunk metadata
            score    — raw BM25 score (higher = more relevant)

        Results are sorted by score descending.
        Chunks with score=0 (no keyword overlap) are excluded.
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call build_index() or load() first.")

        tokenised_query = query.lower().split()
        scores = self.bm25.get_scores(tokenised_query)

        scored_chunks = [
            {
                "id":       self.chunks[i]["id"],
                "text":     self.chunks[i]["text"],
                "metadata": self.chunks[i]["metadata"],
                "score":    round(float(scores[i]), 4),
            }
            for i in range(len(self.chunks))
            if scores[i] > 0
        ]

        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:n_results]

    # ─────────────────────────────────────────
    # Persist
    # ─────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Pickle the full BM25 payload — index + chunks — to disk.
        Creates the directory if it doesn't exist.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        payload = {
            "bm25":   self.bm25,
            "chunks": self.chunks,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

        print(f"BM25 index saved → {path}")

    def load(self, path: str) -> None:
        """
        Load a pickled BM25 payload from disk.
        Restores both the fitted BM25 model and the original chunks.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"BM25 index not found at {path}. "
                f"Run refresh_data.py first."
            )

        with open(path, "rb") as f:
            payload = pickle.load(f)

        self.bm25 = payload["bm25"]
        self.chunks = payload["chunks"]
        print(f"BM25 index loaded ← {path}  ({len(self.chunks)} documents)")

    # ─────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────

    def count(self) -> int:
        """Return the number of documents in the index."""
        return len(self.chunks)

    def rebuild_index(self, chunks: list[dict], path: str) -> None:
        """
        Rebuild the index from a new chunk list and immediately save.
        Called in Week 4 after a file upload adds new chunks to ChromaDB —
        ensures BM25 stays in sync with ChromaDB.
        """
        self.build_index(chunks)
        self.save(path)