import os
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings

from src.backend.embedder import GoogleEmbedder
from src.backend.config import get_settings

load_dotenv()

COLLECTION_NAME = "workmate"
#CHROMA_PATH = os.getenv("CHROMA_PATH", "workmate_db/chroma")
CHROMA_PATH = get_settings().CHROMA_PATH


class ChromaManager:
    def __init__(self):
        self.embedder = GoogleEmbedder()
        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"ChromaDB collection '{COLLECTION_NAME}' ready — {self.collection.count()} chunks")

    def add_documents(self, chunks: list[dict]) -> None:
        """
        Embed and upsert a list of chunks into ChromaDB.

        Each chunk dict must have:
            id       — unique string identifier e.g. 'page_abc123_chunk_0'
            text     — the plain text content to embed
            metadata — dict with at least 'parent_id' and 'title'
        """
        if not chunks:
            return

        ids = [c["id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        print(f"Embedding {len(texts)} chunks...")
        embeddings = self.embedder.embed_batch(texts)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        print(f"Upserted {len(ids)} chunks into ChromaDB ✓")

    def query(self, query_text: str, n_results: int = 10) -> list[dict]:
        """
        Search ChromaDB for the top-n chunks most similar to the query.

        Returns a list of dicts with keys: id, text, metadata, score.
        Score is cosine distance — lower means more similar (0.0 = identical).
        """
        query_embedding = self.embedder.embed_query(query_text)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "id":       results["ids"][0][i],
                "text":     results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score":    round(1 - results["distances"][0][i], 4),
            })
        return output

    def get_by_parent(self, parent_id: str) -> list[dict]:
        """
        Fetch all chunks belonging to a single Notion page by its parent_id.
        Used in Week 2 for sibling expansion — if a matched chunk is short,
        fetch its neighbours for more context.
        """
        results = self.collection.get(
            where={"parent_id": {"$eq": parent_id}},
            include=["documents", "metadatas"],
        )

        output = []
        for i in range(len(results["ids"])):
            output.append({
                "id":       results["ids"][i],
                "text":     results["documents"][i],
                "metadata": results["metadatas"][i],
            })
        return output

    def reset(self) -> None:
        """
        Delete and recreate the collection. Wipes all chunks.
        Called by refresh_data.py (Step 1.5) before re-ingesting from Notion.
        """
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"ChromaDB collection reset ✓")

    def count(self) -> int:
        """Return the total number of chunks currently stored."""
        return self.collection.count()