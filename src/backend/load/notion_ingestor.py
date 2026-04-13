import json
import os
from dotenv import load_dotenv
from src.backend.load.bm25_manager import BM25Manager

from src.backend.chroma_manager import ChromaManager
from src.backend.embedder import GoogleEmbedder

load_dotenv()

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
NOTION_DATA_PATH = os.getenv("NOTION_DATA_PATH", "src/data/notion_data.json")
BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "workmate_db/bm25_index.pkl")


class NotionIngestor:
    """
    Reads notion_data.json, chunks each document, embeds the chunks,
    and stores them in ChromaDB + BM25.
    """

    def __init__(self):
        self.chroma = ChromaManager()
        self.embedder = GoogleEmbedder()

    def chunk_document(self, doc: dict) -> list[dict]:
        """
        Split a document's content into overlapping chunks.
        Each chunk inherits the parent document's metadata.

        chunk_size=500    — short enough for precise retrieval
        chunk_overlap=100 — prevents sentences split at boundaries losing context
        """
        content = doc.get("content", "").strip()
        if not content:
            return []

        title = doc.get("title", "Untitled")
        doc_id = doc.get("id", "unknown")
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(content):
            end = start + CHUNK_SIZE
            chunk_text = content[start:end].strip()

            if chunk_text:
                chunks.append({
                    "id": f"{doc_id}_chunk_{chunk_index}",
                    "text": f"{title}\n{chunk_text}",
                    "metadata": {
                        "parent_id":         doc_id,
                        "title":             title,
                        "url":               doc.get("url", ""),
                        "source_type":       doc.get("source_type", "page"),
                        "created_time":      doc.get("created_time", ""),
                        "last_edited_time":  doc.get("last_edited_time", ""),
                        "chunk_index":       chunk_index,
                    },
                })
                chunk_index += 1

            if end >= len(content):
                break
            start = end - CHUNK_OVERLAP

        return chunks

    def _load_data(self) -> list[dict]:
        """Load documents from notion_data.json."""
        if not os.path.exists(NOTION_DATA_PATH):
            raise FileNotFoundError(
                f"notion_data.json not found at {NOTION_DATA_PATH}. "
                f"Run refresh_data.py first."
            )
        with open(NOTION_DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_bm25(self, chunks: list[dict]) -> None:
        """
        Build a BM25 index from all chunks and save to disk.
        Tokenises as '{title} {chunk_text}'.lower().split()
        so title keywords boost keyword search results.
        """
        bm25_manager = BM25Manager()
        bm25_manager.build_index(chunks)
        bm25_manager.save(BM25_INDEX_PATH)

    def run_pipeline(self, reset: bool = False) -> None:
        """
        Full ingestion pipeline:
        1. Load notion_data.json
        2. Chunk each document
        3. Reset ChromaDB if requested
        4. Embed + upsert all chunks into ChromaDB
        5. Build and save BM25 index
        """
        print("\n=== Ingesting into ChromaDB + BM25 ===")
        documents = self._load_data()
        print(f"  Loaded {len(documents)} documents from JSON")

        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        print(f"  Total chunks: {len(all_chunks)}")

        if reset:
            print("  Resetting ChromaDB collection...")
            self.chroma.reset()

        print("  Embedding and upserting into ChromaDB...")
        for i, chunk in enumerate(all_chunks):
            print(f"  [{i+1:03}/{len(all_chunks)}] '{chunk['metadata']['title']}' — chunk {chunk['metadata']['chunk_index']}")
            self.chroma.add_documents([chunk])

        print(f"\n  ChromaDB chunks : {self.chroma.count()}")

        self._build_bm25(all_chunks)
        print(f"  BM25 documents  : {len(all_chunks)}")
        print("\n  Pipeline complete — ready to query!")