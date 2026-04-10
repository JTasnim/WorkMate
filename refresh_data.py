import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.backend.notion.notion_fetcher import NotionFetcher
from src.backend.load.notion_ingestor import NotionIngestor

NOTION_DATA_PATH = os.getenv("NOTION_DATA_PATH", "src/data/notion_data.json")


def main():
    # ── Step 1: Fetch from Notion API ──────────────────────────
    print("=== Step 1: Fetching from Notion API ===")
    fetcher = NotionFetcher()
    documents = fetcher.fetch_all(include_database_content=True)
    fetcher.save_to_json(documents, NOTION_DATA_PATH)
    print(f"  Total documents fetched: {len(documents)}")

    # ── Step 2: Ingest into ChromaDB + BM25 ────────────────────
    ingestor = NotionIngestor()
    ingestor.run_pipeline(reset=True)


if __name__ == "__main__":
    main()