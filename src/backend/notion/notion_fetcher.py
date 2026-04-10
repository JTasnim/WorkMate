import json
import os
from dotenv import load_dotenv

from src.backend.notion.client import NotionClient
from src.backend.notion.page_fetcher import PageFetcher
from src.backend.notion.models.document import NotionDocument

load_dotenv()


class NotionFetcher:
    """
    Orchestrates fetching all content from the Notion workspace.
    Combines pages and database rows into a single list of NotionDocuments.
    Saves results to notion_data.json.
    """

    def __init__(self):
        self.client = NotionClient()
        self.page_fetcher = PageFetcher(self.client)

    def fetch_all(self, include_database_content: bool = True) -> list[NotionDocument]:
        """
        Fetch all pages and optionally all database rows from the workspace.
        Returns a combined list of NotionDocument objects.
        """
        all_documents = []

        # Fetch all pages
        print("  Fetching pages...")
        pages = self.page_fetcher.fetch_all_pages()
        all_documents.extend(pages)
        print(f"  Pages fetched: {len(pages)}")

        # Fetch all database rows
        if include_database_content:
            print("  Fetching databases...")
            databases = self.client.search(filter_type="database")
            for db in databases:
                try:
                    rows = self.client.query_database(db["id"])
                    for row in rows:
                        try:
                            doc = self.page_fetcher.fetch_page(row["id"])
                            if doc:
                                doc.source_type = "database"
                                all_documents.append(doc)
                                print(f"  Fetched DB row: '{doc.title}'")
                        except Exception as e:
                            print(f"  Skipped DB row {row['id']}: {e}")
                except Exception as e:
                    print(f"  Skipped database {db['id']}: {e}")

        return all_documents

    def save_to_json(self, documents: list[NotionDocument], filepath: str) -> None:
        """
        Serialise all NotionDocument objects to JSON and save to disk.
        Creates the directory if it doesn't exist.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        data = [doc.to_dict() for doc in documents]

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"  Saved {len(data)} documents to {filepath}")