import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

NOTION_API_BASE = "https://api.notion.com/v1"
RATE_LIMIT_DELAY = 0.35   # 3 req/sec free tier — 0.35s keeps safely under


class NotionClient:
    def __init__(self):
        token = os.getenv("NOTION_TOKEN")
        if not token:
            raise ValueError("NOTION_TOKEN not found in environment")

        version = os.getenv("NOTION_API_VERSION", "2022-06-28")

        self.headers = {
            "Authorization": f"Bearer {token}",
            "Notion-Version": version,
            "Content-Type": "application/json",
        }

    # ─────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────

    def _rate_limit(self) -> None:
        """Sleep between requests to stay under the free-tier rate limit."""
        time.sleep(RATE_LIMIT_DELAY)

    def _request(self, method: str, endpoint: str, payload: dict = None) -> dict:
        """
        Make a single authenticated request to the Notion API.
        Handles 429 Too Many Requests by reading Retry-After header.
        Raises on any other non-200 status.
        """
        url = f"{NOTION_API_BASE}/{endpoint}"
        self._rate_limit()

        response = requests.request(
            method=method,
            url=url,
            headers=self.headers,
            json=payload,
        )

        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 10))
            print(f"Rate limited — waiting {retry_after}s before retrying...")
            time.sleep(retry_after)
            return self._request(method, endpoint, payload)

        response.raise_for_status()
        return response.json()

    def _paginate(self, endpoint: str, payload: dict = None, method: str = "POST") -> list[dict]:
        """
        Handle Notion's cursor-based pagination.
        Keeps fetching pages until has_more is False.
        Yields individual result objects one at a time.
        """
        if payload is None:
            payload = {}

        while True:
            data = self._request(method, endpoint, payload)

            for result in data.get("results", []):
                yield result

            if not data.get("has_more"):
                break

            payload["start_cursor"] = data["next_cursor"]

    # ─────────────────────────────────────────
    # Public API methods
    # ─────────────────────────────────────────

    def search(self, filter_type: str = "page") -> list[dict]:
        """
        Search the entire workspace for all pages or databases.
        filter_type: 'page' or 'database'
        """
        payload = {
            "filter": {"property": "object", "value": filter_type},
            "page_size": 100,
        }
        return list(self._paginate("search", payload, method="POST"))

    def get_page(self, page_id: str) -> dict:
        """Fetch metadata for a single page by its ID."""
        return self._request("GET", f"pages/{page_id}")

    def get_block_children(self, block_id: str) -> list[dict]:
        """
        Fetch all child blocks of a page or block.
        Uses GET with pagination — block children use query params not body.
        """
        results = []
        endpoint = f"blocks/{block_id}/children"
        cursor = None

        while True:
            self._rate_limit()
            url = f"{NOTION_API_BASE}/{endpoint}"
            params = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor

            response = requests.get(url, headers=self.headers, params=params)

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 10))
                print(f"Rate limited — waiting {retry_after}s...")
                time.sleep(retry_after)
                continue

            response.raise_for_status()
            data = response.json()
            results.extend(data.get("results", []))

            if not data.get("has_more"):
                break
            cursor = data["next_cursor"]

        return results

    def query_database(self, database_id: str) -> list[dict]:
        """
        Fetch all rows from a Notion database.
        Each row is a page object with properties.
        """
        payload = {"page_size": 100}
        return list(self._paginate(f"databases/{database_id}/query", payload, method="POST"))