from src.backend.notion.client import NotionClient
from src.backend.notion.block_parser import BlockParser
from src.backend.notion.models.document import NotionDocument


class PageFetcher:
    """
    Sits on top of NotionClient.
    Converts raw Notion API page responses into clean NotionDocument objects.
    """

    def __init__(self, client: NotionClient):
        self.client = client
        self.parser = BlockParser()

    def fetch_all_pages(self) -> list[NotionDocument]:
        """
        Search the entire workspace for all pages.
        Returns a list of NotionDocument objects.
        """
        raw_pages = self.client.search(filter_type="page")
        documents = []

        for raw_page in raw_pages:
            try:
                doc = self.fetch_page(raw_page["id"])
                if doc:
                    documents.append(doc)
                    print(f"  Fetched: '{doc.title}'")
            except Exception as e:
                print(f"  Skipped page {raw_page['id']}: {e}")

        return documents

    def fetch_page(self, page_id: str) -> NotionDocument | None:
        """
        Fetch a single page by ID.
        Gets metadata, extracts title and parent, fetches and parses all blocks.
        Returns None if the page has no content.
        """
        page_data = self.client.get_page(page_id)

        title = self._extract_title(page_data)
        parent_id = self._extract_parent_id(page_data)

        blocks = self._fetch_blocks_recursive(page_id)
        parsed_lines = []
        for block in blocks:
            text = self.parser.parse_block(block)
            if text.strip():
                parsed_lines.append(text)

        content = "\n".join(parsed_lines)

        if not content.strip():
            return None

        return NotionDocument(
            id=page_id,
            title=title,
            content=content,
            parent_id=parent_id,
            url=page_data.get("url", ""),
            source_type=page_data.get("object", "page"),
            properties=page_data.get("properties", {}),
            created_time=page_data.get("created_time", ""),
            last_edited_time=page_data.get("last_edited_time", ""),
        )

    def _fetch_blocks_recursive(
        self,
        block_id: str,
        depth: int = 0,
        max_depth: int = 5,
    ) -> list[dict]:
        """
        Recursively fetch all blocks under a given block or page.
        Caps at max_depth=5 to prevent infinite loops on deeply nested pages.
        """
        if depth > max_depth:
            return []

        blocks = self.client.get_block_children(block_id)
        all_blocks = []

        for block in blocks:
            all_blocks.append(block)
            if block.get("has_children"):
                child_blocks = self._fetch_blocks_recursive(
                    block["id"],
                    depth=depth + 1,
                    max_depth=max_depth,
                )
                all_blocks.extend(child_blocks)

        return all_blocks

    def _extract_title(self, page_data: dict) -> str:
        """
        Extract the page title from the properties dict.
        The title property has type='title' and contains a rich_text array.
        """
        properties = page_data.get("properties", {})
        for prop_name, prop_value in properties.items():
            if prop_value.get("type") == "title":
                rich_text = prop_value.get("title", [])
                return "".join(rt.get("plain_text", "") for rt in rich_text)
        return "Untitled"

    def _extract_parent_id(self, page_data: dict) -> str:
        """
        Extract the parent ID from the page data.
        Parent can be a page, a database, or the workspace root.
        """
        parent = page_data.get("parent", {})
        parent_type = parent.get("type", "")

        if parent_type == "page_id":
            return parent["page_id"]
        elif parent_type == "database_id":
            return parent["database_id"]
        elif parent_type == "workspace":
            return "workspace"
        else:
            return "unknown"